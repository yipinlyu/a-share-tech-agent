"""SQLite-backed anonymous caches with explicit lifecycle rules."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

CACHE_VERSION: Final = 1
ANALYSIS_TTL: Final = timedelta(days=90)
AI_TTL: Final = timedelta(days=30)
LATEST_MARKET_TTL: Final = timedelta(hours=6)
HISTORICAL_MARKET_TTL: Final = timedelta(days=30)
STOCK_MASTER_TTL: Final = timedelta(hours=24)

_TABLES: Final = (
    "analysis_cache",
    "ai_cache",
    "market_cache",
    "stock_master_cache",
)
_PROCESS_WRITE_LOCK = threading.RLock()
_UNAVAILABLE_MESSAGE = "memory repository is unavailable"


class MemoryRepositoryError(RuntimeError):
    """A cache failure callers may catch without exposing SQLite details."""


def market_data_ttl(
    requested_end_date: date | datetime | str | None = None,
    *,
    now: datetime,
) -> timedelta:
    """Choose TTL from request intent, independent of the last trading day."""

    requested = _coerce_date(requested_end_date)
    current = _coerce_datetime(now).date()
    if requested is None or requested == current:
        return LATEST_MARKET_TTL
    return HISTORICAL_MARKET_TTL


class SQLiteMemory:
    """Own four anonymous JSON caches and their SQLite connection lifecycle."""

    def __init__(
        self,
        database: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._connection_lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

        try:
            if str(database) != ":memory:":
                Path(database).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                str(database),
                timeout=5.0,
                check_same_thread=False,
            )
            connection.execute("PRAGMA busy_timeout = 5000")
            self._connection = connection
            self._enable_wal_when_supported()
            self._create_schema()
            self.cleanup_expired()
        except MemoryRepositoryError:
            self.close()
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            self.close()
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc

    def put_analysis(
        self,
        cache_key: str,
        payload: Any,
        *,
        ttl: timedelta = ANALYSIS_TTL,
    ) -> None:
        self._put("analysis_cache", cache_key, payload, ttl)

    def get_analysis(self, cache_key: str) -> Any | None:
        return self._get("analysis_cache", cache_key)

    def put_ai(
        self,
        cache_key: str,
        payload: Any,
        *,
        ttl: timedelta = AI_TTL,
    ) -> None:
        self._put("ai_cache", cache_key, payload, ttl)

    def get_ai(self, cache_key: str) -> Any | None:
        return self._get("ai_cache", cache_key)

    def put_market(
        self,
        cache_key: str,
        payload: Any,
        *,
        requested_end_date: date | datetime | str | None = None,
        ttl: timedelta | None = None,
    ) -> None:
        now = self._now()
        effective_ttl = (
            ttl if ttl is not None else market_data_ttl(requested_end_date, now=now)
        )
        self._put("market_cache", cache_key, payload, effective_ttl, now=now)

    def get_market(self, cache_key: str) -> Any | None:
        return self._get("market_cache", cache_key)

    def put_stock_master(
        self,
        cache_key: str,
        payload: Any,
        *,
        ttl: timedelta = STOCK_MASTER_TTL,
    ) -> None:
        self._put("stock_master_cache", cache_key, payload, ttl)

    def get_stock_master(self, cache_key: str) -> Any | None:
        return self._get("stock_master_cache", cache_key)

    def cleanup_expired(self) -> int:
        """Delete rows whose expiry is not strictly later than the injected clock."""

        now = self._timestamp(self._now())
        deleted = 0
        connection = self._require_connection()
        try:
            with _PROCESS_WRITE_LOCK, self._connection_lock, connection:
                for table in _TABLES:
                    cursor = connection.execute(
                        f"DELETE FROM {table} WHERE expires_at <= ?",
                        (now,),
                    )
                    deleted += max(cursor.rowcount, 0)
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc
        return deleted

    def close(self) -> None:
        """Close the owned connection; repeated closes are harmless."""

        with _PROCESS_WRITE_LOCK, self._connection_lock:
            connection, self._connection = self._connection, None
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass

    def __enter__(self) -> SQLiteMemory:
        self._require_connection()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _enable_wal_when_supported(self) -> None:
        connection = self._require_connection()
        try:
            with _PROCESS_WRITE_LOCK, self._connection_lock:
                connection.execute("PRAGMA journal_mode = WAL").fetchone()
        except sqlite3.Error:
            # In-memory and restricted filesystems may not support WAL.
            pass

    def _create_schema(self) -> None:
        connection = self._require_connection()
        try:
            with _PROCESS_WRITE_LOCK, self._connection_lock, connection:
                for table in _TABLES:
                    connection.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {table} (
                            cache_key TEXT PRIMARY KEY,
                            payload TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            expires_at REAL NOT NULL,
                            version INTEGER NOT NULL
                        )
                        """
                    )
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc

    def _put(
        self,
        table: str,
        cache_key: str,
        payload: Any,
        ttl: timedelta,
        *,
        now: datetime | None = None,
    ) -> None:
        _require_table(table)
        key = _validate_cache_key(cache_key)
        serialized = _compact_json(payload)
        created_at = self._timestamp(now or self._now())
        try:
            ttl_seconds = ttl.total_seconds()
        except (AttributeError, OverflowError) as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc
        if not math.isfinite(ttl_seconds):
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE)
        expires_at = created_at + ttl_seconds

        connection = self._require_connection()
        try:
            with _PROCESS_WRITE_LOCK, self._connection_lock, connection:
                connection.execute(
                    f"""
                    INSERT INTO {table} (
                        cache_key, payload, created_at, expires_at, version
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload = excluded.payload,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        version = excluded.version
                    """,
                    (key, serialized, created_at, expires_at, CACHE_VERSION),
                )
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc

    def _get(self, table: str, cache_key: str) -> Any | None:
        _require_table(table)
        key = _validate_cache_key(cache_key)
        connection = self._require_connection()
        try:
            with self._connection_lock:
                row = connection.execute(
                    f"SELECT payload, expires_at, version FROM {table} WHERE cache_key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc

        if row is None:
            return None
        payload, expires_at, version = row
        try:
            fresh = self._timestamp(self._now()) < float(expires_at)
            current_version = int(version) == CACHE_VERSION
            decoded = _load_json(payload) if fresh and current_version else None
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
            fresh = False
            current_version = False

        if not fresh or not current_version or decoded is None:
            self._evict(table, key)
            return None
        return decoded

    def _evict(self, table: str, cache_key: str) -> None:
        connection = self._require_connection()
        try:
            with _PROCESS_WRITE_LOCK, self._connection_lock, connection:
                connection.execute(
                    f"DELETE FROM {table} WHERE cache_key = ?",
                    (cache_key,),
                )
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE)
        return self._connection

    def _now(self) -> datetime:
        try:
            return _coerce_datetime(self._clock())
        except (TypeError, ValueError, OverflowError) as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc

    @staticmethod
    def _timestamp(value: datetime) -> float:
        try:
            result = value.timestamp()
        except (AttributeError, OSError, OverflowError, ValueError) as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc
        if not math.isfinite(result):
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE)
        return result


def _require_table(table: str) -> None:
    if table not in _TABLES:
        raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE)


def _validate_cache_key(cache_key: str) -> str:
    if not isinstance(cache_key, str) or not cache_key:
        raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE)
    return cache_key


def _compact_json(payload: Any) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc


def _load_json(payload: object) -> Any:
    if not isinstance(payload, (str, bytes, bytearray)):
        raise TypeError("invalid JSON payload")

    def reject_non_finite(value: str) -> None:
        raise ValueError(value)

    return json.loads(payload, parse_constant=reject_non_finite)


def _coerce_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("clock must return datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE) from exc
    raise MemoryRepositoryError(_UNAVAILABLE_MESSAGE)
