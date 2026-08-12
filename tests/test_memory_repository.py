from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pytest

from stock_agent.memory.repository import (
    CACHE_VERSION,
    MemoryRepositoryError,
    SQLiteMemory,
    market_data_ttl,
)


@dataclass
class MutableClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current

    def advance(self, **delta: float) -> None:
        self.current += timedelta(**delta)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock(datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc))


def _columns(database: object, table: str) -> list[str]:
    with sqlite3.connect(database) as connection:
        return [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]


def _row_count(database: object, table: str) -> int:
    with sqlite3.connect(database) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_schema_contains_only_the_four_anonymous_cache_tables(tmp_path, clock) -> None:
    database = tmp_path / "memory.db"
    repo = SQLiteMemory(database, clock=clock)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE ?",
                ("table", "sqlite_%"),
            )
        }

    assert tables == {
        "analysis_cache",
        "ai_cache",
        "market_cache",
        "stock_master_cache",
    }
    expected_columns = ["cache_key", "payload", "created_at", "expires_at", "version"]
    for table in tables:
        assert _columns(database, table) == expected_columns

    repo.close()


def test_upsert_is_stable_and_stores_compact_json(tmp_path, clock) -> None:
    database = tmp_path / "memory.db"
    repo = SQLiteMemory(database, clock=clock)
    repo.put_analysis("same", {"summary": "旧", "nested": {"score": 1}})
    clock.advance(seconds=1)

    repo.put_analysis("same", {"summary": "新", "nested": {"score": 2}})

    assert repo.get_analysis("same") == {"nested": {"score": 2}, "summary": "新"}
    assert _row_count(database, "analysis_cache") == 1
    with sqlite3.connect(database) as connection:
        payload = connection.execute(
            "SELECT payload FROM analysis_cache WHERE cache_key = ?", ("same",)
        ).fetchone()[0]
    assert payload == '{"nested":{"score":2},"summary":"新"}'
    repo.close()


def test_cache_keys_are_bound_parameters(tmp_path, clock) -> None:
    database = tmp_path / "memory.db"
    repo = SQLiteMemory(database, clock=clock)
    hostile_key = "x'); DROP TABLE ai_cache; --"

    repo.put_ai(hostile_key, {"summary": "safe"})

    assert repo.get_ai(hostile_key) == {"summary": "safe"}
    assert _columns(database, "ai_cache")
    repo.close()


def test_exact_expiry_is_stale_and_evicts_only_that_row(tmp_path, clock) -> None:
    database = tmp_path / "memory.db"
    repo = SQLiteMemory(database, clock=clock)
    repo.put_ai("expires", {"summary": "x"}, ttl=timedelta(seconds=10))
    repo.put_ai("fresh", {"summary": "y"}, ttl=timedelta(seconds=11))
    clock.advance(seconds=10)

    assert repo.get_ai("expires") is None
    assert repo.get_ai("fresh") == {"summary": "y"}
    assert _row_count(database, "ai_cache") == 1
    repo.close()


@pytest.mark.parametrize(
    ("put_name", "get_name", "ttl"),
    [
        ("put_analysis", "get_analysis", timedelta(days=90)),
        ("put_ai", "get_ai", timedelta(days=30)),
        ("put_stock_master", "get_stock_master", timedelta(hours=24)),
    ],
)
def test_default_cache_ttls(
    tmp_path,
    clock,
    put_name: str,
    get_name: str,
    ttl: timedelta,
) -> None:
    repo = SQLiteMemory(tmp_path / f"{put_name}.db", clock=clock)
    getattr(repo, put_name)("key", {"value": put_name})
    clock.advance(seconds=ttl.total_seconds() - 1)
    assert getattr(repo, get_name)("key") == {"value": put_name}

    clock.advance(seconds=1)
    assert getattr(repo, get_name)("key") is None
    repo.close()


def test_latest_market_request_uses_six_hours_even_on_weekend(tmp_path) -> None:
    saturday = MutableClock(datetime(2026, 8, 15, 9, 30, tzinfo=timezone.utc))
    assert market_data_ttl(requested_end_date=None, now=saturday()) == timedelta(hours=6)
    assert market_data_ttl(requested_end_date=saturday().date(), now=saturday()) == timedelta(
        hours=6
    )

    repo = SQLiteMemory(tmp_path / "weekend.db", clock=saturday)
    repo.put_market("latest", {"last_trade_date": "2026-08-14"})
    saturday.advance(hours=6)
    assert repo.get_market("latest") is None
    repo.close()


def test_explicit_historical_market_request_uses_thirty_days(tmp_path, clock) -> None:
    requested_end_date = date(2026, 8, 11)
    assert market_data_ttl(requested_end_date=requested_end_date, now=clock()) == timedelta(days=30)

    repo = SQLiteMemory(tmp_path / "history.db", clock=clock)
    repo.put_market(
        "history",
        {"last_trade_date": "2026-08-11"},
        requested_end_date=requested_end_date,
    )
    clock.advance(days=30)
    assert repo.get_market("history") is None
    repo.close()


def test_cleanup_removes_all_and_only_expired_rows(tmp_path, clock) -> None:
    database = tmp_path / "memory.db"
    repo = SQLiteMemory(database, clock=clock)
    repo.put_analysis("expired-analysis", {"value": 1}, ttl=timedelta(seconds=5))
    repo.put_ai("expired-ai", {"value": 2}, ttl=timedelta(seconds=5))
    repo.put_market("fresh-market", {"value": 3}, ttl=timedelta(seconds=6))
    repo.put_stock_master("fresh-master", {"value": 4}, ttl=timedelta(seconds=6))
    clock.advance(seconds=5)

    assert repo.cleanup_expired() == 2
    assert repo.get_analysis("expired-analysis") is None
    assert repo.get_ai("expired-ai") is None
    assert repo.get_market("fresh-market") == {"value": 3}
    assert repo.get_stock_master("fresh-master") == {"value": 4}
    repo.close()


def test_corrupt_json_is_evicted_without_disturbing_other_rows(tmp_path, clock) -> None:
    database = tmp_path / "memory.db"
    repo = SQLiteMemory(database, clock=clock)
    repo.put_analysis("corrupt", {"value": 1})
    repo.put_analysis("healthy", {"value": 2})
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE analysis_cache SET payload = ? WHERE cache_key = ?",
            ("{broken", "corrupt"),
        )

    assert repo.get_analysis("corrupt") is None
    assert repo.get_analysis("healthy") == {"value": 2}
    assert _row_count(database, "analysis_cache") == 1
    repo.close()


def test_old_version_is_evicted_without_disturbing_other_rows(tmp_path, clock) -> None:
    database = tmp_path / "memory.db"
    repo = SQLiteMemory(database, clock=clock)
    repo.put_ai("old", {"value": 1})
    repo.put_ai("current", {"value": 2})
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ai_cache SET version = ? WHERE cache_key = ?",
            (CACHE_VERSION - 1, "old"),
        )

    assert repo.get_ai("old") is None
    assert repo.get_ai("current") == {"value": 2}
    assert _row_count(database, "ai_cache") == 1
    repo.close()


def test_close_is_idempotent_and_subsequent_failures_are_safe(tmp_path, clock) -> None:
    repo = SQLiteMemory(tmp_path / "memory.db", clock=clock)
    repo.close()
    repo.close()

    with pytest.raises(MemoryRepositoryError, match="memory repository is unavailable"):
        repo.get_analysis("key")
