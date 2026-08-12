"""Injected Tushare adapter with point-in-time qfq and strict quality checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pandas.errors import MergeError

from stock_agent.domain.models import AgentError, DataQuality

STOCK_BASIC_FIELDS = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date"
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
ADJ_FACTOR_FIELDS = "ts_code,trade_date,adj_factor"

_STOCK_COLUMNS = STOCK_BASIC_FIELDS.split(",")
_DAILY_COLUMNS = DAILY_FIELDS.split(",")
_FACTOR_COLUMNS = ADJ_FACTOR_FIELDS.split(",")
_PRICE_COLUMNS = ["open", "high", "low", "close", "pre_close"]
_NUMERIC_DAILY_COLUMNS = [*_PRICE_COLUMNS, "change", "pct_chg", "vol", "amount"]
_EXCHANGES = ("SSE", "SZSE", "BSE")
_DATE_FORMAT = "%Y%m%d"
_STOCK_RESPONSE_LIMIT = 6000

_TOKEN_LIKE = re.compile(r"(?i)\b(?:sk-[a-z0-9_-]+|bearer\s+\S+|token\s*[:=]\s*\S+)")
_AUTHORIZATION = re.compile(r"(?i)authorization\s*:\s*\S+(?:\s+\S+)?")


@runtime_checkable
class ProLike(Protocol):
    """Only the Tushare Pro operations the application is allowed to call."""

    def stock_basic(self, **kwargs: object) -> pd.DataFrame: ...

    def daily(self, **kwargs: object) -> pd.DataFrame: ...

    def adj_factor(self, **kwargs: object) -> pd.DataFrame: ...


class TushareAdapterError(Exception):
    """Safe adapter failure carrying the application's structured error contract."""

    def __init__(self, error: AgentError) -> None:
        self.error = error
        super().__init__(error.user_message)


@dataclass(frozen=True)
class DailyDataResult:
    """Full indicator input plus a report describing the display window."""

    frame: pd.DataFrame
    quality: DataQuality


class TushareDataClient:
    """Tushare adapter that never mutates global SDK authentication state."""

    def __init__(self, pro: ProLike | None = None, *, token: str | None = None) -> None:
        if pro is not None:
            self.pro = pro
            return

        clean_token = token.strip() if token is not None else ""
        if not clean_token:
            raise _adapter_error("CONFIG", "未配置 Tushare Token，无法获取股票数据。", False)

        try:
            import tushare as ts

            self.pro = ts.pro_api(token=clean_token, timeout=30)
        except TushareAdapterError:
            raise
        except BaseException as exc:
            raise _map_upstream_error(exc) from None

    def fetch_stock_master(self) -> pd.DataFrame:
        """Fetch and normalize the complete listed-stock master from all A-share exchanges."""

        frames: list[pd.DataFrame] = []
        for exchange in _EXCHANGES:
            try:
                response = self.pro.stock_basic(
                    exchange=exchange,
                    list_status="L",
                    fields=STOCK_BASIC_FIELDS,
                )
            except BaseException as exc:
                raise _map_upstream_error(exc) from None

            frame = _require_dataframe(response, endpoint="股票主数据")
            _require_columns(frame, _STOCK_COLUMNS, endpoint="股票主数据")
            if len(frame) >= _STOCK_RESPONSE_LIMIT:
                raise _adapter_error(
                    "DATA",
                    "股票主数据可能被服务端截断，请稍后重试。",
                    False,
                )
            frames.append(frame.loc[:, _STOCK_COLUMNS].copy(deep=True))

        stock_master = pd.concat(frames, ignore_index=True)
        stock_master = _deduplicate(stock_master, ["ts_code"], label="股票主数据")
        if stock_master.empty:
            return _empty_stock_master()

        for column in [
            "ts_code",
            "symbol",
            "name",
            "area",
            "industry",
            "market",
            "exchange",
            "list_status",
        ]:
            stock_master[column] = stock_master[column].astype("string").str.strip()
        stock_master["ts_code"] = stock_master["ts_code"].str.upper()
        stock_master["exchange"] = stock_master["exchange"].str.upper()
        stock_master["list_status"] = stock_master["list_status"].str.upper()
        stock_master["list_date"] = _parse_date_series(
            stock_master["list_date"], label="股票主数据上市日期"
        )

        required_text = ["ts_code", "symbol", "name", "market", "exchange", "list_status"]
        if (
            stock_master[required_text].isna().any().any()
            or (stock_master[required_text] == "").any().any()
        ):
            raise _adapter_error("DATA", "股票主数据含无效必要字段。", False)
        if not stock_master["exchange"].isin(_EXCHANGES).all():
            raise _adapter_error("DATA", "股票主数据含无法识别的交易所。", False)
        if not stock_master["list_status"].eq("L").all():
            raise _adapter_error("DATA", "股票主数据含非上市状态记录。", False)

        return stock_master.sort_values("ts_code", kind="stable").reset_index(drop=True)

    def fetch_daily(
        self,
        ts_code: str,
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Fetch inclusive raw daily data and construct point-in-time qfq prices."""

        start, end = _validate_request_dates(start_date, end_date)
        code = ts_code.strip().upper()
        if not code:
            raise _adapter_error("VALIDATION", "股票代码不能为空。", False)

        request_kwargs = {
            "ts_code": code,
            "start_date": start.strftime(_DATE_FORMAT),
            "end_date": end.strftime(_DATE_FORMAT),
        }
        try:
            raw_daily = self.pro.daily(**request_kwargs, fields=DAILY_FIELDS)
        except BaseException as exc:
            raise _map_upstream_error(exc) from None

        daily = _require_dataframe(raw_daily, endpoint="日线行情")
        _require_columns(daily, _DAILY_COLUMNS, endpoint="日线行情")
        daily = daily.loc[:, _DAILY_COLUMNS].copy(deep=True)
        if daily.empty:
            return _empty_daily_frame()

        try:
            raw_factors = self.pro.adj_factor(**request_kwargs, fields=ADJ_FACTOR_FIELDS)
        except BaseException as exc:
            raise _map_upstream_error(exc) from None

        factors = _require_dataframe(raw_factors, endpoint="复权因子")
        _require_columns(factors, _FACTOR_COLUMNS, endpoint="复权因子")
        factors = factors.loc[:, _FACTOR_COLUMNS].copy(deep=True)

        daily = _normalize_daily(daily, requested_code=code, start=start, end=end)
        factors = _normalize_factors(factors, requested_code=code, start=start, end=end)

        try:
            merged = daily.merge(
                factors,
                on=["ts_code", "trade_date"],
                how="left",
                validate="one_to_one",
                sort=False,
            )
        except MergeError:
            raise _adapter_error("DATA", "日线行情与复权因子无法唯一对应。", False) from None

        if merged["adj_factor"].isna().any():
            raise _adapter_error("DATA", "部分交易日缺少复权因子，无法安全计算前复权行情。", False)

        anchor = merged.loc[merged["trade_date"] <= pd.Timestamp(end)].iloc[-1]
        anchor_factor = float(anchor["adj_factor"])
        if not np.isfinite(anchor_factor) or anchor_factor <= 0:
            raise _adapter_error("DATA", "截止交易日的复权因子无效。", False)

        multiplier = merged["adj_factor"] / anchor_factor
        for column in _PRICE_COLUMNS:
            merged[column] = merged[column] * multiplier
        merged["change"] = merged["close"] - merged["pre_close"]
        merged["pct_chg"] = merged["change"] / merged["pre_close"] * 100.0

        return merged.loc[:, _DAILY_COLUMNS].reset_index(drop=True)

    def fetch_daily_with_quality(
        self,
        ts_code: str,
        start_date: date | str,
        end_date: date | str,
        *,
        display_start_date: date | str | None = None,
    ) -> DailyDataResult:
        """Return the full prewarmed frame together with an explicit quality report."""

        frame = self.fetch_daily(ts_code, start_date, end_date)
        quality = assess_data_quality(frame, display_start_date=display_start_date)
        return DailyDataResult(frame=frame, quality=quality)


def assess_data_quality(
    frame: pd.DataFrame,
    *,
    display_start_date: date | str | None = None,
) -> DataQuality:
    """Assess required fields, numerical invariants, dates, and prewarm sufficiency."""

    raw_count = len(frame)
    warnings: list[str] = []
    valid = True

    missing_columns = [column for column in _DAILY_COLUMNS if column not in frame.columns]
    missing_values = {
        column: raw_count if column in missing_columns else int(frame[column].isna().sum())
        for column in _DAILY_COLUMNS
        if column in missing_columns or frame[column].isna().any()
    }
    if missing_columns:
        warnings.append(f"缺少必要字段：{', '.join(missing_columns)}。")
        valid = False
    if any(count > 0 for count in missing_values.values()):
        warnings.append("必要字段存在缺失值。")
        valid = False

    dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if "trade_date" in frame:
        dates = pd.to_datetime(frame["trade_date"], format=_DATE_FORMAT, errors="coerce")
        if dates.isna().any():
            warnings.append("交易日期无效。")
            valid = False
        elif not dates.is_monotonic_increasing or dates.duplicated().any():
            warnings.append("交易日期必须严格递增。")
            valid = False

    for column in _PRICE_COLUMNS:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
            warnings.append(f"{column} 必须是严格大于 0 的有限数。")
            valid = False

    for column in ("vol", "amount"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            warnings.append(f"{column} 必须是非负有限数。")
            valid = False

    if all(column in frame for column in ("open", "high", "low", "close")):
        open_price = pd.to_numeric(frame["open"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        invalid_ohlc = (high < pd.concat([open_price, close, low], axis=1).max(axis=1)) | (
            low > pd.concat([open_price, close, high], axis=1).min(axis=1)
        )
        if invalid_ohlc.any():
            warnings.append("OHLC 高低价关系无效。")
            valid = False

    display_start: date | None = None
    if display_start_date is not None:
        try:
            display_start = _coerce_date(display_start_date, name="展示起始日")
        except TushareAdapterError:
            warnings.append("展示起始日无效。")
            valid = False

    if raw_count == 0:
        warnings.append("请求区间无交易日数据，数据不足。")
        valid = False

    parseable_dates = dates.dropna()
    last_trade_date = parseable_dates.iloc[-1].date() if not parseable_dates.empty else None
    if display_start is None:
        prewarm_count = 0
        display_count = raw_count
    else:
        prewarm_count = int((dates.dt.date < display_start).sum())
        display_count = int((dates.dt.date >= display_start).sum())
        if display_count == 0:
            warnings.append("展示区间无交易日数据，数据不足。")
            valid = False
        if prewarm_count < 60:
            warnings.append("预热数据少于 60 个交易日，数据不足。")
            valid = False
        elif prewarm_count < 120:
            warnings.append("预热数据少于 120 个交易日，长周期指标可能不稳定。")

    return DataQuality(
        raw_row_count=raw_count,
        display_row_count=display_count,
        prewarm_row_count=prewarm_count,
        last_trade_date=last_trade_date,
        missing_values=missing_values,
        warnings=warnings,
        valid=valid,
    )


def _normalize_daily(
    frame: pd.DataFrame,
    *,
    requested_code: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    normalized = _deduplicate(frame, ["ts_code", "trade_date"], label="日线行情")
    normalized["ts_code"] = normalized["ts_code"].astype("string").str.strip().str.upper()
    normalized["trade_date"] = _parse_date_series(normalized["trade_date"], label="交易日期")
    for column in _NUMERIC_DAILY_COLUMNS:
        normalized[column] = _parse_numeric_series(normalized[column], label=column)

    if not normalized["ts_code"].eq(requested_code).all():
        raise _adapter_error("DATA", "日线行情包含非请求股票代码。", False)
    _require_inclusive_range(normalized["trade_date"], start=start, end=end, label="日线行情")
    return normalized.sort_values("trade_date", kind="stable").reset_index(drop=True)


def _normalize_factors(
    frame: pd.DataFrame,
    *,
    requested_code: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized["ts_code"] = normalized["ts_code"].astype("string").str.strip().str.upper()
    normalized["trade_date"] = _parse_date_series(normalized["trade_date"], label="复权因子日期")
    normalized = normalized.loc[
        (normalized["trade_date"].dt.date >= start) & (normalized["trade_date"].dt.date <= end)
    ].copy()
    normalized = _deduplicate(normalized, ["ts_code", "trade_date"], label="复权因子")
    normalized["adj_factor"] = _parse_numeric_series(normalized["adj_factor"], label="adj_factor")
    if not normalized["ts_code"].eq(requested_code).all():
        raise _adapter_error("DATA", "复权因子包含非请求股票代码。", False)
    if (normalized["adj_factor"] <= 0).any():
        raise _adapter_error("DATA", "复权因子必须严格大于 0。", False)
    return normalized.sort_values("trade_date", kind="stable").reset_index(drop=True)


def _deduplicate(frame: pd.DataFrame, key_columns: list[str], *, label: str) -> pd.DataFrame:
    exact = frame.drop_duplicates().reset_index(drop=True)
    if exact.duplicated(subset=key_columns, keep=False).any():
        raise _adapter_error("DATA", f"{label}存在冲突重复记录。", False)
    return exact


def _parse_date_series(series: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, format=_DATE_FORMAT, errors="coerce")
    if parsed.isna().any():
        raise _adapter_error("DATA", f"{label}包含无效日期。", False)
    return parsed


def _parse_numeric_series(series: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce").astype("float64")
    if parsed.isna().any() or not np.isfinite(parsed).all():
        raise _adapter_error("DATA", f"{label} 包含缺失或非有限数值。", False)
    return parsed


def _require_inclusive_range(series: pd.Series, *, start: date, end: date, label: str) -> None:
    outside = (series.dt.date < start) | (series.dt.date > end)
    if outside.any():
        raise _adapter_error("DATA", f"{label}返回了请求日期范围外的数据。", False)


def _validate_request_dates(start: date | str, end: date | str) -> tuple[date, date]:
    parsed_start = _coerce_date(start, name="开始日期")
    parsed_end = _coerce_date(end, name="截止日期")
    if parsed_start > parsed_end:
        raise _adapter_error("VALIDATION", "开始日期不能晚于截止日期。", False)
    return parsed_start, parsed_end


def _coerce_date(value: date | str, *, name: str) -> date:
    if isinstance(value, datetime):
        raise _adapter_error("VALIDATION", f"{name}必须是 YYYYMMDD 日期。", False)
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise _adapter_error("VALIDATION", f"{name}必须是 YYYYMMDD 日期。", False)
    try:
        return datetime.strptime(value, _DATE_FORMAT).date()
    except ValueError:
        raise _adapter_error("VALIDATION", f"{name}必须是有效 YYYYMMDD 日期。", False) from None


def _require_dataframe(value: object, *, endpoint: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise _adapter_error("DATA", f"{endpoint}返回了无效数据类型。", False)
    return value


def _require_columns(frame: pd.DataFrame, columns: list[str], *, endpoint: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise _adapter_error("DATA", f"{endpoint}缺少必要字段。", False)


def _empty_daily_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=_DAILY_COLUMNS)
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    for column in _NUMERIC_DAILY_COLUMNS:
        frame[column] = frame[column].astype("float64")
    return frame


def _empty_stock_master() -> pd.DataFrame:
    frame = pd.DataFrame(columns=_STOCK_COLUMNS)
    for column in _STOCK_COLUMNS[:-1]:
        frame[column] = frame[column].astype("string")
    frame["list_date"] = pd.to_datetime(frame["list_date"])
    return frame


def _adapter_error(code: str, message: str, retryable: bool) -> TushareAdapterError:
    return TushareAdapterError(
        AgentError(code=code, user_message=_sanitize(message), retryable=retryable)  # type: ignore[arg-type]
    )


def _map_upstream_error(exc: BaseException) -> TushareAdapterError:
    raw = _sanitize(str(exc)).lower()
    if any(marker in raw for marker in ("权限", "积分", "permission", "privilege", "points")):
        return _adapter_error(
            "AUTH",
            "Tushare 接口权限或积分不足，请确认账号至少有 2000 积分。",
            False,
        )
    if any(marker in raw for marker in ("频率", "每分钟", "rate limit", "too many", "429")):
        return _adapter_error("RATE_LIMIT", "Tushare 请求过于频繁，请稍后重试。", True)
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        marker in raw
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "network",
            "server",
            "500",
            "502",
            "503",
            "504",
        )
    ):
        return _adapter_error("DATA", "Tushare 服务暂时不可用，请稍后重试。", True)
    if any(marker in raw for marker in ("token", "auth", "unauthorized", "认证")):
        return _adapter_error("AUTH", "Tushare Token 无效，请检查配置。", False)
    return _adapter_error("DATA", "Tushare 数据请求失败，请稍后重试。", True)


def _sanitize(message: str) -> str:
    sanitized = _AUTHORIZATION.sub("[REDACTED]", message)
    sanitized = _TOKEN_LIKE.sub("[REDACTED]", sanitized)
    return sanitized


__all__ = [
    "ADJ_FACTOR_FIELDS",
    "DAILY_FIELDS",
    "STOCK_BASIC_FIELDS",
    "DailyDataResult",
    "ProLike",
    "TushareAdapterError",
    "TushareDataClient",
    "assess_data_quality",
]
