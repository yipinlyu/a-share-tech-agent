"""Deterministic, causal technical indicators over validated ascending OHLCV."""

from __future__ import annotations

from datetime import date, datetime
from typing import Final

import numpy as np
import pandas as pd

INDICATOR_COLUMNS: Final = (
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ema12",
    "ema26",
    "macd",
    "macd_signal",
    "macd_hist",
    "boll_mid",
    "boll_upper",
    "boll_lower",
    "bandwidth",
    "rsi14",
    "rsv",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "tr",
    "atr14",
    "obv",
    "vol_ma20",
    "volume_ratio",
    "return_5d",
    "return_10d",
    "return_20d",
)

SNAPSHOT_COLUMNS: Final = (
    "trade_date",
    "close",
    "pct_chg",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "macd",
    "macd_signal",
    "macd_hist",
    "boll_upper",
    "boll_mid",
    "boll_lower",
    "rsi14",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "atr14",
    "atr_ratio",
    "obv",
    "volume_ratio",
)


def compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the frozen indicator set without mutating or backfilling input data."""

    result = frame.copy(deep=True)
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    volume = result["vol"].astype(float)

    for window in (5, 10, 20, 60):
        result[f"ma{window}"] = close.rolling(window, min_periods=window).mean()

    result["ema12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    result["ema26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = result["ema12"] - result["ema26"]
    result["macd_signal"] = (
        result["macd"]
        .dropna()
        .ewm(span=9, adjust=False, min_periods=9)
        .mean()
        .reindex(result.index)
    )
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    result["boll_mid"] = close.rolling(20, min_periods=20).mean()
    boll_std = close.rolling(20, min_periods=20).std(ddof=0)
    result["boll_upper"] = result["boll_mid"] + 2.0 * boll_std
    result["boll_lower"] = result["boll_mid"] - 2.0 * boll_std
    result["bandwidth"] = ((result["boll_upper"] - result["boll_lower"]) / result["boll_mid"]).mask(
        result["boll_mid"] == 0
    )

    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    result["rsi14"] = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    result["rsi14"] = result["rsi14"].mask((average_loss == 0) & (average_gain > 0), 100.0)
    result["rsi14"] = result["rsi14"].mask((average_loss == 0) & (average_gain == 0), 50.0)

    rolling_low = low.rolling(9, min_periods=9).min()
    rolling_high = high.rolling(9, min_periods=9).max()
    kdj_range = rolling_high - rolling_low
    result["rsv"] = (close - rolling_low) / kdj_range * 100.0
    result["rsv"] = result["rsv"].mask(kdj_range == 0, 50.0)
    result["kdj_k"], result["kdj_d"] = _smooth_kdj(result["rsv"])
    result["kdj_j"] = 3.0 * result["kdj_k"] - 2.0 * result["kdj_d"]

    previous_close = close.shift(1)
    result["tr"] = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    if not result.empty:
        result.iloc[0, result.columns.get_loc("tr")] = high.iloc[0] - low.iloc[0]
    result["atr14"] = result["tr"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    direction = np.sign(delta).fillna(0.0)
    result["obv"] = (direction * volume).cumsum()
    if not result.empty:
        result.iloc[0, result.columns.get_loc("obv")] = 0.0

    result["vol_ma20"] = volume.rolling(20, min_periods=20).mean()
    result["volume_ratio"] = volume / result["vol_ma20"]
    for days in (5, 10, 20):
        result[f"return_{days}d"] = close / close.shift(days) - 1.0

    result.loc[:, list(INDICATOR_COLUMNS)] = result.loc[:, list(INDICATOR_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )
    return result


def latest_snapshot(frame: pd.DataFrame) -> dict[str, str | float | None]:
    """Serialize the latest allowlisted values as an ISO date and finite floats/nulls."""

    if frame.empty:
        return {}

    latest = frame.iloc[-1]
    snapshot: dict[str, str | float | None] = {}
    for column in SNAPSHOT_COLUMNS:
        if column == "trade_date":
            snapshot[column] = _json_date(latest.get(column))
        elif column == "atr_ratio":
            snapshot[column] = _finite_ratio(latest.get("atr14"), latest.get("close"))
        else:
            snapshot[column] = _finite_float(latest.get(column))
    return snapshot


def _smooth_kdj(rsv: pd.Series) -> tuple[pd.Series, pd.Series]:
    k_values = np.full(len(rsv), np.nan, dtype=float)
    d_values = np.full(len(rsv), np.nan, dtype=float)
    previous_k = 50.0
    previous_d = 50.0
    for position, value in enumerate(rsv.to_numpy(dtype=float)):
        if np.isnan(value):
            continue
        current_k = (2.0 / 3.0) * previous_k + (1.0 / 3.0) * value
        current_d = (2.0 / 3.0) * previous_d + (1.0 / 3.0) * current_k
        k_values[position] = current_k
        d_values[position] = current_d
        previous_k = current_k
        previous_d = current_d
    return (
        pd.Series(k_values, index=rsv.index, dtype=float),
        pd.Series(d_values, index=rsv.index, dtype=float),
    )


def _json_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    return str(value)


def _finite_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _finite_ratio(numerator: object, denominator: object) -> float | None:
    top = _finite_float(numerator)
    bottom = _finite_float(denominator)
    if top is None or bottom in (None, 0.0):
        return None
    return _finite_float(top / bottom)
