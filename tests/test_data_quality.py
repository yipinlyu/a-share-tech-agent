from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_agent.data.tushare_client import (
    DAILY_FIELDS,
    DailyDataResult,
    TushareDataClient,
    assess_data_quality,
)

from fakes import FakeProApi


def quality_frame(prewarm_rows: int = 120, display_rows: int = 3) -> tuple[pd.DataFrame, date]:
    dates = pd.bdate_range("2023-01-02", periods=prewarm_rows + display_rows)
    frame = pd.DataFrame(
        {
            "ts_code": pd.Series(["600519.SH"] * len(dates), dtype="string"),
            "trade_date": dates,
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "pre_close": 10.5,
            "change": 0.5,
            "pct_chg": 100 * 0.5 / 10.5,
            "vol": 100.0,
            "amount": 1100.0,
        }
    )
    return frame, dates[prewarm_rows].date()


@pytest.mark.parametrize(
    ("prewarm_rows", "valid", "expects_warning"),
    [(59, False, True), (60, True, True), (119, True, True), (120, True, False)],
)
def test_prewarm_quality_boundaries(prewarm_rows: int, valid: bool, expects_warning: bool) -> None:
    frame, display_start = quality_frame(prewarm_rows=prewarm_rows)

    quality = assess_data_quality(frame, display_start_date=display_start)

    assert quality.raw_row_count == prewarm_rows + 3
    assert quality.prewarm_row_count == prewarm_rows
    assert quality.display_row_count == 3
    assert quality.valid is valid
    has_prewarm_warning = any(
        "预热" in warning or "长周期" in warning for warning in quality.warnings
    )
    assert has_prewarm_warning is expects_warning


def test_empty_daily_is_invalid_data_quality_not_synthetic_success() -> None:
    frame = pd.DataFrame(columns=DAILY_FIELDS.split(","))

    quality = assess_data_quality(frame, display_start_date=date(2024, 1, 1))

    assert quality.raw_row_count == 0
    assert quality.display_row_count == 0
    assert quality.prewarm_row_count == 0
    assert quality.last_trade_date is None
    assert quality.valid is False
    assert any("无交易日" in warning or "数据不足" in warning for warning in quality.warnings)


def test_display_window_with_no_rows_is_invalid() -> None:
    frame, _ = quality_frame()

    quality = assess_data_quality(frame, display_start_date=date(2025, 1, 1))

    assert quality.display_row_count == 0
    assert quality.valid is False
    assert any("展示区间" in warning for warning in quality.warnings)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "pre_close"])
def test_zero_or_negative_prices_fail_quality(column: str) -> None:
    frame, _ = quality_frame()
    frame.loc[0, column] = 0.0

    quality = assess_data_quality(frame)

    assert quality.valid is False
    assert any(column in warning for warning in quality.warnings)


@pytest.mark.parametrize("column", ["vol", "amount"])
def test_negative_volume_or_amount_fails_quality(column: str) -> None:
    frame, _ = quality_frame()
    frame.loc[0, column] = -1.0

    quality = assess_data_quality(frame)

    assert quality.valid is False
    assert any(column in warning for warning in quality.warnings)


@pytest.mark.parametrize(
    ("column", "value"),
    [("high", 10.5), ("low", 11.5)],
)
def test_impossible_ohlc_relationship_fails_quality(column: str, value: float) -> None:
    frame, _ = quality_frame()
    frame.loc[0, column] = value

    quality = assess_data_quality(frame)

    assert quality.valid is False
    assert any("OHLC" in warning for warning in quality.warnings)


def test_missing_required_column_and_value_fail_quality() -> None:
    frame, _ = quality_frame()
    frame = frame.drop(columns="amount")
    frame.loc[0, "close"] = float("nan")

    quality = assess_data_quality(frame)

    assert quality.valid is False
    assert quality.missing_values == {"close": 1, "amount": len(frame)}
    assert any("必要字段" in warning for warning in quality.warnings)


@pytest.mark.parametrize("mutation", ["descending", "duplicate", "invalid"])
def test_dates_must_be_valid_and_strictly_increasing(mutation: str) -> None:
    frame, _ = quality_frame()
    if mutation == "descending":
        frame = frame.sort_values("trade_date", ascending=False).reset_index(drop=True)
    elif mutation == "duplicate":
        frame.loc[1, "trade_date"] = frame.loc[0, "trade_date"]
    else:
        frame["trade_date"] = frame["trade_date"].astype("object")
        frame.loc[0, "trade_date"] = "not-a-date"

    quality = assess_data_quality(frame)

    assert quality.valid is False
    assert any("日期" in warning for warning in quality.warnings)


def test_fetch_with_quality_preserves_prewarm_frame_for_indicators() -> None:
    full, display_start = quality_frame(prewarm_rows=120, display_rows=3)
    raw = full.copy()
    raw["trade_date"] = raw["trade_date"].dt.strftime("%Y%m%d")
    factors = raw[["ts_code", "trade_date"]].copy()
    factors["adj_factor"] = 1.0
    pro = FakeProApi(daily_frame=raw, adj_factor_frame=factors)

    result = TushareDataClient(pro=pro).fetch_daily_with_quality(
        "600519.SH",
        raw.iloc[0]["trade_date"],
        raw.iloc[-1]["trade_date"],
        display_start_date=display_start,
    )

    assert isinstance(result, DailyDataResult)
    assert len(result.frame) == 123
    assert result.quality.prewarm_row_count == 120
    assert result.quality.display_row_count == 3
    assert result.quality.valid is True
