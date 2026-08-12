from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fixtures.ohlcv import (
    flat_then_move as _flat_then_move_fixture,  # noqa: F401
)
from fixtures.ohlcv import make_ohlcv
from fixtures.ohlcv import (
    ohlcv_160 as _ohlcv_160_fixture,  # noqa: F401
)
from stock_agent.indicators.engine import (
    INDICATOR_COLUMNS,
    SNAPSHOT_COLUMNS,
    compute_indicators,
    latest_snapshot,
)

EXPECTED_INDICATOR_COLUMNS = [
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
]


def test_compute_indicators_preserves_input_and_appends_frozen_columns(
    ohlcv_160: pd.DataFrame,
) -> None:
    before = ohlcv_160.copy(deep=True)

    result = compute_indicators(ohlcv_160)

    pd.testing.assert_frame_equal(result[before.columns], before)
    pd.testing.assert_frame_equal(ohlcv_160, before)
    assert list(INDICATOR_COLUMNS) == EXPECTED_INDICATOR_COLUMNS
    assert result.columns.tolist() == [*before.columns, *EXPECTED_INDICATOR_COLUMNS]
    assert result.index.equals(before.index)


def test_sma_volume_ratio_and_returns_follow_frozen_windows(ohlcv_160: pd.DataFrame) -> None:
    result = compute_indicators(ohlcv_160)

    for window in (5, 10, 20, 60):
        expected = ohlcv_160["close"].rolling(window, min_periods=window).mean()
        pd.testing.assert_series_equal(result[f"ma{window}"], expected, check_names=False)
    expected_vol_ma = ohlcv_160["vol"].rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(result["vol_ma20"], expected_vol_ma, check_names=False)
    pd.testing.assert_series_equal(
        result["volume_ratio"], ohlcv_160["vol"] / expected_vol_ma, check_names=False
    )
    for days in (5, 10, 20):
        expected_return = ohlcv_160["close"] / ohlcv_160["close"].shift(days) - 1.0
        pd.testing.assert_series_equal(
            result[f"return_{days}d"], expected_return, check_names=False
        )


def test_ema_macd_signal_and_histogram_use_exact_pandas_contract(
    ohlcv_160: pd.DataFrame,
) -> None:
    result = compute_indicators(ohlcv_160)
    close = ohlcv_160["close"]
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.dropna().ewm(span=9, adjust=False, min_periods=9).mean().reindex(macd.index)

    pd.testing.assert_series_equal(result["ema12"], ema12, check_names=False)
    pd.testing.assert_series_equal(result["ema26"], ema26, check_names=False)
    pd.testing.assert_series_equal(result["macd"], macd, check_names=False)
    pd.testing.assert_series_equal(result["macd_signal"], signal, check_names=False)
    pd.testing.assert_series_equal(result["macd_hist"], macd - signal, check_names=False)


def test_bollinger_uses_population_std_and_zero_mid_has_missing_bandwidth() -> None:
    frame = make_ohlcv([0.0] * 20 + [1.0] * 5)
    result = compute_indicators(frame)
    close = frame["close"]
    middle = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)

    pd.testing.assert_series_equal(result["boll_mid"], middle, check_names=False)
    pd.testing.assert_series_equal(result["boll_upper"], middle + 2.0 * std, check_names=False)
    pd.testing.assert_series_equal(result["boll_lower"], middle - 2.0 * std, check_names=False)
    assert pd.isna(result.loc[19, "bandwidth"])
    assert result.loc[24, "bandwidth"] == pytest.approx(
        (result.loc[24, "boll_upper"] - result.loc[24, "boll_lower"]) / result.loc[24, "boll_mid"]
    )


@pytest.mark.parametrize(
    ("close", "expected"),
    [
        (list(range(1, 18)), 100.0),
        (list(range(18, 1, -1)), 0.0),
        ([10.0] * 17, 50.0),
    ],
)
def test_rsi_wilder_zero_loss_and_zero_movement_edges(
    close: list[float],
    expected: float,
) -> None:
    result = compute_indicators(make_ohlcv(close))

    assert result["rsi14"].iloc[-1] == pytest.approx(expected)
    assert result["rsi14"].iloc[:14].isna().all()


def test_rsi14_matches_wilder_ewm_on_nontrivial_path(ohlcv_160: pd.DataFrame) -> None:
    result = compute_indicators(ohlcv_160)
    delta = ohlcv_160["close"].diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    expected = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    expected = expected.mask((average_loss == 0) & (average_gain > 0), 100.0)
    expected = expected.mask((average_loss == 0) & (average_gain == 0), 50.0)

    pd.testing.assert_series_equal(result["rsi14"], expected, check_names=False)


def test_kdj_first_valid_value_updates_from_seed_50(flat_then_move: pd.DataFrame) -> None:
    result = compute_indicators(flat_then_move)
    first = result["kdj_k"].first_valid_index()

    assert first == 8
    expected_k = (2 / 3) * 50.0 + (1 / 3) * result.loc[first, "rsv"]
    expected_d = (2 / 3) * 50.0 + (1 / 3) * expected_k
    assert result.loc[first, "kdj_k"] == pytest.approx(expected_k)
    assert result.loc[first, "kdj_d"] == pytest.approx(expected_d)
    assert result.loc[first, "kdj_j"] == pytest.approx(3 * expected_k - 2 * expected_d)
    assert result.loc[:7, ["rsv", "kdj_k", "kdj_d", "kdj_j"]].isna().all().all()


def test_kdj_flat_window_uses_rsv_50_and_does_not_clip_j() -> None:
    flat = make_ohlcv([10.0] * 12)
    flat[["open", "high", "low", "close"]] = 10.0
    result = compute_indicators(flat)

    assert result.loc[8, "rsv"] == pytest.approx(50.0)
    assert result.loc[8, "kdj_k"] == pytest.approx(50.0)
    moved = make_ohlcv([10.0] * 8 + [20.0] * 8)
    moved.loc[:7, ["open", "high", "low", "close"]] = 10.0
    moved.loc[8:, ["open", "high", "low", "close"]] = [20.0, 20.0, 10.0, 20.0]
    moved_result = compute_indicators(moved)
    assert (moved_result["kdj_j"].dropna() > 100.0).any()


def test_true_range_atr_and_obv_follow_frozen_formulas() -> None:
    frame = make_ohlcv([10, 12, 11, 11, 14, 13, 15, 14, 16, 18, 17, 19, 20, 18, 21, 22])
    result = compute_indicators(frame)
    previous_close = frame["close"].shift(1)
    expected_tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    expected_tr.iloc[0] = frame.loc[0, "high"] - frame.loc[0, "low"]
    expected_atr = expected_tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    direction = np.sign(frame["close"].diff()).fillna(0.0)
    expected_obv = (direction * frame["vol"]).cumsum()
    expected_obv.iloc[0] = 0.0

    pd.testing.assert_series_equal(result["tr"], expected_tr, check_names=False)
    pd.testing.assert_series_equal(result["atr14"], expected_atr, check_names=False)
    pd.testing.assert_series_equal(result["obv"], expected_obv, check_names=False)


def test_indicator_prefix_is_invariant_to_future_spike(ohlcv_160: pd.DataFrame) -> None:
    prefix = compute_indicators(ohlcv_160.iloc[:120]).iloc[-1]
    mutated = ohlcv_160.copy()
    mutated.loc[mutated.index[120] :, ["open", "high", "low", "close"]] *= 100
    full = compute_indicators(mutated).iloc[119]

    pd.testing.assert_series_equal(
        prefix[list(INDICATOR_COLUMNS)],
        full[list(INDICATOR_COLUMNS)],
    )


def test_future_spike_cannot_fill_an_earlier_warmup_value(ohlcv_160: pd.DataFrame) -> None:
    baseline = compute_indicators(ohlcv_160.iloc[:30])
    spiked = ohlcv_160.copy()
    spiked.loc[30:, ["open", "high", "low", "close", "vol"]] *= 1_000
    full = compute_indicators(spiked)

    pd.testing.assert_frame_equal(
        baseline[list(INDICATOR_COLUMNS)],
        full.loc[:29, list(INDICATOR_COLUMNS)],
    )
    assert full.loc[:58, "ma60"].isna().all()


def test_infinite_derived_values_become_nan_without_backfill() -> None:
    frame = make_ohlcv([0.0] * 25)
    frame["vol"] = 0.0

    result = compute_indicators(frame)

    assert not np.isinf(result[list(INDICATOR_COLUMNS)].to_numpy(dtype=float)).any()
    assert pd.isna(result.loc[19, "volume_ratio"])
    assert result.loc[:18, "vol_ma20"].isna().all()


def test_latest_snapshot_is_allowlisted_finite_and_json_safe(ohlcv_160: pd.DataFrame) -> None:
    result = compute_indicators(ohlcv_160)
    result["secret"] = "must-not-leak"
    result.loc[result.index[-1], "ma60"] = np.inf

    snapshot = latest_snapshot(result)

    assert set(snapshot).issubset(SNAPSHOT_COLUMNS)
    assert "secret" not in snapshot
    assert snapshot["trade_date"] == result.iloc[-1]["trade_date"].date().isoformat()
    assert snapshot["close"] == pytest.approx(float(result.iloc[-1]["close"]))
    assert snapshot["ma60"] is None
    assert isinstance(json.dumps(snapshot, allow_nan=False), str)


def test_latest_snapshot_handles_date_object_and_empty_frame(ohlcv_160: pd.DataFrame) -> None:
    result = compute_indicators(ohlcv_160.iloc[:1])
    result.loc[result.index[-1], "trade_date"] = pd.Timestamp(date(2024, 1, 2))

    assert latest_snapshot(result)["trade_date"] == "2024-01-02"
    assert latest_snapshot(result.iloc[:0]) == {}
