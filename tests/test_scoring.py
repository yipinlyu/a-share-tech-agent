from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_agent.scoring.rules import (
    CAPACITY,
    MIN_EVALUABLE,
    RULE_DEFINITIONS,
    WEIGHTS,
    build_analysis_id,
    build_watch_levels,
    canonical_json,
    evaluate_rule,
    label_for_score,
    score_signals,
)


def _neutral_frame(rows: int = 25) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "ts_code": ["600519.SH"] * rows,
            "trade_date": pd.date_range("2024-01-02", periods=rows, freq="B"),
            "open": close,
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": close,
            "pre_close": close,
            "vol": np.full(rows, 1_000.0),
            "ma5": close,
            "ma10": close,
            "ma20": close,
            "ma60": close,
            "macd": np.zeros(rows),
            "macd_signal": np.zeros(rows),
            "macd_hist": np.zeros(rows),
            "rsi14": np.full(rows, 50.0),
            "kdj_k": np.full(rows, 50.0),
            "kdj_d": np.full(rows, 50.0),
            "obv": np.zeros(rows),
            "volume_ratio": np.ones(rows),
            "boll_upper": np.full(rows, 110.0),
            "boll_mid": close,
            "boll_lower": np.full(rows, 90.0),
            "atr14": np.full(rows, 3.0),
        }
    )


def _set_latest(frame: pd.DataFrame, **values: float) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for column, value in values.items():
        result.loc[result.index[-1], column] = value
    return result


def _bullish_frame() -> pd.DataFrame:
    frame = _neutral_frame()
    last = frame.index[-1]
    frame["close"] = np.linspace(100.0, 106.0, len(frame))
    frame["pre_close"] = frame["close"].shift(1).fillna(frame["close"])
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 1.0
    frame["low"] = frame["close"] - 1.0
    frame["ma20"] = np.linspace(98.0, 100.0, len(frame))
    frame.loc[frame.index[-6], "ma20"] = 99.0
    frame.loc[last, ["ma5", "ma10", "ma20", "ma60"]] = [104.0, 103.0, 100.0, 95.0]
    frame.loc[last, ["macd", "macd_signal", "macd_hist"]] = [2.0, 1.0, 1.0]
    frame.loc[last - 1, "macd_hist"] = 0.5
    frame.loc[last, "rsi14"] = 60.0
    frame.loc[last - 1, ["kdj_k", "kdj_d"]] = [49.0, 50.0]
    frame.loc[last, ["kdj_k", "kdj_d"]] = [60.0, 55.0]
    frame["obv"] = np.arange(len(frame), dtype=float) * 1_000.0
    frame.loc[last, "volume_ratio"] = 1.3
    frame.loc[last, ["boll_mid", "boll_upper", "boll_lower"]] = [100.0, 110.0, 90.0]
    frame.loc[last, "atr14"] = 3.0
    return frame


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (40, "偏多"),
        (39.999, "中性偏多"),
        (15, "中性偏多"),
        (14.999, "中性"),
        (-14.999, "中性"),
        (-15, "中性偏空"),
        (-39.999, "中性偏空"),
        (-40, "偏空"),
    ],
)
def test_score_boundaries(value: float, label: str) -> None:
    assert label_for_score(value) == label


def test_group_contract_constants_and_data_driven_capacity() -> None:
    assert CAPACITY == {"trend": 40, "momentum": 30, "volume_volatility": 27}
    assert WEIGHTS == {"trend": 40, "momentum": 30, "volume_volatility": 30}
    assert MIN_EVALUABLE == {"trend": 20, "momentum": 15, "volume_volatility": 14}
    assert sum(rule.capacity for rule in RULE_DEFINITIONS) == 97
    assert {
        group: sum(rule.capacity for rule in RULE_DEFINITIONS if rule.group == group)
        for group in CAPACITY
    } == CAPACITY


@pytest.mark.parametrize(
    ("rule_key", "bullish", "bearish", "neutral"),
    [
        ("price_ma20", {"close": 101, "ma20": 100}, {"close": 99, "ma20": 100}, {"close": 100, "ma20": 100}),
        ("price_ma60", {"close": 101, "ma60": 100}, {"close": 99, "ma60": 100}, {"close": 100, "ma60": 100}),
        (
            "ma_alignment",
            {"ma5": 104, "ma10": 103, "ma20": 102, "ma60": 101},
            {"ma5": 101, "ma10": 102, "ma20": 103, "ma60": 104},
            {"ma5": 104, "ma10": 104, "ma20": 102, "ma60": 101},
        ),
        (
            "macd_line",
            {"macd": 2, "macd_signal": 1, "macd_hist": 0.1},
            {"macd": 1, "macd_signal": 2, "macd_hist": -0.1},
            {"macd": 2, "macd_signal": 1, "macd_hist": 0},
        ),
        ("rsi14", {"rsi14": 55}, {"rsi14": 45}, {"rsi14": 50}),
        (
            "volume_confirmation",
            {"close": 101, "pre_close": 100, "volume_ratio": 1.2},
            {"close": 99, "pre_close": 100, "volume_ratio": 1.2},
            {"close": 101, "pre_close": 100, "volume_ratio": 1.199999},
        ),
        (
            "boll_position",
            {"close": 110, "boll_mid": 100, "boll_upper": 110, "boll_lower": 90},
            {"close": 90, "boll_mid": 100, "boll_upper": 110, "boll_lower": 90},
            {"close": 100, "boll_mid": 100, "boll_upper": 110, "boll_lower": 90},
        ),
        (
            "boll_breakout",
            {"close": 110.01, "boll_upper": 110, "boll_lower": 90},
            {"close": 89.99, "boll_upper": 110, "boll_lower": 90},
            {"close": 110, "boll_upper": 110, "boll_lower": 90},
        ),
    ],
)
def test_latest_value_rules_cover_bullish_bearish_and_strict_neutral_boundaries(
    rule_key: str,
    bullish: dict[str, float],
    bearish: dict[str, float],
    neutral: dict[str, float],
) -> None:
    assert evaluate_rule(rule_key, _set_latest(_neutral_frame(), **bullish)).status == "bullish"
    assert evaluate_rule(rule_key, _set_latest(_neutral_frame(), **bearish)).status == "bearish"
    assert evaluate_rule(rule_key, _set_latest(_neutral_frame(), **neutral)).status == "neutral"


@pytest.mark.parametrize("rsi", [55.0, 70.0])
def test_rsi_bullish_range_is_inclusive(rsi: float) -> None:
    result = evaluate_rule("rsi14", _set_latest(_neutral_frame(), rsi14=rsi))
    assert (result.status, result.signed_points) == ("bullish", 10)


@pytest.mark.parametrize("rsi", [30.0, 45.0])
def test_rsi_bearish_range_is_inclusive(rsi: float) -> None:
    result = evaluate_rule("rsi14", _set_latest(_neutral_frame(), rsi14=rsi))
    assert (result.status, result.signed_points) == ("bearish", -10)


@pytest.mark.parametrize(
    ("latest", "five_days_ago", "status"),
    [
        (100.500001, 100.0, "bullish"),
        (100.5, 100.0, "neutral"),
        (99.499999, 100.0, "bearish"),
        (99.5, 100.0, "neutral"),
    ],
)
def test_ma20_slope_uses_strict_half_percent_boundary(
    latest: float, five_days_ago: float, status: str
) -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-6], "ma20"] = five_days_ago
    frame.loc[frame.index[-1], "ma20"] = latest
    assert evaluate_rule("ma20_slope", frame).status == status


@pytest.mark.parametrize(
    ("previous", "latest", "status"),
    [(0.1, 0.2, "bullish"), (0.2, 0.1, "neutral"), (-0.1, -0.2, "bearish"), (-0.2, -0.1, "neutral"), (0.1, 0.1, "neutral")],
)
def test_macd_hist_momentum_requires_directional_change_and_same_sign(
    previous: float, latest: float, status: str
) -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-2], "macd_hist"] = previous
    frame.loc[frame.index[-1], "macd_hist"] = latest
    assert evaluate_rule("macd_hist_momentum", frame).status == status


@pytest.mark.parametrize(
    ("latest", "twenty_days_ago", "status"),
    [
        (103.000001, 100.0, "bullish"),
        (103.0, 100.0, "neutral"),
        (96.999999, 100.0, "bearish"),
        (97.0, 100.0, "neutral"),
    ],
)
def test_twenty_day_return_uses_strict_three_percent_boundary(
    latest: float, twenty_days_ago: float, status: str
) -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-21], "close"] = twenty_days_ago
    frame.loc[frame.index[-1], "close"] = latest
    assert evaluate_rule("return_20d", frame).status == status


@pytest.mark.parametrize(
    ("obv_then", "obv_now", "close_then", "close_now", "status"),
    [
        (0, 5, 100, 101, "bullish"),
        (5, 0, 100, 99, "bearish"),
        (0, 5, 100, 99, "neutral"),
        (0, 0, 100, 101, "neutral"),
    ],
)
def test_obv_slope_requires_price_confirmation_and_strict_nonzero_directions(
    obv_then: float, obv_now: float, close_then: float, close_now: float, status: str
) -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-6], ["obv", "close"]] = [obv_then, close_then]
    frame.loc[frame.index[-1], ["obv", "close"]] = [obv_now, close_now]
    assert evaluate_rule("obv_price_trend", frame).status == status


def test_kdj_uses_only_most_recent_qualified_cross_in_three_valid_days() -> None:
    frame = _neutral_frame()
    tail = frame.index[-4:]
    frame.loc[tail, "kdj_k"] = [50, 60, 40, 55]
    frame.loc[tail, "kdj_d"] = [50, 50, 50, 50]

    result = evaluate_rule("kdj_cross", frame)

    assert (result.status, result.signed_points) == ("bullish", 8)
    assert result.observed_value == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("previous_k", "latest_k", "d", "status"),
    [(70, 80, 75, "neutral"), (30, 20, 25, "neutral"), (70, 79.999, 75, "bullish"), (30, 20.001, 25, "bearish")],
)
def test_kdj_cross_threshold_uses_cross_day_k_strictly(
    previous_k: float, latest_k: float, d: float, status: str
) -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-4:-1], "kdj_k"] = previous_k
    frame.loc[frame.index[-4:-1], "kdj_d"] = d
    frame.loc[frame.index[-1], ["kdj_k", "kdj_d"]] = [latest_k, d]
    assert evaluate_rule("kdj_cross", frame).status == status


@pytest.mark.parametrize(
    ("rule_key", "column", "row_offset"),
    [
        ("price_ma20", "ma20", -1),
        ("price_ma60", "ma60", -1),
        ("ma_alignment", "ma10", -1),
        ("macd_line", "macd_hist", -1),
        ("ma20_slope", "ma20", -6),
        ("rsi14", "rsi14", -1),
        ("kdj_cross", "kdj_k", -1),
        ("macd_hist_momentum", "macd_hist", -2),
        ("return_20d", "close", -21),
        ("obv_price_trend", "obv", -6),
        ("volume_confirmation", "volume_ratio", -1),
        ("boll_position", "boll_mid", -1),
        ("boll_breakout", "boll_upper", -1),
    ],
)
def test_each_direction_rule_reports_missing_when_a_required_input_is_missing(
    rule_key: str, column: str, row_offset: int
) -> None:
    frame = _neutral_frame()
    if rule_key == "kdj_cross":
        frame[["kdj_k", "kdj_d"]] = np.nan
    else:
        frame.loc[frame.index[row_offset], column] = np.nan
    result = evaluate_rule(rule_key, frame)
    assert (result.status, result.signed_points) == ("missing", 0)


def test_all_evaluable_but_neutral_is_zero_not_insufficient() -> None:
    score = score_signals(_neutral_frame(), data_warnings=[])

    assert score is not None
    assert score.total == 0
    assert score.signal == "中性"
    assert score.evaluable_capacity == CAPACITY
    assert score.completeness == 1
    assert score.consistency == 0
    assert score.bullish_points == 0
    assert score.bearish_points == 0


def test_group_capacity_normalizes_raw_score_by_only_evaluable_capacity() -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-1], ["close", "ma20"]] = [101.0, 100.0]
    frame.loc[frame.index[-1], ["ma5", "ma10"]] = np.nan
    frame.loc[frame.index[-1], "ma60"] = 101.0
    frame.loc[frame.index[-6], "ma20"] = np.nan
    frame.loc[frame.index[-1], ["macd", "macd_signal", "macd_hist"]] = np.nan

    score = score_signals(frame, data_warnings=[])

    assert score is not None
    assert score.evaluable_capacity["trend"] == 15
    assert score.group_scores["trend"] is None
    assert "trend" not in score.usable_groups
    assert score.raw_group_scores["trend"] == 8


def test_less_than_two_usable_groups_is_insufficient() -> None:
    frame = _neutral_frame()
    frame.loc[:, ["rsi14", "kdj_k", "kdj_d", "macd_hist"]] = np.nan
    frame.loc[:, ["obv", "volume_ratio", "boll_upper", "boll_mid", "boll_lower"]] = np.nan

    assert score_signals(frame, data_warnings=[]) is None


def test_usable_weight_sixty_is_rescaled_to_full_total() -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-1], ["close", "ma20", "ma60"]] = [101.0, 100.0, 100.0]
    frame.loc[:, ["rsi14", "kdj_k", "kdj_d", "macd_hist"]] = np.nan

    score = score_signals(frame, data_warnings=[])

    assert score is not None
    assert score.usable_groups == ["trend", "volume_volatility"]
    assert sum(WEIGHTS[group] for group in score.usable_groups) == 70
    expected = (score.group_scores["trend"] + score.group_scores["volume_volatility"]) * 100 / 70  # type: ignore[operator]
    assert score.total == pytest.approx(expected)


def test_exactly_sixty_usable_weight_is_allowed_and_rescaled() -> None:
    frame = _neutral_frame()
    frame.loc[:, ["ma5", "ma10", "ma20", "ma60", "macd", "macd_signal"]] = np.nan

    score = score_signals(frame, data_warnings=[])

    assert score is not None
    assert score.usable_groups == ["momentum", "volume_volatility"]
    assert sum(WEIGHTS[group] for group in score.usable_groups) == 60
    assert score.total == 0


def test_completeness_uses_fixed_ninety_seven_point_denominator() -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-1], "ma60"] = np.nan

    score = score_signals(frame, data_warnings=[])

    assert score is not None
    assert sum(score.evaluable_capacity.values()) == 78
    assert score.completeness == pytest.approx(78 / 97)


def test_consistency_uses_hit_raw_points_and_zero_denominator_is_zero() -> None:
    neutral = score_signals(_neutral_frame(), data_warnings=[])
    mixed = _neutral_frame()
    mixed.loc[mixed.index[-1], ["close", "ma20", "ma60", "boll_mid"]] = [
        101.0,
        100.0,
        102.0,
        101.0,
    ]
    mixed_score = score_signals(mixed, data_warnings=[])

    assert neutral is not None and neutral.consistency == 0
    assert mixed_score is not None
    assert mixed_score.bullish_points == 8
    assert mixed_score.bearish_points == 7
    assert mixed_score.consistency == pytest.approx(1 / 15)


@pytest.mark.parametrize(
    ("atr_ratio", "expected_points", "expected_level"),
    [(0.04, 0, "低"), (0.040001, 1, "低"), (0.06, 1, "低"), (0.060001, 2, "中")],
)
def test_atr_risk_thresholds_are_strict_at_four_and_six_percent(
    atr_ratio: float, expected_points: int, expected_level: str
) -> None:
    frame = _set_latest(_neutral_frame(), close=100.0, atr14=atr_ratio * 100.0)
    score = score_signals(frame, data_warnings=[])
    assert score is not None
    assert score.risk_score == expected_points
    assert score.risk_level == expected_level


@pytest.mark.parametrize(("rsi", "risk_type"), [(75, None), (75.001, "overbought"), (25, None), (24.999, "oversold")])
def test_rsi_extremes_are_risk_only_and_strict(rsi: float, risk_type: str | None) -> None:
    score = score_signals(_set_latest(_neutral_frame(), rsi14=rsi), data_warnings=[])
    assert score is not None
    assert score.raw_group_scores["momentum"] == 0
    assert ([risk.risk_type for risk in score.risks] or [None]) == [risk_type]
    assert score.risk_score == (2 if risk_type else 0)


def test_bollinger_break_divergence_and_each_data_warning_add_only_market_risk() -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-11], ["close", "obv"]] = [100.0, 10.0]
    frame.loc[frame.index[-1], ["close", "obv", "boll_upper"]] = [111.0, 0.0, 110.0]

    score = score_signals(frame, data_warnings=["行情缺口", "复权数据警告"])

    assert score is not None
    assert score.risk_score == 5
    assert score.risk_level == "高"
    assert [risk.risk_type for risk in score.risks] == [
        "volatility",
        "signal_conflict",
        "data_quality",
        "data_quality",
    ]
    assert [item.source_key for item in score.conflict_evidence] == ["price_obv_divergence_10d"]


def test_direction_evidence_is_partitioned_and_keeps_declaration_order() -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-1], ["close", "ma20", "ma60", "boll_mid"]] = [
        101.0,
        100.0,
        102.0,
        101.0,
    ]

    score = score_signals(frame, data_warnings=[])

    assert score is not None
    assert [evidence.source_key for evidence in score.positive_evidence] == ["price_ma20"]
    assert [evidence.source_key for evidence in score.negative_evidence] == ["price_ma60"]


def test_watch_levels_use_only_allowed_keys_positive_prices_and_recent_twenty_days() -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-20] :, "low"] = np.arange(20, dtype=float) + 80.0
    frame.loc[frame.index[-20] :, "high"] = np.arange(20, dtype=float) + 120.0
    frame.loc[frame.index[-1], ["close", "ma20", "boll_upper", "boll_lower", "atr14"]] = [
        100.0,
        99.0,
        110.0,
        90.0,
        4.0,
    ]

    levels = build_watch_levels(frame)

    assert [level.basis_key for level in levels] == [
        "recent_20d_low",
        "recent_20d_high",
        "ma20",
        "boll_upper",
        "boll_lower",
        "close_minus_atr",
        "close_plus_atr",
    ]
    assert levels[0].price == 80.0
    assert levels[1].price == 139.0
    assert all(level.price > 0 and np.isfinite(level.price) for level in levels)


def test_nonpositive_or_missing_watch_prices_are_omitted() -> None:
    frame = _neutral_frame()
    frame.loc[frame.index[-1], ["close", "ma20", "boll_lower", "atr14"]] = [1, np.nan, -1, 2]
    levels = build_watch_levels(frame)
    by_key = {level.basis_key: level.price for level in levels}

    assert "ma20" not in by_key
    assert "boll_lower" not in by_key
    assert "close_minus_atr" not in by_key


def test_volume_group_realized_maximum_and_total_theoretical_realized_maximum() -> None:
    score = score_signals(_bullish_frame(), data_warnings=[])

    assert score is not None
    assert score.raw_group_scores == {"trend": 40.0, "momentum": 30.0, "volume_volatility": 22.0}
    assert score.group_scores["volume_volatility"] == pytest.approx(22 / 27 * 30)
    assert score.total == pytest.approx(94.44444444444444)


def test_bollinger_position_and_breakout_are_mutually_exclusive() -> None:
    score = score_signals(
        _set_latest(_neutral_frame(), close=111, boll_mid=100, boll_upper=110, boll_lower=90),
        data_warnings=[],
    )
    assert score is not None
    keys = [item.source_key for item in score.positive_evidence]
    assert "boll_breakout" in keys
    assert "boll_position" not in keys


def test_canonical_json_sorts_keys_preserves_unicode_dates_null_and_float_precision() -> None:
    payload = {"z": 0.1, "a": "贵州茅台", "date": date(2024, 2, 5), "missing": None}

    assert canonical_json(payload) == (
        '{"a":"贵州茅台","date":"2024-02-05","missing":null,"z":0.10000000000000001}'
    )


def test_analysis_id_is_stable_and_ignores_rows_after_resolved_date() -> None:
    frame = _neutral_frame(10)
    resolved = frame.loc[frame.index[-2], "trade_date"].date()
    arguments = {
        "ts_code": "600519.SH",
        "resolved_end_date": resolved,
        "lookback_months": 12,
        "frame": frame.iloc[:-1],
    }

    first = build_analysis_id(**arguments)
    second = build_analysis_id(**arguments)
    with_future_suffix = build_analysis_id(**{**arguments, "frame": frame})

    assert first == second == with_future_suffix
    assert len(first) == 64
    int(first, 16)


@pytest.mark.parametrize("field", ["ts_code", "resolved_end_date", "lookback_months", "frame"])
def test_analysis_id_changes_when_any_as_of_input_changes(field: str) -> None:
    frame = _neutral_frame(10)
    arguments: dict[str, object] = {
        "ts_code": "600519.SH",
        "resolved_end_date": frame.loc[frame.index[-1], "trade_date"].date(),
        "lookback_months": 12,
        "frame": frame,
    }
    baseline = build_analysis_id(**arguments)  # type: ignore[arg-type]
    changed = dict(arguments)
    if field == "ts_code":
        changed[field] = "000001.SZ"
    elif field == "resolved_end_date":
        changed[field] = frame.loc[frame.index[-2], "trade_date"].date()
    elif field == "lookback_months":
        changed[field] = 6
    else:
        mutated = frame.copy()
        mutated.loc[mutated.index[-1], "close"] += 0.01
        changed[field] = mutated

    assert build_analysis_id(**changed) != baseline  # type: ignore[arg-type]
