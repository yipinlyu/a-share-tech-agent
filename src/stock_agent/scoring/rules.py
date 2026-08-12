"""Explainable, deterministic score-v1 rules over an as-of indicator frame."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal, TypeAlias

import numpy as np
import pandas as pd

from stock_agent.domain.models import Evidence, Risk, ScoreResult, WatchLevel

Group: TypeAlias = Literal["trend", "momentum", "volume_volatility"]
RuleStatus: TypeAlias = Literal["bullish", "bearish", "neutral", "missing"]

CAPACITY: Final[dict[Group, int]] = {
    "trend": 40,
    "momentum": 30,
    "volume_volatility": 27,
}
WEIGHTS: Final[dict[Group, int]] = {
    "trend": 40,
    "momentum": 30,
    "volume_volatility": 30,
}
MIN_EVALUABLE: Final[dict[Group, int]] = {
    "trend": 20,
    "momentum": 15,
    "volume_volatility": 14,
}
GROUP_ORDER: Final[tuple[Group, ...]] = ("trend", "momentum", "volume_volatility")
TOTAL_CAPACITY: Final = sum(CAPACITY.values())


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    key: str
    group: Group
    capacity: int
    bullish_text: str
    bearish_text: str


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    key: str
    group: Group
    capacity: int
    status: RuleStatus
    signed_points: int
    observed_value: float | None
    interpretation: str | None


RULE_DEFINITIONS: Final[tuple[RuleDefinition, ...]] = (
    RuleDefinition("price_ma20", "trend", 8, "收盘价高于 MA20", "收盘价低于 MA20"),
    RuleDefinition("price_ma60", "trend", 7, "收盘价高于 MA60", "收盘价低于 MA60"),
    RuleDefinition("ma_alignment", "trend", 12, "均线呈严格多头排列", "均线呈严格空头排列"),
    RuleDefinition("macd_line", "trend", 8, "MACD 线与柱值共同偏多", "MACD 线与柱值共同偏空"),
    RuleDefinition("ma20_slope", "trend", 5, "MA20 五日升幅超过 0.5%", "MA20 五日降幅超过 0.5%"),
    RuleDefinition("rsi14", "momentum", 10, "RSI 位于偏强区间", "RSI 位于偏弱区间"),
    RuleDefinition("kdj_cross", "momentum", 8, "KDJ 最近出现合格上穿", "KDJ 最近出现合格下穿"),
    RuleDefinition("macd_hist_momentum", "momentum", 7, "正 MACD 柱继续增强", "负 MACD 柱继续减弱"),
    RuleDefinition("return_20d", "momentum", 5, "二十日涨幅超过 3%", "二十日跌幅超过 3%"),
    RuleDefinition(
        "obv_price_trend", "volume_volatility", 8, "OBV 与价格五日同升", "OBV 与价格五日同降"
    ),
    RuleDefinition(
        "volume_confirmation", "volume_volatility", 7, "放量上涨得到确认", "放量下跌得到确认"
    ),
    RuleDefinition(
        "boll_position",
        "volume_volatility",
        7,
        "价格位于布林中轨上方轨内",
        "价格位于布林中轨下方轨内",
    ),
    RuleDefinition("boll_breakout", "volume_volatility", 5, "价格突破布林上轨", "价格跌破布林下轨"),
)
_RULES: Final = {rule.key: rule for rule in RULE_DEFINITIONS}


def label_for_score(score: float) -> str:
    """Map a finite total to the frozen five-band signal boundaries."""

    if score >= 40:
        return "偏多"
    if score >= 15:
        return "中性偏多"
    if score > -15:
        return "中性"
    if score > -40:
        return "中性偏空"
    return "偏空"


def evaluate_rule(rule_key: str, frame: pd.DataFrame) -> RuleEvaluation:
    """Evaluate one declared direction rule without filling missing inputs."""

    definition = _RULES[rule_key]
    status, observed = _PREDICATES[rule_key](frame)
    points = (
        definition.capacity
        if status == "bullish"
        else -definition.capacity
        if status == "bearish"
        else 0
    )
    interpretation = (
        definition.bullish_text
        if status == "bullish"
        else definition.bearish_text
        if status == "bearish"
        else None
    )
    return RuleEvaluation(
        key=definition.key,
        group=definition.group,
        capacity=definition.capacity,
        status=status,
        signed_points=points,
        observed_value=observed,
        interpretation=interpretation,
    )


def score_signals(
    frame: pd.DataFrame,
    data_warnings: Sequence[str] = (),
) -> ScoreResult | None:
    """Return a successful score, or ``None`` when fewer than two groups are usable."""

    evaluations = [evaluate_rule(rule.key, frame) for rule in RULE_DEFINITIONS]
    capacities: dict[Group, int] = {}
    raw_scores: dict[Group, float] = {}
    group_scores: dict[Group, float | None] = {}
    usable_groups: list[Group] = []
    for group in GROUP_ORDER:
        group_results = [item for item in evaluations if item.group == group]
        evaluable = sum(item.capacity for item in group_results if item.status != "missing")
        raw = float(sum(item.signed_points for item in group_results))
        capacities[group] = evaluable
        raw_scores[group] = raw
        if evaluable >= MIN_EVALUABLE[group]:
            normalized = _clamp(raw / evaluable * WEIGHTS[group], -WEIGHTS[group], WEIGHTS[group])
            group_scores[group] = normalized
            usable_groups.append(group)
        else:
            group_scores[group] = None

    usable_weight = sum(WEIGHTS[group] for group in usable_groups)
    if len(usable_groups) < 2 or usable_weight < 60:
        return None

    total = sum(group_scores[group] or 0.0 for group in usable_groups) * 100.0 / usable_weight
    total = _clamp(total, -100.0, 100.0)
    bullish_points = float(
        sum(item.signed_points for item in evaluations if item.signed_points > 0)
    )
    bearish_points = float(
        -sum(item.signed_points for item in evaluations if item.signed_points < 0)
    )
    hit_points = bullish_points + bearish_points
    consistency = abs(bullish_points - bearish_points) / hit_points if hit_points else 0.0

    positive = [_evidence(item) for item in evaluations if item.status == "bullish"]
    negative = [_evidence(item) for item in evaluations if item.status == "bearish"]
    conflict_evidence, risks, risk_score = _risk_results(frame, data_warnings)
    risk_level = "低" if risk_score <= 1 else "中" if risk_score <= 3 else "高"
    return ScoreResult(
        total=total,
        group_scores=group_scores,
        signal=label_for_score(total),
        raw_group_scores=raw_scores,
        evaluable_capacity=capacities,
        usable_groups=usable_groups,
        bullish_points=bullish_points,
        bearish_points=bearish_points,
        positive_evidence=positive,
        negative_evidence=negative,
        conflict_evidence=conflict_evidence,
        risks=risks,
        watch_levels=build_watch_levels(frame),
        completeness=sum(capacities.values()) / TOTAL_CAPACITY,
        consistency=consistency,
        risk_score=risk_score,
        risk_level=risk_level,
    )


def build_watch_levels(frame: pd.DataFrame) -> list[WatchLevel]:
    """Build the seven allowlisted, strictly-positive observation levels in stable order."""

    if frame.empty:
        return []
    latest = frame.iloc[-1]
    close = _number(latest.get("close"))
    atr = _number(latest.get("atr14"))
    finite_lows = _recent_finite(frame, "low", 20)
    finite_highs = _recent_finite(frame, "high", 20)
    candidates: tuple[tuple[str, str, float | None, str], ...] = (
        (
            "recent_20d_low",
            "支撑观察",
            min(finite_lows) if finite_lows else None,
            "最近二十个有效交易日低点",
        ),
        (
            "recent_20d_high",
            "压力观察",
            max(finite_highs) if finite_highs else None,
            "最近二十个有效交易日高点",
        ),
        (
            "ma20",
            _relative_label(_number(latest.get("ma20")), close),
            _number(latest.get("ma20")),
            "MA20 观察位",
        ),
        ("boll_upper", "压力观察", _number(latest.get("boll_upper")), "布林带上轨观察位"),
        ("boll_lower", "支撑观察", _number(latest.get("boll_lower")), "布林带下轨观察位"),
        (
            "close_minus_atr",
            "波动参考",
            close - atr if close is not None and atr is not None else None,
            "收盘价减 ATR14 波动参考",
        ),
        (
            "close_plus_atr",
            "波动参考",
            close + atr if close is not None and atr is not None else None,
            "收盘价加 ATR14 波动参考",
        ),
    )
    return [
        WatchLevel(label=label, price=price, basis_key=key, rationale=rationale)
        for key, label, price, rationale in candidates
        if price is not None and price > 0
    ]


def canonical_json(value: object) -> str:
    """Serialize canonical JSON with sorted keys and exact finite-float formatting."""

    return _canonical(value)


def build_analysis_id(
    ts_code: str,
    resolved_end_date: date,
    lookback_months: int,
    frame: pd.DataFrame,
) -> str:
    """Hash the frozen identity fields and OHLCV rows available by the resolved date."""

    as_of = frame
    if "trade_date" in as_of.columns:
        dates = pd.to_datetime(as_of["trade_date"], errors="coerce")
        as_of = as_of.loc[dates.dt.date <= resolved_end_date]
    rows: list[dict[str, object]] = []
    volume_key = "vol" if "vol" in as_of.columns else "volume"
    selected = ("trade_date", "open", "high", "low", "close", volume_key)
    for _, row in as_of.loc[:, list(selected)].iterrows():
        rows.append(
            {
                "trade_date": row["trade_date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row[volume_key],
            }
        )
    market_digest = hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()
    payload = {
        "code": ts_code,
        "resolved_date": resolved_end_date,
        "lookback": lookback_months,
        "ohlcv_digest": market_digest,
        "indicator_version": "indicators-v1",
        "score_version": "score-v1",
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _latest_values(
    frame: pd.DataFrame, columns: Sequence[str], offset: int = -1
) -> list[float] | None:
    if len(frame) < abs(offset) or any(column not in frame.columns for column in columns):
        return None
    values = [_number(frame.iloc[offset].get(column)) for column in columns]
    return None if any(value is None for value in values) else [float(value) for value in values]


def _compare_pair(frame: pd.DataFrame, reference: str) -> tuple[RuleStatus, float | None]:
    values = _latest_values(frame, ("close", reference))
    if values is None:
        return "missing", None
    difference = values[0] - values[1]
    return ("bullish" if difference > 0 else "bearish" if difference < 0 else "neutral", difference)


def _price_ma20(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    return _compare_pair(frame, "ma20")


def _price_ma60(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    return _compare_pair(frame, "ma60")


def _ma_alignment(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    values = _latest_values(frame, ("ma5", "ma10", "ma20", "ma60"))
    if values is None:
        return "missing", None
    bullish = values[0] > values[1] > values[2] > values[3]
    bearish = values[0] < values[1] < values[2] < values[3]
    spread = min(values[index] - values[index + 1] for index in range(3))
    return ("bullish" if bullish else "bearish" if bearish else "neutral", spread)


def _macd_line(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    values = _latest_values(frame, ("macd", "macd_signal", "macd_hist"))
    if values is None:
        return "missing", None
    bullish = values[0] > values[1] and values[2] > 0
    bearish = values[0] < values[1] and values[2] < 0
    return ("bullish" if bullish else "bearish" if bearish else "neutral", values[2])


def _ratio_rule(
    frame: pd.DataFrame, column: str, shift: int, boundary: float
) -> tuple[RuleStatus, float | None]:
    latest = _latest_values(frame, (column,))
    past = _latest_values(frame, (column,), -(shift + 1))
    if latest is None or past is None:
        return "missing", None
    if past[0] == 0:
        return "neutral", 0.0
    change = latest[0] / past[0] - 1.0
    difference = latest[0] - past[0]
    threshold = abs(past[0]) * boundary
    return (
        "bullish"
        if difference > threshold
        else "bearish"
        if difference < -threshold
        else "neutral",
        change,
    )


def _ma20_slope(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    return _ratio_rule(frame, "ma20", 5, 0.005)


def _rsi14(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    values = _latest_values(frame, ("rsi14",))
    if values is None:
        return "missing", None
    value = values[0]
    return (
        "bullish" if 55 <= value <= 70 else "bearish" if 30 <= value <= 45 else "neutral",
        value,
    )


def _kdj_cross(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    if frame.empty or any(column not in frame.columns for column in ("kdj_k", "kdj_d")):
        return "missing", None
    if _latest_values(frame, ("kdj_k", "kdj_d")) is None:
        return "missing", None
    pairs = frame.loc[:, ["kdj_k", "kdj_d"]].apply(pd.to_numeric, errors="coerce")
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna().iloc[-4:]
    if len(pairs) < 4:
        return "missing", None
    differences = pairs["kdj_k"] - pairs["kdj_d"]
    qualified: list[tuple[int, RuleStatus]] = []
    for position in range(1, len(pairs)):
        current = float(differences.iloc[position])
        previous = float(differences.iloc[position - 1])
        cross_k = float(pairs["kdj_k"].iloc[position])
        if current > 0 and previous <= 0 and cross_k < 80:
            qualified.append((position, "bullish"))
        elif current < 0 and previous >= 0 and cross_k > 20:
            qualified.append((position, "bearish"))
    observed = float(differences.iloc[-1])
    return (qualified[-1][1] if qualified else "neutral", observed)


def _macd_hist_momentum(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    latest = _latest_values(frame, ("macd_hist",))
    previous = _latest_values(frame, ("macd_hist",), -2)
    if latest is None or previous is None:
        return "missing", None
    change = latest[0] - previous[0]
    bullish = latest[0] > 0 and change > 0
    bearish = latest[0] < 0 and change < 0
    return ("bullish" if bullish else "bearish" if bearish else "neutral", change)


def _return_20d(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    return _ratio_rule(frame, "close", 20, 0.03)


def _obv_price_trend(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    latest = _latest_values(frame, ("obv", "close"))
    past = _latest_values(frame, ("obv", "close"), -6)
    if latest is None or past is None:
        return "missing", None
    obv_slope = (latest[0] - past[0]) / 5.0
    if past[1] == 0:
        return "neutral", obv_slope
    price_return = latest[1] / past[1] - 1.0
    bullish = obv_slope > 0 and price_return > 0
    bearish = obv_slope < 0 and price_return < 0
    return ("bullish" if bullish else "bearish" if bearish else "neutral", obv_slope)


def _volume_confirmation(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    values = _latest_values(frame, ("close", "pre_close", "volume_ratio"))
    if values is None:
        return "missing", None
    bullish = values[0] > values[1] and values[2] >= 1.2
    bearish = values[0] < values[1] and values[2] >= 1.2
    return ("bullish" if bullish else "bearish" if bearish else "neutral", values[2])


def _boll_position(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    values = _latest_values(frame, ("close", "boll_mid", "boll_upper", "boll_lower"))
    if values is None:
        return "missing", None
    bullish = values[1] < values[0] <= values[2]
    bearish = values[3] <= values[0] < values[1]
    return ("bullish" if bullish else "bearish" if bearish else "neutral", values[0] - values[1])


def _boll_breakout(frame: pd.DataFrame) -> tuple[RuleStatus, float | None]:
    values = _latest_values(frame, ("close", "boll_upper", "boll_lower"))
    if values is None:
        return "missing", None
    if values[0] > values[1]:
        return "bullish", values[0] - values[1]
    if values[0] < values[2]:
        return "bearish", values[0] - values[2]
    return "neutral", 0.0


_PREDICATES: Final = {
    "price_ma20": _price_ma20,
    "price_ma60": _price_ma60,
    "ma_alignment": _ma_alignment,
    "macd_line": _macd_line,
    "ma20_slope": _ma20_slope,
    "rsi14": _rsi14,
    "kdj_cross": _kdj_cross,
    "macd_hist_momentum": _macd_hist_momentum,
    "return_20d": _return_20d,
    "obv_price_trend": _obv_price_trend,
    "volume_confirmation": _volume_confirmation,
    "boll_position": _boll_position,
    "boll_breakout": _boll_breakout,
}


def _evidence(item: RuleEvaluation) -> Evidence:
    if item.observed_value is None or item.interpretation is None:
        raise ValueError("hit direction rules require finite evidence")
    return Evidence(
        source_key=item.key,
        observed_value=item.observed_value,
        interpretation=item.interpretation,
    )


def _risk_results(
    frame: pd.DataFrame, data_warnings: Sequence[str]
) -> tuple[list[Evidence], list[Risk], int]:
    conflicts: list[Evidence] = []
    risks: list[Risk] = []
    points = 0
    atr_ratio = _atr_ratio(frame)
    if atr_ratio is not None and atr_ratio > 0.04:
        atr_points = 2 if atr_ratio > 0.06 else 1
        risks.append(
            Risk(
                risk_type="volatility", evidence_key="atr_ratio", description="ATR14 相对收盘价偏高"
            )
        )
        points += atr_points

    breakout = _boll_breakout(frame)[0]
    if breakout in ("bullish", "bearish"):
        risks.append(
            Risk(
                risk_type="volatility",
                evidence_key="boll_breakout",
                description="收盘价位于布林带轨外",
            )
        )
        points += 2

    rsi = _latest_values(frame, ("rsi14",))
    if rsi is not None and (rsi[0] > 75 or rsi[0] < 25):
        risk_type = "overbought" if rsi[0] > 75 else "oversold"
        description = "RSI 进入超买区" if rsi[0] > 75 else "RSI 进入超卖区"
        risks.append(Risk(risk_type=risk_type, evidence_key="rsi14", description=description))
        points += 2

    divergence = _divergence(frame)
    if divergence is not None:
        conflicts.append(
            Evidence(
                source_key="price_obv_divergence_10d",
                observed_value=divergence,
                interpretation="价格与 OBV 十日方向相反",
            )
        )
        risks.append(
            Risk(
                risk_type="signal_conflict",
                evidence_key="price_obv_divergence_10d",
                description="价格与 OBV 十日方向背离",
            )
        )
        points += 1

    for warning in data_warnings:
        text = str(warning).strip()
        if text:
            risks.append(
                Risk(risk_type="data_quality", evidence_key="data_quality", description=text[:120])
            )
            points += 1
    return conflicts, risks, points


def _atr_ratio(frame: pd.DataFrame) -> float | None:
    values = _latest_values(frame, ("atr14", "close"))
    if values is None or values[1] <= 0:
        return None
    ratio = values[0] / values[1]
    return ratio if np.isfinite(ratio) else None


def _divergence(frame: pd.DataFrame) -> float | None:
    latest = _latest_values(frame, ("close", "obv"))
    past = _latest_values(frame, ("close", "obv"), -11)
    if latest is None or past is None or past[0] == 0:
        return None
    price_return = latest[0] / past[0] - 1.0
    obv_change = latest[1] - past[1]
    if price_return * obv_change < 0:
        return price_return
    return None


def _relative_label(level: float | None, close: float | None) -> str:
    if level is not None and close is not None and level <= close:
        return "支撑观察"
    return "压力观察"


def _number(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _recent_finite(frame: pd.DataFrame, column: str, count: int) -> list[float]:
    if column not in frame.columns:
        return []
    values = [_number(value) for value in frame[column]]
    return [value for value in values if value is not None][-count:]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _canonical(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return "null"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return "null" if pd.isna(timestamp) else json.dumps(timestamp.date().isoformat())
    if isinstance(value, datetime):
        return json.dumps(value.isoformat(), ensure_ascii=False)
    if isinstance(value, date):
        return json.dumps(value.isoformat(), ensure_ascii=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return "null"
        if not np.isfinite(number):
            raise ValueError("canonical JSON does not permit infinite floats")
        return format(number, ".17g")
    if isinstance(value, Mapping):
        items = sorted(((str(key), item) for key, item in value.items()), key=lambda pair: pair[0])
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{_canonical(item)}" for key, item in items
            )
            + "}"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")
