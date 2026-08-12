from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from stock_agent.llm.schemas import (
    NUMERIC_SOURCE_KEYS,
    SemanticValidationError,
    ai_cache_key,
    flatten_numeric_snapshot,
    parse_and_validate_interpretation,
)


SNAPSHOT = {
    "close": 100.0,
    "pct_chg": 1.25,
    "ma20": 99.5,
    "rsi14": 55.0,
    "atr14": 2.0,
    "atr_ratio": 0.02,
    "trade_date": "2026-08-11",
    "nested": {"close": 999.0},
}
WATCH_LEVELS = {
    "recent_20d_low": 95.0,
    "recent_20d_high": 105.0,
    "ma20": 99.5,
    "boll_upper": 106.0,
    "boll_lower": 94.0,
    "close_minus_atr": 98.0,
    "close_plus_atr": 102.0,
}


def raw_payload() -> dict[str, object]:
    return {
        "model_signal": "中性偏多",
        "summary": "价格位于二十日均线上方，但仍应关注波动。",
        "evidence": [
            {"source_key": "close", "observed_value": 100.0, "interpretation": "最新收盘价"},
            {"source_key": "ma20", "observed_value": 99.5, "interpretation": "二十日均线"},
        ],
        "risks": [
            {
                "risk_type": "volatility",
                "evidence_key": "atr_ratio",
                "description": "波动仍需观察",
            }
        ],
        "watch_levels": [
            {
                "label": "支撑观察",
                "price": 99.5,
                "basis_key": "ma20",
                "rationale": "二十日均线附近",
            }
        ],
    }


def parse(payload: dict[str, object]):
    return parse_and_validate_interpretation(
        json.dumps(payload, ensure_ascii=False, allow_nan=True),
        snapshot=SNAPSHOT,
        watch_levels=WATCH_LEVELS,
    )


def test_ai_cache_key_is_model_and_prompt_bound() -> None:
    baseline = ai_cache_key("analysis", "deepseek-v4-flash", "prompt-v1")
    assert baseline != ai_cache_key("analysis", "deepseek-v4-pro", "prompt-v1")
    assert baseline != ai_cache_key("analysis", "deepseek-v4-flash", "prompt-v2")
    assert len(baseline) == 64


def test_snapshot_flattening_uses_only_frozen_top_level_finite_numbers() -> None:
    snapshot = {
        **SNAPSHOT,
        "macd": float("nan"),
        "obv": float("inf"),
        "volume_ratio": True,
        "unknown": 123.0,
    }

    flattened = flatten_numeric_snapshot(snapshot)

    assert flattened == {
        "close": 100.0,
        "pct_chg": 1.25,
        "ma20": 99.5,
        "rsi14": 55.0,
        "atr14": 2.0,
        "atr_ratio": 0.02,
    }
    assert set(flattened) <= set(NUMERIC_SOURCE_KEYS)


def test_valid_minimum_payload_is_grounded() -> None:
    raw = parse(raw_payload())
    assert raw.evidence[0].source_key == "close"
    assert raw.watch_levels[0].basis_key == "ma20"


def test_valid_maximum_arrays_are_accepted() -> None:
    payload = raw_payload()
    payload["evidence"] = [
        {"source_key": "close", "observed_value": 100.0, "interpretation": str(i)}
        for i in range(6)
    ]
    payload["risks"] = [
        {"risk_type": "other", "evidence_key": "close", "description": str(i)}
        for i in range(6)
    ]
    payload["watch_levels"] = [
        {
            "label": "波动参考",
            "price": 102.0,
            "basis_key": "close_plus_atr",
            "rationale": str(i),
        }
        for i in range(6)
    ]
    assert len(parse(payload).evidence) == 6


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": "forged"},
        {"rule_signal": "偏空"},
        {"model_signal": "强烈买入"},
        {"summary": ""},
        {"evidence": []},
        {"risks": []},
        {"watch_levels": []},
    ],
)
def test_raw_schema_rejects_extra_invalid_or_out_of_bounds_fields(change) -> None:
    with pytest.raises(ValidationError):
        parse({**raw_payload(), **change})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_raw_schema_rejects_non_finite_json_numbers(invalid: float) -> None:
    payload = raw_payload()
    payload["evidence"][0]["observed_value"] = invalid  # type: ignore[index]
    with pytest.raises((ValueError, ValidationError)):
        parse(payload)


def test_semantics_reject_hallucinated_or_missing_evidence_keys() -> None:
    payload = raw_payload()
    payload["evidence"][0]["source_key"] = "pe_ratio"  # type: ignore[index]
    with pytest.raises(SemanticValidationError, match="source_key"):
        parse(payload)

    payload = raw_payload()
    payload["risks"][0]["evidence_key"] = "macd"  # type: ignore[index]
    with pytest.raises(SemanticValidationError, match="evidence_key"):
        parse(payload)


@pytest.mark.parametrize("observed", [100.001, 99.999])
def test_semantics_reject_numeric_mismatch_beyond_exact_tolerance(observed: float) -> None:
    payload = raw_payload()
    payload["evidence"][0]["observed_value"] = observed  # type: ignore[index]
    with pytest.raises(SemanticValidationError, match="observed_value"):
        parse(payload)


def test_semantics_accept_numeric_value_on_tolerance_boundary() -> None:
    payload = raw_payload()
    payload["evidence"][0]["observed_value"] = 100.0001  # type: ignore[index]
    assert parse(payload).evidence[0].observed_value == pytest.approx(100.0001)


def test_semantics_reject_unknown_or_mismatched_watch_level() -> None:
    payload = raw_payload()
    payload["watch_levels"][0]["price"] = 99.511  # type: ignore[index]
    with pytest.raises(SemanticValidationError, match="price"):
        parse(payload)

    payload = raw_payload()
    payload["watch_levels"][0]["basis_key"] = "recent_20d_low"  # type: ignore[index]
    with pytest.raises(SemanticValidationError, match="price"):
        parse(payload)


def test_nested_extra_property_and_duplicate_json_key_are_rejected() -> None:
    payload = raw_payload()
    payload["evidence"][0]["invented"] = 1  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse(payload)

    duplicate = json.dumps(raw_payload(), ensure_ascii=False)[:-1] + ',"summary":"伪造"}'
    with pytest.raises(ValueError, match="duplicate"):
        parse_and_validate_interpretation(
            duplicate,
            snapshot=SNAPSHOT,
            watch_levels=WATCH_LEVELS,
        )
