"""Strict model-output parsing and grounding checks for DeepSeek."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Final

from stock_agent.domain.models import AIRawInterpretation

NUMERIC_SOURCE_KEYS: Final[tuple[str, ...]] = (
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
WATCH_BASIS_KEYS: Final[tuple[str, ...]] = (
    "recent_20d_low",
    "recent_20d_high",
    "ma20",
    "boll_upper",
    "boll_lower",
    "close_minus_atr",
    "close_plus_atr",
)
RISK_TYPES: Final[tuple[str, ...]] = (
    "volatility",
    "overbought",
    "oversold",
    "signal_conflict",
    "data_quality",
    "other",
)
MODEL_RESPONSE_JSON_SCHEMA: Final[dict[str, Any]] = AIRawInterpretation.model_json_schema()


class SemanticValidationError(ValueError):
    """The JSON is structurally valid but cites facts outside server evidence."""


def ai_cache_key(analysis_id: str, model: str, prompt_version: str) -> str:
    """Bind an interpretation cache entry to data, actual model, and prompt."""

    parts = (analysis_id, model, prompt_version)
    if any(not isinstance(part, str) or not part.strip() for part in parts):
        raise ValueError("cache key inputs must be non-empty strings")
    canonical = "|".join(part.strip() for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def flatten_numeric_snapshot(snapshot: Mapping[str, object]) -> dict[str, float]:
    """Return only whitelisted, top-level, finite numeric snapshot values."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    flattened: dict[str, float] = {}
    for key in NUMERIC_SOURCE_KEYS:
        value = snapshot.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            flattened[key] = number
    return flattened


def watch_level_mapping(
    watch_levels: Mapping[str, object] | Iterable[Mapping[str, object] | object],
) -> dict[str, float]:
    """Normalize the seven server-computed observation prices by basis key."""

    values: dict[str, float] = {}
    if isinstance(watch_levels, Mapping):
        items: Iterable[tuple[object, object]] = watch_levels.items()
    else:
        normalized: list[tuple[object, object]] = []
        for item in watch_levels:
            if isinstance(item, Mapping):
                normalized.append((item.get("basis_key"), item.get("price")))
            else:
                normalized.append((getattr(item, "basis_key", None), getattr(item, "price", None)))
        items = normalized

    for raw_key, raw_value in items:
        if raw_key not in WATCH_BASIS_KEYS:
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        if math.isfinite(value) and value > 0:
            values[str(raw_key)] = value
    return values


def normalize_server_risks(
    risks: Iterable[Mapping[str, object] | object],
    *,
    snapshot: Mapping[str, object],
) -> list[dict[str, object]]:
    """Expose only server-owned risk identities and safe descriptions to the model."""

    facts = flatten_numeric_snapshot(snapshot)
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str | None, str]] = set()
    for item in risks:
        if isinstance(item, Mapping):
            risk_type = item.get("risk_type")
            evidence_key = item.get("evidence_key")
            description = item.get("description")
        else:
            risk_type = getattr(item, "risk_type", None)
            evidence_key = getattr(item, "evidence_key", None)
            description = getattr(item, "description", None)
        if (
            risk_type not in RISK_TYPES
            or not isinstance(description, str)
            or not description.strip()
        ):
            continue
        grounded_key = (
            evidence_key if isinstance(evidence_key, str) and evidence_key in facts else None
        )
        clean_description = description.strip()[:120]
        identity = (str(risk_type), grounded_key, clean_description)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(
            {
                "risk_type": str(risk_type),
                "evidence_key": grounded_key,
                "description": clean_description,
            }
        )
    return normalized[:6]


def parse_and_validate_interpretation(
    content: str,
    *,
    snapshot: Mapping[str, object],
    watch_levels: Mapping[str, object] | Iterable[Mapping[str, object] | object],
    server_risks: Iterable[Mapping[str, object] | object] = (),
) -> AIRawInterpretation:
    """Parse strict JSON, validate the raw schema, then ground every citation."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response must be non-empty JSON text")
    payload = json.loads(
        content,
        parse_constant=_reject_non_finite,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise ValueError("model response JSON root must be an object")
    raw = AIRawInterpretation.model_validate(payload)
    return validate_interpretation_semantics(
        raw,
        snapshot=snapshot,
        watch_levels=watch_levels,
        server_risks=server_risks,
    )


def validate_interpretation_semantics(
    raw: AIRawInterpretation,
    *,
    snapshot: Mapping[str, object],
    watch_levels: Mapping[str, object] | Iterable[Mapping[str, object] | object],
    server_risks: Iterable[Mapping[str, object] | object] = (),
) -> AIRawInterpretation:
    """Require every AI number and evidence key to match server-owned facts."""

    if not isinstance(raw, AIRawInterpretation):
        raise TypeError("raw must be an AIRawInterpretation")
    facts = flatten_numeric_snapshot(snapshot)
    levels = watch_level_mapping(watch_levels)
    allowed_risks = normalize_server_risks(server_risks, snapshot=snapshot)
    allowed_risk_counts = Counter(
        (str(item["risk_type"]), item["evidence_key"], str(item["description"]))
        for item in allowed_risks
    )

    seen_evidence: set[str] = set()
    for evidence in raw.evidence:
        if evidence.source_key in seen_evidence:
            raise SemanticValidationError("evidence source_key must be unique")
        seen_evidence.add(evidence.source_key)
        expected = facts.get(evidence.source_key)
        if expected is None:
            raise SemanticValidationError("source_key is not an available whitelisted fact")
        tolerance = max(1e-6 * abs(expected), 1e-6)
        difference = abs(evidence.observed_value - expected)
        if difference > tolerance and not math.isclose(
            difference, tolerance, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise SemanticValidationError("observed_value does not match the cited fact")

    cited_risk_counts: Counter[tuple[str, str | None, str]] = Counter()
    for risk in raw.risks:
        identity = (risk.risk_type, risk.evidence_key, risk.description)
        cited_risk_counts[identity] += 1
        if cited_risk_counts[identity] > allowed_risk_counts[identity]:
            raise SemanticValidationError("risk is not present in the server risk facts")

    seen_levels: set[str] = set()
    for level in raw.watch_levels:
        if level.basis_key in seen_levels:
            raise SemanticValidationError("watch-level basis_key must be unique")
        seen_levels.add(level.basis_key)
        expected = levels.get(level.basis_key)
        if expected is None:
            raise SemanticValidationError("basis_key is not an available server observation")
        tolerance = max(0.01, 1e-6 * abs(expected))
        difference = abs(level.price - expected)
        if difference > tolerance and not math.isclose(
            difference, tolerance, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise SemanticValidationError("watch-level price does not match its basis_key")
    return raw


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


__all__ = [
    "MODEL_RESPONSE_JSON_SCHEMA",
    "NUMERIC_SOURCE_KEYS",
    "WATCH_BASIS_KEYS",
    "SemanticValidationError",
    "ai_cache_key",
    "flatten_numeric_snapshot",
    "parse_and_validate_interpretation",
    "validate_interpretation_semantics",
    "watch_level_mapping",
]
