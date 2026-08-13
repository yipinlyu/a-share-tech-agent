from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from stock_agent.domain.models import AIInterpretation, AgentError, INVESTMENT_DISCLAIMER
from stock_agent.llm.deepseek_client import DeepSeekClient
from stock_agent.llm.schemas import ai_cache_key


NOW = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
SNAPSHOT = {"close": 100.0, "ma20": 99.5, "atr_ratio": 0.02}
WATCH_LEVELS = [{"label": "支撑观察", "price": 99.5, "basis_key": "ma20", "rationale": "均线"}]
ANALYSIS = {
    "stock": {"ts_code": "600519.SH", "name": "贵州茅台"},
    "snapshot": SNAPSHOT,
    "score": {
        "signal": "中性偏多",
        "risks": [
            {
                "risk_type": "volatility",
                "evidence_key": "atr_ratio",
                "description": "波动风险",
            }
        ],
        "watch_levels": WATCH_LEVELS,
    },
}


def valid_raw(**overrides: object) -> dict[str, object]:
    payload = {
        "model_signal": "中性偏多",
        "summary": "趋势略偏强，但仍需关注波动。",
        "evidence": [
            {"source_key": "close", "observed_value": 100.0, "interpretation": "收盘价"},
            {"source_key": "ma20", "observed_value": 99.5, "interpretation": "均线"},
        ],
        "risks": [
            {"risk_type": "volatility", "evidence_key": "atr_ratio", "description": "波动风险"}
        ],
        "watch_levels": WATCH_LEVELS,
    }
    return {**payload, **overrides}


def response(payload: dict[str, object] | str):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(
        id="completion-1",
        model="deepseek-v4-flash",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


class FakeCompletions:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeOpenAI:
    def __init__(self, outcomes) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))

    def factory(self, **kwargs):
        self.init_calls.append(kwargs)
        return self


class FakeCache:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.put_calls: list[tuple[str, object]] = []

    def get_ai(self, key: str):
        return self.values.get(key)

    def put_ai(self, key: str, payload: object) -> None:
        self.put_calls.append((key, payload))
        self.values[key] = payload


class FakeStatusError(Exception):
    def __init__(self, status_code: int, secret: str = "sk-do-not-leak") -> None:
        super().__init__(secret)
        self.status_code = status_code


def make_client(outcomes, *, cache=None, model="deepseek-v4-flash"):
    fake = FakeOpenAI(outcomes)
    client = DeepSeekClient(
        api_key="sk-test-only",
        model=model,
        prompt_version="prompt-v2",
        cache=cache,
        openai_factory=fake.factory,
        clock=lambda: NOW,
    )
    return client, fake


def interpret(client, **kwargs):
    return client.interpret("analysis-1", ANALYSIS, **kwargs)


def test_initializes_official_openai_client_and_makes_bounded_json_call() -> None:
    client, fake = make_client([response(valid_raw())])

    result = interpret(client)

    assert isinstance(result, AIInterpretation)
    assert fake.init_calls == [
        {
            "api_key": "sk-test-only",
            "base_url": "https://api.deepseek.com",
            "timeout": 30,
            "max_retries": 2,
        }
    ]
    call = fake.chat.completions.calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert len(fake.chat.completions.calls) == 1
    system_prompt = call["messages"][0]["content"]
    assert "不得重新计算" in system_prompt
    assert "个性化" in system_prompt
    assert "仓位" in system_prompt


def test_server_owns_all_enrichment_and_caches_only_validated_raw_result() -> None:
    forged = valid_raw(
        rule_signal="偏空",
        disclaimer="立即满仓",
        model="forged",
        cache_hit=True,
    )
    cache = FakeCache()
    client, fake = make_client([response(forged), response(valid_raw())], cache=cache)

    result = interpret(client)

    assert isinstance(result, AIInterpretation)
    assert len(fake.chat.completions.calls) == 2
    assert result.rule_signal == "中性偏多"
    assert result.consistency_status == "consistent"
    assert result.disclaimer == INVESTMENT_DISCLAIMER
    assert result.model == "deepseek-v4-flash"
    assert result.prompt_version == "prompt-v2"
    assert result.cache_hit is False
    assert result.generated_at == NOW
    assert len(cache.put_calls) == 1
    cached = cache.put_calls[0][1]
    assert set(cached) == {"raw", "generated_at"}


def test_valid_cache_hit_avoids_external_call_and_marks_result() -> None:
    cache = FakeCache()
    first, first_fake = make_client([response(valid_raw())], cache=cache)
    generated = interpret(first)
    second, second_fake = make_client([], cache=cache)

    cached = interpret(second)

    assert isinstance(generated, AIInterpretation)
    assert isinstance(cached, AIInterpretation)
    assert first_fake.chat.completions.calls
    assert second_fake.chat.completions.calls == []
    assert cached.cache_hit is True
    assert cached.generated_at == generated.generated_at


def test_cache_key_is_bound_to_configured_actual_model() -> None:
    cache = FakeCache()
    first, _ = make_client([response(valid_raw())], cache=cache)
    interpret(first)
    second, second_fake = make_client([response(valid_raw())], cache=cache, model="deepseek-v4-pro")

    result = interpret(second)

    assert isinstance(result, AIInterpretation)
    assert second_fake.chat.completions.calls[0]["model"] == "deepseek-v4-pro"
    assert set(cache.values) == {
        ai_cache_key("analysis-1", "deepseek-v4-flash", "prompt-v2"),
        ai_cache_key("analysis-1", "deepseek-v4-pro", "prompt-v2"),
    }


@pytest.mark.parametrize(
    "invalid",
    [
        "not-json",
        valid_raw(evidence=[]),
        valid_raw(
            evidence=[
                {"source_key": "close", "observed_value": 999.0, "interpretation": "伪造"},
                {"source_key": "ma20", "observed_value": 99.5, "interpretation": "均线"},
            ]
        ),
    ],
)
def test_schema_or_semantic_failure_gets_exactly_one_repair(invalid) -> None:
    client, fake = make_client([response(invalid), response(valid_raw())])

    result = interpret(client)

    assert isinstance(result, AIInterpretation)
    assert len(fake.chat.completions.calls) == 2
    repair_messages = fake.chat.completions.calls[1]["messages"]
    assert "仅修复 JSON" in repair_messages[-1]["content"]


def test_interpret_prompt_supplies_schema_and_exact_grounding_contract() -> None:
    client, fake = make_client([response("bad"), response(valid_raw())])

    result = interpret(client)

    assert isinstance(result, AIInterpretation)
    initial_user_prompt = fake.chat.completions.calls[0]["messages"][-1]["content"]
    repair_prompt = fake.chat.completions.calls[1]["messages"][-1]["content"]
    for prompt in (initial_user_prompt, repair_prompt):
        assert "JSON Schema" in prompt
        assert "observed_value 必须逐字复制" in prompt
        assert "price 必须逐字复制" in prompt


def test_seven_server_watch_levels_are_accepted_without_repair() -> None:
    basis_and_prices = [
        ("recent_20d_low", 95.0),
        ("recent_20d_high", 105.0),
        ("ma20", 99.5),
        ("boll_upper", 106.0),
        ("boll_lower", 94.0),
        ("close_minus_atr", 98.0),
        ("close_plus_atr", 102.0),
    ]
    levels = [
        {
            "label": "波动参考",
            "price": price,
            "basis_key": basis,
            "rationale": basis,
        }
        for basis, price in basis_and_prices
    ]
    analysis = {
        **ANALYSIS,
        "score": {"signal": "中性偏多", "watch_levels": levels, "risks": []},
    }
    client, fake = make_client([response(valid_raw(risks=[], watch_levels=levels))])

    result = client.interpret("analysis-1", analysis)

    assert isinstance(result, AIInterpretation)
    assert len(result.watch_levels) == 7
    assert len(fake.chat.completions.calls) == 1


def test_interpretation_prompt_isolates_rule_keys_and_requires_server_risks() -> None:
    analysis = {
        **ANALYSIS,
        "score": {
            "signal": "中性偏多",
            "positive_evidence": [
                {
                    "source_key": "price_ma20",
                    "observed_value": 1.0,
                    "interpretation": "收盘价高于 MA20",
                }
            ],
            "conflict_evidence": [
                {
                    "source_key": "price_obv_divergence_10d",
                    "observed_value": -0.01,
                    "interpretation": "价格与 OBV 方向背离",
                }
            ],
            "risks": [
                {
                    "risk_type": "signal_conflict",
                    "evidence_key": "price_obv_divergence_10d",
                    "description": "价格与 OBV 方向背离",
                }
            ],
            "watch_levels": WATCH_LEVELS,
        },
    }
    model_risk = {
        "risk_type": "signal_conflict",
        "evidence_key": None,
        "description": "价格与 OBV 方向背离",
    }
    client, fake = make_client([response(valid_raw(risks=[model_risk]))])

    result = client.interpret("analysis-1", analysis)

    assert isinstance(result, AIInterpretation)
    prompt = fake.chat.completions.calls[0]["messages"][-1]["content"]
    payload = json.loads(prompt.split("请解释以下服务端结构化分析：", 1)[1])
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    assert "source_key" not in serialized_payload
    assert "evidence_key" not in serialized_payload
    assert "price_ma20" not in serialized_payload
    assert "price_obv_divergence_10d" not in serialized_payload
    assert payload["rule_observations"]["positive"] == ["收盘价高于 MA20"]


def test_model_cannot_invent_risk_when_server_risk_list_is_empty() -> None:
    invented = valid_raw(
        risks=[{"risk_type": "other", "evidence_key": None, "description": "未知风险"}]
    )
    client, fake = make_client([response(invented), response(valid_raw(risks=[]))])

    analysis = {**ANALYSIS, "score": {**ANALYSIS["score"], "risks": []}}
    result = client.interpret("analysis-1", analysis)

    assert isinstance(result, AIInterpretation)
    assert result.risks == []
    assert len(fake.chat.completions.calls) == 2


def test_repair_failure_returns_safe_error_and_is_not_cached() -> None:
    cache = FakeCache()
    client, fake = make_client([response("bad"), response("still bad")], cache=cache)

    result = interpret(client)

    assert isinstance(result, AgentError)
    assert result.code == "MODEL"
    assert result.retryable is True
    assert len(fake.chat.completions.calls) == 2
    assert cache.put_calls == []
    assert "bad" not in result.user_message


@pytest.mark.parametrize(
    ("status", "code", "retryable", "hint"),
    [
        (401, "AUTH", False, "认证"),
        (402, "MODEL", False, "余额"),
        (429, "RATE_LIMIT", True, "稍后"),
        (500, "MODEL", True, "稍后"),
        (503, "MODEL", True, "稍后"),
    ],
)
def test_status_errors_are_safely_mapped_without_repair(
    status: int, code: str, retryable: bool, hint: str
) -> None:
    cache = FakeCache()
    client, fake = make_client([FakeStatusError(status)], cache=cache)

    result = interpret(client)

    assert isinstance(result, AgentError)
    assert (result.code, result.retryable) == (code, retryable)
    assert hint in result.user_message
    assert "sk-do-not-leak" not in result.user_message
    assert len(fake.chat.completions.calls) == 1
    assert cache.put_calls == []


def test_transport_error_does_not_enter_repair_or_cache() -> None:
    cache = FakeCache()
    client, fake = make_client([TimeoutError("sk-timeout")], cache=cache)

    result = interpret(client)

    assert isinstance(result, AgentError)
    assert result.code == "MODEL"
    assert result.retryable is True
    assert "sk-timeout" not in result.user_message
    assert len(fake.chat.completions.calls) == 1
    assert cache.put_calls == []
