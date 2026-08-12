"""Bounded DeepSeek adapter for grounded interpretations and follow-up chat."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Protocol
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError
from stock_agent.domain.models import (
    AIInterpretation,
    AIRawInterpretation,
    AgentError,
    ChatResponse,
)
from stock_agent.llm.schemas import (
    ai_cache_key,
    flatten_numeric_snapshot,
    parse_and_validate_interpretation,
    watch_level_mapping,
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_PROMPT_VERSION = "prompt-v1"
MAX_CHAT_PAIRS = 10

_INTERPRET_SYSTEM_PROMPT = """你是受约束的 A 股技术指标解释器。仅依据用户提供的结构化事实输出 JSON。
不得重新计算、覆盖或伪造任何指标、规则信号和观察位；不得引入外部事实。
不得提供个性化买卖建议、收益承诺、自动交易、仓位或头寸规模建议。
只输出约定字段：model_signal、summary、evidence、risks、watch_levels。"""
_CHAT_SYSTEM_PROMPT = """你只解释当前这份结构化 A 股技术分析及已有问答。
不得引入外部事实、重新计算指标、提供个性化买卖决定、仓位或收益承诺。
只输出 JSON 对象 {\"answer\": \"中文回答\"}，必要时重申仅供学习研究、不构成投资建议。"""
_REPAIR_PROMPT = "仅修复 JSON，使其严格符合字段、数量、数值和证据约束；不要解释或添加事实。"


class AICache(Protocol):
    def get_ai(self, cache_key: str) -> object | None: ...

    def put_ai(self, cache_key: str, payload: object) -> None: ...


class _ChatRaw(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


def append_turn(
    history: Sequence[Mapping[str, str]],
    question: str,
    answer: str,
    *,
    max_pairs: int = MAX_CHAT_PAIRS,
) -> list[dict[str, str]]:
    """Append one complete pair and evict only whole oldest pairs."""

    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer cannot be empty")
    copied = [dict(message) for message in history]
    copied.extend(
        [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    )
    return copied[-2 * max_pairs :]


@dataclass
class ChatMemory:
    """Session-only state for one current structured analysis."""

    analysis_id: str | None = None
    structured_analysis: Mapping[str, object] | None = None
    thread_id: str = field(default_factory=lambda: str(uuid4()))
    history: list[dict[str, str]] = field(default_factory=list)

    def set_analysis(self, analysis_id: str, structured_analysis: Mapping[str, object]) -> str:
        clean_id = analysis_id.strip() if isinstance(analysis_id, str) else ""
        if not clean_id or not isinstance(structured_analysis, Mapping):
            raise ValueError("a current structured analysis is required")
        if clean_id != self.analysis_id:
            self.analysis_id = clean_id
            self.thread_id = str(uuid4())
            self.history = []
        self.structured_analysis = structured_analysis
        return self.thread_id

    def append(self, question: str, answer: str) -> None:
        self.history = append_turn(self.history, question, answer)

    @property
    def turn_count(self) -> int:
        return min(len(self.history) // 2, MAX_CHAT_PAIRS)


class DeepSeekClient:
    """OpenAI-compatible client with strict validation and no implicit calls."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = DEFAULT_MODEL,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        cache: AICache | None = None,
        openai_factory: Callable[..., object] = OpenAI,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if not clean_key:
            raise ValueError("DeepSeek API key is required")
        if base_url.rstrip("/") != DEEPSEEK_BASE_URL:
            raise ValueError("DeepSeek base URL must use the official HTTPS endpoint")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version must be a non-empty string")

        self.model = model.strip()
        self.prompt_version = prompt_version.strip()
        self._cache = cache
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._client = openai_factory(
            api_key=clean_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=30,
            max_retries=2,
        )

    def interpret(
        self,
        analysis_id: str,
        structured_analysis: Mapping[str, object] | object,
        *,
        force_refresh: bool = False,
    ) -> AIInterpretation | AgentError:
        """Generate or retrieve one validated interpretation on explicit demand."""

        try:
            analysis = _as_mapping(structured_analysis)
            clean_id, snapshot, levels, rule_signal = _analysis_facts(analysis_id, analysis)
            key = ai_cache_key(clean_id, self.model, self.prompt_version)
        except (TypeError, ValueError, ValidationError):
            return _error("VALIDATION", "当前结构化分析无效，无法生成 AI 解读。", False)

        if not force_refresh:
            cached = self._read_cache(key, snapshot=snapshot, levels=levels)
            if cached is not None:
                raw, generated_at = cached
                return AIInterpretation.from_raw(
                    raw,
                    rule_signal=rule_signal,  # type: ignore[arg-type]
                    model=self.model,
                    prompt_version=self.prompt_version,
                    generated_at=generated_at,
                    cache_hit=True,
                )

        messages = [
            {"role": "system", "content": _INTERPRET_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请解释以下服务端结构化分析："
                + _json_text(_safe_analysis_payload(analysis)),
            },
        ]
        try:
            content = self._complete(messages)
        except BaseException as exc:
            return _map_model_error(exc)

        try:
            raw = parse_and_validate_interpretation(
                content,
                snapshot=snapshot,
                watch_levels=levels,
            )
        except (ValueError, TypeError, ValidationError):
            repair_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": _REPAIR_PROMPT},
            ]
            try:
                repaired = self._complete(repair_messages)
            except BaseException as exc:
                return _map_model_error(exc)
            try:
                raw = parse_and_validate_interpretation(
                    repaired,
                    snapshot=snapshot,
                    watch_levels=levels,
                )
            except (ValueError, TypeError, ValidationError):
                return _error("MODEL", "AI 返回内容无法安全校验，请稍后重试。", True)

        generated_at = _aware_datetime(self._clock())
        self._write_cache(key, raw, generated_at)
        return AIInterpretation.from_raw(
            raw,
            rule_signal=rule_signal,  # type: ignore[arg-type]
            model=self.model,
            prompt_version=self.prompt_version,
            generated_at=generated_at,
            cache_hit=False,
        )

    def follow_up(self, memory: ChatMemory, question: str) -> ChatResponse:
        """Answer from current analysis plus at most ten successful complete pairs."""

        if not isinstance(memory, ChatMemory):
            raise TypeError("memory must be ChatMemory")
        clean_question = question.strip() if isinstance(question, str) else ""
        if not clean_question or len(clean_question) > 500:
            return _chat_failure(memory, "VALIDATION", "追问须为 1–500 个字符。", False)
        if not memory.analysis_id or memory.structured_analysis is None:
            return _chat_failure(memory, "VALIDATION", "请先完成当前股票分析再追问。", False)

        try:
            context = _json_text(_safe_analysis_payload(_as_mapping(memory.structured_analysis)))
        except (TypeError, ValueError):
            return _chat_failure(memory, "VALIDATION", "当前结构化分析无效，请重新分析。", False)

        messages = [
            {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
            {"role": "system", "content": "当前结构化分析：" + context},
            *[dict(message) for message in memory.history[-2 * MAX_CHAT_PAIRS :]],
            {"role": "user", "content": clean_question},
        ]
        try:
            content = self._complete(messages)
        except BaseException as exc:
            return _chat_failure_from_error(memory, _map_model_error(exc))

        try:
            answer = _parse_chat(content).answer
        except (ValueError, TypeError, ValidationError):
            repair_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": _REPAIR_PROMPT},
            ]
            try:
                repaired = self._complete(repair_messages)
                answer = _parse_chat(repaired).answer
            except BaseException as exc:
                if isinstance(exc, (ValueError, TypeError, ValidationError)):
                    error = _error("MODEL", "AI 追问内容无法安全校验，请稍后重试。", True)
                else:
                    error = _map_model_error(exc)
                return _chat_failure_from_error(memory, error)

        memory.append(clean_question, answer)
        return ChatResponse(
            answer=answer,
            thread_id=memory.thread_id,
            turn_count=memory.turn_count,
            model=self.model,
            error=None,
        )

    def _complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("model response has no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model response content is empty")
        return content

    def _read_cache(
        self,
        key: str,
        *,
        snapshot: Mapping[str, object],
        levels: Mapping[str, float],
    ) -> tuple[AIRawInterpretation, datetime] | None:
        if self._cache is None:
            return None
        try:
            payload = self._cache.get_ai(key)
            if not isinstance(payload, Mapping):
                return None
            raw_payload = payload.get("raw")
            generated_value = payload.get("generated_at")
            raw = parse_and_validate_interpretation(
                _json_text(raw_payload),
                snapshot=snapshot,
                watch_levels=levels,
            )
            generated_at = _parse_datetime(generated_value)
            return raw, generated_at
        except BaseException:
            return None

    def _write_cache(self, key: str, raw: AIRawInterpretation, generated_at: datetime) -> None:
        if self._cache is None:
            return
        payload = {
            "raw": raw.model_dump(mode="json"),
            "generated_at": generated_at.isoformat(),
        }
        try:
            self._cache.put_ai(key, payload)
        except BaseException:
            pass


def _analysis_facts(
    analysis_id: str,
    analysis: Mapping[str, object],
) -> tuple[str, dict[str, float], dict[str, float], str]:
    clean_id = analysis_id.strip() if isinstance(analysis_id, str) else ""
    if not clean_id:
        raise ValueError("analysis_id is required")
    snapshot_raw = analysis.get("snapshot")
    score_raw = analysis.get("score")
    if not isinstance(snapshot_raw, Mapping) or not isinstance(score_raw, Mapping):
        raise ValueError("analysis snapshot and score are required")
    snapshot = flatten_numeric_snapshot(snapshot_raw)
    if not snapshot:
        raise ValueError("analysis snapshot has no finite whitelisted facts")
    signal = score_raw.get("signal")
    if signal not in ("偏多", "中性偏多", "中性", "中性偏空", "偏空"):
        raise ValueError("rule signal is invalid")
    levels = watch_level_mapping(score_raw.get("watch_levels", ()))  # type: ignore[arg-type]
    return clean_id, snapshot, levels, str(signal)


def _safe_analysis_payload(analysis: Mapping[str, object]) -> dict[str, object]:
    allowed = ("stock", "period", "data_quality", "snapshot", "score", "analysis_id", "warnings")
    return {key: _jsonable(analysis[key]) for key in allowed if key in analysis}


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return,union-attr]
    raise TypeError("analysis contains a non-serializable value")


def _as_mapping(value: Mapping[str, object] | object) -> Mapping[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return value
    raise TypeError("structured analysis must be a mapping or Pydantic model")


def _json_text(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _parse_chat(content: str) -> _ChatRaw:
    def reject_constant(value: str) -> None:
        raise ValueError(value)

    payload = json.loads(content, parse_constant=reject_constant)
    return _ChatRaw.model_validate(payload)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("cached timestamp is invalid")
    return _aware_datetime(datetime.fromisoformat(value))


def _aware_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("clock must return datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _status_code(exc: BaseException) -> int | None:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _map_model_error(exc: BaseException) -> AgentError:
    status = _status_code(exc)
    if status == 401:
        return _error("AUTH", "DeepSeek API 认证失败，请检查服务端配置。", False)
    if status == 402:
        return _error("MODEL", "DeepSeek 账户余额不足，请充值后重试。", False)
    if status == 429:
        return _error("RATE_LIMIT", "DeepSeek 请求过于频繁，请稍后重试。", True)
    if status in (500, 503) or status is None:
        return _error("MODEL", "DeepSeek 服务暂时不可用，请稍后重试。", True)
    return _error("MODEL", "DeepSeek 请求失败，请稍后重试。", status >= 500)


def _error(code: str, message: str, retryable: bool) -> AgentError:
    return AgentError(code=code, user_message=message, retryable=retryable)  # type: ignore[arg-type]


def _chat_failure(
    memory: ChatMemory,
    code: str,
    message: str,
    retryable: bool,
) -> ChatResponse:
    return _chat_failure_from_error(memory, _error(code, message, retryable))


def _chat_failure_from_error(memory: ChatMemory, error: AgentError) -> ChatResponse:
    return ChatResponse(
        answer=None,
        thread_id=memory.thread_id,
        turn_count=memory.turn_count,
        model=None,
        error=error,
    )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_VERSION",
    "DEEPSEEK_BASE_URL",
    "MAX_CHAT_PAIRS",
    "ChatMemory",
    "DeepSeekClient",
    "append_turn",
]
