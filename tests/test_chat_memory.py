from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import UUID

from stock_agent.domain.models import ChatResponse
from stock_agent.llm.deepseek_client import ChatMemory, DeepSeekClient, append_turn


ANALYSIS = {
    "analysis_id": "analysis-a",
    "stock": {"ts_code": "600519.SH", "name": "贵州茅台"},
    "snapshot": {"close": 100.0, "rsi14": 55.0},
    "score": {"signal": "中性", "watch_levels": []},
}


class FakeCompletions:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        content = json.dumps({"answer": outcome}, ensure_ascii=False)
        return SimpleNamespace(
            id="chat-1",
            model="deepseek-v4-flash",
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class FakeOpenAI:
    def __init__(self, outcomes) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))

    def factory(self, **_kwargs):
        return self


def client_with(outcomes):
    fake = FakeOpenAI(outcomes)
    client = DeepSeekClient(api_key="sk-test", openai_factory=fake.factory)
    return client, fake


def test_followup_keeps_ten_complete_pairs() -> None:
    history = []
    for i in range(11):
        history = append_turn(history, f"q{i}", f"a{i}", max_pairs=10)

    assert len(history) == 20
    assert history[0] == {"role": "user", "content": "q1"}
    assert history[-1] == {"role": "assistant", "content": "a10"}


def test_switching_analysis_creates_uuid_thread_and_clears_history() -> None:
    memory = ChatMemory()
    first = memory.set_analysis("analysis-a", ANALYSIS)
    memory.append("问题", "回答")

    unchanged = memory.set_analysis("analysis-a", ANALYSIS)
    second = memory.set_analysis("analysis-b", {**ANALYSIS, "analysis_id": "analysis-b"})

    assert unchanged == first
    assert second != first
    assert str(UUID(first)) == first
    assert str(UUID(second)) == second
    assert memory.analysis_id == "analysis-b"
    assert memory.history == []
    assert memory.turn_count == 0


def test_empty_question_fails_locally_without_call_or_half_turn() -> None:
    client, fake = client_with([])
    memory = ChatMemory()
    memory.set_analysis("analysis-a", ANALYSIS)

    result = client.follow_up(memory, "   ")

    assert isinstance(result, ChatResponse)
    assert result.error is not None
    assert result.error.code == "VALIDATION"
    assert result.turn_count == 0
    assert fake.chat.completions.calls == []
    assert memory.history == []


def test_missing_current_analysis_fails_locally_without_call() -> None:
    client, fake = client_with([])
    memory = ChatMemory()

    result = client.follow_up(memory, "RSI 是什么意思？")

    assert result.error is not None
    assert result.error.code == "VALIDATION"
    assert fake.chat.completions.calls == []
    assert memory.history == []


def test_success_sends_only_current_analysis_and_last_ten_complete_pairs() -> None:
    client, fake = client_with(["新回答"])
    memory = ChatMemory()
    memory.set_analysis("analysis-a", ANALYSIS)
    for i in range(11):
        memory.append(f"q{i}", f"a{i}")

    result = client.follow_up(memory, "为什么是中性？")

    assert result.error is None
    assert result.answer == "新回答"
    assert result.turn_count == 10
    assert memory.history[0]["content"] == "q2"
    assert memory.history[-1] == {"role": "assistant", "content": "新回答"}
    call = fake.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    messages = call["messages"]
    assert json.loads(messages[1]["content"].split("：", 1)[1]) == ANALYSIS
    history_messages = messages[2:-1]
    assert len(history_messages) == 20
    assert history_messages[0] == {"role": "user", "content": "q1"}
    assert messages[-1] == {"role": "user", "content": "为什么是中性？"}


def test_failed_followup_does_not_append_question_or_partial_answer() -> None:
    client, fake = client_with([TimeoutError("secret")])
    memory = ChatMemory()
    memory.set_analysis("analysis-a", ANALYSIS)
    memory.append("旧问题", "旧回答")
    before = list(memory.history)

    result = client.follow_up(memory, "新问题")

    assert result.error is not None
    assert result.answer is None
    assert result.turn_count == 1
    assert len(fake.chat.completions.calls) == 1
    assert memory.history == before
