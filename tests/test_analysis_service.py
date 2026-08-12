from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from stock_agent.agent.graph import GraphDependencies
from stock_agent.domain.models import (
    AIInterpretation,
    AIRequest,
    AgentError,
    AnalysisRequest,
    AnalysisResult,
    ChatRequest,
    ChatResponse,
    StockInfo,
    StockQuery,
    StockSearchResult,
)
from stock_agent.llm.deepseek_client import ChatMemory
from stock_agent.services.analysis_service import AnalysisService

from test_agent_graph import (
    END_DATE,
    STOCK,
    FakeIndicators,
    FakeMarketData,
    FakeRepository,
    FakeScorer,
    StepClock,
    market_frame,
)


def stock_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "600519.SH",
                "symbol": "600519",
                "name": "贵州茅台",
                "market": "主板",
                "industry": "白酒",
            },
            {
                "ts_code": "000001.SZ",
                "symbol": "000001",
                "name": "平安银行",
                "market": "主板",
                "industry": "银行",
            },
            {
                "ts_code": "601318.SH",
                "symbol": "601318",
                "name": "中国平安",
                "market": "主板",
                "industry": "保险",
            },
        ]
    )


class ServiceMarket(FakeMarketData):
    def __init__(self) -> None:
        super().__init__(market_frame())
        self.master_calls = 0

    def fetch_stock_master(self) -> pd.DataFrame:
        self.master_calls += 1
        return stock_master().copy(deep=True)


class ServiceRepository(FakeRepository):
    def __init__(self, *, fail_reads: bool = False) -> None:
        super().__init__()
        self.fail_reads = fail_reads
        self.master_payload: object | None = None
        self.master_put_calls = 0

    def get_stock_master(self, cache_key: str) -> object | None:
        if self.fail_reads:
            raise RuntimeError("sqlite secret path")
        return self.master_payload

    def put_stock_master(self, cache_key: str, payload: object) -> None:
        self.master_put_calls += 1
        self.master_payload = payload


class SpyAI:
    model = "fake-v1"

    def __init__(self) -> None:
        self.interpret_calls: list[tuple[str, object, bool]] = []
        self.follow_calls: list[tuple[ChatMemory, str]] = []

    def interpret(self, analysis_id: str, analysis: object, *, force_refresh: bool = False):
        self.interpret_calls.append((analysis_id, analysis, force_refresh))
        return AgentError(
            code="MODEL",
            user_message="AI 服务暂时不可用。",
            retryable=True,
        )

    def follow_up(self, memory: ChatMemory, question: str) -> ChatResponse:
        self.follow_calls.append((memory, question))
        return ChatResponse(
            answer="仅解释当前结构化分析。",
            thread_id=memory.thread_id,
            turn_count=memory.turn_count,
            model=self.model,
        )


def make_service(*, fail_master_reads: bool = False, ai: SpyAI | None = None):
    market = ServiceMarket()
    repository = ServiceRepository(fail_reads=fail_master_reads)
    deps = GraphDependencies(
        market_data=market,
        indicators=FakeIndicators(),
        scorer=FakeScorer(),
        repository=repository,
        stock=STOCK,
        monotonic=StepClock(),
        today=lambda: END_DATE,
    )
    service = AnalysisService(
        market_data=market,
        repository=repository,
        graph_dependencies=deps,
        ai_client=ai,
    )
    return service, market, repository


def test_search_uses_stock_master_ttl_cache_and_returns_domain_result() -> None:
    service, market, repository = make_service()

    first = service.search(StockQuery(query="贵州茅台", lookback_months=3))
    second = service.search("600519")

    assert isinstance(first, StockSearchResult)
    assert first.status == "resolved"
    assert second.status == "resolved"
    assert market.master_calls == 1
    assert repository.master_put_calls == 1


def test_stock_master_cache_read_failure_refetches_safely() -> None:
    service, market, _ = make_service(fail_master_reads=True)

    result = service.search("贵州茅台")

    assert result.status == "resolved"
    assert market.master_calls == 1


def test_ambiguous_search_stays_outside_graph_and_ai() -> None:
    ai = SpyAI()
    service, market, _ = make_service(ai=ai)

    result = service.search("平安")

    assert result.status == "ambiguous"
    assert market.calls == []
    assert ai.interpret_calls == []


def test_analyze_resolves_candidate_outside_graph_without_automatic_ai_call() -> None:
    ai = SpyAI()
    service, _, _ = make_service(ai=ai)

    result = service.analyze(
        AnalysisRequest(
            ts_code="600519.SH",
            lookback_months=3,
            requested_end_date=END_DATE,
        )
    )

    assert isinstance(result, AnalysisResult)
    assert result.status == "success"
    assert result.stock == STOCK
    assert ai.interpret_calls == []


def test_ai_interpretation_is_explicit_and_rejects_mismatched_analysis_id() -> None:
    ai = SpyAI()
    service, _, _ = make_service(ai=ai)
    result = service.analyze(
        AnalysisRequest(
            ts_code="600519.SH",
            lookback_months=3,
            requested_end_date=END_DATE,
        )
    )
    assert result.analysis_id is not None

    mismatch = service.interpret(AIRequest(analysis_id="different"), result)
    interpreted = service.interpret(AIRequest(analysis_id=result.analysis_id), result)

    assert isinstance(mismatch, AgentError)
    assert mismatch.code == "VALIDATION"
    assert isinstance(interpreted, AgentError)
    assert len(ai.interpret_calls) == 1


def test_follow_up_adapter_requires_current_matching_session_context() -> None:
    ai = SpyAI()
    service, _, _ = make_service(ai=ai)
    result = service.analyze(
        AnalysisRequest(
            ts_code="600519.SH",
            lookback_months=3,
            requested_end_date=END_DATE,
        )
    )
    memory = ChatMemory()
    thread_id = service.activate_analysis(result, memory)
    assert result.analysis_id is not None

    wrong_thread = service.follow_up(
        ChatRequest(
            thread_id="1e9f2977-7c1d-4db6-9754-66539c5a20a7",
            question="RSI 是什么？",
            analysis_id=result.analysis_id,
        ),
        memory,
    )
    response = service.follow_up(
        ChatRequest(
            thread_id=thread_id,
            question="RSI 是什么？",
            analysis_id=result.analysis_id,
        ),
        memory,
    )

    assert wrong_thread.error is not None
    assert wrong_thread.error.code == "VALIDATION"
    assert response.answer == "仅解释当前结构化分析。"
    assert len(ai.follow_calls) == 1


def test_missing_ai_configuration_returns_domain_errors_not_streamlit_calls() -> None:
    service, _, _ = make_service(ai=None)
    result = service.analyze(
        AnalysisRequest(
            ts_code="600519.SH",
            lookback_months=3,
            requested_end_date=END_DATE,
        )
    )
    assert result.analysis_id is not None

    interpretation = service.interpret(AIRequest(analysis_id=result.analysis_id), result)

    assert isinstance(interpretation, AgentError)
    assert interpretation.code == "CONFIG"
    assert "Streamlit" not in type(service).__module__
