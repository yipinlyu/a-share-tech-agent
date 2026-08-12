"""UI-independent application facade for search, analysis, AI, and follow-up."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Protocol

import pandas as pd

from stock_agent.agent.graph import GraphDependencies, run_analysis_graph
from stock_agent.data.stock_search import search_stocks
from stock_agent.data.tushare_client import TushareAdapterError
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


STOCK_MASTER_CACHE_KEY = "a-share-listed-v1"


class StockMasterPort(Protocol):
    def fetch_stock_master(self) -> pd.DataFrame: ...


class RepositoryPort(Protocol):
    def get_stock_master(self, cache_key: str) -> object | None: ...

    def put_stock_master(self, cache_key: str, payload: object) -> None: ...


class AIPort(Protocol):
    def interpret(
        self,
        analysis_id: str,
        structured_analysis: Mapping[str, object] | object,
        *,
        force_refresh: bool = False,
    ) -> AIInterpretation | AgentError: ...

    def follow_up(self, memory: ChatMemory, question: str) -> ChatResponse: ...


def _error(code: str, message: str, retryable: bool = False) -> AgentError:
    return AgentError(code=code, user_message=message, retryable=retryable)  # type: ignore[arg-type]


def _failed_chat(memory: ChatMemory, error: AgentError) -> ChatResponse:
    return ChatResponse(
        answer=None,
        thread_id=memory.thread_id,
        turn_count=memory.turn_count,
        model=None,
        error=error,
    )


class AnalysisService:
    """Return domain objects only; candidate choice remains outside the graph."""

    def __init__(
        self,
        *,
        market_data: StockMasterPort,
        repository: RepositoryPort | None,
        graph_dependencies: GraphDependencies,
        ai_client: AIPort | None = None,
        searcher: Callable[[str, pd.DataFrame], StockSearchResult] = search_stocks,
        graph_runner: Callable[
            [AnalysisRequest | Mapping[str, Any], GraphDependencies], AnalysisResult
        ] = run_analysis_graph,
    ) -> None:
        self._market_data = market_data
        self._repository = repository
        self._graph_dependencies = graph_dependencies
        self._ai = ai_client
        self._searcher = searcher
        self._graph_runner = graph_runner
        self._stock_master: pd.DataFrame | None = None

    def _load_stock_master(self) -> pd.DataFrame:
        if self._stock_master is not None:
            return self._stock_master.copy(deep=True)
        payload: object | None = None
        if self._repository is not None:
            try:
                payload = self._repository.get_stock_master(STOCK_MASTER_CACHE_KEY)
            except BaseException:
                payload = None
        if isinstance(payload, list):
            cached = pd.DataFrame(payload)
            if not cached.empty:
                self._stock_master = cached
                return cached.copy(deep=True)
        frame = self._market_data.fetch_stock_master()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("stock master adapter must return a DataFrame")
        self._stock_master = frame.copy(deep=True)
        if self._repository is not None:
            try:
                self._repository.put_stock_master(
                    STOCK_MASTER_CACHE_KEY,
                    frame.to_dict(orient="records"),
                )
            except BaseException:
                pass
        return frame.copy(deep=True)

    def search(self, query: StockQuery | str) -> StockSearchResult:
        text = query.query if isinstance(query, StockQuery) else query
        try:
            return self._searcher(text, self._load_stock_master())
        except TushareAdapterError as exc:
            return StockSearchResult(status="error", error=exc.error)
        except BaseException:
            return StockSearchResult(
                status="error",
                error=_error("DATA", "股票主数据暂时无法获取，请稍后重试。", True),
            )

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        resolved = self.search(request.ts_code)
        if resolved.status == "resolved":
            stock = resolved.candidates[0]
            return self._graph_runner(request, replace(self._graph_dependencies, stock=stock))
        error = resolved.error or _error("VALIDATION", "未找到已选股票，请重新搜索并选择候选项。")
        return AnalysisResult(
            status="error",
            stock=StockInfo(ts_code=request.ts_code, name=request.ts_code, market="未知"),
            error=error,
        )

    def interpret(
        self,
        request: AIRequest,
        analysis: AnalysisResult,
    ) -> AIInterpretation | AgentError:
        if self._ai is None:
            return _error("CONFIG", "未配置 DeepSeek，量化分析仍可正常使用。")
        if (
            analysis.status != "success"
            or analysis.analysis_id is None
            or analysis.analysis_id != request.analysis_id
        ):
            return _error("VALIDATION", "AI 请求与当前结构化分析不匹配。")
        return self._ai.interpret(
            request.analysis_id,
            analysis,
            force_refresh=request.force_refresh,
        )

    def activate_analysis(self, analysis: AnalysisResult, memory: ChatMemory) -> str:
        if analysis.status != "success" or analysis.analysis_id is None:
            raise ValueError("a successful analysis is required")
        return memory.set_analysis(
            analysis.analysis_id,
            analysis.model_dump(mode="python"),
        )

    def follow_up(self, request: ChatRequest, memory: ChatMemory) -> ChatResponse:
        if self._ai is None:
            return _failed_chat(memory, _error("CONFIG", "未配置 DeepSeek，无法使用追问。"))
        if (
            request.thread_id != memory.thread_id
            or request.analysis_id != memory.analysis_id
            or memory.structured_analysis is None
        ):
            return _failed_chat(
                memory,
                _error("VALIDATION", "追问与当前分析线程不匹配，请重新分析。"),
            )
        return self._ai.follow_up(memory, request.question)

    # Explicit names used by UI composition without importing adapter details.
    search_stocks = search
    run_analysis = analyze
    interpret_with_ai = interpret
    answer_followup = follow_up


__all__ = ["AIPort", "AnalysisService", "RepositoryPort", "STOCK_MASTER_CACHE_KEY"]
