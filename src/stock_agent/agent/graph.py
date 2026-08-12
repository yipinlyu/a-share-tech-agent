"""Synchronous LangGraph orchestration for deterministic stock analysis."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from stock_agent.agent.state import AnalysisState, PlanTrace
from stock_agent.data.tushare_client import TushareAdapterError, assess_data_quality
from stock_agent.domain.models import (
    AgentError,
    AnalysisRequest,
    AnalysisResult,
    DataQuality,
    PeriodInfo,
    ScoreResult,
    StockInfo,
)
from stock_agent.indicators.engine import compute_indicators, latest_snapshot
from stock_agent.scoring.rules import build_analysis_id, score_signals


class MarketDataPort(Protocol):
    def fetch_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame: ...


class IndicatorPort(Protocol):
    def compute(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    def snapshot(self, frame: pd.DataFrame) -> Mapping[str, object]: ...


class ScoringPort(Protocol):
    def score(self, frame: pd.DataFrame, warnings: Sequence[str]) -> ScoreResult | None: ...


class AnalysisRepositoryPort(Protocol):
    def put_analysis(self, cache_key: str, payload: object) -> None: ...


@dataclass(frozen=True, slots=True)
class _IndicatorFunctions:
    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        return compute_indicators(frame)

    def snapshot(self, frame: pd.DataFrame) -> Mapping[str, object]:
        return latest_snapshot(frame)


@dataclass(frozen=True, slots=True)
class _ScoringFunctions:
    def score(self, frame: pd.DataFrame, warnings: Sequence[str]) -> ScoreResult | None:
        return score_signals(frame, warnings)


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    """All effects used by the graph, including its clocks, are explicitly injected."""

    market_data: MarketDataPort
    repository: AnalysisRepositoryPort | None
    stock: StockInfo
    indicators: IndicatorPort = field(default_factory=_IndicatorFunctions)
    scorer: ScoringPort = field(default_factory=_ScoringFunctions)
    monotonic: Callable[[], float] = time.monotonic
    today: Callable[[], date] = date.today
    analysis_id_builder: Callable[[str, date, int, pd.DataFrame], str] = build_analysis_id


GraphDeps = GraphDependencies


def _error(code: str, message: str, retryable: bool = False) -> AgentError:
    return AgentError(code=code, user_message=message, retryable=retryable)  # type: ignore[arg-type]


def _trace(deps: GraphDependencies, node: str, status: str, started: float) -> list[PlanTrace]:
    try:
        elapsed = max(0.0, (float(deps.monotonic()) - started) * 1000.0)
    except BaseException:
        elapsed = 0.0
    return [{"node": node, "status": status, "elapsed_ms": round(elapsed, 3)}]


def _start(deps: GraphDependencies) -> float:
    try:
        return float(deps.monotonic())
    except BaseException:
        return 0.0


def _display_start(end_date: date, months: int) -> date:
    return (pd.Timestamp(end_date) - pd.DateOffset(months=months)).date()


def _period(
    request: AnalysisRequest,
    display_start: date,
    quality: DataQuality,
    frame: pd.DataFrame,
    fallback_end: date,
) -> PeriodInfo:
    dates = (
        pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
        if "trade_date" in frame.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    displayed = dates.loc[dates.dt.date >= display_start]
    last = quality.last_trade_date or (dates.iloc[-1].date() if not dates.empty else fallback_end)
    first = displayed.iloc[0].date() if not displayed.empty else min(display_start, last)
    return PeriodInfo(
        requested_end_date=request.requested_end_date,
        resolved_end_date=last,
        actual_start_date=first,
        actual_end_date=last,
        last_trade_date=last,
        adjustment="qfq",
    )


def _analysis_payload(state: AnalysisState, deps: GraphDependencies) -> dict[str, object]:
    return {
        "analysis_id": state["analysis_id"],
        "stock": deps.stock.model_dump(mode="json"),
        "period": state["period"].model_dump(mode="json"),
        "data_quality": state["data_quality"].model_dump(mode="json"),
        "snapshot": state["snapshot"],
        "score": state["score"].model_dump(mode="json"),
        "warnings": list(state.get("warnings", [])),
    }


def build_analysis_graph(deps: GraphDependencies):
    """Compile a fresh graph without a process-wide or cross-user checkpointer."""

    def plan_analysis(state: AnalysisState) -> dict[str, object]:
        started = _start(deps)
        try:
            request = AnalysisRequest.model_validate(state["request_input"])
            if request.ts_code != deps.stock.ts_code:
                raise ValueError("selected stock does not match request")
            requested_end = request.requested_end_date or deps.today()
            display_start = _display_start(requested_end, request.lookback_months)
            # About 140 business days in ordinary calendars, safely exceeding 120.
            fetch_start = display_start - timedelta(days=200)
            return {
                "request": request,
                "requested_end_date": requested_end,
                "display_start_date": display_start,
                "fetch_start_date": fetch_start,
                "warnings": [],
                "plan_trace": _trace(deps, "plan_analysis", "success", started),
            }
        except (ValidationError, TypeError, ValueError, OverflowError):
            return {
                "terminal": "error",
                "error": _error("VALIDATION", "分析请求无效，请检查股票与时间参数。"),
                "warnings": [],
                "plan_trace": _trace(deps, "plan_analysis", "error", started),
            }

    def fetch_market_data(state: AnalysisState) -> dict[str, object]:
        started = _start(deps)
        request = state["request"]
        try:
            frame = deps.market_data.fetch_daily(
                request.ts_code,
                state["fetch_start_date"],
                state["requested_end_date"],
            )
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("market adapter must return a DataFrame")
            return {
                "raw_series": frame.copy(deep=True),
                "plan_trace": _trace(deps, "fetch_market_data", "success", started),
            }
        except TushareAdapterError as exc:
            return {
                "terminal": "error",
                "error": exc.error,
                "plan_trace": _trace(deps, "fetch_market_data", "error", started),
            }
        except BaseException:
            return {
                "terminal": "error",
                "error": _error("DATA", "行情数据暂时无法获取，请稍后重试。", True),
                "plan_trace": _trace(deps, "fetch_market_data", "error", started),
            }

    def validate_data(state: AnalysisState) -> dict[str, object]:
        started = _start(deps)
        frame = state["raw_series"]
        request = state["request"]
        try:
            quality = assess_data_quality(
                frame,
                display_start_date=state["display_start_date"],
            )
            period = _period(
                request,
                state["display_start_date"],
                quality,
                frame,
                state["requested_end_date"],
            )
            delta: dict[str, object] = {
                "data_quality": quality,
                "period": period,
                "warnings": list(quality.warnings),
            }
            lacks_samples = (
                quality.raw_row_count == 0
                or quality.display_row_count == 0
                or quality.prewarm_row_count < 60
            )
            if quality.valid:
                status = "success"
            elif lacks_samples:
                status = "insufficient_data"
                delta["terminal"] = "insufficient_data"
            else:
                status = "error"
                delta["terminal"] = "error"
                delta["error"] = _error("DATA", "行情数据未通过完整性校验，无法安全分析。")
            delta["plan_trace"] = _trace(deps, "validate_data", status, started)
            return delta
        except BaseException:
            return {
                "terminal": "error",
                "error": _error("DATA", "行情数据校验失败，请稍后重试。"),
                "plan_trace": _trace(deps, "validate_data", "error", started),
            }

    def compute_indicator_node(state: AnalysisState) -> dict[str, object]:
        started = _start(deps)
        try:
            full = deps.indicators.compute(state["raw_series"])
            dates = pd.to_datetime(full["trade_date"], errors="coerce")
            displayed = full.loc[dates.dt.date >= state["display_start_date"]].copy()
            if displayed.empty:
                raise ValueError("display series is empty")
            snapshot = dict(deps.indicators.snapshot(displayed))
            if not snapshot:
                raise ValueError("snapshot is empty")
            return {
                "indicator_series": full,
                "display_series": displayed.reset_index(drop=True),
                "snapshot": snapshot,
                "plan_trace": _trace(deps, "compute_indicators", "success", started),
            }
        except BaseException:
            return {
                "terminal": "error",
                "error": _error("INTERNAL", "技术指标计算失败，请稍后重试。"),
                "plan_trace": _trace(deps, "compute_indicators", "error", started),
            }

    def score_signal_node(state: AnalysisState) -> dict[str, object]:
        started = _start(deps)
        try:
            score = deps.scorer.score(state["display_series"], state.get("warnings", []))
            if score is None:
                return {
                    "terminal": "insufficient_data",
                    "plan_trace": _trace(deps, "score_signals", "insufficient_data", started),
                }
            return {
                "score": score,
                "plan_trace": _trace(deps, "score_signals", "success", started),
            }
        except BaseException:
            return {
                "terminal": "error",
                "error": _error("INTERNAL", "规则评分失败，请稍后重试。"),
                "plan_trace": _trace(deps, "score_signals", "error", started),
            }

    def build_result_node(state: AnalysisState) -> dict[str, object]:
        started = _start(deps)
        try:
            analysis_id = deps.analysis_id_builder(
                state["request"].ts_code,
                state["period"].resolved_end_date,
                state["request"].lookback_months,
                state["raw_series"],
            )
            if not isinstance(analysis_id, str) or not analysis_id.strip():
                raise ValueError("analysis id is empty")
            return {
                "analysis_id": analysis_id,
                "terminal": "success",
                "plan_trace": _trace(deps, "build_result", "success", started),
            }
        except BaseException:
            return {
                "terminal": "error",
                "error": _error("INTERNAL", "分析结果构造失败，请稍后重试。"),
                "plan_trace": _trace(deps, "build_result", "error", started),
            }

    def write_memory(state: AnalysisState) -> dict[str, object]:
        started = _start(deps)
        try:
            if deps.repository is None:
                raise RuntimeError("repository unavailable")
            deps.repository.put_analysis(state["analysis_id"], _analysis_payload(state, deps))
            return {"plan_trace": _trace(deps, "write_memory", "success", started)}
        except BaseException:
            warnings = list(state.get("warnings", []))
            warnings.append("persistence unavailable; analysis remains available in this session.")
            return {
                "warnings": warnings,
                "plan_trace": _trace(deps, "write_memory", "warning", started),
            }

    def continue_or_end(state: AnalysisState) -> str:
        return "end" if state.get("terminal") in ("error", "insufficient_data") else "next"

    graph = StateGraph(AnalysisState)
    graph.add_node("plan_analysis", plan_analysis)
    graph.add_node("fetch_market_data", fetch_market_data)
    graph.add_node("validate_data", validate_data)
    graph.add_node("compute_indicators", compute_indicator_node)
    graph.add_node("score_signals", score_signal_node)
    graph.add_node("build_result", build_result_node)
    graph.add_node("write_memory", write_memory)
    graph.add_edge(START, "plan_analysis")
    for source, target in (
        ("plan_analysis", "fetch_market_data"),
        ("fetch_market_data", "validate_data"),
        ("validate_data", "compute_indicators"),
        ("compute_indicators", "score_signals"),
        ("score_signals", "build_result"),
        ("build_result", "write_memory"),
    ):
        graph.add_conditional_edges(source, continue_or_end, {"next": target, "end": END})
    graph.add_edge("write_memory", END)
    return graph.compile()


def run_analysis_graph(
    request: AnalysisRequest | Mapping[str, Any],
    deps: GraphDependencies,
) -> AnalysisResult:
    """Run the bounded graph and convert its terminal state to one domain result."""

    state: AnalysisState = build_analysis_graph(deps).invoke(
        {"request_input": request, "plan_trace": []}
    )
    status = state.get("terminal", "error")
    common = {
        "stock": deps.stock,
        "period": state.get("period"),
        "data_quality": state.get("data_quality"),
        "plan_trace": state.get("plan_trace", []),
        "warnings": state.get("warnings", []),
    }
    if status == "success":
        return AnalysisResult(
            status="success",
            series=state["display_series"],
            snapshot=state["snapshot"],
            score=state["score"],
            analysis_id=state["analysis_id"],
            **common,
        )
    if status == "insufficient_data":
        return AnalysisResult(status="insufficient_data", **common)
    return AnalysisResult(
        status="error",
        error=state.get("error")
        or _error("INTERNAL", "分析流程未能安全完成，请稍后重试。"),
        **common,
    )


__all__ = [
    "AnalysisRepositoryPort",
    "GraphDependencies",
    "GraphDeps",
    "IndicatorPort",
    "MarketDataPort",
    "ScoringPort",
    "build_analysis_graph",
    "run_analysis_graph",
]
