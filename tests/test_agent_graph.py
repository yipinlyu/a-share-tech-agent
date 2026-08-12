from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_agent.agent.graph import GraphDependencies, run_analysis_graph
from stock_agent.data.tushare_client import TushareAdapterError
from stock_agent.domain.models import AgentError, AnalysisRequest, StockInfo
from stock_agent.indicators.engine import compute_indicators, latest_snapshot
from stock_agent.scoring.rules import build_analysis_id, score_signals


END_DATE = date(2024, 8, 30)
DISPLAY_START = date(2024, 5, 30)
STOCK = StockInfo(
    ts_code="600519.SH",
    name="贵州茅台",
    market="主板",
    industry="白酒",
)


def market_frame(*, prewarm_rows: int = 120) -> pd.DataFrame:
    before = pd.bdate_range(
        end=pd.Timestamp(DISPLAY_START) - pd.Timedelta(days=1), periods=prewarm_rows
    )
    displayed = pd.bdate_range(DISPLAY_START, END_DATE)
    dates = before.append(displayed)
    positions = np.arange(len(dates), dtype=float)
    close = 100.0 + positions * 0.15 + np.sin(positions / 7.0)
    open_price = close - 0.2
    high = close + 1.0
    low = open_price - 1.0
    pre_close = np.r_[close[0], close[:-1]]
    change = close - pre_close
    return pd.DataFrame(
        {
            "ts_code": pd.Series([STOCK.ts_code] * len(dates), dtype="string"),
            "trade_date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pre_close,
            "change": change,
            "pct_chg": change / pre_close * 100.0,
            "vol": 1_000.0 + positions * 5.0,
            "amount": (1_000.0 + positions * 5.0) * close,
        }
    )


class FakeMarketData:
    def __init__(self, frame: pd.DataFrame, error: BaseException | None = None) -> None:
        self.frame = frame
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    def fetch_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        self.calls.append((ts_code, start_date, end_date))
        if self.error is not None:
            raise self.error
        return self.frame.copy(deep=True)


class FakeIndicators:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.call_count = 0
        self.input_lengths: list[int] = []

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.call_count += 1
        self.input_lengths.append(len(frame))
        if self.error is not None:
            raise self.error
        return compute_indicators(frame)

    def snapshot(self, frame: pd.DataFrame) -> dict[str, str | float | None]:
        return latest_snapshot(frame)


class FakeScorer:
    def __init__(self, *, no_coverage: bool = False) -> None:
        self.no_coverage = no_coverage
        self.call_count = 0

    def score(self, frame: pd.DataFrame, warnings: list[str]):
        self.call_count += 1
        return None if self.no_coverage else score_signals(frame, warnings)


class FakeRepository:
    def __init__(self, *, fail_writes: bool = False) -> None:
        self.fail_writes = fail_writes
        self.put_calls: list[tuple[str, object]] = []

    def put_analysis(self, cache_key: str, payload: object) -> None:
        if self.fail_writes:
            raise RuntimeError("sqlite /private/path secret=sk-never-trace")
        self.put_calls.append((cache_key, payload))


class StepClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


@pytest.fixture
def analysis_request() -> AnalysisRequest:
    return AnalysisRequest(
        ts_code=STOCK.ts_code,
        lookback_months=3,
        requested_end_date=END_DATE,
    )


@pytest.fixture
def graph_deps() -> GraphDependencies:
    return GraphDependencies(
        market_data=FakeMarketData(market_frame()),
        indicators=FakeIndicators(),
        scorer=FakeScorer(),
        repository=FakeRepository(),
        stock=STOCK,
        monotonic=StepClock(),
        today=lambda: END_DATE,
    )


def test_success_path_is_ordered_sanitized_and_crops_only_after_indicators(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    result = run_analysis_graph(analysis_request, graph_deps)

    assert result.status == "success"
    assert result.series is not None
    assert result.series.iloc[0]["trade_date"].date() == DISPLAY_START
    assert graph_deps.indicators.input_lengths == [len(market_frame())]
    assert len(result.series) < graph_deps.indicators.input_lengths[0]
    assert [entry["node"] for entry in result.plan_trace] == [
        "plan_analysis",
        "fetch_market_data",
        "validate_data",
        "compute_indicators",
        "score_signals",
        "build_result",
        "write_memory",
    ]
    assert all(set(entry) == {"node", "status", "elapsed_ms"} for entry in result.plan_trace)
    assert all(entry["elapsed_ms"] >= 0 for entry in result.plan_trace)
    serialized_trace = repr(result.plan_trace).lower()
    assert "sk-never" not in serialized_trace
    assert "prompt" not in serialized_trace
    assert "/private/" not in serialized_trace


def test_persistence_failure_preserves_analysis_id(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    graph_deps.repository.fail_writes = True

    result = run_analysis_graph(analysis_request, graph_deps)

    assert result.status == "success"
    assert result.analysis_id
    assert "persistence" in " ".join(result.warnings).lower()
    assert result.plan_trace[-1]["status"] == "warning"


def test_tushare_error_stops_before_indicators(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    safe_error = AgentError(
        code="RATE_LIMIT",
        user_message="Tushare 请求过于频繁，请稍后重试。",
        retryable=True,
    )
    graph_deps.market_data.error = TushareAdapterError(safe_error)

    result = run_analysis_graph(analysis_request, graph_deps)

    assert result.status == "error"
    assert result.error == safe_error
    assert graph_deps.indicators.call_count == 0
    assert graph_deps.scorer.call_count == 0
    assert [entry["node"] for entry in result.plan_trace] == [
        "plan_analysis",
        "fetch_market_data",
    ]


def test_59_prewarm_rows_route_to_insufficient_without_indicators(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    graph_deps.market_data.frame = market_frame(prewarm_rows=59)

    result = run_analysis_graph(analysis_request, graph_deps)

    assert result.status == "insufficient_data"
    assert result.data_quality is not None
    assert result.data_quality.prewarm_row_count == 59
    assert graph_deps.indicators.call_count == 0
    assert graph_deps.scorer.call_count == 0


def test_invalid_ohlcv_routes_to_error_without_indicators(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    graph_deps.market_data.frame.loc[0, "high"] = 0.1

    result = run_analysis_graph(analysis_request, graph_deps)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "DATA"
    assert graph_deps.indicators.call_count == 0


def test_indicator_exception_stops_before_score_and_persistence(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    graph_deps.indicators.error = ArithmeticError("stack /secret/path sk-do-not-leak")

    result = run_analysis_graph(analysis_request, graph_deps)

    assert result.status == "error"
    assert graph_deps.scorer.call_count == 0
    assert graph_deps.repository.put_calls == []
    assert "sk-do-not-leak" not in result.error.user_message


def test_score_coverage_failure_routes_to_insufficient_without_persistence(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    graph_deps.scorer.no_coverage = True

    result = run_analysis_graph(analysis_request, graph_deps)

    assert result.status == "insufficient_data"
    assert graph_deps.repository.put_calls == []


def test_build_failure_is_safe_and_does_not_persist(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    def fail_build(*args, **kwargs):
        raise ValueError("raw model output and sk-build-secret")

    deps = replace(graph_deps, analysis_id_builder=fail_build)

    result = run_analysis_graph(analysis_request, deps)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "INTERNAL"
    assert deps.repository.put_calls == []
    assert "sk-build-secret" not in repr(result.plan_trace)


def test_invalid_request_stops_in_plan_node(graph_deps: GraphDependencies) -> None:
    result = run_analysis_graph(
        {"ts_code": "not-a-code", "lookback_months": 7},
        graph_deps,
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "VALIDATION"
    assert graph_deps.market_data.calls == []
    assert [entry["node"] for entry in result.plan_trace] == ["plan_analysis"]


def test_default_analysis_id_builder_is_bound_to_full_prewarm_input(
    graph_deps: GraphDependencies,
    analysis_request: AnalysisRequest,
) -> None:
    result = run_analysis_graph(analysis_request, graph_deps)

    expected = build_analysis_id(
        STOCK.ts_code,
        END_DATE,
        3,
        graph_deps.market_data.frame,
    )
    assert result.analysis_id == expected
