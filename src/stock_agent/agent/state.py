"""Typed state shared by the synchronous deterministic analysis graph."""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd

from stock_agent.domain.models import (
    AgentError,
    AnalysisRequest,
    DataQuality,
    PeriodInfo,
    ScoreResult,
)


class PlanTrace(TypedDict):
    """The deliberately small, display-safe trace record."""

    node: str
    status: str
    elapsed_ms: float


class AnalysisState(TypedDict, total=False):
    """Internal graph state; nodes append a delta instead of mutating this mapping."""

    request_input: Any
    request: AnalysisRequest
    display_start_date: date
    fetch_start_date: date
    requested_end_date: date
    raw_series: pd.DataFrame
    indicator_series: pd.DataFrame
    display_series: pd.DataFrame
    data_quality: DataQuality
    period: PeriodInfo
    snapshot: dict[str, str | int | float | bool | None]
    score: ScoreResult
    analysis_id: str
    warnings: list[str]
    error: AgentError
    terminal: Literal["success", "insufficient_data", "error"]
    plan_trace: Annotated[list[PlanTrace], operator.add]


__all__ = ["AnalysisState", "PlanTrace"]
