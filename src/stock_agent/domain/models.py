"""Strict Pydantic contracts shared by adapters, services, and the UI."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID, uuid4

import pandas as pd
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

INVESTMENT_DISCLAIMER = "仅供学习研究，不构成投资建议"

LookbackMonths = Literal[3, 6, 12, 24, 36]
Signal = Literal["偏多", "中性偏多", "中性", "中性偏空", "偏空"]
SearchStatus = Literal["resolved", "ambiguous", "not_found", "error"]
AnalysisStatus = Literal["success", "insufficient_data", "error"]
ErrorCode = Literal["CONFIG", "AUTH", "RATE_LIMIT", "DATA", "VALIDATION", "MODEL", "INTERNAL"]
ConsistencyStatus = Literal["consistent", "mismatch"]
RiskLevel = Literal["低", "中", "高"]
ScoreGroup = Literal["trend", "momentum", "volume_volatility"]
WatchLabel = Literal["支撑观察", "压力观察", "波动参考"]
BasisKey = Literal[
    "recent_20d_low",
    "recent_20d_high",
    "ma20",
    "boll_upper",
    "boll_lower",
    "close_minus_atr",
    "close_plus_atr",
]
RiskType = Literal[
    "volatility",
    "overbought",
    "oversold",
    "signal_conflict",
    "data_quality",
    "other",
]

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
TraceValue = str | int | FiniteFloat | bool | None
SnapshotValue = str | int | FiniteFloat | bool | None

_A_SHARE_CODE = re.compile(
    r"^(?:(?:60|68)\d{4}\.SH|(?:00|30)\d{4}\.SZ|(?:4|8)\d{5}\.BJ|92\d{4}\.BJ)$"
)
_AI_ENRICHMENT_CONTEXT = "server_enrichment"
_AI_ENRICHMENT_TOKEN = object()


class DomainModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )


def _validate_ts_code(value: str) -> str:
    normalized = value.strip().upper()
    if not _A_SHARE_CODE.fullmatch(normalized):
        raise ValueError("ts_code must be a supported A-share Tushare code")
    return normalized


def _validate_not_future(value: date | None) -> date | None:
    if value is not None and value > date.today():
        raise ValueError("date cannot be in the future")
    return value


def _canonical_uuid(value: str | UUID) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("thread_id must be a UUID string") from exc


class StockInfo(DomainModel):
    ts_code: str
    symbol: str | None = None
    name: NonEmptyStr
    market: NonEmptyStr
    industry: NonEmptyStr | None = None

    @field_validator("ts_code")
    @classmethod
    def valid_ts_code(cls, value: str) -> str:
        return _validate_ts_code(value)

    @model_validator(mode="after")
    def symbol_matches_ts_code(self) -> Self:
        expected = self.ts_code[:6]
        if self.symbol is not None and self.symbol != expected:
            raise ValueError("symbol must match ts_code")
        self.symbol = expected
        return self


class StockQuery(DomainModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=30)]
    lookback_months: LookbackMonths = 12
    end_date: date | None = None

    _valid_end_date = field_validator("end_date")(_validate_not_future)


class IndicatorConfig(DomainModel):
    """The only indicator parameter set supported by the first release."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    ma_windows: tuple[Literal[5], Literal[10], Literal[20], Literal[60]] = (5, 10, 20, 60)
    macd: tuple[Literal[12], Literal[26], Literal[9]] = (12, 26, 9)
    bollinger: tuple[Literal[20], Literal[2.0]] = (20, 2.0)
    rsi_window: Literal[14] = 14
    kdj: tuple[Literal[9], Literal[3], Literal[3]] = (9, 3, 3)
    atr_window: Literal[14] = 14
    volume_window: Literal[20] = 20


class AnalysisRequest(DomainModel):
    ts_code: str
    lookback_months: LookbackMonths = 12
    requested_end_date: date | None = None
    indicator_config: IndicatorConfig = Field(default_factory=IndicatorConfig)

    @field_validator("ts_code")
    @classmethod
    def valid_ts_code(cls, value: str) -> str:
        return _validate_ts_code(value)

    _valid_requested_end_date = field_validator("requested_end_date")(_validate_not_future)


class AIRequest(DomainModel):
    analysis_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]
    force_refresh: bool = False


class ChatRequest(DomainModel):
    thread_id: str
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    analysis_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]

    _valid_thread_id = field_validator("thread_id", mode="before")(_canonical_uuid)


class AgentError(DomainModel):
    code: ErrorCode
    user_message: NonEmptyStr
    retryable: bool
    trace_id: NonEmptyStr = Field(default_factory=lambda: str(uuid4()))


class StockSearchResult(DomainModel):
    status: SearchStatus
    candidates: list[StockInfo] = Field(default_factory=list, max_length=10)
    error: AgentError | None = None

    @model_validator(mode="after")
    def valid_status_shape(self) -> Self:
        count = len(self.candidates)
        if self.status == "resolved" and (count != 1 or self.error is not None):
            raise ValueError("resolved search requires exactly one candidate and no error")
        if self.status == "ambiguous" and (count < 2 or self.error is not None):
            raise ValueError("ambiguous search requires two or more candidates and no error")
        if self.status == "not_found" and (count != 0 or self.error is not None):
            raise ValueError("not_found search cannot contain candidates or an error")
        if self.status == "error" and (count != 0 or self.error is None):
            raise ValueError("error search requires an error and no candidates")
        return self


class ChatResponse(DomainModel):
    answer: NonEmptyStr | None
    thread_id: str
    turn_count: Annotated[int, Field(ge=0, le=10)]
    model: NonEmptyStr | None
    error: AgentError | None = None

    _valid_thread_id = field_validator("thread_id", mode="before")(_canonical_uuid)

    @model_validator(mode="after")
    def valid_response_shape(self) -> Self:
        if self.error is None and (self.answer is None or self.model is None):
            raise ValueError("successful chat response requires answer and model")
        if self.error is not None and self.answer is not None:
            raise ValueError("failed chat response cannot contain a partial answer")
        return self


class DataQuality(DomainModel):
    raw_row_count: NonNegativeInt
    display_row_count: NonNegativeInt
    prewarm_row_count: NonNegativeInt
    last_trade_date: date | None = None
    missing_values: dict[str, NonNegativeInt] = Field(default_factory=dict)
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    valid: bool = Field(default=True, validation_alias=AliasChoices("valid", "is_valid"))

    @property
    def is_valid(self) -> bool:
        return self.valid

    @model_validator(mode="after")
    def valid_row_counts(self) -> Self:
        if self.display_row_count + self.prewarm_row_count > self.raw_row_count:
            raise ValueError("display and prewarm rows cannot exceed raw rows")
        if self.valid and self.display_row_count > 0 and self.last_trade_date is None:
            raise ValueError("valid displayed data requires a last trade date")
        if self.last_trade_date is not None:
            _validate_not_future(self.last_trade_date)
        return self


class PeriodInfo(DomainModel):
    requested_end_date: date | None
    resolved_end_date: date
    actual_start_date: date
    actual_end_date: date
    last_trade_date: date
    adjustment: Literal["qfq"] = "qfq"

    @model_validator(mode="after")
    def valid_period_order(self) -> Self:
        for value in (
            self.requested_end_date,
            self.resolved_end_date,
            self.actual_start_date,
            self.actual_end_date,
            self.last_trade_date,
        ):
            _validate_not_future(value)
        if self.actual_start_date > self.actual_end_date:
            raise ValueError("actual start date cannot be after actual end date")
        if self.requested_end_date is not None and self.resolved_end_date > self.requested_end_date:
            raise ValueError("resolved end date cannot be after requested end date")
        if self.actual_end_date > self.resolved_end_date:
            raise ValueError("actual end date cannot be after resolved end date")
        if self.last_trade_date > self.resolved_end_date:
            raise ValueError("last trade date cannot be after resolved end date")
        if self.last_trade_date != self.actual_end_date:
            raise ValueError("last trade date must equal actual end date")
        return self


class Evidence(DomainModel):
    source_key: NonEmptyStr
    observed_value: FiniteFloat
    interpretation: ShortText


class Risk(DomainModel):
    risk_type: RiskType
    evidence_key: NonEmptyStr | None = None
    description: ShortText


class WatchLevel(DomainModel):
    label: WatchLabel
    price: PositiveFiniteFloat
    basis_key: BasisKey
    rationale: ShortText


class ScoreResult(DomainModel):
    total: Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]
    group_scores: dict[ScoreGroup, FiniteFloat | None] = Field(default_factory=dict)
    signal: Signal
    raw_group_scores: dict[ScoreGroup, FiniteFloat] = Field(default_factory=dict)
    evaluable_capacity: dict[ScoreGroup, NonNegativeInt] = Field(default_factory=dict)
    usable_groups: list[ScoreGroup] = Field(default_factory=list)
    bullish_points: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    bearish_points: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    positive_evidence: list[Evidence] = Field(default_factory=list)
    negative_evidence: list[Evidence] = Field(default_factory=list)
    conflict_evidence: list[Evidence] = Field(
        default_factory=list,
        validation_alias=AliasChoices("conflict_evidence", "conflicts"),
    )
    risks: list[Risk] = Field(default_factory=list)
    watch_levels: list[WatchLevel] = Field(default_factory=list)
    completeness: UnitFloat = Field(
        default=0,
        validation_alias=AliasChoices("completeness", "data_completeness"),
    )
    consistency: UnitFloat = Field(
        default=0,
        validation_alias=AliasChoices("consistency", "signal_consistency"),
    )
    risk_score: NonNegativeInt
    risk_level: RiskLevel
    rule_version: Literal["score-v1"] = Field(
        default="score-v1",
        validation_alias=AliasChoices("rule_version", "version"),
    )

    @property
    def conflicts(self) -> list[Evidence]:
        return self.conflict_evidence

    @property
    def data_completeness(self) -> float:
        return self.completeness

    @property
    def signal_consistency(self) -> float:
        return self.consistency

    @property
    def version(self) -> str:
        return self.rule_version

    @model_validator(mode="after")
    def signal_and_risk_match_numeric_values(self) -> Self:
        expected_signal: Signal
        if self.total >= 40:
            expected_signal = "偏多"
        elif self.total >= 15:
            expected_signal = "中性偏多"
        elif self.total > -15:
            expected_signal = "中性"
        elif self.total > -40:
            expected_signal = "中性偏空"
        else:
            expected_signal = "偏空"
        if self.signal != expected_signal:
            raise ValueError("signal must match the score-v1 total boundary")

        expected_risk: RiskLevel
        if self.risk_score <= 1:
            expected_risk = "低"
        elif self.risk_score <= 3:
            expected_risk = "中"
        else:
            expected_risk = "高"
        if self.risk_level != expected_risk:
            raise ValueError("risk level must match risk score")
        return self


class AnalysisResult(DomainModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    status: AnalysisStatus
    stock: StockInfo
    period: PeriodInfo | None = None
    data_quality: DataQuality | None = None
    series: pd.DataFrame | None = Field(default=None, exclude=True)
    snapshot: dict[str, SnapshotValue] | None = None
    score: ScoreResult | None = None
    plan_trace: list[dict[str, TraceValue]] = Field(default_factory=list)
    analysis_id: NonEmptyStr | None = None
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    error: AgentError | None = None

    @model_validator(mode="after")
    def valid_terminal_state(self) -> Self:
        analysis_fields = (self.series, self.snapshot, self.score, self.analysis_id)
        if self.status == "success":
            required = (self.period, self.data_quality, *analysis_fields)
            if any(value is None for value in required) or self.error is not None:
                raise ValueError("success requires complete analysis fields and no error")
            assert self.period is not None
            assert self.data_quality is not None
            assert self.series is not None
            assert self.snapshot is not None
            if not self.data_quality.valid:
                raise ValueError("success requires valid data quality")
            if self.data_quality.raw_row_count <= 0 or self.data_quality.display_row_count <= 0:
                raise ValueError("success requires positive raw and display row counts")
            if self.series.empty:
                raise ValueError("success requires a non-empty series")
            if not self.snapshot:
                raise ValueError("success requires a non-empty snapshot")
            if self.data_quality.last_trade_date != self.period.last_trade_date:
                raise ValueError("data quality and period last trade dates must match")
        elif self.status == "insufficient_data":
            if self.period is None or self.data_quality is None:
                raise ValueError("insufficient_data requires period and data quality")
            if any(value is not None for value in analysis_fields) or self.error is not None:
                raise ValueError("insufficient_data cannot contain partial analysis or an error")
        elif self.error is None:
            raise ValueError("error status requires an AgentError")
        elif any(value is not None for value in analysis_fields):
            raise ValueError("error status cannot contain partial analysis")
        return self


StrictFiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
StrictPositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


class AIRawEvidence(DomainModel):
    source_key: NonEmptyStr
    observed_value: StrictFiniteFloat
    interpretation: ShortText


class AIRawRisk(DomainModel):
    risk_type: RiskType
    evidence_key: NonEmptyStr | None = None
    description: ShortText


class AIRawWatchLevel(DomainModel):
    label: WatchLabel
    price: StrictPositiveFiniteFloat
    basis_key: BasisKey
    rationale: ShortText


class AIRawInterpretation(DomainModel):
    """The only fields accepted from an AI model response."""

    model_signal: Signal
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=280)]
    evidence: list[AIRawEvidence] = Field(min_length=2, max_length=6)
    risks: list[AIRawRisk] = Field(default_factory=list, max_length=6)
    watch_levels: list[AIRawWatchLevel] = Field(min_length=1, max_length=7)


class AIInterpretation(DomainModel):
    """A validated AI response enriched exclusively with server-owned metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    model_signal: Signal
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=280)]
    evidence: list[Evidence] = Field(min_length=2, max_length=6)
    risks: list[Risk] = Field(default_factory=list, max_length=6)
    watch_levels: list[WatchLevel] = Field(min_length=1, max_length=7)
    rule_signal: Signal
    consistency_status: ConsistencyStatus
    disclaimer: Literal[INVESTMENT_DISCLAIMER] = INVESTMENT_DISCLAIMER
    model: NonEmptyStr
    prompt_version: NonEmptyStr
    cache_hit: bool = False
    generated_at: datetime

    @classmethod
    def from_raw(
        cls,
        raw: AIRawInterpretation,
        *,
        rule_signal: Signal,
        model: NonEmptyStr,
        prompt_version: NonEmptyStr,
        generated_at: datetime,
        cache_hit: bool = False,
    ) -> Self:
        """Attach trusted server metadata to an already validated model response."""

        if not isinstance(raw, AIRawInterpretation):
            raise TypeError("raw must be an AIRawInterpretation")
        consistency: ConsistencyStatus = (
            "consistent" if raw.model_signal == rule_signal else "mismatch"
        )
        return cls.model_validate(
            {
                **raw.model_dump(),
                "rule_signal": rule_signal,
                "consistency_status": consistency,
                "disclaimer": INVESTMENT_DISCLAIMER,
                "model": model,
                "prompt_version": prompt_version,
                "cache_hit": cache_hit,
                "generated_at": generated_at,
            },
            context={_AI_ENRICHMENT_CONTEXT: _AI_ENRICHMENT_TOKEN},
        )

    @model_validator(mode="before")
    @classmethod
    def require_enrichment_factory(cls, data: Any, info: ValidationInfo) -> Any:
        context = info.context or {}
        if context.get(_AI_ENRICHMENT_CONTEXT) is not _AI_ENRICHMENT_TOKEN:
            raise ValueError("AIInterpretation must be created with from_raw")
        return data

    @model_validator(mode="after")
    def valid_consistency_and_timestamp(self) -> Self:
        signals_match = self.model_signal == self.rule_signal
        expected: ConsistencyStatus = "consistent" if signals_match else "mismatch"
        if self.consistency_status != expected:
            raise ValueError("consistency status must reflect model and rule signals")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return self
