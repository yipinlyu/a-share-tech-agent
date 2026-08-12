from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
import pytest
from pydantic import ValidationError

from stock_agent.domain.models import (
    INVESTMENT_DISCLAIMER,
    AIInterpretation,
    AIRawInterpretation,
    AIRequest,
    AgentError,
    AnalysisRequest,
    AnalysisResult,
    ChatRequest,
    ChatResponse,
    DataQuality,
    Evidence,
    PeriodInfo,
    Risk,
    ScoreResult,
    StockInfo,
    StockQuery,
    StockSearchResult,
    WatchLevel,
)


THREAD_ID = "4fdb57c8-508a-43f0-aa26-4a01389d567e"


def stock() -> StockInfo:
    return StockInfo(
        ts_code="600519.SH",
        symbol="600519",
        name="贵州茅台",
        market="主板",
        industry="白酒",
    )


def period() -> PeriodInfo:
    return PeriodInfo(
        requested_end_date=date(2024, 12, 31),
        resolved_end_date=date(2024, 12, 31),
        actual_start_date=date(2024, 1, 2),
        actual_end_date=date(2024, 12, 31),
        last_trade_date=date(2024, 12, 31),
        adjustment="qfq",
    )


def quality() -> DataQuality:
    return DataQuality(
        raw_row_count=360,
        display_row_count=242,
        prewarm_row_count=118,
        last_trade_date=date(2024, 12, 31),
        missing_values={},
        warnings=["长周期指标可能不稳定"],
        is_valid=True,
    )


def evidence() -> Evidence:
    return Evidence(source_key="rsi14", observed_value=60.0, interpretation="RSI 处于偏强区间")


def watch_level() -> WatchLevel:
    return WatchLevel(
        label="支撑观察",
        price=100.5,
        basis_key="ma20",
        rationale="20 日均线附近",
    )


def score() -> ScoreResult:
    return ScoreResult(
        total=25.0,
        group_scores={"trend": 20.0, "momentum": 5.0, "volume_volatility": 0.0},
        signal="中性偏多",
        positive_evidence=[evidence()],
        negative_evidence=[],
        conflict_evidence=[],
        watch_levels=[watch_level()],
        completeness=0.9,
        consistency=0.75,
        risk_score=1,
        risk_level="低",
        rule_version="score-v1",
    )


def success_result_data() -> dict[str, object]:
    return {
        "status": "success",
        "stock": stock(),
        "period": period(),
        "data_quality": quality(),
        "series": pd.DataFrame({"close": [100.0, 101.0]}),
        "snapshot": {"close": 101.0, "rsi14": 60.0},
        "score": score(),
        "plan_trace": [{"node": "score", "status": "success", "elapsed_ms": 1.2}],
        "analysis_id": "analysis-123",
        "warnings": [],
        "error": None,
    }


@pytest.mark.parametrize(
    "ts_code",
    ["600519.SH", "000001.SZ", "300750.SZ", "688981.SH", "430047.BJ", "920002.BJ"],
)
def test_stock_info_accepts_supported_a_share_code_families(ts_code: str) -> None:
    info = StockInfo(ts_code=ts_code, name="测试", market="主板", industry=None)

    assert info.ts_code == ts_code


@pytest.mark.parametrize(
    "ts_code",
    ["600519", "600519.SZ", "000001.SH", "200001.SZ", "600519.HK", "ABCDEF.SH"],
)
def test_stock_info_rejects_non_a_share_tushare_codes(ts_code: str) -> None:
    with pytest.raises(ValidationError):
        StockInfo(ts_code=ts_code, name="测试", market="主板", industry=None)


def test_stock_info_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        StockInfo(
            ts_code="600519.SH",
            name="贵州茅台",
            market="主板",
            industry="白酒",
            secret_note="unexpected",
        )


@pytest.mark.parametrize("lookback", [3, 6, 12, 24, 36])
def test_stock_query_accepts_supported_lookbacks_and_strips_query(lookback: int) -> None:
    query = StockQuery(query="  贵州茅台  ", lookback_months=lookback)

    assert query.query == "贵州茅台"
    assert query.lookback_months == lookback
    assert query.end_date is None


@pytest.mark.parametrize("lookback", [0, 1, 18, 48])
def test_stock_query_rejects_unsupported_lookbacks(lookback: int) -> None:
    with pytest.raises(ValidationError):
        StockQuery(query="600519", lookback_months=lookback)


def test_stock_query_enforces_text_bounds_and_no_future_date() -> None:
    with pytest.raises(ValidationError):
        StockQuery(query="   ")
    with pytest.raises(ValidationError):
        StockQuery(query="x" * 31)
    with pytest.raises(ValidationError):
        StockQuery(query="600519", end_date=date.today() + timedelta(days=1))


def test_analysis_request_has_frozen_default_indicator_parameters() -> None:
    request = AnalysisRequest(ts_code="600519.SH", lookback_months=12)

    assert request.requested_end_date is None
    assert request.indicator_config.ma_windows == (5, 10, 20, 60)
    assert request.indicator_config.macd == (12, 26, 9)
    assert request.indicator_config.bollinger == (20, 2.0)
    assert request.indicator_config.rsi_window == 14
    assert request.indicator_config.kdj == (9, 3, 3)
    assert request.indicator_config.atr_window == 14
    assert request.indicator_config.volume_window == 20


def test_analysis_request_rejects_unsupported_code_lookback_date_and_indicator_override() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(ts_code="600519.SZ", lookback_months=12)
    with pytest.raises(ValidationError):
        AnalysisRequest(ts_code="600519.SH", lookback_months=18)
    with pytest.raises(ValidationError):
        AnalysisRequest(
            ts_code="600519.SH",
            lookback_months=12,
            requested_end_date=date.today() + timedelta(days=1),
        )
    with pytest.raises(ValidationError):
        AnalysisRequest(
            ts_code="600519.SH",
            lookback_months=12,
            indicator_config={"rsi_window": 7},
        )


def test_ai_request_requires_analysis_id_and_defaults_to_cached_behavior() -> None:
    request = AIRequest(analysis_id="analysis-123")

    assert request.force_refresh is False
    with pytest.raises(ValidationError):
        AIRequest(analysis_id="  ")


def test_chat_request_keeps_uuid_string_and_strips_bounded_question() -> None:
    request = ChatRequest(
        thread_id=THREAD_ID, analysis_id="analysis-123", question="  如何理解 RSI？  "
    )

    assert request.thread_id == THREAD_ID
    assert isinstance(request.thread_id, str)
    assert request.question == "如何理解 RSI？"


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_chat_request_enforces_question_length(question: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(thread_id=THREAD_ID, analysis_id="analysis-123", question=question)


def test_chat_request_rejects_non_uuid_thread_and_empty_analysis() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="thread", analysis_id="analysis-123", question="question")
    with pytest.raises(ValidationError):
        ChatRequest(thread_id=THREAD_ID, analysis_id=" ", question="question")


def test_agent_error_rejects_unknown_codes_and_extra_context() -> None:
    error = AgentError(
        code="RATE_LIMIT", user_message="请稍后重试", retryable=True, trace_id="trace-1"
    )

    assert error.retryable is True
    with pytest.raises(ValidationError):
        AgentError(code="NETWORK", user_message="x", retryable=True, trace_id="trace-1")
    with pytest.raises(ValidationError):
        AgentError(
            code="INTERNAL",
            user_message="x",
            retryable=False,
            trace_id="trace-1",
            stack_trace="secret path",
        )


def test_stock_search_result_enforces_each_status_shape() -> None:
    error = AgentError(code="DATA", user_message="数据不可用", retryable=True, trace_id="trace-1")

    assert StockSearchResult(status="resolved", candidates=[stock()]).error is None
    assert StockSearchResult(status="ambiguous", candidates=[stock(), stock()]).error is None
    assert StockSearchResult(status="not_found", candidates=[]).candidates == []
    assert StockSearchResult(status="error", candidates=[], error=error).error == error

    invalid_shapes = [
        {"status": "resolved", "candidates": []},
        {"status": "resolved", "candidates": [stock(), stock()]},
        {"status": "ambiguous", "candidates": [stock()]},
        {"status": "not_found", "candidates": [stock()]},
        {"status": "not_found", "candidates": [], "error": error},
        {"status": "error", "candidates": []},
        {"status": "error", "candidates": [stock()], "error": error},
    ]
    for values in invalid_shapes:
        with pytest.raises(ValidationError):
            StockSearchResult(**values)


def test_stock_search_result_limits_candidates_to_ten() -> None:
    with pytest.raises(ValidationError):
        StockSearchResult(status="ambiguous", candidates=[stock()] * 11)


def test_chat_response_enforces_success_and_failure_shapes() -> None:
    success = ChatResponse(
        answer="RSI 显示动量偏强。",
        thread_id=THREAD_ID,
        turn_count=1,
        model="deepseek-v4-flash",
        error=None,
    )
    error = AgentError(code="MODEL", user_message="模型暂时不可用", retryable=True, trace_id="t")
    failure = ChatResponse(
        answer=None,
        thread_id=THREAD_ID,
        turn_count=0,
        model=None,
        error=error,
    )

    assert success.answer is not None
    assert failure.answer is None
    assert failure.model is None
    with pytest.raises(ValidationError):
        ChatResponse(
            answer=None,
            thread_id=THREAD_ID,
            turn_count=0,
            model=None,
            error=None,
        )
    with pytest.raises(ValidationError):
        ChatResponse(
            answer="partial",
            thread_id=THREAD_ID,
            turn_count=0,
            model=None,
            error=error,
        )


@pytest.mark.parametrize("turn_count", [-1, 11])
def test_chat_response_enforces_turn_count_bounds(turn_count: int) -> None:
    with pytest.raises(ValidationError):
        ChatResponse(
            answer="ok",
            thread_id=THREAD_ID,
            turn_count=turn_count,
            model="deepseek-v4-flash",
        )


def test_data_quality_rejects_negative_counts_and_missing_totals() -> None:
    report = quality()

    assert report.prewarm_row_count == 118
    with pytest.raises(ValidationError):
        DataQuality(
            raw_row_count=-1,
            display_row_count=0,
            prewarm_row_count=0,
            last_trade_date=date(2024, 12, 31),
            missing_values={},
            warnings=[],
            is_valid=False,
        )
    with pytest.raises(ValidationError):
        DataQuality(
            raw_row_count=10,
            display_row_count=8,
            prewarm_row_count=5,
            last_trade_date=date(2024, 12, 31),
            missing_values={},
            warnings=[],
            is_valid=True,
        )


def test_period_info_rejects_inverted_or_future_ranges() -> None:
    assert period().adjustment == "qfq"

    with pytest.raises(ValidationError):
        PeriodInfo(
            requested_end_date=date(2024, 12, 31),
            resolved_end_date=date(2024, 12, 31),
            actual_start_date=date(2025, 1, 1),
            actual_end_date=date(2024, 12, 31),
            last_trade_date=date(2024, 12, 31),
        )
    with pytest.raises(ValidationError):
        PeriodInfo(
            requested_end_date=date.today() + timedelta(days=1),
            resolved_end_date=date.today(),
            actual_start_date=date.today(),
            actual_end_date=date.today(),
            last_trade_date=date.today(),
        )


def test_evidence_and_watch_level_match_grounded_schema() -> None:
    assert evidence().observed_value == 60.0
    assert watch_level().basis_key == "ma20"

    with pytest.raises(ValidationError):
        Evidence(source_key="rsi14", observed_value=float("nan"), interpretation="bad")
    with pytest.raises(ValidationError):
        WatchLevel(label="目标价", price=100, basis_key="ma20", rationale="bad")
    with pytest.raises(ValidationError):
        WatchLevel(label="支撑观察", price=0, basis_key="ma20", rationale="bad")
    with pytest.raises(ValidationError):
        WatchLevel(label="支撑观察", price=100, basis_key="invented", rationale="bad")


def test_score_result_enforces_score_probability_and_risk_bounds() -> None:
    assert score().signal == "中性偏多"

    for changes in (
        {"total": 100.01},
        {"completeness": 1.01},
        {"consistency": -0.01},
        {"risk_score": -1},
        {"risk_level": "极高"},
    ):
        values = score().model_dump()
        values.update(changes)
        with pytest.raises(ValidationError):
            ScoreResult(**values)


def test_analysis_result_accepts_the_success_shape() -> None:
    result = AnalysisResult(**success_result_data())

    assert result.status == "success"
    assert result.analysis_id == "analysis-123"
    assert isinstance(result.series, pd.DataFrame)
    assert result.error is None


@pytest.mark.parametrize(
    "missing_field",
    ["period", "data_quality", "series", "snapshot", "score", "analysis_id"],
)
def test_success_analysis_requires_every_success_only_field(missing_field: str) -> None:
    values = success_result_data()
    values[missing_field] = None

    with pytest.raises(ValidationError):
        AnalysisResult(**values)


def test_insufficient_analysis_has_quality_but_no_partial_analysis() -> None:
    result = AnalysisResult(
        status="insufficient_data",
        stock=stock(),
        period=period(),
        data_quality=quality(),
        series=None,
        snapshot=None,
        score=None,
        plan_trace=[{"node": "validate", "status": "insufficient_data"}],
        analysis_id=None,
        warnings=["预热样本少于 60 条"],
        error=None,
    )

    assert result.status == "insufficient_data"
    values = result.model_dump()
    values["score"] = score()
    with pytest.raises(ValidationError):
        AnalysisResult(**values)


def test_error_analysis_requires_error_and_rejects_partial_score() -> None:
    error = AgentError(code="DATA", user_message="数据获取失败", retryable=True, trace_id="trace-1")
    values = {
        "status": "error",
        "stock": stock(),
        "period": None,
        "data_quality": None,
        "series": None,
        "snapshot": None,
        "score": None,
        "plan_trace": [{"node": "fetch", "status": "error"}],
        "analysis_id": None,
        "warnings": [],
        "error": error,
    }

    assert AnalysisResult(**values).error == error
    with pytest.raises(ValidationError):
        AnalysisResult(**{**values, "error": None})
    with pytest.raises(ValidationError):
        AnalysisResult(**{**values, "score": score()})
    with pytest.raises(ValidationError):
        AnalysisResult(**{**success_result_data(), "error": error})


def test_ai_interpretation_enforces_schema_and_server_consistency() -> None:
    raw = AIRawInterpretation(
        model_signal="中性偏多",
        summary="趋势偏强，但仍需关注波动。",
        evidence=[
            evidence(),
            Evidence(source_key="ma20", observed_value=100.5, interpretation="价格高于均线"),
        ],
        risks=[Risk(risk_type="volatility", evidence_key="atr_ratio", description="波动率偏高")],
        watch_levels=[watch_level()],
    )
    interpretation = AIInterpretation.from_raw(
        raw,
        rule_signal="中性偏多",
        model="deepseek-v4-flash",
        prompt_version="prompt-v1",
        cache_hit=False,
        generated_at=datetime(2024, 12, 31, 8, tzinfo=timezone.utc),
    )

    assert interpretation.disclaimer == INVESTMENT_DISCLAIMER
    assert interpretation.consistency_status == "consistent"

    with pytest.raises(ValidationError):
        AIRawInterpretation(**raw.model_dump(), unexpected="model-controlled")


def test_raw_ai_contract_rejects_server_fields_and_requires_enrichment_factory() -> None:
    raw_payload = {
        "model_signal": "中性偏多",
        "summary": "趋势偏强，但仍需关注波动。",
        "evidence": [
            evidence().model_dump(),
            Evidence(
                source_key="ma20",
                observed_value=100.5,
                interpretation="价格高于均线",
            ).model_dump(),
        ],
        "risks": [
            Risk(
                risk_type="volatility",
                evidence_key="atr_ratio",
                description="波动率偏高",
            ).model_dump()
        ],
        "watch_levels": [watch_level().model_dump()],
    }
    server_fields = {
        "rule_signal": "偏空",
        "consistency_status": "consistent",
        "disclaimer": "伪造免责声明",
        "model": "forged-model",
        "prompt_version": "forged-prompt",
        "cache_hit": True,
        "generated_at": datetime.now(timezone.utc),
    }

    for field_name, forged_value in server_fields.items():
        with pytest.raises(ValidationError):
            AIRawInterpretation.model_validate({**raw_payload, field_name: forged_value})

    raw = AIRawInterpretation.model_validate(raw_payload)
    generated_at = datetime(2024, 12, 31, 8, tzinfo=timezone.utc)
    enriched = AIInterpretation.from_raw(
        raw,
        rule_signal="中性偏多",
        model="deepseek-v4-flash",
        prompt_version="prompt-v1",
        cache_hit=True,
        generated_at=generated_at,
    )

    assert enriched.rule_signal == "中性偏多"
    assert enriched.consistency_status == "consistent"
    assert enriched.disclaimer == INVESTMENT_DISCLAIMER
    assert enriched.model == "deepseek-v4-flash"
    assert enriched.prompt_version == "prompt-v1"
    assert enriched.cache_hit is True
    assert enriched.generated_at == generated_at

    with pytest.raises(ValidationError):
        AIInterpretation(
            **raw_payload,
            rule_signal="中性偏多",
            consistency_status="consistent",
            disclaimer=INVESTMENT_DISCLAIMER,
            model="deepseek-v4-flash",
            prompt_version="prompt-v1",
            cache_hit=False,
            generated_at=generated_at,
        )

    with pytest.raises(TypeError):
        AIInterpretation.from_raw(
            {**raw_payload, **server_fields},  # type: ignore[arg-type]
            rule_signal="中性偏多",
            model="deepseek-v4-flash",
            prompt_version="prompt-v1",
            generated_at=generated_at,
        )


def test_ai_interpretation_accepts_an_explicit_mismatch() -> None:
    raw = AIRawInterpretation(
        model_signal="偏空",
        summary="模型解读与规则信号不一致。",
        evidence=[
            evidence(),
            Evidence(source_key="ma20", observed_value=100.5, interpretation="均线依据"),
        ],
        risks=[Risk(risk_type="signal_conflict", evidence_key=None, description="信号冲突")],
        watch_levels=[watch_level()],
    )
    interpretation = AIInterpretation.from_raw(
        raw,
        rule_signal="中性偏多",
        model="deepseek-v4-flash",
        prompt_version="prompt-v1",
        generated_at=datetime.now(timezone.utc),
    )

    assert interpretation.consistency_status == "mismatch"


def test_ai_interpretation_enforces_array_and_text_bounds() -> None:
    base = {
        "model_signal": "中性",
        "summary": "ok",
        "evidence": [evidence(), evidence()],
        "risks": [Risk(risk_type="other", description="risk")],
        "watch_levels": [watch_level()],
    }

    with pytest.raises(ValidationError):
        AIRawInterpretation(**{**base, "evidence": [evidence()]})
    with pytest.raises(ValidationError):
        AIRawInterpretation(**{**base, "risks": []})
    with pytest.raises(ValidationError):
        AIRawInterpretation(**{**base, "watch_levels": []})
    with pytest.raises(ValidationError):
        AIRawInterpretation(**{**base, "summary": "x" * 281})


def test_models_do_not_silently_accept_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AIRequest(analysis_id="analysis-123", api_key="sk-secret")
    with pytest.raises(ValidationError):
        ChatRequest(
            thread_id=str(uuid4()),
            analysis_id="analysis-123",
            question="question",
            api_key="sk-secret",
        )
