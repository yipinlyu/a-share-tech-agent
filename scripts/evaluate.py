#!/usr/bin/env python3
"""Deterministic, credential-safe evaluation for the A-share technical agent."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from stock_agent.config import Settings
from stock_agent.data.stock_search import search_stocks
from stock_agent.data.tushare_client import (
    TushareAdapterError,
    TushareDataClient,
    assess_data_quality,
)
from stock_agent.indicators.engine import INDICATOR_COLUMNS, compute_indicators
from stock_agent.llm.deepseek_client import DeepSeekClient
from stock_agent.llm.schemas import parse_and_validate_interpretation

REPORT_SCHEMA_VERSION = "offline-evaluation-report-v1"
REFERENCE_FORMULA_VERSION = "evaluation-reference-v1"
INDICATOR_FORMULA_VERSION = "indicators-v1"
SCORING_RULE_VERSION = "score-v1"
RTOL = 1e-6
ATOL = 1e-8
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/evaluation-report.json"
SEARCH_CASES_PATH = PROJECT_ROOT / "tests/evaluation/search_cases.json"
ERROR_CASES_PATH = PROJECT_ROOT / "tests/evaluation/error_cases.json"

_VALID_AI_PAYLOAD = {
    "model_signal": "中性偏多",
    "summary": "趋势略偏强，但仍需关注波动。",
    "evidence": [
        {"source_key": "close", "observed_value": 100.0, "interpretation": "收盘价"},
        {"source_key": "ma20", "observed_value": 99.5, "interpretation": "二十日均线"},
    ],
    "risks": [
        {
            "risk_type": "volatility",
            "evidence_key": "atr_ratio",
            "description": "波动风险仍需观察",
        }
    ],
    "watch_levels": [
        {"label": "支撑观察", "price": 99.5, "basis_key": "ma20", "rationale": "均线"}
    ],
}
_AI_ANALYSIS = {
    "stock": {"ts_code": "600519.SH", "name": "贵州茅台"},
    "snapshot": {"close": 100.0, "ma20": 99.5, "atr_ratio": 0.02},
    "score": {
        "signal": "中性偏多",
        "watch_levels": _VALID_AI_PAYLOAD["watch_levels"],
    },
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 A 股智研台确定性离线评测。")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="机器可读 JSON 报告路径（默认 artifacts/evaluation-report.json）。",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="读取本地配置并报告就绪状态；不会输出凭据，也不会替代发布前真实 smoke test。",
    )
    return parser


def run_evaluation() -> dict[str, Any]:
    """Run all deterministic suites without network access or paid model calls."""

    started = datetime.now(timezone.utc)
    suite_results: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    for name, evaluator in (
        ("search_top5", _evaluate_search),
        ("indicator_reference", _evaluate_indicators),
        ("data_quality_detection", _evaluate_data_quality),
        ("ai_grounding", _evaluate_ai_grounding),
        ("cache_skip", _evaluate_cache_skip),
        ("safe_degradation", _evaluate_safe_degradation),
    ):
        tick = time.perf_counter()
        suite_results[name] = evaluator()
        timings[name] = round(time.perf_counter() - tick, 6)

    search_fixture = _read_json(SEARCH_CASES_PATH)
    error_fixture = _read_json(ERROR_CASES_PATH)
    data_quality_count = sum(
        case["kind"] == "data_quality" for case in error_fixture["cases"]
    )
    thresholds = {
        "search_top5": 0.95,
        "indicator_reference": 1.0,
        "data_quality_detection": 1.0,
        "ai_grounding": 1.0,
        "cache_skip": 1.0,
        "safe_degradation": 1.0,
    }
    metrics: dict[str, dict[str, Any]] = {}
    for name, result in suite_results.items():
        metrics[name] = _metric(
            correct=result["correct"],
            total=result["total"],
            threshold=thresholds[name],
            details=result["details"],
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "offline",
        "generated_at": started.isoformat(),
        "datasets": {
            "search_queries": len(search_fixture["cases"]),
            "market_rows": 500,
            "market_patterns": 5,
            "data_quality_cases": data_quality_count,
            "ai_grounding_cases": 50,
            "cache_cases": 10,
            "exception_cases": len(error_fixture["cases"]),
        },
        "dataset_versions": {
            "search": search_fixture["dataset_version"],
            "exceptions": error_fixture["dataset_version"],
        },
        "disclosures": {
            "indicator_formula_version": INDICATOR_FORMULA_VERSION,
            "scoring_rule_version": SCORING_RULE_VERSION,
            "reference_formula_version": REFERENCE_FORMULA_VERSION,
            "rtol": RTOL,
            "atol": ATOL,
            "historical_return_ci_gate": False,
            "historical_evaluation_limitations": [
                "离线夹具不代表完整 A 股横截面、真实停牌、涨跌停或流动性条件。",
                "未来 5/20 日收益仅可作描述性教学回看，不代表未来表现。",
                "手续费、滑点、停牌和样本选择会显著改变历史回看结果。",
            ],
        },
        "metrics": metrics,
        "timings_seconds": timings,
        "timing_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine() or "unknown",
            "processor": platform.processor() or "unknown",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "clock": "time.perf_counter",
            "network": "disabled",
            "timings_are_ci_gates": False,
            "note": "耗时随机器和共享 CI 负载变化，仅记录不设阈值。",
        },
        "historical_evaluation": {
            "performed": False,
            "ci_gate": False,
            "future_horizons_trading_days": [5, 20],
            "planned_statistics": [
                "方向命中率",
                "平均收益",
                "最大不利变动",
                "样本数",
            ],
            "reason": "本任务不联网，未用有限静态夹具制造收益结论。",
        },
    }
    report["passed"] = all(metric["passed"] for metric in metrics.values())
    return report


def public_live_configuration(settings: Settings) -> dict[str, str | bool]:
    """Expose only capability booleans and the non-secret model identifier."""

    return {
        "tushare_configured": settings.data_enabled,
        "deepseek_configured": settings.ai_enabled,
        "deepseek_model": settings.deepseek_model,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_live_settings() -> Settings:
    secrets_path = PROJECT_ROOT / ".streamlit/secrets.toml"
    secrets: dict[str, object] = {}
    if secrets_path.is_file():
        with secrets_path.open("rb") as handle:
            loaded = tomllib.load(handle)
        secrets = {str(key): value for key, value in loaded.items()}
    return Settings.from_sources(secrets=secrets)


def _metric(
    *, correct: int, total: int, threshold: float, details: dict[str, Any]
) -> dict[str, Any]:
    value = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "value": value,
        "threshold": threshold,
        "passed": total > 0 and value >= threshold,
        "details": details,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation fixture must be an object: {path.name}")
    return payload


def _evaluate_search() -> dict[str, Any]:
    fixture = _read_json(SEARCH_CASES_PATH)
    master = pd.DataFrame(fixture["stock_master"])
    correct = 0
    misses: list[dict[str, str]] = []
    for case in fixture["cases"]:
        result = search_stocks(case["query"], master, limit=5)
        top5 = [candidate.ts_code for candidate in result.candidates]
        if case["expected_ts_code"] in top5:
            correct += 1
        else:
            misses.append(
                {"id": case["id"], "expected": case["expected_ts_code"], "actual": ",".join(top5)}
            )
    return {
        "correct": correct,
        "total": len(fixture["cases"]),
        "details": {"top_k": 5, "misses": misses[:20]},
    }


def _market_patterns() -> dict[str, pd.DataFrame]:
    positions = np.arange(100, dtype=float)
    paths = {
        "uptrend": 80.0 + positions * 0.25 + np.sin(positions / 7.0),
        "downtrend": 140.0 - positions * 0.22 + np.cos(positions / 6.0),
        "sideways": 100.0 + 3.0 * np.sin(positions / 5.0),
        "volatile": 105.0 + 8.0 * np.sin(positions / 2.5) + 2.0 * np.cos(positions / 7.0),
        "gap_volume": 90.0
        + positions * 0.1
        + np.sin(positions / 4.0)
        + np.where(positions >= 30, 5.0, 0.0)
        - np.where(positions >= 70, 4.0, 0.0),
    }
    return {
        name: _make_market_frame(close, index=index)
        for index, (name, close) in enumerate(paths.items(), start=1)
    }


def _make_market_frame(close: np.ndarray, *, index: int) -> pd.DataFrame:
    size = len(close)
    positions = np.arange(size, dtype=float)
    open_price = close + np.sin(positions / 3.0) * 0.35
    high = np.maximum(open_price, close) + 0.8 + (positions % 3) * 0.03
    low = np.minimum(open_price, close) - 0.8 - (positions % 2) * 0.03
    pre_close = np.r_[close[0], close[:-1]]
    change = close - pre_close
    pct_chg = np.divide(change * 100.0, pre_close, out=np.zeros_like(change), where=pre_close != 0)
    volume = 1_000.0 + positions * 5.0 + (positions % 13) * 29.0
    if index == 5:
        volume = volume + np.where((positions % 17) == 0, 1_500.0, 0.0)
    return pd.DataFrame(
        {
            "ts_code": pd.Series([f"{index:06d}.SZ"] * size, dtype="string"),
            "trade_date": pd.date_range("2023-01-02", periods=size, freq="B"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pre_close,
            "change": change,
            "pct_chg": pct_chg,
            "vol": volume,
            "amount": volume * close,
        }
    )


def _evaluate_indicators() -> dict[str, Any]:
    correct = 0
    total = 0
    mismatches: list[dict[str, Any]] = []
    for pattern, frame in _market_patterns().items():
        actual = compute_indicators(frame)
        expected = _reference_indicators(frame)
        for position in range(len(frame)):
            total += 1
            actual_row = actual.loc[position, list(INDICATOR_COLUMNS)].to_numpy(dtype=float)
            expected_row = expected.loc[position, list(INDICATOR_COLUMNS)].to_numpy(dtype=float)
            if np.allclose(actual_row, expected_row, rtol=RTOL, atol=ATOL, equal_nan=True):
                correct += 1
            elif len(mismatches) < 20:
                differing = [
                    column
                    for column, left, right in zip(INDICATOR_COLUMNS, actual_row, expected_row)
                    if not math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL)
                    and not (math.isnan(left) and math.isnan(right))
                ]
                mismatches.append({"pattern": pattern, "row": position, "columns": differing})
    return {
        "correct": correct,
        "total": total,
        "details": {
            "patterns": list(_market_patterns()),
            "compared_columns": list(INDICATOR_COLUMNS),
            "mismatches": mismatches,
        },
    }


def _reference_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Independent transcription of the frozen formulas for offline comparison."""

    result = frame.copy(deep=True)
    close = pd.Series(frame["close"].to_numpy(float), index=frame.index)
    high = pd.Series(frame["high"].to_numpy(float), index=frame.index)
    low = pd.Series(frame["low"].to_numpy(float), index=frame.index)
    volume = pd.Series(frame["vol"].to_numpy(float), index=frame.index)

    for window in (5, 10, 20, 60):
        result[f"ma{window}"] = close.rolling(window, min_periods=window).sum() / window
    result["ema12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    result["ema26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = result["ema12"] - result["ema26"]
    valid_macd = result["macd"].dropna()
    result["macd_signal"] = valid_macd.ewm(span=9, adjust=False, min_periods=9).mean().reindex(
        result.index
    )
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    result["boll_mid"] = close.rolling(20, min_periods=20).sum() / 20.0
    deviations = close.rolling(20, min_periods=20).std(ddof=0)
    result["boll_upper"] = result["boll_mid"] + 2.0 * deviations
    result["boll_lower"] = result["boll_mid"] - 2.0 * deviations
    result["bandwidth"] = (
        (result["boll_upper"] - result["boll_lower"]) / result["boll_mid"]
    ).mask(result["boll_mid"] == 0)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).where(delta.notna())
    loss = (-delta.where(delta < 0, 0.0)).where(delta.notna())
    average_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    result["rsi14"] = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    result.loc[(average_loss == 0) & (average_gain > 0), "rsi14"] = 100.0
    result.loc[(average_loss == 0) & (average_gain == 0), "rsi14"] = 50.0

    rolling_low = low.rolling(9, min_periods=9).min()
    rolling_high = high.rolling(9, min_periods=9).max()
    width = rolling_high - rolling_low
    result["rsv"] = ((close - rolling_low) / width * 100.0).mask(width == 0, 50.0)
    k_values = np.full(len(result), np.nan)
    d_values = np.full(len(result), np.nan)
    previous_k = previous_d = 50.0
    for offset, rsv in enumerate(result["rsv"].to_numpy(float)):
        if math.isnan(rsv):
            continue
        previous_k = 2.0 / 3.0 * previous_k + 1.0 / 3.0 * rsv
        previous_d = 2.0 / 3.0 * previous_d + 1.0 / 3.0 * previous_k
        k_values[offset] = previous_k
        d_values[offset] = previous_d
    result["kdj_k"] = k_values
    result["kdj_d"] = d_values
    result["kdj_j"] = 3.0 * result["kdj_k"] - 2.0 * result["kdj_d"]

    previous = close.shift(1)
    result["tr"] = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    if not result.empty:
        result.loc[result.index[0], "tr"] = high.iloc[0] - low.iloc[0]
    result["atr14"] = result["tr"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    signed_volume = np.sign(delta).fillna(0.0) * volume
    result["obv"] = signed_volume.cumsum()
    if not result.empty:
        result.loc[result.index[0], "obv"] = 0.0
    result["vol_ma20"] = volume.rolling(20, min_periods=20).sum() / 20.0
    result["volume_ratio"] = volume / result["vol_ma20"]
    for horizon in (5, 10, 20):
        result[f"return_{horizon}d"] = close / close.shift(horizon) - 1.0
    result.loc[:, list(INDICATOR_COLUMNS)] = result.loc[:, list(INDICATOR_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )
    return result


def _data_quality_frame(scenario: str) -> tuple[pd.DataFrame, date]:
    frame = _market_patterns()["uptrend"].copy(deep=True)
    display_start = frame.loc[60, "trade_date"].date()
    if scenario == "missing_column":
        frame = frame.drop(columns="amount")
    elif scenario == "missing_value":
        frame.loc[70, "close"] = np.nan
    elif scenario == "unsorted_dates":
        first, second = frame.loc[70, "trade_date"], frame.loc[71, "trade_date"]
        frame.loc[70, "trade_date"], frame.loc[71, "trade_date"] = second, first
    elif scenario == "duplicate_date":
        frame.loc[71, "trade_date"] = frame.loc[70, "trade_date"]
    elif scenario == "nonpositive_price":
        frame.loc[70, "low"] = 0.0
    elif scenario == "negative_volume":
        frame.loc[70, "vol"] = -1.0
    elif scenario == "invalid_ohlc":
        frame.loc[70, "high"] = frame.loc[70, "low"] - 1.0
    elif scenario == "empty_data":
        frame = frame.iloc[0:0].copy()
    elif scenario == "prewarm_59":
        display_start = frame.loc[59, "trade_date"].date()
    elif scenario == "empty_display":
        display_start = (frame.iloc[-1]["trade_date"] + pd.Timedelta(days=10)).date()
    else:
        raise ValueError(f"unknown data-quality scenario: {scenario}")
    return frame, display_start


def _evaluate_data_quality() -> dict[str, Any]:
    fixture = _read_json(ERROR_CASES_PATH)
    cases = [case for case in fixture["cases"] if case["kind"] == "data_quality"]
    correct = 0
    misses: list[str] = []
    for case in cases:
        try:
            frame, display_start = _data_quality_frame(case["scenario"])
            quality = assess_data_quality(frame, display_start_date=display_start)
            detected = not quality.valid and bool(quality.warnings)
        except BaseException:
            detected = False
        if detected:
            correct += 1
        else:
            misses.append(case["id"])
    return {"correct": correct, "total": len(cases), "details": {"misses": misses}}


def _ai_validation_cases() -> list[tuple[str, dict[str, Any], bool]]:
    cases: list[tuple[str, dict[str, Any], bool]] = []
    for index in range(25):
        valid = copy.deepcopy(_VALID_AI_PAYLOAD)
        valid["summary"] = f"离线有效 grounding 案例 {index + 1}。"
        cases.append((f"ai-valid-{index + 1:02d}", valid, True))
    for index in range(25):
        invalid = copy.deepcopy(_VALID_AI_PAYLOAD)
        variant = index % 5
        if variant == 0:
            invalid["evidence"][0]["source_key"] = "invented_indicator"
        elif variant == 1:
            invalid["evidence"][0]["observed_value"] = 999.0
        elif variant == 2:
            invalid["watch_levels"][0]["price"] = 1.0
        elif variant == 3:
            invalid["risks"][0]["evidence_key"] = "invented_indicator"
        else:
            invalid["unexpected"] = "forbidden"
        cases.append((f"ai-invalid-{index + 1:02d}", invalid, False))
    return cases


def _evaluate_ai_grounding() -> dict[str, Any]:
    correct = 0
    misses: list[str] = []
    for case_id, payload, expected_valid in _ai_validation_cases():
        try:
            parse_and_validate_interpretation(
                json.dumps(payload, ensure_ascii=False),
                snapshot=_AI_ANALYSIS["snapshot"],
                watch_levels=_AI_ANALYSIS["score"]["watch_levels"],
            )
            actual_valid = True
        except (TypeError, ValueError):
            actual_valid = False
        if actual_valid == expected_valid:
            correct += 1
        else:
            misses.append(case_id)
    return {
        "correct": correct,
        "total": 50,
        "details": {"expected_valid": 25, "expected_rejected": 25, "misses": misses},
    }


class _FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("offline fake received an unexpected model call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeOpenAI:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcomes))

    def factory(self, **_kwargs: object) -> _FakeOpenAI:
        return self


class _FakeCache:
    def __init__(self, *, fail_read: bool = False, fail_write: bool = False) -> None:
        self.values: dict[str, object] = {}
        self.fail_read = fail_read
        self.fail_write = fail_write

    def get_ai(self, key: str) -> object | None:
        if self.fail_read:
            raise RuntimeError("offline cache read failure")
        return self.values.get(key)

    def put_ai(self, key: str, payload: object) -> None:
        if self.fail_write:
            raise RuntimeError("offline cache write failure")
        self.values[key] = payload


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("offline upstream failure")
        self.status_code = status_code


class _FakePro:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def stock_basic(self, **_kwargs: object) -> pd.DataFrame:
        raise self.error

    def daily(self, **_kwargs: object) -> pd.DataFrame:
        raise self.error

    def adj_factor(self, **_kwargs: object) -> pd.DataFrame:
        raise self.error


def _fake_ai_response(payload: dict[str, Any] | str) -> object:
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _offline_deepseek(
    outcomes: list[object], *, cache: _FakeCache | None = None
) -> tuple[DeepSeekClient, _FakeOpenAI]:
    fake = _FakeOpenAI(outcomes)
    client = DeepSeekClient(
        api_key="offline-fixture-not-a-secret",
        cache=cache,
        openai_factory=fake.factory,
        clock=lambda: datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc),
    )
    return client, fake


def _evaluate_cache_skip() -> dict[str, Any]:
    cache = _FakeCache()
    client, fake = _offline_deepseek([_fake_ai_response(_VALID_AI_PAYLOAD)], cache=cache)
    initial = client.interpret("offline-analysis", _AI_ANALYSIS)
    correct = 0
    for _ in range(10):
        repeated = client.interpret("offline-analysis", _AI_ANALYSIS)
        if getattr(repeated, "cache_hit", False) is True and len(fake.chat.completions.calls) == 1:
            correct += 1
    return {
        "correct": correct,
        "total": 10,
        "details": {
            "warmup_model_calls": 1,
            "repeated_model_calls": max(len(fake.chat.completions.calls) - 1, 0),
            "warmup_succeeded": getattr(initial, "cache_hit", None) is False,
        },
    }


def _evaluate_safe_degradation() -> dict[str, Any]:
    fixture = _read_json(ERROR_CASES_PATH)
    correct = 0
    failures: list[str] = []
    for case in fixture["cases"]:
        try:
            safe = _run_exception_case(case)
        except BaseException:
            safe = False
        if safe:
            correct += 1
        else:
            failures.append(case["id"])
    return {
        "correct": correct,
        "total": len(fixture["cases"]),
        "details": {"matrix_version": fixture["dataset_version"], "failures": failures},
    }


def _run_exception_case(case: dict[str, Any]) -> bool:
    kind = case["kind"]
    scenario = case["scenario"]
    if kind == "data_quality":
        frame, display_start = _data_quality_frame(scenario)
        result = assess_data_quality(frame, display_start_date=display_start)
        return not result.valid and bool(result.warnings)
    if kind == "tushare":
        if scenario == "missing_config":
            operation = lambda: TushareDataClient(token=None)
        else:
            upstream = {
                "permission": RuntimeError("权限不足，需要积分"),
                "invalid_token": RuntimeError("invalid token"),
                "rate_limit": RuntimeError("rate limit 429"),
                "timeout": TimeoutError("timed out"),
                "connection": ConnectionError("connection failed"),
                "server": RuntimeError("server 503"),
            }[scenario]
            operation = lambda: TushareDataClient(pro=_FakePro(upstream)).fetch_daily(
                "600519.SH", date(2024, 1, 1), date(2024, 1, 3)
            )
        try:
            operation()
            return False
        except TushareAdapterError as exc:
            return _safe_error_matches(exc.error, case)
    if kind == "deepseek":
        if scenario == "invalid_json_twice":
            outcomes = [_fake_ai_response("bad"), _fake_ai_response("still bad")]
        else:
            outcomes = [_StatusError(int(scenario.removeprefix("status_")))]
        client, _ = _offline_deepseek(outcomes)
        result = client.interpret("offline-analysis", _AI_ANALYSIS)
        return _safe_error_matches(result, case)
    if kind == "cache":
        cache = _FakeCache(
            fail_read=scenario == "read_failure",
            fail_write=scenario == "write_failure",
        )
        client, _ = _offline_deepseek([_fake_ai_response(_VALID_AI_PAYLOAD)], cache=cache)
        result = client.interpret("offline-analysis", _AI_ANALYSIS)
        return getattr(result, "model_signal", None) == "中性偏多"
    return False


def _safe_error_matches(error: object, case: dict[str, Any]) -> bool:
    message = getattr(error, "user_message", "")
    lowered = str(message).lower()
    return (
        getattr(error, "code", None) == case["expected_code"]
        and getattr(error, "retryable", None) is case["expected_retryable"]
        and bool(message)
        and "offline-fixture-not-a-secret" not in lowered
        and "authorization" not in lowered
        and "traceback" not in lowered
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = run_evaluation()
    if args.live:
        report["mode"] = "live-config-check"
        report["live_configuration"] = public_live_configuration(_read_live_settings())
        report["live_note"] = (
            "仅检查配置是否就绪；付费 API 的最小真实 smoke test 在发布验收阶段单独运行。"
        )
    write_report(report, args.output)
    summary = {
        "mode": report["mode"],
        "passed": report["passed"],
        "report": str(Path(args.output)),
        "metrics": {
            name: f"{metric['correct']}/{metric['total']}"
            for name, metric in report["metrics"].items()
        },
    }
    if args.live:
        summary["live_configuration"] = report["live_configuration"]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
