from __future__ import annotations

import json
from pathlib import Path

from stock_agent.config import Settings

from scripts.evaluate import (
    PROJECT_ROOT,
    REPORT_SCHEMA_VERSION,
    build_argument_parser,
    public_live_configuration,
    run_evaluation,
    write_report,
)


def test_offline_evaluation_report_has_required_denominators_and_disclosures(tmp_path) -> None:
    report = run_evaluation()

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["mode"] == "offline"
    assert report["datasets"] == {
        "search_queries": 100,
        "market_rows": 500,
        "market_patterns": 5,
        "data_quality_cases": 10,
        "ai_grounding_cases": 50,
        "cache_cases": 10,
        "exception_cases": 25,
    }

    disclosures = report["disclosures"]
    assert disclosures["indicator_formula_version"] == "indicators-v1"
    assert disclosures["scoring_rule_version"] == "score-v1"
    assert disclosures["reference_formula_version"] == "evaluation-reference-v1"
    assert disclosures["rtol"] == 1e-6
    assert disclosures["atol"] == 1e-8
    assert disclosures["historical_return_ci_gate"] is False
    assert disclosures["historical_evaluation_limitations"]
    assert report["timing_environment"]["timings_are_ci_gates"] is False

    expected = {
        "search_top5": (100, 100, 0.95),
        "indicator_reference": (500, 500, 1.0),
        "data_quality_detection": (10, 10, 1.0),
        "ai_grounding": (50, 50, 1.0),
        "cache_skip": (10, 10, 1.0),
        "safe_degradation": (25, 25, 1.0),
    }
    for name, (correct, total, threshold) in expected.items():
        metric = report["metrics"][name]
        assert (metric["correct"], metric["total"]) == (correct, total)
        assert metric["threshold"] == threshold
        assert metric["passed"] is True

    assert report["historical_evaluation"]["ci_gate"] is False
    assert report["historical_evaluation"]["performed"] is False
    assert report["passed"] is True

    output = tmp_path / "evaluation-report.json"
    write_report(report, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_live_flag_reports_capabilities_without_exposing_credentials() -> None:
    parser = build_argument_parser()
    assert parser.parse_args(["--live"]).live is True

    tushare_marker = "private-tushare-evaluation-marker"
    deepseek_marker = "private-deepseek-evaluation-marker"
    settings = Settings.from_sources(
        environ={
            "TUSHARE_TOKEN": tushare_marker,
            "DEEPSEEK_API_KEY": deepseek_marker,
            "DEEPSEEK_MODEL": "deepseek-v4-flash",
        }
    )
    public = public_live_configuration(settings)
    rendered = json.dumps(public, ensure_ascii=False)

    assert public == {
        "tushare_configured": True,
        "deepseek_configured": True,
        "deepseek_model": "deepseek-v4-flash",
    }
    assert tushare_marker not in rendered
    assert deepseek_marker not in rendered


def test_coursework_documents_follow_the_frozen_submission_contract() -> None:
    assignment = (PROJECT_ROOT / "docs/assignment/final-project.md").read_text(encoding="utf-8")
    top_level = [line for line in assignment.splitlines() if line.startswith("# ")]
    assert top_level == [
        "# 1 选题名称",
        "# 2 选题背景",
        "# 3 需求说明文档",
        "# 4 技术设计文档（技术/模型选型）",
        "# 5 评测报告",
        "# 6 用户使用手册",
        "# 7 作品体验链接",
    ]
    for required in (
        "LangGraph",
        "轻量 Python 编排",
        "全自主工具调用 Agent",
        "短期记忆",
        "长期记忆",
        "```mermaid",
        "2000 积分",
        "访客无需输入",
        "仅供学习研究，不构成投资建议",
    ):
        assert required in assignment

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        ".venv",
        "TUSHARE_TOKEN",
        "DEEPSEEK_API_KEY",
        "数据最后交易日",
        "scripts/evaluate.py",
        "docs/architecture.md",
        "Streamlit Community Cloud",
        "2000 积分",
        "仅供学习研究，不构成投资建议",
    ):
        assert required in readme

    for relative in (
        "docs/architecture.md",
        "docs/evaluation-report.md",
        "docs/user-guide.md",
    ):
        assert (Path(PROJECT_ROOT) / relative).is_file()
