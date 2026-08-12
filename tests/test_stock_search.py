from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.stock_search import search_stocks
from stock_agent.domain.models import StockInfo


@pytest.fixture
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
            {
                "ts_code": "000002.SZ",
                "symbol": "000002",
                "name": "万科A",
                "market": "主板",
                "industry": "全国地产",
            },
            {
                "ts_code": "300750.SZ",
                "symbol": "300750",
                "name": "宁德时代",
                "market": "创业板",
                "industry": "电池",
            },
        ]
    )


@pytest.mark.parametrize(
    ("query", "expected_code"),
    [
        (" 600519.sh ", "600519.SH"),
        ("600519", "600519.SH"),
        ("贵州茅台", "600519.SH"),
    ],
)
def test_exact_search_resolves_normalized_code_symbol_or_name(
    stock_master: pd.DataFrame,
    query: str,
    expected_code: str,
) -> None:
    result = search_stocks(query, stock_master)

    assert result.status == "resolved"
    assert result.candidates == [
        StockInfo(
            ts_code=expected_code,
            name="贵州茅台",
            market="主板",
            industry="白酒",
        )
    ]


def test_ascii_name_matching_is_case_insensitive(stock_master: pd.DataFrame) -> None:
    result = search_stocks(" 万科a ", stock_master)

    assert result.status == "resolved"
    assert result.candidates[0].ts_code == "000002.SZ"


def test_name_substring_returns_ranked_candidates(stock_master: pd.DataFrame) -> None:
    result = search_stocks("平安", stock_master)

    assert result.status == "ambiguous"
    assert [candidate.ts_code for candidate in result.candidates] == [
        "000001.SZ",
        "601318.SH",
    ]


def test_minor_name_typo_uses_sequence_matcher(stock_master: pd.DataFrame) -> None:
    result = search_stocks("贵州矛台", stock_master)

    assert result.status == "resolved"
    assert result.candidates[0].ts_code == "600519.SH"


@pytest.mark.parametrize("query", ["", "   ", "x" * 31, "火星公司"])
def test_empty_overlong_or_unmatched_query_is_not_found(
    stock_master: pd.DataFrame,
    query: str,
) -> None:
    result = search_stocks(query, stock_master)

    assert result.status == "not_found"
    assert result.candidates == []
    assert result.error is None


def test_match_tier_precedes_similarity_and_ties_break_by_ts_code() -> None:
    stock_master = pd.DataFrame(
        [
            {
                "ts_code": "600003.SH",
                "symbol": "600003",
                "name": "丙测试",
                "market": "主板",
                "industry": None,
            },
            {
                "ts_code": "600002.SH",
                "symbol": "600002",
                "name": "乙测试",
                "market": "主板",
                "industry": None,
            },
            {
                "ts_code": "600001.SH",
                "symbol": "600001",
                "name": "测试",
                "market": "主板",
                "industry": None,
            },
        ]
    )

    result = search_stocks("测试", stock_master)

    assert result.status == "ambiguous"
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600001.SH",
        "600002.SH",
        "600003.SH",
    ]


def test_top_five_limit_is_stable() -> None:
    stock_master = pd.DataFrame(
        [
            {
                "ts_code": f"60000{code}.SH",
                "symbol": f"60000{code}",
                "name": f"{label}银行",
                "market": "主板",
                "industry": "银行",
            }
            for code, label in zip(range(7), "甲乙丙丁戊己庚", strict=True)
        ]
    )

    result = search_stocks("银行", stock_master, limit=5)

    assert result.status == "ambiguous"
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600000.SH",
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
    ]
