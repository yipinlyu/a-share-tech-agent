"""Pure, deterministic stock-master search without external calls or guessing."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Final

import pandas as pd

from stock_agent.domain.models import StockInfo, StockSearchResult

FUZZY_THRESHOLD: Final = 0.55
MAX_CANDIDATES: Final = 10
MAX_QUERY_LENGTH: Final = 30


def search_stocks(
    query: str,
    stock_master: pd.DataFrame,
    *,
    limit: int = MAX_CANDIDATES,
) -> StockSearchResult:
    """Return candidates ranked by exactness, similarity, then Tushare code."""

    normalized_query = _normalize(query)
    if not normalized_query or len(normalized_query) > MAX_QUERY_LENGTH:
        return StockSearchResult(status="not_found", candidates=[])

    candidate_limit = min(max(int(limit), 1), MAX_CANDIDATES)
    matches: list[tuple[int, float, str, StockInfo]] = []
    query_is_symbol = len(normalized_query) == 6 and normalized_query.isdecimal()

    for row in stock_master.to_dict(orient="records"):
        ts_code = _normalize(row.get("ts_code"))
        symbol = _normalize(row.get("symbol"))
        name = _normalize(row.get("name"))
        if not ts_code or not name:
            continue

        similarity = SequenceMatcher(None, normalized_query, name, autojunk=False).ratio()
        if normalized_query == ts_code:
            tier = 0
            similarity = 1.0
        elif query_is_symbol and normalized_query == symbol:
            tier = 1
            similarity = 1.0
        elif normalized_query == name:
            tier = 2
            similarity = 1.0
        elif normalized_query in name:
            tier = 3
        elif similarity >= FUZZY_THRESHOLD:
            tier = 4
        else:
            continue

        matches.append((tier, -similarity, ts_code, _stock_info(row)))

    matches.sort(key=lambda item: item[:3])
    candidates = [item[3] for item in matches[:candidate_limit]]
    if not candidates:
        status = "not_found"
    elif len(candidates) == 1:
        status = "resolved"
    else:
        status = "ambiguous"
    return StockSearchResult(status=status, candidates=candidates)


def _normalize(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    return str(value).strip().casefold()


def _optional_text(value: object) -> str | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _stock_info(row: dict[str, object]) -> StockInfo:
    return StockInfo(
        ts_code=str(row["ts_code"]),
        symbol=_optional_text(row.get("symbol")),
        name=str(row["name"]),
        market=str(row["market"]),
        industry=_optional_text(row.get("industry")),
    )
