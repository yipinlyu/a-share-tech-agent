"""A 股技术分析智能体的 Streamlit 单页工作台。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from stock_agent.agent.graph import GraphDependencies
from stock_agent.config import Settings
from stock_agent.data.tushare_client import TushareAdapterError, TushareDataClient
from stock_agent.domain.models import (
    AIRequest,
    AnalysisRequest,
    AnalysisResult,
    ChatRequest,
    StockInfo,
)
from stock_agent.llm.deepseek_client import ChatMemory, DeepSeekClient
from stock_agent.memory.repository import MemoryRepositoryError, SQLiteMemory
from stock_agent.services.analysis_service import AnalysisService
from stock_agent.ui.charts import (
    create_atr_obv_chart,
    create_kline_chart,
    create_macd_chart,
    create_rsi_kdj_chart,
)
from stock_agent.ui.components import (
    apply_theme,
    candidate_label,
    compact_stock,
    render_ai_interpretation,
    render_config_status,
    render_disclaimer,
    render_score_panel,
    render_summary,
    render_trace,
)

st.set_page_config(
    page_title="格物·A股技术研究台",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATE_DEFAULTS: dict[str, Any] = {
    "candidates": [],
    "selected_code": None,
    "current_analysis": None,
    "ai_interpretation": None,
    "recent_searches": [],
    "watchlist": [],
    "chat_memory": None,
    "last_analysis_id": None,
    "search_message": None,
}
LOOKBACK_LABELS = {3: "3 个月", 6: "6 个月", 12: "12 个月", 24: "24 个月", 36: "36 个月"}


def _secrets() -> dict[str, object]:
    try:
        return {key: value for key, value in st.secrets.items()}
    except (FileNotFoundError, RuntimeError):
        return {}


@st.cache_resource(show_spinner=False)
def load_settings() -> Settings:
    """Cache only deployment configuration, never user/session personalization."""

    return Settings.from_sources(_secrets(), os.environ)


@st.cache_resource(show_spinner=False)
def build_service(settings: Settings) -> AnalysisService | None:
    """Build production Tushare/SQLite/DeepSeek adapters for process lifetime."""

    if not settings.data_enabled:
        return None
    token = settings.tushare_token.get_secret_value() if settings.tushare_token else None
    market = TushareDataClient(token=token)
    try:
        repository: SQLiteMemory | None = SQLiteMemory(Path(".cache") / "stock_agent.sqlite3")
    except MemoryRepositoryError:
        repository = None
    ai_client = None
    if settings.deepseek_api_key is not None:
        ai_client = DeepSeekClient(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            cache=repository,
        )
    placeholder = StockInfo(ts_code="600000.SH", name="待选择", market="未知")
    dependencies = GraphDependencies(
        market_data=market,
        repository=repository,
        stock=placeholder,
    )
    return AnalysisService(
        market_data=market,
        repository=repository,
        graph_dependencies=dependencies,
        ai_client=ai_client,
    )


def _init_state() -> None:
    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default.copy() if isinstance(default, list) else default
    if not isinstance(st.session_state.chat_memory, ChatMemory):
        st.session_state.chat_memory = ChatMemory()


def _clear_session() -> None:
    for key, default in STATE_DEFAULTS.items():
        st.session_state[key] = default.copy() if isinstance(default, list) else default
    st.session_state.chat_memory = ChatMemory()


def _stock_dict(stock: StockInfo) -> dict[str, object]:
    return {
        "ts_code": stock.ts_code,
        "name": stock.name,
        "market": stock.market,
        "industry": stock.industry,
    }


def _remember(stock: StockInfo) -> None:
    items = [
        item for item in st.session_state.recent_searches if item.get("ts_code") != stock.ts_code
    ]
    st.session_state.recent_searches = [_stock_dict(stock), *items][:20]


def _toggle_watch(stock: StockInfo) -> None:
    items = list(st.session_state.watchlist)
    existing = next(
        (index for index, item in enumerate(items) if item.get("ts_code") == stock.ts_code), None
    )
    if existing is None:
        items.append(_stock_dict(stock))
    else:
        items.pop(existing)
    st.session_state.watchlist = items


def _reset_for_analysis(result: AnalysisResult, service: AnalysisService) -> None:
    if result.status != "success" or result.analysis_id is None:
        return
    if result.analysis_id != st.session_state.last_analysis_id:
        st.session_state.ai_interpretation = None
        service.activate_analysis(result, st.session_state.chat_memory)
        st.session_state.last_analysis_id = result.analysis_id


def _selected_candidate() -> StockInfo | None:
    selected = st.session_state.selected_code
    return next(
        (candidate for candidate in st.session_state.candidates if candidate.ts_code == selected),
        None,
    )


def _show_sidebar(settings: Settings, service: AnalysisService | None) -> tuple[int, object]:
    with st.sidebar:
        st.markdown('<div class="eyebrow">RESEARCH CONSOLE / 01</div>', unsafe_allow_html=True)
        st.markdown("## 标的与区间")
        query = st.text_input("股票搜索", placeholder="600519、贵州茅台或名称片段", max_chars=30)
        search_clicked = st.button(
            "搜索候选",
            key="search_stock",
            use_container_width=True,
            disabled=service is None,
        )
        if search_clicked:
            if not query.strip():
                st.session_state.search_message = "请输入股票代码或中文名称。"
                st.session_state.candidates = []
            else:
                assert service is not None
                with st.spinner("读取 A 股上市主数据…"):
                    result = service.search_stocks(query)
                if result.status in ("resolved", "ambiguous"):
                    st.session_state.candidates = result.candidates
                    st.session_state.selected_code = result.candidates[0].ts_code
                    st.session_state.search_message = (
                        "已唯一匹配。"
                        if result.status == "resolved"
                        else "找到多个候选，请明确选择。"
                    )
                elif result.status == "not_found":
                    st.session_state.candidates = []
                    st.session_state.search_message = (
                        "未找到匹配股票，请尝试 6 位代码或更完整名称。"
                    )
                else:
                    st.session_state.candidates = []
                    st.session_state.search_message = (
                        result.error.user_message if result.error else "股票搜索暂时失败。"
                    )
        if st.session_state.search_message:
            st.caption(st.session_state.search_message)
        if st.session_state.candidates:
            labels = {
                candidate.ts_code: candidate_label(candidate)
                for candidate in st.session_state.candidates
            }
            st.selectbox(
                "明确选择候选股",
                options=list(labels),
                format_func=labels.__getitem__,
                key="selected_code",
            )
        lookback = st.radio(
            "回看周期",
            options=list(LOOKBACK_LABELS),
            format_func=LOOKBACK_LABELS.__getitem__,
            index=2,
            horizontal=True,
        )
        requested_end_date = st.date_input(
            "请求截止日", value=None, help="留空则以今日为请求截止日。"
        )
        selected = _selected_candidate()
        analyze_clicked = st.button(
            "运行规则分析",
            key="run_analysis",
            type="primary",
            use_container_width=True,
            disabled=service is None or selected is None,
        )
        if analyze_clicked and selected is not None and service is not None:
            try:
                request = AnalysisRequest(
                    ts_code=selected.ts_code,
                    lookback_months=lookback,
                    requested_end_date=requested_end_date,
                )
                with st.spinner("正在拉取前复权日线并执行规则图…"):
                    analysis = service.run_analysis(request)
                st.session_state.current_analysis = analysis
                if analysis.status == "success":
                    _remember(selected)
                    _reset_for_analysis(analysis, service)
            except ValidationError:
                st.session_state.search_message = "日期或周期参数无效。"
        selected = _selected_candidate()
        if selected is not None:
            followed = any(
                item.get("ts_code") == selected.ts_code for item in st.session_state.watchlist
            )
            if st.button("移出自选" if followed else "加入本会话自选", use_container_width=True):
                _toggle_watch(selected)
                st.rerun()
        st.divider()
        render_config_status(settings)
        st.markdown("#### 最近查询 · 最多 20")
        if st.session_state.recent_searches:
            for item in st.session_state.recent_searches:
                st.caption(compact_stock(item))
        else:
            st.caption("当前会话尚无记录")
        st.markdown("#### 本会话自选")
        if st.session_state.watchlist:
            for item in st.session_state.watchlist:
                st.caption("★ " + compact_stock(item))
        else:
            st.caption("尚未收藏")
        if st.button("清除本会话记忆", key="clear_session", use_container_width=True):
            _clear_session()
            st.rerun()
    return lookback, requested_end_date


def _render_analysis(result: AnalysisResult) -> None:
    if result.status == "error":
        st.error(result.error.user_message if result.error else "分析未能安全完成。")
        return
    if result.status == "insufficient_data":
        date_text = result.period.last_trade_date.isoformat() if result.period else "未解析"
        st.warning(f"截至 {date_text} 的有效交易日不足，不强行生成信号。")
        for warning in result.warnings:
            st.caption(f"• {warning}")
        return
    render_summary(result)
    for warning in result.warnings:
        st.warning(warning)
    st.plotly_chart(
        create_kline_chart(result.series, title=f"{result.stock.name} · 价格结构"),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )
    macd_tab, rsi_tab, risk_tab = st.tabs(["MACD", "RSI / KDJ", "ATR / OBV 与量能"])
    with macd_tab:
        st.plotly_chart(
            create_macd_chart(result.series),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with rsi_tab:
        st.plotly_chart(
            create_rsi_kdj_chart(result.series),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with risk_tab:
        st.plotly_chart(
            create_atr_obv_chart(result.series),
            use_container_width=True,
            config={"displaylogo": False},
        )
    render_score_panel(result)
    with st.expander("查看 Agent 执行轨迹"):
        render_trace(result)


def _render_ai(
    service: AnalysisService | None, settings: Settings, analysis: AnalysisResult | None
) -> None:
    st.markdown("## AI 解读 · 按需产生")
    can_ai = (
        service is not None
        and settings.ai_enabled
        and isinstance(analysis, AnalysisResult)
        and analysis.status == "success"
        and analysis.analysis_id is not None
    )
    if st.button("生成AI解读", key="generate_ai", type="primary", disabled=not can_ai):
        assert service is not None and analysis is not None and analysis.analysis_id is not None
        with st.spinner("正在请求 DeepSeek 并校验结构化证据…"):
            st.session_state.ai_interpretation = service.interpret_with_ai(
                AIRequest(analysis_id=analysis.analysis_id), analysis
            )
    render_ai_interpretation(st.session_state.ai_interpretation)
    st.markdown("### 当前分析追问")
    memory: ChatMemory = st.session_state.chat_memory
    for message in memory.history:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])
    question = st.chat_input(
        "询问指标含义、信号冲突或观察位依据…",
        disabled=not can_ai,
        max_chars=500,
    )
    if question and can_ai:
        assert service is not None and analysis is not None and analysis.analysis_id is not None
        response = service.answer_followup(
            ChatRequest(
                thread_id=memory.thread_id,
                analysis_id=analysis.analysis_id,
                question=question,
            ),
            memory,
        )
        if response.error:
            st.error(response.error.user_message)
        st.rerun()
    st.caption(f"当前分析线程保留最近 {memory.turn_count} / 10 组完整问答。")


def main() -> None:
    apply_theme()
    _init_state()
    settings = load_settings()
    service: AnalysisService | None = None
    if settings.data_enabled:
        try:
            service = build_service(settings)
        except (TushareAdapterError, MemoryRepositoryError, ValueError) as exc:
            message = (
                exc.error.user_message
                if isinstance(exc, TushareAdapterError)
                else "生产服务初始化失败，请检查部署配置。"
            )
            st.session_state.search_message = message
    _show_sidebar(settings, service)
    st.markdown('<div class="eyebrow">A-SHARE / TECHNICAL EVIDENCE</div>', unsafe_allow_html=True)
    st.title("格物 · 技术研究台")
    st.markdown(
        '<p class="deck">把前复权日线、冻结指标公式和 score-v1 证据放在同一张桌面上。'
        "规则先于模型，截至日期先于结论。</p>",
        unsafe_allow_html=True,
    )
    analysis = st.session_state.current_analysis
    if isinstance(analysis, AnalysisResult):
        _render_analysis(analysis)
    else:
        st.info("从侧栏搜索并明确选择 A 股。未运行分析前，页面不会请求 DeepSeek。")
    st.divider()
    _render_ai(service, settings, analysis if isinstance(analysis, AnalysisResult) else None)
    render_disclaimer()


if __name__ == "__main__":
    main()
