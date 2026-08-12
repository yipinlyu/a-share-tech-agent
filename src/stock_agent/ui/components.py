"""Reusable Streamlit rendering components for the Chinese research workbench."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from stock_agent.config import Settings
from stock_agent.domain.models import (
    AIInterpretation,
    AgentError,
    AnalysisResult,
    INVESTMENT_DISCLAIMER,
)

GROUP_LABELS = {
    "trend": "趋势结构",
    "momentum": "动量强弱",
    "volume_volatility": "量价与波动",
}
SOURCE_LABELS = {
    "price_ma20": "价格 / MA20",
    "price_ma60": "价格 / MA60",
    "ma_alignment": "均线排列",
    "macd_line": "MACD 结构",
    "ma20_slope": "MA20 斜率",
    "rsi14": "RSI14",
    "kdj_cross": "KDJ 交叉",
    "macd_hist_momentum": "MACD 柱动能",
    "return_20d": "20 日收益",
    "obv_price_trend": "OBV / 价格",
    "volume_confirmation": "成交量确认",
    "boll_position": "布林带位置",
    "boll_breakout": "布林带突破",
}


def apply_theme() -> None:
    """Apply the restrained editorial/terminal visual system."""

    st.markdown(
        """
        <style>
        :root { --ink:#17221f; --muted:#69736e; --paper:#f7f5ed; --line:#d8d4c8;
                --brick:#b64b3c; --jade:#16756e; --gold:#b38a38; }
        .stApp { background:
          linear-gradient(90deg, rgba(23,34,31,.025) 1px, transparent 1px) 0 0/28px 28px,
          #f7f5ed; color:var(--ink); }
        [data-testid="stSidebar"] { background:#ece9df; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div { padding-top:1.4rem; }
        .block-container { max-width:1280px; padding-top:2rem; padding-bottom:6rem; }
        h1,h2,h3 { font-family:"Songti SC","STSong","Noto Serif CJK SC",serif !important;
                   letter-spacing:-.02em; color:var(--ink) !important; }
        h1 { font-size:clamp(2rem,5vw,4.4rem) !important; line-height:.96 !important; }
        .eyebrow { font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:var(--jade);
                   font-weight:700; margin-bottom:.5rem; }
        .deck { color:var(--muted); max-width:46rem; font-size:1rem; line-height:1.75; }
        .asof { display:inline-flex; border:1px solid var(--line); border-radius:999px; padding:.34rem .75rem;
                font-size:.76rem; letter-spacing:.04em; color:var(--muted); background:#fffdf7; }
        .signal-card { border-top:3px solid var(--ink); border-bottom:1px solid var(--line);
                       padding:1rem .2rem .9rem; min-height:7rem; }
        .signal-label { color:var(--muted); font-size:.74rem; letter-spacing:.08em; }
        .signal-value { font-family:"Songti SC","STSong",serif; font-size:1.7rem; margin-top:.4rem; }
        .signal-sub { color:var(--muted); font-size:.78rem; margin-top:.3rem; }
        .evidence { border-left:3px solid var(--line); padding:.25rem 0 .25rem .8rem; margin:.55rem 0; }
        .evidence.positive { border-color:var(--brick); }
        .evidence.negative { border-color:var(--jade); }
        .evidence.conflict { border-color:var(--gold); }
        .evidence small { color:var(--muted); }
        .watch-level { display:flex; justify-content:space-between; gap:1rem; border-bottom:1px dotted var(--line);
                       padding:.62rem 0; }
        .fixed-disclaimer { position:fixed; left:0; right:0; bottom:0; z-index:9999; text-align:center;
                            background:rgba(23,34,31,.96); color:#fffdf7; padding:.68rem 1rem;
                            font-size:.78rem; letter-spacing:.04em; }
        .status-dot { font-size:.82rem; margin:.35rem 0; }
        div.stButton > button { border-radius:2px; font-weight:650; min-height:2.65rem; }
        div.stButton > button[kind="primary"] { background:var(--ink); border-color:var(--ink); }
        [data-testid="stMetric"] { border-top:1px solid var(--line); padding-top:.65rem; }
        [data-testid="stPlotlyChart"] { border:1px solid rgba(216,212,200,.75); background:rgba(255,253,247,.6); }
        @media (max-width:700px) {
          .block-container { padding:1.2rem .85rem 5.5rem; }
          h1 { font-size:2.35rem !important; }
          .signal-card { min-height:auto; }
          .fixed-disclaimer { font-size:.7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="fixed-disclaimer">{INVESTMENT_DISCLAIMER} · 技术指标存在滞后与失效风险</div>',
        unsafe_allow_html=True,
    )


def render_config_status(settings: Settings) -> None:
    """Render capabilities only; never render credential values."""

    st.caption("运行配置")
    if settings.data_enabled:
        st.markdown('<div class="status-dot">● Tushare 数据已连接</div>', unsafe_allow_html=True)
    else:
        st.warning("未配置 Tushare Token，股票搜索与分析已停用。")
    if settings.ai_enabled:
        st.markdown(
            f'<div class="status-dot">● DeepSeek 已连接 · {settings.deepseek_model}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("未配置 DeepSeek，量化分析仍可使用；AI 解读与追问已停用。")


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _signal_mark(signal: str) -> str:
    if signal in ("偏多", "中性偏多"):
        return "▲"
    if signal in ("偏空", "中性偏空"):
        return "▼"
    return "◆"


def render_summary(result: AnalysisResult) -> None:
    """Render date-anchored headline facts without colour-only semantics."""

    snapshot = result.snapshot or {}
    score = result.score
    period = result.period
    last_date = period.last_trade_date.isoformat() if period else "—"
    st.markdown(f'<span class="asof">数据截至 {last_date} · 前复权</span>', unsafe_allow_html=True)
    st.subheader(f"{result.stock.name}  /  {result.stock.ts_code}")
    close = _finite(snapshot.get("close"))
    change = _finite(snapshot.get("pct_chg"))
    signal = score.signal if score else "数据不足"
    cells = (
        ("最新收盘", f"{close:,.2f} 元" if close is not None else "—", "最近有效交易日"),
        (
            "当日涨跌",
            f"{'▲' if (change or 0) >= 0 else '▼'} {change:+.2f}%" if change is not None else "—",
            "符号 + 数字双重表达",
        ),
        (
            "规则信号",
            f"{_signal_mark(signal)} {signal}",
            f"score-v1 · {score.total:+.1f}" if score else "—",
        ),
        (
            "风险等级",
            f"◆ {score.risk_level}" if score else "—",
            f"风险分 {score.risk_score}" if score else "—",
        ),
    )
    columns = st.columns(4)
    for column, (label, value, subtitle) in zip(columns, cells, strict=True):
        column.markdown(
            f'<div class="signal-card"><div class="signal-label">{label}</div>'
            f'<div class="signal-value">{value}</div><div class="signal-sub">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )


def _evidence_block(items: list[Any], kind: str, empty_text: str) -> None:
    if not items:
        st.caption(empty_text)
        return
    for item in items:
        label = SOURCE_LABELS.get(item.source_key, item.source_key)
        st.markdown(
            f'<div class="evidence {kind}"><b>{item.interpretation}</b><br>'
            f"<small>{label} · 观测值 {item.observed_value:,.4g}</small></div>",
            unsafe_allow_html=True,
        )


def render_score_panel(result: AnalysisResult) -> None:
    score = result.score
    if score is None:
        st.info("当前指标覆盖不足，未生成规则评分。")
        return
    st.markdown("### 规则评分与证据")
    score_columns = st.columns(3)
    for column, key in zip(score_columns, GROUP_LABELS, strict=True):
        value = score.group_scores.get(key)  # type: ignore[arg-type]
        column.metric(GROUP_LABELS[key], "未覆盖" if value is None else f"{value:+.1f}")
    st.caption(
        f"数据完整度 {score.completeness:.0%} · 信号一致度 {score.consistency:.0%} · "
        "分值是规则聚合，不是收益概率。"
    )
    positive, negative, conflicts = st.columns(3)
    with positive:
        st.markdown("#### ▲ 偏多证据")
        _evidence_block(score.positive_evidence, "positive", "暂无生效偏多规则")
    with negative:
        st.markdown("#### ▼ 偏空证据")
        _evidence_block(score.negative_evidence, "negative", "暂无生效偏空规则")
    with conflicts:
        st.markdown("#### ◆ 冲突 / 风险")
        _evidence_block(score.conflict_evidence, "conflict", "暂无规则冲突")
        for risk in score.risks:
            st.caption(f"• {risk.description}")
    st.markdown("#### 观察位（非目标价）")
    if not score.watch_levels:
        st.caption("暂无可用观察位")
    for level in score.watch_levels:
        st.markdown(
            f'<div class="watch-level"><span><b>{level.label}</b><br><small>{level.rationale}</small></span>'
            f"<span>{level.price:,.2f} 元</span></div>",
            unsafe_allow_html=True,
        )


def render_trace(result: AnalysisResult) -> None:
    if not result.plan_trace:
        return
    labels = {
        "plan_analysis": "规划",
        "fetch_market_data": "行情工具",
        "validate_data": "质量校验",
        "compute_indicators": "指标计算",
        "score_signals": "规则评分",
        "build_result": "结果封装",
        "write_memory": "匿名记忆",
    }
    rows = [
        {
            "步骤": labels.get(str(entry.get("node")), str(entry.get("node"))),
            "状态": entry.get("status"),
            "耗时（毫秒）": entry.get("elapsed_ms"),
        }
        for entry in result.plan_trace
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_ai_interpretation(interpretation: AIInterpretation | AgentError | None) -> None:
    if interpretation is None:
        st.caption("AI 不会随页面刷新自动调用。点击按钮后才生成一次结构化解读。")
        return
    if isinstance(interpretation, AgentError):
        st.error(interpretation.user_message)
        return
    if interpretation.consistency_status == "mismatch":
        st.warning("AI 复述与规则信号不一致；页面仍以规则信号为准。")
    cache_label = "缓存命中" if interpretation.cache_hit else "本次生成"
    st.markdown(f"#### {_signal_mark(interpretation.rule_signal)} {interpretation.rule_signal}")
    st.write(interpretation.summary)
    st.caption(
        f"{interpretation.model} · {cache_label} · {interpretation.generated_at:%Y-%m-%d %H:%M}"
    )
    for evidence in interpretation.evidence:
        st.markdown(
            f"- **{SOURCE_LABELS.get(evidence.source_key, evidence.source_key)}**：{evidence.interpretation}"
        )
    with st.expander("AI 风险与观察位"):
        for risk in interpretation.risks:
            st.markdown(f"- {risk.description}")
        for level in interpretation.watch_levels:
            st.markdown(f"- {level.label} `{level.price:,.2f} 元`：{level.rationale}")


def candidate_label(candidate: Any) -> str:
    industry = f" · {candidate.industry}" if candidate.industry else ""
    return f"{candidate.name} · {candidate.ts_code} · {candidate.market}{industry}"


def compact_stock(item: Mapping[str, object]) -> str:
    return f"{item.get('name', '未知')} · {item.get('ts_code', '')}"


__all__ = [
    "apply_theme",
    "candidate_label",
    "compact_stock",
    "render_ai_interpretation",
    "render_config_status",
    "render_disclaimer",
    "render_score_panel",
    "render_summary",
    "render_trace",
]
