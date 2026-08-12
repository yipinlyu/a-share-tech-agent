"""Pure, no-data-safe Plotly factories for the technical-analysis workbench."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK = "#17221f"
MUTED = "#66736d"
GRID = "rgba(23,34,31,.09)"
PAPER = "rgba(0,0,0,0)"
UP = "#b64b3c"
DOWN = "#16756e"
GOLD = "#b38a38"
BLUE = "#416d85"
PLUM = "#76556e"


def _frame(value: pd.DataFrame | None) -> pd.DataFrame:
    return value.copy(deep=False) if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _x(frame: pd.DataFrame) -> pd.Series:
    if "trade_date" in frame:
        return pd.to_datetime(frame["trade_date"], errors="coerce")
    return pd.Series(frame.index, index=frame.index)


def _values(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame:
        return None
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _usable(frame: pd.DataFrame, columns: Iterable[str]) -> bool:
    required = tuple(columns)
    return bool(len(frame)) and all(column in frame for column in required)


def _base_layout(figure: go.Figure, *, title: str, height: int) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 17}},
        height=height,
        margin={"l": 52, "r": 28, "t": 60, "b": 42},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font={"family": "PingFang SC, Noto Sans CJK SC, sans-serif", "color": INK},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#fffdf6", "font_color": INK, "bordercolor": "#d8d4c8"},
        legend={"orientation": "h", "y": 1.03, "x": 1, "xanchor": "right"},
        modebar={"orientation": "v"},
    )
    figure.update_xaxes(
        showgrid=False,
        rangeslider_visible=False,
        tickformat="%Y-%m-%d",
        title_text="交易日",
    )
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zerolinecolor=GRID)
    return figure


def _empty(title: str, *, height: int) -> go.Figure:
    figure = go.Figure()
    _base_layout(figure, title=title, height=height)
    figure.add_annotation(
        text="暂无可展示数据",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": MUTED},
    )
    return figure


def create_kline_chart(
    frame: pd.DataFrame | None, *, title: str = "价格结构 · 前复权日线"
) -> go.Figure:
    """Build K-line, moving-average, and Bollinger overlays with Chinese units."""

    data = _frame(frame)
    if not _usable(data, ("open", "high", "low", "close")):
        return _empty(title, height=570)
    dates = _x(data)
    hover = [
        f"{date:%Y-%m-%d}<br>开 {open_:,.2f} 元<br>高 {high:,.2f} 元"
        f"<br>低 {low:,.2f} 元<br>收 {close:,.2f} 元"
        for date, open_, high, low, close in zip(
            dates,
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            strict=False,
        )
    ]
    figure = go.Figure(
        go.Candlestick(
            x=dates,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="K线（▲涨 / ▼跌）",
            increasing={"line": {"color": UP, "width": 1.2}, "fillcolor": UP},
            decreasing={"line": {"color": DOWN, "width": 1.2}, "fillcolor": DOWN},
            hovertext=hover,
            hoverinfo="text",
        )
    )
    line_specs = (
        ("ma5", "MA5", UP, "solid"),
        ("ma10", "MA10", GOLD, "solid"),
        ("ma20", "MA20", BLUE, "solid"),
        ("ma60", "MA60", PLUM, "solid"),
        ("boll_upper", "布林上轨", MUTED, "dot"),
        ("boll_mid", "布林中轨", "#8a7560", "dash"),
        ("boll_lower", "布林下轨", MUTED, "dot"),
    )
    for column, label, color, dash in line_specs:
        values = _values(data, column)
        if values is None or not values.notna().any():
            continue
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                name=label,
                mode="lines",
                line={"color": color, "width": 1.35, "dash": dash},
                hovertemplate=f"{label} %{{y:,.2f}} 元<extra></extra>",
            )
        )
    _base_layout(figure, title=title, height=570)
    figure.update_yaxes(title_text="价格（元）")
    return figure


def create_macd_chart(frame: pd.DataFrame | None, *, title: str = "MACD · 趋势动能") -> go.Figure:
    """Build DIF/DEA lines and symbol-labelled positive/negative histogram bars."""

    data = _frame(frame)
    if not _usable(data, ("macd", "macd_signal", "macd_hist")):
        return _empty(title, height=390)
    dates = _x(data)
    histogram = _values(data, "macd_hist")
    assert histogram is not None
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=dates,
            y=histogram.where(histogram >= 0),
            name="正柱 ▲",
            marker_color=UP,
            hovertemplate="正柱 ▲ %{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=dates,
            y=histogram.where(histogram < 0),
            name="负柱 ▼",
            marker_color=DOWN,
            hovertemplate="负柱 ▼ %{y:.4f}<extra></extra>",
        )
    )
    for column, label, color in (("macd", "DIF", INK), ("macd_signal", "DEA", GOLD)):
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=_values(data, column),
                name=label,
                mode="lines",
                line={"color": color, "width": 1.7},
                hovertemplate=f"{label} %{{y:.4f}}<extra></extra>",
            )
        )
    figure.add_hline(y=0, line_width=1, line_color=GRID)
    _base_layout(figure, title=title, height=390)
    figure.update_yaxes(title_text="指数值")
    figure.update_layout(barmode="overlay")
    return figure


def create_rsi_kdj_chart(
    frame: pd.DataFrame | None, *, title: str = "RSI / KDJ · 强弱与摆动"
) -> go.Figure:
    """Build RSI and K/D/J lines with explicit 30/70 reference labels."""

    data = _frame(frame)
    if data.empty or not any(column in data for column in ("rsi14", "kdj_k", "kdj_d", "kdj_j")):
        return _empty(title, height=420)
    dates = _x(data)
    figure = go.Figure()
    for column, label, color, dash in (
        ("rsi14", "RSI14", INK, "solid"),
        ("kdj_k", "K", UP, "solid"),
        ("kdj_d", "D", GOLD, "solid"),
        ("kdj_j", "J", BLUE, "dot"),
    ):
        values = _values(data, column)
        if values is None or not values.notna().any():
            continue
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                name=label,
                mode="lines",
                line={"color": color, "width": 1.6, "dash": dash},
                hovertemplate=f"{label} %{{y:.2f}}<extra></extra>",
            )
        )
    if not figure.data:
        return _empty(title, height=420)
    figure.add_hrect(y0=70, y1=100, fillcolor="rgba(182,75,60,.05)", line_width=0)
    figure.add_hrect(y0=0, y1=30, fillcolor="rgba(22,117,110,.05)", line_width=0)
    figure.add_hline(y=70, line_dash="dot", line_color=UP, annotation_text="70 偏热观察")
    figure.add_hline(y=30, line_dash="dot", line_color=DOWN, annotation_text="30 偏弱观察")
    _base_layout(figure, title=title, height=420)
    figure.update_yaxes(title_text="指标值", range=[0, 100])
    return figure


def create_atr_obv_chart(
    frame: pd.DataFrame | None, *, title: str = "ATR / OBV / 成交量 · 风险与确认"
) -> go.Figure:
    """Build a two-row volatility and volume-confirmation figure."""

    data = _frame(frame)
    if data.empty or not any(column in data for column in ("atr14", "obv", "vol", "volume")):
        return _empty(title, height=520)
    dates = _x(data)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[[{}], [{"secondary_y": True}]],
        row_heights=[0.42, 0.58],
    )
    atr = _values(data, "atr14")
    if atr is not None and atr.notna().any():
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=atr,
                name="ATR14 波动幅度",
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(179,138,56,.12)",
                line={"color": GOLD, "width": 1.8},
                hovertemplate="ATR14 %{y:,.2f} 元<extra></extra>",
            ),
            row=1,
            col=1,
        )
    volume_key = "vol" if "vol" in data else "volume" if "volume" in data else None
    volume = _values(data, volume_key) if volume_key else None
    if volume is not None and volume.notna().any():
        close = _values(data, "close")
        open_ = _values(data, "open")
        colors = [MUTED] * len(data)
        symbols = ["·"] * len(data)
        if close is not None and open_ is not None:
            colors = [UP if c >= o else DOWN for c, o in zip(close, open_, strict=False)]
            symbols = ["▲" if c >= o else "▼" for c, o in zip(close, open_, strict=False)]
        figure.add_trace(
            go.Bar(
                x=dates,
                y=volume,
                name="成交量（▲涨 / ▼跌）",
                marker_color=colors,
                customdata=symbols,
                hovertemplate="%{customdata} 成交量 %{y:,.0f} 手<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
    obv = _values(data, "obv")
    if obv is not None and obv.notna().any():
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=obv,
                name="OBV 能量潮",
                mode="lines",
                line={"color": BLUE, "width": 1.7},
                hovertemplate="OBV %{y:,.0f} 手<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
    if not figure.data:
        return _empty(title, height=520)
    _base_layout(figure, title=title, height=520)
    figure.update_yaxes(title_text="ATR（元）", row=1, col=1)
    figure.update_yaxes(title_text="成交量（手）", row=2, col=1, secondary_y=False)
    figure.update_yaxes(title_text="OBV（手）", row=2, col=1, secondary_y=True, showgrid=False)
    figure.update_xaxes(title_text="", row=1, col=1)
    figure.update_xaxes(title_text="交易日", row=2, col=1)
    return figure


# Stable descriptive aliases for callers that prefer ``build_*`` factory names.
build_price_chart = create_kline_chart
build_kline_chart = create_kline_chart
build_macd_chart = create_macd_chart
build_rsi_kdj_chart = create_rsi_kdj_chart
build_atr_obv_chart = create_atr_obv_chart


__all__ = [
    "build_atr_obv_chart",
    "build_kline_chart",
    "build_macd_chart",
    "build_price_chart",
    "build_rsi_kdj_chart",
    "create_atr_obv_chart",
    "create_kline_chart",
    "create_macd_chart",
    "create_rsi_kdj_chart",
]
