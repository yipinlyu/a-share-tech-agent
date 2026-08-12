from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from stock_agent.ui.charts import (
    create_atr_obv_chart,
    create_kline_chart,
    create_macd_chart,
    create_rsi_kdj_chart,
)


def chart_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [10.0, 10.4, 10.2],
            "high": [10.8, 10.7, 10.6],
            "low": [9.8, 10.1, 9.9],
            "close": [10.5, 10.2, 10.4],
            "ma5": [10.1, 10.2, 10.3],
            "boll_upper": [11.0, 11.0, 11.0],
            "boll_mid": [10.0, 10.0, 10.0],
            "boll_lower": [9.0, 9.0, 9.0],
            "macd": [0.1, 0.08, 0.12],
            "macd_signal": [0.05, 0.07, 0.09],
            "macd_hist": [0.05, -0.01, 0.03],
            "rsi14": [48.0, 52.0, 56.0],
            "kdj_k": [44.0, 51.0, 59.0],
            "kdj_d": [46.0, 49.0, 54.0],
            "kdj_j": [40.0, 55.0, 69.0],
            "atr14": [0.5, 0.55, 0.52],
            "obv": [0.0, -900.0, 200.0],
            "vol": [1000.0, 900.0, 1100.0],
        }
    )


def test_four_chart_factories_return_accessible_plotly_figures() -> None:
    frame = chart_frame()

    figures = (
        create_kline_chart(frame, title="测试股价"),
        create_macd_chart(frame),
        create_rsi_kdj_chart(frame),
        create_atr_obv_chart(frame),
    )

    assert all(isinstance(figure, go.Figure) for figure in figures)
    assert "元" in str(figures[0].layout.yaxis.title.text)
    assert any("正" in str(trace.name) for trace in figures[1].data)
    assert "RSI" in str(figures[2].layout.title.text)
    assert "ATR" in str(figures[3].layout.title.text)


def test_chart_factories_are_safe_for_empty_or_missing_data() -> None:
    for factory in (
        create_kline_chart,
        create_macd_chart,
        create_rsi_kdj_chart,
        create_atr_obv_chart,
    ):
        figure = factory(pd.DataFrame())
        assert isinstance(figure, go.Figure)
        assert any("暂无可展示数据" in annotation.text for annotation in figure.layout.annotations)
