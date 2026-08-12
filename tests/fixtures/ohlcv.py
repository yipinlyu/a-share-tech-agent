from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(close: list[float] | np.ndarray) -> pd.DataFrame:
    """Build a deterministic daily frame around an explicit close path."""

    close_values = np.asarray(close, dtype=float)
    size = len(close_values)
    positions = np.arange(size, dtype=float)
    open_values = close_values + np.sin(positions / 3.0) * 0.4
    high = np.maximum(open_values, close_values) + 1.0 + (positions % 3) * 0.05
    low = np.minimum(open_values, close_values) - 1.0 - (positions % 2) * 0.05
    pre_close = np.r_[close_values[0], close_values[:-1]] if size else close_values.copy()
    change = close_values - pre_close
    pct_chg = np.divide(
        change * 100.0,
        pre_close,
        out=np.zeros_like(change),
        where=pre_close != 0,
    )
    vol = 1_000.0 + positions * 3.0 + (positions % 17) * 37.0

    return pd.DataFrame(
        {
            "ts_code": pd.Series(["600519.SH"] * size, dtype="string"),
            "trade_date": pd.date_range("2023-01-02", periods=size, freq="B"),
            "open": open_values,
            "high": high,
            "low": low,
            "close": close_values,
            "pre_close": pre_close,
            "change": change,
            "pct_chg": pct_chg,
            "vol": vol,
            "amount": vol * close_values,
        }
    )


@pytest.fixture(name="ohlcv_160")
def ohlcv_160() -> pd.DataFrame:
    positions = np.arange(160, dtype=float)
    close = 100.0 + positions * 0.35 + np.sin(positions / 5.0) * 4.0
    return make_ohlcv(close)


@pytest.fixture(name="flat_then_move")
def flat_then_move() -> pd.DataFrame:
    frame = make_ohlcv([10.0] * 8 + [12.0, 11.0, 13.0, 12.5])
    frame.loc[:7, ["open", "high", "low", "close"]] = 10.0
    frame.loc[8, ["open", "high", "low", "close"]] = [11.0, 13.0, 9.0, 12.0]
    return frame
