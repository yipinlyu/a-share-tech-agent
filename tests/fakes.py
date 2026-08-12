from __future__ import annotations

from copy import deepcopy

import pandas as pd


class FakeProApi:
    """Small exact-call fake for the three Tushare endpoints used by the adapter."""

    def __init__(
        self,
        *,
        stock_basic_frames: dict[str, pd.DataFrame] | None = None,
        daily_frame: pd.DataFrame | None = None,
        adj_factor_frame: pd.DataFrame | None = None,
        exceptions: dict[str, BaseException] | None = None,
    ) -> None:
        self.stock_basic_frames = deepcopy(stock_basic_frames or {})
        self.daily_frame = deepcopy(daily_frame) if daily_frame is not None else pd.DataFrame()
        self.adj_factor_frame = (
            deepcopy(adj_factor_frame) if adj_factor_frame is not None else pd.DataFrame()
        )
        self.exceptions = dict(exceptions or {})
        self.calls: list[tuple[str, dict[str, object]]] = []

    def stock_basic(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("stock_basic", deepcopy(kwargs)))
        self._raise_if_configured("stock_basic")
        exchange = str(kwargs["exchange"])
        return deepcopy(self.stock_basic_frames.get(exchange, pd.DataFrame()))

    def daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("daily", deepcopy(kwargs)))
        self._raise_if_configured("daily")
        return deepcopy(self.daily_frame)

    def adj_factor(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("adj_factor", deepcopy(kwargs)))
        self._raise_if_configured("adj_factor")
        return deepcopy(self.adj_factor_frame)

    def add_future_factor(self, trade_date: str, adj_factor: float) -> None:
        row = pd.DataFrame(
            [
                {
                    "ts_code": "600519.SH",
                    "trade_date": trade_date.replace("-", ""),
                    "adj_factor": adj_factor,
                }
            ]
        )
        self.adj_factor_frame = pd.concat([self.adj_factor_frame, row], ignore_index=True)

    def _raise_if_configured(self, endpoint: str) -> None:
        error = self.exceptions.get(endpoint)
        if error is not None:
            raise error


FakePro = FakeProApi
