"""Public data-adapter interfaces."""

from stock_agent.data.tushare_client import (
    DailyDataResult,
    ProLike,
    TushareAdapterError,
    TushareDataClient,
    assess_data_quality,
)

__all__ = [
    "DailyDataResult",
    "ProLike",
    "TushareAdapterError",
    "TushareDataClient",
    "assess_data_quality",
]
