from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype

from fakes import FakeProApi
from stock_agent.data.tushare_client import (
    ADJ_FACTOR_FIELDS,
    DAILY_FIELDS,
    STOCK_BASIC_FIELDS,
    TushareAdapterError,
    TushareDataClient,
)


def daily_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "600519.SH",
                "trade_date": "20240103",
                "open": 11.0,
                "high": 13.0,
                "low": 10.0,
                "close": 12.0,
                "pre_close": 11.0,
                "change": 1.0,
                "pct_chg": 9.0909,
                "vol": 300.0,
                "amount": 3600.0,
            },
            {
                "ts_code": "600519.SH",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 11.0,
                "pre_close": 10.0,
                "change": 1.0,
                "pct_chg": 10.0,
                "vol": 200.0,
                "amount": 2200.0,
            },
            {
                "ts_code": "600519.SH",
                "trade_date": "20240101",
                "open": 9.0,
                "high": 11.0,
                "low": 8.0,
                "close": 10.0,
                "pre_close": 9.0,
                "change": 1.0,
                "pct_chg": 11.1111,
                "vol": 100.0,
                "amount": 1000.0,
            },
        ]
    )


def factor_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ts_code": "600519.SH", "trade_date": "20240103", "adj_factor": 2.0},
            {"ts_code": "600519.SH", "trade_date": "20240102", "adj_factor": 1.0},
            {"ts_code": "600519.SH", "trade_date": "20240101", "adj_factor": 1.0},
        ]
    )


def fake_daily_client(
    *,
    daily: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    exceptions: dict[str, BaseException] | None = None,
) -> tuple[TushareDataClient, FakeProApi]:
    pro = FakeProApi(
        daily_frame=daily if daily is not None else daily_rows(),
        adj_factor_frame=factors if factors is not None else factor_rows(),
        exceptions=exceptions,
    )
    return TushareDataClient(pro=pro), pro


def stock_row(code: str, exchange: str) -> dict[str, object]:
    return {
        "ts_code": code,
        "symbol": code[:6],
        "name": {"SSE": "贵州茅台", "SZSE": "平安银行", "BSE": "北交测试"}[exchange],
        "area": "测试地区",
        "industry": "测试行业",
        "market": "主板" if exchange != "BSE" else "北交所",
        "exchange": exchange,
        "list_status": "L",
        "list_date": "20010101",
    }


def stock_frames() -> dict[str, pd.DataFrame]:
    return {
        "SSE": pd.DataFrame([stock_row("600519.SH", "SSE")]),
        "SZSE": pd.DataFrame([stock_row("000001.SZ", "SZSE")]),
        "BSE": pd.DataFrame([stock_row("430047.BJ", "BSE")]),
    }


def test_stock_master_calls_each_exchange_once_with_exact_fields() -> None:
    pro = FakeProApi(stock_basic_frames=stock_frames())

    result = TushareDataClient(pro=pro).fetch_stock_master()

    assert pro.calls == [
        (
            "stock_basic",
            {"exchange": exchange, "list_status": "L", "fields": STOCK_BASIC_FIELDS},
        )
        for exchange in ("SSE", "SZSE", "BSE")
    ]
    assert result["ts_code"].tolist() == ["000001.SZ", "430047.BJ", "600519.SH"]
    assert is_string_dtype(result["ts_code"])
    assert is_datetime64_any_dtype(result["list_date"])


def test_stock_master_deduplicates_identical_rows() -> None:
    frames = stock_frames()
    frames["SSE"] = pd.concat([frames["SSE"], frames["SSE"]], ignore_index=True)

    result = TushareDataClient(pro=FakeProApi(stock_basic_frames=frames)).fetch_stock_master()

    assert len(result) == 3


def test_stock_master_rejects_conflicting_duplicate_codes() -> None:
    frames = stock_frames()
    conflict = frames["SSE"].copy()
    conflict.loc[0, "name"] = "冲突名称"
    frames["SSE"] = pd.concat([frames["SSE"], conflict], ignore_index=True)

    with pytest.raises(TushareAdapterError) as caught:
        TushareDataClient(pro=FakeProApi(stock_basic_frames=frames)).fetch_stock_master()

    assert caught.value.error.code == "DATA"
    assert caught.value.error.retryable is False


def test_stock_master_rejects_a_response_at_the_server_row_limit() -> None:
    frames = stock_frames()
    frames["SSE"] = pd.concat([frames["SSE"]] * 6000, ignore_index=True)

    with pytest.raises(TushareAdapterError, match="截断"):
        TushareDataClient(pro=FakeProApi(stock_basic_frames=frames)).fetch_stock_master()


def test_stock_master_rejects_missing_fields() -> None:
    frames = stock_frames()
    frames["BSE"] = frames["BSE"].drop(columns="industry")

    with pytest.raises(TushareAdapterError) as caught:
        TushareDataClient(pro=FakeProApi(stock_basic_frames=frames)).fetch_stock_master()

    assert caught.value.error.code == "DATA"


def test_daily_and_factor_calls_use_inclusive_yyyymmdd_and_exact_fields() -> None:
    client, pro = fake_daily_client()

    client.fetch_daily("600519.SH", date(2024, 1, 1), date(2024, 1, 3))

    assert pro.calls == [
        (
            "daily",
            {
                "ts_code": "600519.SH",
                "start_date": "20240101",
                "end_date": "20240103",
                "fields": DAILY_FIELDS,
            },
        ),
        (
            "adj_factor",
            {
                "ts_code": "600519.SH",
                "start_date": "20240101",
                "end_date": "20240103",
                "fields": ADJ_FACTOR_FIELDS,
            },
        ),
    ]


def test_qfq_uses_last_actual_trade_on_or_before_requested_end_as_anchor() -> None:
    client, _ = fake_daily_client()

    result = client.fetch_daily("600519.SH", "20240101", "20240105")

    assert result["trade_date"].is_monotonic_increasing
    assert result["trade_date"].dt.strftime("%Y%m%d").tolist() == [
        "20240101",
        "20240102",
        "20240103",
    ]
    assert result.iloc[0]["close"] == pytest.approx(5.0)
    assert result.iloc[-1]["close"] == pytest.approx(12.0)
    assert result.iloc[0]["pre_close"] == pytest.approx(4.5)
    assert result.iloc[0]["change"] == pytest.approx(0.5)
    assert result.iloc[0]["pct_chg"] == pytest.approx(100 * 0.5 / 4.5)
    assert result["vol"].tolist() == [100.0, 200.0, 300.0]
    assert result["amount"].tolist() == [1000.0, 2200.0, 3600.0]


def test_future_factor_cannot_change_historical_qfq() -> None:
    client, pro = fake_daily_client()
    baseline = client.fetch_daily("600519.SH", "20240101", "20240103")

    pro.add_future_factor("2024-01-10", 99.0)
    repeated = client.fetch_daily("600519.SH", "20240101", "20240103")

    pd.testing.assert_frame_equal(baseline, repeated)


def test_conflicting_future_factors_cannot_break_historical_qfq() -> None:
    client, pro = fake_daily_client()
    baseline = client.fetch_daily("600519.SH", "20240101", "20240103")
    pro.add_future_factor("2024-01-10", 99.0)
    pro.add_future_factor("2024-01-10", 100.0)

    repeated = client.fetch_daily("600519.SH", "20240101", "20240103")

    pd.testing.assert_frame_equal(baseline, repeated)


def test_identical_daily_duplicates_are_removed() -> None:
    daily = pd.concat([daily_rows(), daily_rows().iloc[[0]]], ignore_index=True)
    client, _ = fake_daily_client(daily=daily)

    result = client.fetch_daily("600519.SH", "20240101", "20240103")

    assert len(result) == 3


def test_conflicting_daily_duplicates_raise_data_error() -> None:
    conflict = daily_rows().iloc[[0]].copy()
    conflict.loc[:, "close"] = 99.0
    daily = pd.concat([daily_rows(), conflict], ignore_index=True)
    client, _ = fake_daily_client(daily=daily)

    with pytest.raises(TushareAdapterError) as caught:
        client.fetch_daily("600519.SH", "20240101", "20240103")

    assert caught.value.error.code == "DATA"
    assert caught.value.error.retryable is False


def test_conflicting_factor_duplicates_raise_data_error() -> None:
    conflict = factor_rows().iloc[[0]].copy()
    conflict.loc[:, "adj_factor"] = 99.0
    factors = pd.concat([factor_rows(), conflict], ignore_index=True)
    client, _ = fake_daily_client(factors=factors)

    with pytest.raises(TushareAdapterError) as caught:
        client.fetch_daily("600519.SH", "20240101", "20240103")

    assert caught.value.error.code == "DATA"


def test_missing_factor_raises_data_error_without_filling() -> None:
    factors = factor_rows().loc[lambda frame: frame["trade_date"] != "20240102"]
    client, _ = fake_daily_client(factors=factors)

    with pytest.raises(TushareAdapterError) as caught:
        client.fetch_daily("600519.SH", "20240101", "20240103")

    assert caught.value.error.code == "DATA"
    assert "复权因子" in caught.value.error.user_message


def test_suspension_days_are_not_synthesized() -> None:
    daily = daily_rows().loc[lambda frame: frame["trade_date"] != "20240102"]
    factors = factor_rows().loc[lambda frame: frame["trade_date"] != "20240102"]
    client, _ = fake_daily_client(daily=daily, factors=factors)

    result = client.fetch_daily("600519.SH", "20240101", "20240103")

    assert result["trade_date"].dt.strftime("%Y%m%d").tolist() == ["20240101", "20240103"]


def test_empty_daily_returns_an_empty_frame_for_quality_handling() -> None:
    client, pro = fake_daily_client(daily=pd.DataFrame(columns=DAILY_FIELDS.split(",")))

    result = client.fetch_daily("600519.SH", "20240101", "20240103")

    assert result.empty
    assert result.columns.tolist() == DAILY_FIELDS.split(",")
    assert [endpoint for endpoint, _ in pro.calls] == ["daily"]


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [("2024-01-01", "20240103"), ("20240230", "20240301"), ("20240103", "20240101")],
)
def test_invalid_request_dates_raise_validation_error(start_date: str, end_date: str) -> None:
    client, _ = fake_daily_client()

    with pytest.raises(TushareAdapterError) as caught:
        client.fetch_daily("600519.SH", start_date, end_date)

    assert caught.value.error.code == "VALIDATION"
    assert caught.value.error.retryable is False


def test_production_client_uses_pro_api_token_and_timeout_without_global_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pro = FakeProApi()
    calls: list[dict[str, object]] = []

    def pro_api(**kwargs: object) -> FakeProApi:
        calls.append(dict(kwargs))
        return pro

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"forbidden global helper: {args!r}, {kwargs!r}")

    monkeypatch.setitem(
        sys.modules,
        "tushare",
        SimpleNamespace(pro_api=pro_api, set_token=forbidden, pro_bar=forbidden),
    )

    client = TushareDataClient(token="token-example-value")

    assert calls == [{"token": "token-example-value", "timeout": 30}]
    assert client.pro is pro


def test_missing_token_is_a_nonretryable_config_error() -> None:
    with pytest.raises(TushareAdapterError) as caught:
        TushareDataClient(token="   ")

    assert caught.value.error.code == "CONFIG"
    assert caught.value.error.retryable is False
    assert "Token" in caught.value.error.user_message


@pytest.mark.parametrize(
    ("upstream_error", "code", "retryable"),
    [
        (RuntimeError("抱歉，您输入的 TOKEN 无效: token-example-value"), "AUTH", False),
        (RuntimeError("每分钟访问频率超过限制"), "RATE_LIMIT", True),
        (TimeoutError("request timeout token-example-value"), "DATA", True),
        (ConnectionError("connection reset"), "DATA", True),
        (RuntimeError("HTTP 503 server error"), "DATA", True),
    ],
)
def test_upstream_errors_map_to_safe_agent_errors(
    upstream_error: BaseException, code: str, retryable: bool
) -> None:
    client, _ = fake_daily_client(exceptions={"daily": upstream_error})

    with pytest.raises(TushareAdapterError) as caught:
        client.fetch_daily("600519.SH", "20240101", "20240103")

    assert caught.value.error.code == code
    assert caught.value.error.retryable is retryable
    rendered = f"{caught.value} {caught.value.error.model_dump_json()}"
    assert "token-example-value" not in rendered
    assert "Authorization" not in rendered


def test_permission_error_is_nonretryable_auth_and_explains_2000_points() -> None:
    client, _ = fake_daily_client(
        exceptions={
            "daily": RuntimeError(
                "没有访问该接口的权限，积分不足; Authorization: Bearer secret-example"
            )
        }
    )

    with pytest.raises(TushareAdapterError) as caught:
        client.fetch_daily("600519.SH", "20240101", "20240103")

    assert caught.value.error.code == "AUTH"
    assert caught.value.error.retryable is False
    assert "2000" in caught.value.error.user_message
    assert "secret-example" not in str(caught.value)
