from __future__ import annotations

from streamlit.testing.v1 import AppTest

from app import _request_ai_interpretation
from stock_agent.domain.models import AgentError


def test_app_starts_without_secrets_and_disables_external_actions() -> None:
    app = AppTest.from_file("app.py")
    app.secrets = {"TUSHARE_TOKEN": "", "DEEPSEEK_API_KEY": ""}
    app = app.run(timeout=15)

    assert not app.exception
    assert any("Tushare" in warning.value for warning in app.warning)
    assert any("不构成投资建议" in markdown.value for markdown in app.markdown)
    assert app.button(key="search_stock").disabled
    assert app.button(key="run_analysis").disabled
    assert app.button(key="generate_ai").disabled


def test_clear_session_memory_removes_personalized_state() -> None:
    app = AppTest.from_file("app.py")
    app.session_state["recent_searches"] = [{"ts_code": "600519.SH", "name": "贵州茅台"}]
    app.session_state["watchlist"] = [{"ts_code": "600519.SH", "name": "贵州茅台"}]
    app.run(timeout=15)

    app.button(key="clear_session").click().run(timeout=15)

    assert app.session_state["recent_searches"] == []
    assert app.session_state["watchlist"] == []
    assert app.session_state["current_analysis"] is None


def test_stale_ai_button_event_returns_safe_error_instead_of_asserting() -> None:
    result = _request_ai_interpretation(None, None)

    assert isinstance(result, AgentError)
    assert result.code == "VALIDATION"
    assert result.retryable is False
