# A股技术分析智能体 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建并发布一个可通过代码或中文名称模糊检索 A 股、用 Tushare 前复权日线计算可解释技术指标与规则评分、由 DeepSeek-V4-Flash 按需解读并具有明确短期/长期记忆生命周期的 Streamlit Agent。

**Architecture:** 根目录 `app.py` 只负责 Streamlit 组合，业务代码采用 `src/stock_agent` 包。Search Service 在 LangGraph 外完成候选消歧；LangGraph 以 `AnalysisRequest` 为输入，顺序执行规划、Tushare 取数、质量校验、指标、评分、结果构造和匿名 SQLite 写入，并用条件边安全停止。DeepSeek 适配器是独立的按需服务，规则信号永远由确定性引擎产生，模型只解释结构化事实。

**Tech Stack:** Python 3.11、Streamlit 1.60、Pandas 2.x、NumPy、Plotly 6.x、Tushare 1.4.29、LangGraph 1.x、OpenAI Python 2.x、Pydantic 2.x、SQLite、pytest 9、Ruff、GitHub Actions、Streamlit Community Cloud。

**Primary spec:** `docs/superpowers/specs/2026-08-12-a-share-agent-design.md`

---

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | 包元数据、运行/开发依赖、Ruff 与 pytest 配置 |
| `requirements.txt` | Community Cloud 安装当前项目的单一入口 |
| `app.py` | Streamlit 页面编排、Session State、按钮事件 |
| `src/stock_agent/config.py` | Secrets/env 配置读取与非敏感状态 |
| `src/stock_agent/domain/models.py` | Pydantic 输入输出、错误、评分和 AI 契约 |
| `src/stock_agent/data/tushare_client.py` | Tushare 适配器、qfq、异常映射、质量校验 |
| `src/stock_agent/data/stock_search.py` | 查询归一化、匹配分层、候选排序 |
| `src/stock_agent/indicators/engine.py` | 纯函数技术指标与快照 |
| `src/stock_agent/scoring/rules.py` | `score-v1`、证据、风险和观察位 |
| `src/stock_agent/memory/repository.py` | SQLite 表、TTL、匿名分析/行情/AI 缓存 |
| `src/stock_agent/llm/schemas.py` | 模型 JSON Schema、语义校验、白名单 |
| `src/stock_agent/llm/deepseek_client.py` | DeepSeek 调用、一次修复、追问 |
| `src/stock_agent/agent/state.py` | LangGraph TypedDict 状态与 trace 类型 |
| `src/stock_agent/agent/graph.py` | 确定性分析状态图与失败路由 |
| `src/stock_agent/services/analysis_service.py` | UI 面向的搜索、分析、AI 与聊天门面 |
| `src/stock_agent/ui/charts.py` | Plotly K 线与指标图纯构造函数 |
| `src/stock_agent/ui/components.py` | Streamlit 展示组件与主题 CSS |
| `tests/fixtures/ohlcv.py` | 可复现行情、复权、涨跌/横盘夹具 |
| `tests/fakes.py` | Fake Tushare、Fake OpenAI 与时钟 |
| `tests/test_*.py` | 分模块单元、契约、状态图和 UI Smoke Test |
| `docs/assignment/final-project.md` | 按提交格式组织的完整结项材料 |
| `docs/architecture.md` | 架构、记忆生命周期与数据流图 |
| `docs/evaluation-report.md` | 可复跑评测方法、结果和限制 |
| `docs/user-guide.md` | 用户使用与常见故障手册 |
| `.github/workflows/ci.yml` | Python 3.11 lint/test/secret scan/UI smoke |

## Task 1: Project skeleton, package contracts, and safe configuration

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.python-version`
- Create: `.streamlit/config.toml`
- Create: `.streamlit/secrets.toml.example`
- Create: `src/stock_agent/__init__.py`
- Create: `src/stock_agent/config.py`
- Create: `src/stock_agent/domain/__init__.py`
- Create: `src/stock_agent/domain/models.py`
- Test: `tests/test_config.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing configuration and domain-contract tests**

```python
# tests/test_config.py
from stock_agent.config import Settings


def test_settings_never_expose_secret_in_repr():
    settings = Settings(tushare_token="t-secret", deepseek_api_key="sk-secret")
    rendered = repr(settings)
    assert "t-secret" not in rendered
    assert "sk-secret" not in rendered
    assert settings.deepseek_model == "deepseek-v4-flash"


def test_settings_report_missing_capabilities():
    settings = Settings(tushare_token=None, deepseek_api_key=None)
    assert settings.data_enabled is False
    assert settings.ai_enabled is False
```

```python
# tests/test_models.py
import pytest
from pydantic import ValidationError
from stock_agent.domain.models import AnalysisRequest, ChatRequest


def test_analysis_request_accepts_only_supported_lookbacks():
    request = AnalysisRequest(ts_code="600519.SH", lookback_months=12)
    assert request.lookback_months == 12
    with pytest.raises(ValidationError):
        AnalysisRequest(ts_code="600519.SH", lookback_months=18)


def test_chat_question_cannot_be_empty():
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="4fdb57c8-508a-43f0-aa26-4a01389d567e", analysis_id="a", question="")
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_models.py -q`

Expected: collection fails because `stock_agent` does not exist.

- [ ] **Step 3: Add Python 3.11 package metadata and contracts**

Use `pyproject.toml` with `requires-python = ">=3.11,<3.12"`, runtime dependencies from the plan header, optional `dev` dependencies `pytest>=9.1,<10` and `ruff==0.15.22`, setuptools `src` discovery, Ruff line length 100, and pytest `pythonpath=["src"]`. `requirements.txt` contains only `.` so Community Cloud installs the package rather than relying on `sys.path` mutation.

Implement `Settings` as a frozen Pydantic model with secret fields declared `repr=False`, defaults `https://api.deepseek.com` and `deepseek-v4-flash`, and `from_sources(secrets, environ)` where Streamlit Secrets override environment values without logging them. Implement the spec models: `StockInfo`, `StockQuery`, `AnalysisRequest`, `AIRequest`, `ChatRequest`, `StockSearchResult`, `ChatResponse`, `AgentError`, `DataQuality`, `PeriodInfo`, `Evidence`, `WatchLevel`, `ScoreResult`, `AnalysisResult`, and `AIInterpretation`.

- [ ] **Step 4: Add only placeholder Secrets and safe Streamlit configuration**

```toml
# .streamlit/secrets.toml.example
TUSHARE_TOKEN = "replace-with-your-tushare-token"
DEEPSEEK_API_KEY = "replace-with-your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
```

Set Streamlit headless mode and disable usage statistics in `.streamlit/config.toml`. Do not create the real `secrets.toml` in this task.

- [ ] **Step 5: Run GREEN checks and commit**

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_models.py -q
.venv/bin/python -m ruff check src tests
git diff --check
```

Expected: all tests pass and both static checks exit 0.

Commit: `git commit -m "build: scaffold typed stock agent package"`

## Task 2: Tushare stock master, qfq adapter, and quality gate

**Files:**
- Create: `src/stock_agent/data/__init__.py`
- Create: `src/stock_agent/data/tushare_client.py`
- Create: `tests/fakes.py`
- Test: `tests/test_tushare_client.py`
- Test: `tests/test_data_quality.py`

- [ ] **Step 1: Write failing adapter contract tests**

Create `FakeProApi` with `stock_basic`, `daily`, and `adj_factor`; record exact kwargs, return deep copies, and allow injected exceptions.

```python
def test_qfq_uses_requested_end_date_anchor(fake_pro):
    client = TushareDataClient(pro=fake_pro)
    out = client.fetch_daily("600519.SH", date(2024, 1, 1), date(2024, 1, 3))
    assert out.trade_date.is_monotonic_increasing
    assert out.iloc[0].close == pytest.approx(5.0)
    assert out.iloc[-1].close == pytest.approx(12.0)


def test_future_factor_cannot_change_historical_qfq(fake_pro):
    baseline = TushareDataClient(fake_pro).fetch_daily(
        "600519.SH", date(2024, 1, 1), date(2024, 1, 3)
    )
    fake_pro.add_future_factor("2024-01-10", 99.0)
    repeated = TushareDataClient(fake_pro).fetch_daily(
        "600519.SH", date(2024, 1, 1), date(2024, 1, 3)
    )
    pd.testing.assert_frame_equal(baseline, repeated)
```

Also cover exact fields, SSE/SZSE/BSE master calls, identical/conflicting duplicates, missing factors, strict positive OHLC, OHLC relations, negative volume, empty daily, 59/60/119/120 prewarm boundaries, and exception mappings for missing 2000-point permission, invalid token, rate limit, timeout, and server failures.

- [ ] **Step 2: Run adapter tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_tushare_client.py tests/test_data_quality.py -q`

Expected: module import failures.

- [ ] **Step 3: Implement the injected adapter**

Define a `ProLike` protocol and construct production clients with `ts.pro_api(token=token, timeout=30)`; never call `ts.set_token`. Fetch stock master separately for `SSE`, `SZSE`, `BSE` with `list_status="L"` and explicit fields. Fetch daily and adjustment factors with explicit fields, parse dates/numerics, reject conflicting duplicates, one-to-one merge on `ts_code/trade_date`, and compute qfq with the request-as-of anchor. Do not use `pro_bar`, fill missing factors, round adjusted prices, or synthesize suspension rows.

- [ ] **Step 4: Implement quality reports and safe errors**

Return `DataQuality` with raw row count, display row count, prewarm row count, last trade date, warnings, and validity. Map missing token to `CONFIG`; invalid token to `AUTH`; permission/points to non-retryable `AUTH` with a 2000-point message; frequency messages to retryable `RATE_LIMIT`; and connection/timeout/server failures to retryable `DATA`. Sanitize all user messages/logs with one redaction helper.

- [ ] **Step 5: Run GREEN and commit**

```bash
.venv/bin/python -m pytest tests/test_tushare_client.py tests/test_data_quality.py -q
.venv/bin/python -m ruff check src tests
```

Expected: all pass.

Commit: `git commit -m "feat: add validated Tushare qfq data adapter"`

## Task 3: Stock search and deterministic indicators

**Files:**
- Create: `src/stock_agent/data/stock_search.py`
- Create: `src/stock_agent/indicators/__init__.py`
- Create: `src/stock_agent/indicators/engine.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/ohlcv.py`
- Test: `tests/test_stock_search.py`
- Test: `tests/test_indicators.py`

- [ ] **Step 1: Write failing search and golden-indicator tests**

Search tests cover `600519.SH`, `600519`, exact Chinese name, substring, fuzzy typo, empty text, stable tie breaking, and a Top-5 labeled fixture. Indicator tests independently calculate expected SMA/EMA/MACD/Bollinger/RSI/KDJ/ATR/OBV/volume ratio.

```python
def test_indicator_prefix_is_invariant_to_future_spike(ohlcv_160):
    prefix = compute_indicators(ohlcv_160.iloc[:120]).iloc[-1]
    mutated = ohlcv_160.copy()
    mutated.loc[mutated.index[120]:, ["open", "high", "low", "close"]] *= 100
    full = compute_indicators(mutated).iloc[119]
    pd.testing.assert_series_equal(prefix[INDICATOR_COLUMNS], full[INDICATOR_COLUMNS])


def test_kdj_first_valid_value_updates_from_seed_50(flat_then_move):
    out = compute_indicators(flat_then_move)
    first = out["kdj_k"].first_valid_index()
    expected_k = (2 / 3) * 50 + (1 / 3) * out.loc[first, "rsv"]
    assert out.loc[first, "kdj_k"] == pytest.approx(expected_k)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_stock_search.py tests/test_indicators.py -q`

Expected: module import failures.

- [ ] **Step 3: Implement layered search without LLM guessing**

Normalize whitespace/case and rank by exact `ts_code`, exact symbol, exact name, name substring, then `difflib.SequenceMatcher`. Return at most 10 candidates with deterministic secondary sort by code; exclude fuzzy scores below 0.55. Stock-master caching is not part of this pure module.

- [ ] **Step 4: Implement the frozen formulas**

Use the exact formula and zero-denominator behavior in the spec, including TR first day=`high-low`, Wilder RMA, explicit KDJ loop seeded at 50, and OBV beginning at 0. Replace infinities with NaN and never backfill indicators.

- [ ] **Step 5: Run GREEN and commit**

```bash
.venv/bin/python -m pytest tests/test_stock_search.py tests/test_indicators.py -q
.venv/bin/python -m ruff check src tests
```

Expected: all pass.

Commit: `git commit -m "feat: add fuzzy search and technical indicators"`

## Task 4: Explainable `score-v1` engine

**Files:**
- Create: `src/stock_agent/scoring/__init__.py`
- Create: `src/stock_agent/scoring/rules.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write failing golden scoring tests**

```python
@pytest.mark.parametrize(("value", "label"), [
    (40, "偏多"), (39.999, "中性偏多"), (15, "中性偏多"),
    (14.999, "中性"), (-14.999, "中性"), (-15, "中性偏空"),
    (-39.999, "中性偏空"), (-40, "偏空"),
])
def test_score_boundaries(value, label):
    assert label_for_score(value) == label


def test_all_evaluable_but_neutral_is_zero_not_insufficient(neutral_frame):
    score = score_signals(neutral_frame, data_warnings=[])
    assert score.total == 0
    assert score.signal == "中性"


def test_group_capacity_constants():
    assert CAPACITY == {"trend": 40, "momentum": 30, "volume_volatility": 27}
    assert MIN_EVALUABLE == {"trend": 20, "momentum": 15, "volume_volatility": 14}
```

Also isolate every predicate; test most-recent KDJ crossing, equality neutrality, ATR 4%/6%, RSI extremes, Bollinger breaks, divergence, data-only warnings, capacity normalization, less than two groups, completeness denominator 97, consistency denominator 0, allowed watch keys, the volume-group maximum `22/27*30`, and total theoretical realized maximum near `94.4444`.

- [ ] **Step 2: Run scorer tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement data-driven rule evaluation**

Each direction evaluator returns `bullish`, `bearish`, `neutral`, or `missing`, plus signed points and evidence. Keep raw score, evaluable capacity, normalized group score, evidence, and conflicts separate. Risk-only rules never increase the 97-point denominator. Signal consistency uses actually hit raw absolute points, not normalized group points.

- [ ] **Step 4: Implement observation levels and canonical IDs**

Produce only the seven supported basis keys and sort evidence deterministically. Canonical JSON for `analysis_id` uses sorted keys, UTF-8/unescaped Unicode, ISO dates, explicit null, finite floats formatted with `.17g`, and contains code, resolved date, lookback, selected OHLCV digest, `indicators-v1`, and `score-v1`.

- [ ] **Step 5: Run GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -q`

Expected: all pass.

Commit: `git commit -m "feat: add explainable technical signal scoring"`

## Task 5: SQLite memory and TTL cache

**Files:**
- Create: `src/stock_agent/memory/__init__.py`
- Create: `src/stock_agent/memory/repository.py`
- Test: `tests/test_memory_repository.py`

- [ ] **Step 1: Write failing lifecycle tests with an injected clock**

```python
def test_exact_expiry_is_stale(tmp_path, clock):
    repo = SQLiteMemory(tmp_path / "memory.db", clock=clock)
    repo.put_ai("key", {"summary": "x"}, ttl=timedelta(seconds=10))
    clock.advance(seconds=10)
    assert repo.get_ai("key") is None


def test_latest_request_on_weekend_keeps_six_hour_ttl(saturday):
    ttl = market_data_ttl(requested_end_date=None, now=saturday)
    assert ttl == timedelta(hours=6)
```

Also test schema creation, stable upsert, cleanup, corrupt JSON eviction, analysis 90 days, AI 30 days, latest market 6 hours, explicit historical 30 days, stock master 24 hours, and absence of token/key/prompt/message/user/authorization columns.

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_memory_repository.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement parameterized SQLite operations**

Create separate `analysis_cache`, `ai_cache`, `market_cache`, and `stock_master_cache` tables with `cache_key`, compact JSON payload, `created_at`, `expires_at`, and version. Use transactions, parameterized SQL, WAL when supported, and a process write lock. `now < expires_at` is the only freshness rule; corrupt or old-version rows are individually evicted.

- [ ] **Step 4: Implement cleanup and graceful degradation**

Expose `cleanup_expired()` and `close()`. Repository write exceptions become a service warning and never remove an already-built `analysis_id` or alter market risk. Recent searches, watchlist, thread state, and full chat text are never persisted here.

- [ ] **Step 5: Run GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_memory_repository.py -q`

Expected: all pass.

Commit: `git commit -m "feat: add lifecycle-managed SQLite memory"`

## Task 6: DeepSeek interpretation, validation, cache, and chat memory

**Files:**
- Create: `src/stock_agent/llm/__init__.py`
- Create: `src/stock_agent/llm/schemas.py`
- Create: `src/stock_agent/llm/deepseek_client.py`
- Test: `tests/test_llm_schemas.py`
- Test: `tests/test_deepseek_client.py`
- Test: `tests/test_chat_memory.py`

- [ ] **Step 1: Write failing schema, semantic, cache, and history tests**

Fake the OpenAI client so no network/quota is used. Assert exact model/base URL initialization, JSON mode, thinking disabled, bounded timeout/retry, valid/min/max JSON, extra fields, invalid signal, NaN/Infinity, hallucinated key, numeric/watch mismatch, one repair, repair failure, error not cached, and model-bound key.

```python
def test_ai_cache_key_changes_with_model():
    assert ai_cache_key("analysis", "deepseek-v4-flash", "prompt-v1") != ai_cache_key(
        "analysis", "deepseek-v4-pro", "prompt-v1"
    )


def test_followup_keeps_ten_complete_pairs():
    history = []
    for i in range(11):
        history = append_turn(history, f"q{i}", f"a{i}", max_pairs=10)
    assert len(history) == 20
    assert history[0]["content"] == "q1"
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_llm_schemas.py tests/test_deepseek_client.py tests/test_chat_memory.py -q`

Expected: module import failures.

- [ ] **Step 3: Implement strict models and semantic verification**

Reject extra properties. Freeze the numeric `source_key` whitelist to serialized snapshot keys: `close`, `pct_chg`, `ma5`, `ma10`, `ma20`, `ma60`, `macd`, `macd_signal`, `macd_hist`, `boll_upper`, `boll_mid`, `boll_lower`, `rsi14`, `kdj_k`, `kdj_d`, `kdj_j`, `atr14`, `atr_ratio`, `obv`, `volume_ratio`. Flatten only these top-level finite floats. Verify exact spec tolerances and the seven `basis_key` prices. Server code alone attaches the disclaimer, rule signal, consistency, model, prompt version, cache flag, and timestamp.

- [ ] **Step 4: Implement bounded DeepSeek calls and one repair**

Use `OpenAI(api_key=..., base_url=..., timeout=30, max_retries=2)` and `response_format={"type":"json_object"}`, `extra_body={"thinking":{"type":"disabled"}}`. Prohibit recalculation, invented facts, individualized trades, and position sizing. Schema/semantic failures get exactly one repair request; API transport/status failures do not. Map 401 to `AUTH`, 402 to `MODEL` with balance guidance, and 429/500/503 to retryable errors. Cache only validated responses.

- [ ] **Step 5: Implement follow-up bounds and reset semantics**

Send only the current structured analysis plus the last 10 complete pairs. Switching `analysis_id` creates a UUID thread and clears history. No current analysis or an empty question fails locally. Append a pair only after success, so no half-turn remains.

- [ ] **Step 6: Run GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_llm_schemas.py tests/test_deepseek_client.py tests/test_chat_memory.py -q`

Expected: all pass with zero external calls.

Commit: `git commit -m "feat: add grounded DeepSeek interpretation and chat"`

## Task 7: LangGraph workflow and application service

**Files:**
- Create: `src/stock_agent/agent/__init__.py`
- Create: `src/stock_agent/agent/state.py`
- Create: `src/stock_agent/agent/graph.py`
- Create: `src/stock_agent/services/__init__.py`
- Create: `src/stock_agent/services/analysis_service.py`
- Test: `tests/test_agent_graph.py`
- Test: `tests/test_analysis_service.py`

- [ ] **Step 1: Write failing success and failure-route tests**

```python
def test_persistence_failure_preserves_analysis_id(graph_deps, request):
    graph_deps.repository.fail_writes = True
    result = run_analysis_graph(request, graph_deps)
    assert result.status == "success"
    assert result.analysis_id
    assert "persistence" in " ".join(result.warnings).lower()


def test_tushare_error_stops_before_indicators(graph_deps, request):
    graph_deps.market_data.raise_error = AgentError(...)
    result = run_analysis_graph(request, graph_deps)
    assert result.status == "error"
    assert graph_deps.indicators.call_count == 0
```

Cover success, invalid request, Tushare error, 59 prewarm rows, invalid OHLCV, indicator exception, score coverage failure, build failure, and persistence warning. Assert sanitized ordered `plan_trace` fields and downstream call count zero after termination.

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_agent_graph.py tests/test_analysis_service.py -q`

Expected: module import failures.

- [ ] **Step 3: Implement state and conditional graph**

Build one synchronous graph entering at `plan_analysis`. Each node returns only a state delta. Route explicit statuses to terminal success, insufficient, or error builders. Measure elapsed time with an injected monotonic clock and never put Secrets, raw prompts, stack traces, or full model output in trace. Do not use a cross-user SQLite checkpointer.

- [ ] **Step 4: Implement the UI-facing service**

`AnalysisService` owns stock-master TTL retrieval, search, graph invocation, AI interpretation, and follow-up calls through injected adapters. It returns domain objects, not Streamlit calls. Calculate indicators on the prewarm range, then crop to the requested display range. Search/candidate resolution stays outside the graph.

- [ ] **Step 5: Run GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_agent_graph.py tests/test_analysis_service.py -q`

Expected: all pass.

Commit: `git commit -m "feat: orchestrate analysis with LangGraph"`

## Task 8: Plotly presentation and Streamlit workbench

**Files:**
- Create: `src/stock_agent/ui/__init__.py`
- Create: `src/stock_agent/ui/charts.py`
- Create: `src/stock_agent/ui/components.py`
- Create: `app.py`
- Test: `tests/test_charts.py`
- Test: `tests/test_app_smoke.py`

- [ ] **Step 1: Write failing chart and AppTest tests**

Use Plotly object assertions and `streamlit.testing.v1.AppTest`. Verify four chart factories, no-data behavior, fixed disclaimer/date, disabled analysis without Tushare, disabled AI without DeepSeek, candidate display, no automatic AI call, and session reset.

```python
def test_app_starts_without_secrets():
    app = AppTest.from_file("app.py").run(timeout=15)
    assert not app.exception
    assert any("Tushare" in alert.value for alert in app.warning)
    assert any("不构成投资建议" in md.value for md in app.markdown)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_charts.py tests/test_app_smoke.py -q`

Expected: missing modules/app.

- [ ] **Step 3: Implement accessible Plotly charts**

Create K-line + MA/Bollinger, MACD, RSI/KDJ, and ATR/OBV-volume figures. Chinese titles/hover labels include units; bullish/bearish meaning uses text/symbol plus color. Use responsive heights and disable range sliders on small panels.

- [ ] **Step 4: Implement the approved one-page layout**

Sidebar: search, period, config state, recent searches and session watchlist. Main: summary, charts, rule score/trace evidence, explicit AI button, follow-up chat and fixed disclaimer. Use `st.cache_resource` only for service construction/repository lifecycle; use `st.session_state` for current analysis, candidates, recent 20 searches, watchlist, thread ID, and 10 chat pairs. Never accept or render credentials.

- [ ] **Step 5: Visual QA and GREEN checks**

```bash
.venv/bin/python -m pytest tests/test_charts.py tests/test_app_smoke.py -q
.venv/bin/python -m streamlit run app.py --server.headless=true --server.port=8501
curl --fail http://127.0.0.1:8501/_stcore/health
```

Open `http://127.0.0.1:8501`; inspect desktop and 390×844 narrow view. Correct overflow, clipped labels, contrast, chart sizes, disabled states, and focus. Health must return `ok`.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: build Streamlit stock analysis workbench"`

## Task 9: Evaluation, coursework pack, and user documentation

**Files:**
- Create: `scripts/evaluate.py`
- Create: `tests/evaluation/search_cases.json`
- Create: `tests/evaluation/error_cases.json`
- Test: `tests/test_evaluation.py`
- Create: `docs/architecture.md`
- Create: `docs/evaluation-report.md`
- Create: `docs/user-guide.md`
- Create: `docs/assignment/final-project.md`
- Create: `README.md`

- [ ] **Step 1: Write failing evaluation-harness tests**

Assert metric denominators and disclosure fields. The offline report must include labeled search cases, reference formula version, `rtol=1e-6`, `atol=1e-8`, exception cases, AI grounding cases, cache cases, timing environment, and historical-evaluation limitations.

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_evaluation.py -q`

Expected: missing evaluation module/script.

- [ ] **Step 3: Implement deterministic offline evaluation**

Default mode uses fixtures/fakes and writes machine JSON to ignored `artifacts/`. Exit nonzero when search Top-5 <95%, data-quality detection <100%, AI grounding <100%, cache skip <100%, or any exception does not safely degrade. Include at least 100 labeled search queries, 500 static market rows over five patterns, 50 AI validation cases, and the full exception matrix. Add opt-in `--live` that reads config without printing values. Historical returns are descriptive only; no-return threshold gates CI.

- [ ] **Step 4: Write final coursework material in the requested order**

`docs/assignment/final-project.md` top-level order must be: 1 选题名称、2 选题背景、3 需求说明文档、4 技术设计文档（含技术/模型选型）、5 评测报告、6 用户使用手册、7 作品体验链接. Link Mermaid architecture/memory diagrams, compare LangGraph with two alternatives, and use clearly marked URL placeholders only until Task 11 fills them.

- [ ] **Step 5: Write README and operational user guide**

README quick start uses `.venv`, explains the Tushare 2000-point requirement, Secrets, data-date semantics, safety boundary, tests/evaluation, architecture, and deployment. User guide contains first use, deployment config, analysis, AI, follow-up, memory, troubleshooting, and risk statement.

- [ ] **Step 6: Generate results and commit**

```bash
.venv/bin/python scripts/evaluate.py
.venv/bin/python -m pytest tests/test_evaluation.py -q
```

Expected: thresholds pass and the report contains exact sample counts.

Commit: `git commit -m "docs: add evaluation and coursework package"`

## Task 10: CI, security scan, and local release verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/scan_secrets.py`
- Test: `tests/test_secret_scan.py`
- Modify: `docs/superpowers/specs/2026-08-12-a-share-agent-design.md`

- [ ] **Step 1: Write a failing repository secret-scan test**

The scanner skips `.git`, virtualenv/cache/artifacts, and documented placeholders, but fails on likely live `sk-` keys, 64-character Tushare-like tokens, bearer headers, or secret assignments in tracked files. Test clean and leaking temporary trees.

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_secret_scan.py -q`

Expected: missing scanner.

- [ ] **Step 3: Implement CI and scanning**

CI on push/PR uses Python 3.11, installs `.[dev]`, runs `pip check`, Ruff check/format, pytest, offline evaluation, secret scan, then starts Streamlit and polls `/_stcore/health` with a 20-second bound. It requires no repository Secrets.

- [ ] **Step 4: Run every release gate fresh**

Only after these pass, change the spec status to `已实现并通过本地验收`:

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/scan_secrets.py .
git diff --check
```

- [ ] **Step 5: Run one minimal real API smoke without printing credentials**

Use ignored local Secrets to fetch stock master, analyze one liquid A-share for 3 months, request one interpretation, repeat to prove cache hit, and ask one follow-up. Print only status, code/name, last trade date, row count, rule signal, model, cache flag, and sanitized error code. If points/network/balance blocks it, report the sanitized blocker honestly; never replace it with fake success.

- [ ] **Step 6: Commit**

Commit: `git commit -m "ci: add release and secret-safety gates"`

## Task 11: GitHub and Streamlit Community Cloud deployment

**Files:**
- Modify: `README.md`
- Modify: `docs/assignment/final-project.md`
- Modify: `docs/user-guide.md`
- Modify: `docs/evaluation-report.md`

- [ ] **Step 1: Inspect authentication without exposing tokens**

Verify `gh auth status`, Git identity, repo status, and signed-in browser state. Create public repo `a-share-tech-agent` unless occupied, then use `a-share-tech-agent-coursework`. Never embed credentials in the remote.

- [ ] **Step 2: Publish verified code**

Create/switch to `codex/a-share-tech-agent`, push, use a non-destructive PR/default-branch flow, and confirm the public default branch contains the verified commit. Confirm GitHub Actions passes remotely.

- [ ] **Step 3: Deploy with Python 3.11 and Cloud Secrets**

In Streamlit Community Cloud select the public repository, default branch, root `app.py`, and Python 3.11. Paste the two real credentials only in the Secrets dialog with base URL/model. No screenshot or logs may show that pane's values.

- [ ] **Step 4: Run unauthenticated public smoke tests**

In signed-out/incognito context verify load, fuzzy search, candidate choice, data date, charts, score evidence, explicit AI trigger, cache hit, one follow-up, disclaimer, narrow viewport, and absence of credential input UI. Record URL and sanitized timestamp.

- [ ] **Step 5: Fill and verify final links**

Replace placeholders in README/course materials with GitHub and Streamlit URLs, commit, push, verify both links, and re-check Actions.

- [ ] **Step 6: Final handoff**

Report repo/app links, test/evaluation counts, last live data date, SQLite-on-Cloud persistence limitation, and advise rotating both shared credentials. Do not repeat either key.

Commit: `git commit -m "docs: publish project experience links"`

---

## Plan self-review

- Spec coverage: search through deployment maps to Tasks 1–11; Search Service stays outside LangGraph.
- Type consistency: `analysis_id` precedes persistence; AI key includes analysis/model/prompt; chat count is 0–10.
- Security: real credentials appear only in ignored local/Cloud Secrets; fakes never use them.
- TDD: every production module begins with focused failing tests and a named GREEN command.
- External cost: automated tests/evaluation are offline; Task 10 alone performs a minimal live smoke.
- Numeric transparency: direction capacity is 97; mutually exclusive Bollinger rules make realized extrema about ±94.4444, fixed by golden tests rather than hidden normalization.
- Placeholder scan: only the two deployment URLs remain intentionally pending until Task 11 creates and verifies them.
