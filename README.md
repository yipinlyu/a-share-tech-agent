# A股智研台

基于 Tushare、LangGraph 与 DeepSeek-V4-Flash 的可解释 A 股技术分析智能体。系统用确定性 Python 工具构造前复权行情、计算技术指标和 `score-v1` 规则信号；大模型只在用户点击后解释已校验的结构化事实。

> **仅供学习研究，不构成投资建议**。

## 能力与边界

- 用六位代码、Tushare 代码、中文全名或名称片段检索 A 股，并由用户完成候选消歧。
- 支持 3/6/12/24/36 个月前复权日线，固定显示“数据最后交易日”，不把日线称为实时行情。
- 计算 MA、MACD、布林带、RSI、KDJ、ATR、OBV 和成交量动能，展示分组分、证据、风险与观察位。
- LangGraph 编排规划、取数、质量校验、指标、评分、结果和记忆节点；异常安全停止。
- DeepSeek 按需生成严格 JSON，Schema 与 grounding 通过后才显示/缓存；规则信号始终是主信号。
- Session State 保存短期上下文，匿名 SQLite 保存有 TTL 的摘要和缓存；不保存身份、Key 或完整对话。

不提供自动交易、个性化仓位、收益预测、收益承诺、基本面估值或实时分钟行情。

## 快速开始

要求 Python 3.11。Tushare 账号至少需要 2000 积分，并仍须具备项目使用接口的实际权限。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

在本地 ignored `.streamlit/secrets.toml` 中由部署者配置：

```toml
TUSHARE_TOKEN = "replace-with-your-tushare-token"
DEEPSEEK_API_KEY = "replace-with-your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
```

不要提交真实 Secrets，也不要把它们贴到 Issue、日志、截图或聊天。公开访客无需输入 Key，应用页面不提供凭据输入框。

启动：

```bash
.venv/bin/python -m streamlit run app.py
```

访问终端显示的本地地址，输入股票并确认候选。缺少 DeepSeek Key 时确定性量化功能仍可用；缺少 Tushare Token 时系统明确禁用数据分析，不用假数据替代。

## 数据日期与计算口径

系统调用 Tushare `daily` 与 `adj_factor`，按截止交易日 anchor 自行构造 point-in-time 前复权价格。展示窗口前至少请求 120 个交易日预热；计算后再裁剪。页面“数据最后交易日”是实际取得的最后日线日期，可能因休市、停牌或数据源延迟早于访问日期。

- 指标公式：`indicators-v1`
- 评分规则：`score-v1`
- AI 提示词：`prompt-v1`
- 固定风险声明：仅供学习研究，不构成投资建议

## 测试与离线评测

自动测试与默认评测使用 fixtures/fakes，不访问外部 API：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate.py
```

评测报告写入被忽略的 `artifacts/evaluation-report.json`。冻结规模为 100 个搜索查询、500 行/5 种行情模式、50 个 AI grounding 案例和 25 个异常矩阵案例。强制门槛：搜索 Top-5 ≥95%，数据质量检测、AI grounding、缓存跳过和异常安全降级均为 100%；指标参考以 `rtol=1e-6`、`atol=1e-8` 比较。历史收益只作描述性回看，不设 CI 门槛。

显式配置就绪检查：

```bash
.venv/bin/python scripts/evaluate.py --live
```

`--live` 只报告 Tushare/DeepSeek 是否已配置及非敏感模型名，绝不打印密钥；发布前付费 API 的最小真实 smoke test 单独执行。

详见 [评测报告](docs/evaluation-report.md)。

## 架构与文档

- [架构、状态图、SDD 契约与记忆生命周期](docs/architecture.md)
- [离线评测方法、异常矩阵与历史回看限制](docs/evaluation-report.md)
- [用户使用、配置、追问与故障排查](docs/user-guide.md)
- [课程结项材料](docs/assignment/final-project.md)
- [冻结总体设计](docs/superpowers/specs/2026-08-12-a-share-agent-design.md)

核心边界：Search Service 在 LangGraph 外消歧；分析图只接受已选定的合法股票；Tushare、指标、评分、记忆和 DeepSeek 都通过明确接口隔离。规则结果先于持久化生成，因此 SQLite 写入失败不会抹掉 `analysis_id` 或量化结论。

## Streamlit Community Cloud 部署

1. 将仓库发布为公开 GitHub 项目；默认分支须通过 CI、离线评测和密钥扫描。
2. 在 Streamlit Community Cloud 选择仓库、默认分支、根目录 `app.py` 和 Python 3.11。
3. 只在 Cloud Secrets 后台填写四个配置项；不要把真实值写入仓库。
4. 在未登录浏览器检查搜索、候选、数据最后交易日、图表、规则证据、显式 AI、缓存、追问、免责声明与窄屏布局。
5. Cloud 本地 SQLite 不保证跨重启耐久；如需生产级多用户长期记忆，应迁移到受控外部数据库。

## 项目结构

```text
app.py
src/stock_agent/
  agent/        # LangGraph 状态与节点
  data/         # Tushare 与股票检索
  indicators/   # indicators-v1
  scoring/      # score-v1
  llm/          # DeepSeek、Schema 与 grounding
  memory/       # SQLite TTL
  services/     # UI 门面
  ui/           # Plotly 与 Streamlit 组件
scripts/evaluate.py
tests/evaluation/
docs/
```

## 发布链接

- GitHub：`https://github.com/<owner>/<repository>`（占位符，Task 11 发布后替换）
- Streamlit：`https://<app-name>.streamlit.app`（占位符，Task 11 部署后替换）

## 风险声明

**仅供学习研究，不构成投资建议**。技术指标和规则信号基于历史数据，存在滞后、噪声、停牌、复权、流动性与市场结构变化风险；AI 解读不是收益概率、交易指令或个性化建议，历史表现不保证未来结果。
