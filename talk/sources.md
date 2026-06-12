# 来源台账 · 《AI 智能体与科研工作流》

> 幻灯片上每条事实断言的出处。核实日期 2026-06-13。条目格式: 键名/断言/来源/置信/用于哪页。


<!-- ===== research/sources-math.md ===== -->

# 来源台账 — AI数学能力（slide对应）

格式：键名 → 断言 → 细节 → 来源 → 置信 → slide。逐条增量写入（访问日期均为2026-06-13）。

## [imo-2024-silver] AlphaProof+AlphaGeometry 2 在IMO 2024达银牌标准，28/42，差1分金牌
- 断言细节: DeepMind官方博客2024-07-25发布；解出6题中4题（AlphaProof解P1/P2/P6，AlphaGeometry 2解P4），每题满分7分共28分；当年金牌线29分，故差1分；由Timothy Gowers（菲尔兹奖得主）与Joseph Myers（IMO 2024命题委员会主席）按IMO评分规则打分；两道组合题未解出
- 来源: https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/ （访问 2026-06-13）；第二来源: https://www.nature.com/articles/s41586-025-09833-y （Nature同行评审论文）
- 置信: 高
- 用于: slide-05/07

## [imo-2025-gold] 2025年Gemini Deep Think与OpenAI实验模型均达IMO金牌线（35/42，解出5/6题）
- 断言细节: OpenAI于2025-07-19经官方X与研究员Alexander Wei宣布，35/42，由三位前IMO奖牌得主独立评分（非IMO官方认证），属实验性模型不发布；Google DeepMind于2025-07-21官宣Gemini Deep Think，35/42，IMO官方认证（"officially"），端到端自然语言、4.5小时限时内完成
- 来源: https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/ ；OpenAI一手: https://x.com/OpenAI/status/1946594928945148246 （均访问 2026-06-13）；第二来源: https://simonwillison.net/2025/Jul/19/openai-gold-medal-math-olympiad/
- 置信: 高
- 用于: slide-07

## [imo-2026-pending] 截至2026-06-13无更新的IMO结果；IMO 2026（第67届）于2026年7月10–21日在上海举行
- 断言细节: IMO官网editions页面列明 "Shanghai, People's Republic of China, July 10 - 21"（2026）；故报告日（2026-06-15）时最新AI-IMO成绩仍为2025年金牌线，IMO 2026尚未举行，逻辑上不可能有更新结果
- 来源: https://www.imo-official.org/editions/2026/ （访问 2026-06-13）；第二来源: https://www.imo-official.org/ （官网主页）
- 置信: 高
- 用于: slide-07（口头说明/脚注）

## [alphaevolve-matmul] AlphaEvolve（2025-05-14）以48次标量乘法实现4×4复矩阵乘法，破Strassen 1969的49次（56年来首次）
- 断言细节: DeepMind博客2025-05-14；适用于复数域矩阵；同时在50+个开放数学问题上测试：约75%重新发现已知最优解，20%改进了已知最优解；11维kissing number新下界593（此前592）；后续有数学家在arXiv:2506.13242将其改进为48次非复数乘法
- 来源: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ （访问 2026-06-13）；第二来源: https://the-decoder.com/alphaevolve-is-google-deepminds-new-ai-system-that-autonomously-creates-better-algorithms/
- 置信: 高
- 用于: slide-08

## [funsearch-capset] FunSearch（Nature, 2023-12-14）发现8维cap set新下界（512>496），DeepMind称之为LLM对开放科学/数学问题的首个新发现
- 断言细节: 博客原话 "first time a new discovery has been made for challenging open problems in science or mathematics using LLMs"；8维cap set 512（此前496），为20年来cap set规模最大增幅；输出形式是生成解的程序；论文 Romera-Paredes et al., Nature 625, 468–475，2023-12-14在线
- 来源: https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/ （访问 2026-06-13）；第二来源: https://www.nature.com/articles/s41586-023-06924-6 （论文页，cookie墙）+ 作者版PDF https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf
- 置信: 高（512/496数字经Wikipedia与作者版PDF双重确认）
- 用于: slide-08

## [tao-lean-quote] 陶哲轩ETP：4694条等式律、22,028,942条蕴涵全部解决并于2025年4月在Lean中完全形式化；另有两句可引用AI评语
- 断言细节: ETP于2024年9月启动，约2个月非形式解决全部蕴涵、再5个月完成Lean形式化（启动后200余天，2025-04）；论文arXiv:2512.07087（2025-12，陶+33人）；陶明言"modern AI tools did not play a major role in this project"（主力是自动定理证明器）。引语1（2024-09-13评o1）: "...roughly on par with trying to advise a mediocre, but not completely incompetent, (static simulation of a) graduate student."（后有澄清帖）；引语2（2023-06-12）: "I expect, say, 2026-level AI, when used properly, will be a trustworthy co-author in mathematical research"
- 来源: https://terrytao.wordpress.com/2025/12/09/the-equational-theories-project-advancing-collaborative-mathematical-research-at-scale/ ；https://mathstodon.xyz/@tao/113132503432772494 （经API取原文）；https://unlocked.microsoft.com/ai-anthology/terence-tao/ （均访问 2026-06-13）；第二来源: https://arxiv.org/abs/2512.07087
- 置信: 高
- 用于: slide-09

## [frontiermath-sota] FrontierMath（Epoch AI）：发布时SOTA<2%（2024-11），2026年中最高分GPT-5.5 Pro约52.4%；但2026-05官方称约1/3题目被标记致命错误、分数待修订
- 断言细节: 论文arXiv:2411.04872（2024-11-07）原话"under 2%"；结构=Tier1-3共300题+Tier4共50题研究级；最高分52.4%=GPT-5.5 Pro(high)（Epoch数据2026-04，OWID CSV与benchlm.ai 2026-06-09快照一致；GPT-5.5=51.7%，GPT-5.4 Pro=50%，Kimi K2.6=39%）；2026-05-11 Epoch官方X宣布AI辅助审查标记约1/3题目有fatal errors，修订分数未出
- 来源: https://arxiv.org/abs/2411.04872 ；https://ourworldindata.org/grapher/ai-frontiermath-over-time.csv （Epoch AI数据）；https://x.com/EpochAIResearch/status/2053995435870892048 （均访问 2026-06-13）；第二来源: https://benchlm.ai/benchmarks/frontierMath
- 置信: 高（<2%与审查事件）/ 中高（52.4%——官方复核中，引用须加"修订前"限定）
- 用于: slide-10

## [erdos-incident] 2025年10月：OpenAI副总裁Kevin Weil宣称GPT-5"解决10个未解Erdős问题"，实为找到已有文献，删帖澄清；Hassabis批"embarrassing"
- 断言细节: Weil原帖（约2025-10-17/18）"GPT-5 found solutions to 10 (!) previously unsolved Erdős problems"；维护者Thomas Bloom斥为"a dramatic misrepresentation"，其站"open"仅指"我个人不知道有论文解决它"，实情"GPT-5 found references...that I personally was unaware of"；Hassabis（10-19）"This is embarrassing"，LeCun讥讽；Bubeck承认仅是文献检索但强调其价值；正面后续：2026-05-02报道GPT-5.4对某Erdős问题给出前人未试的真实新解法（陶哲轩验证），但原始输出质量差需专家解读
- 来源: https://techcrunch.com/2025/10/19/openais-embarrassing-math/ （访问 2026-06-13）；第二来源: https://futurism.com/artificial-intelligence/openai-researcher-deletes-tweet ；正面后续: https://futurism.com/artificial-intelligence/mathematicians-claim-significant-discovery-using-chatgpt
- 置信: 高（事件本身多源一致）/ 中（Weil发帖精确日期未单独核到，记为"约10-17/18，10-19前已删"）
- 用于: slide-10/35

## [cn-provers] 国产形式化证明器2024–25领跑Lean赛道：DeepSeek-Prover-V2 miniF2F 88.9%、Kimina-Prover 80.7%→92.2%、字节Seed-Prover饱和miniF2F并形式化证明IMO 2025的5/6题
- 断言细节: DeepSeek-Prover-V2-671B（arXiv:2504.21801，2025-04-30）miniF2F-test 88.9%、PutnamBench 49/658、开源；Kimina-Prover Preview（Numina & Kimi团队/月之暗面，arXiv:2504.11354，2025-04-15）miniF2F 80.7% pass@8192（首破80%），72B版2025-07-10发布、配TTRL报92.2%；Seed-Prover（字节Seed，arXiv:2507.23726，2025-07-31）"saturates MiniF2F"、past IMO 78.1%、IMO 2025赛题Lean全证5/6
- 来源: https://arxiv.org/abs/2504.21801 ；https://arxiv.org/abs/2504.11354 ；https://arxiv.org/abs/2507.23726 （均访问 2026-06-13）；第二来源: https://github.com/deepseek-ai/DeepSeek-Prover-V2 ；https://github.com/MoonshotAI/Kimina-Prover-Preview ；92.2%佐证 https://arxiv.org/pdf/2511.03108
- 置信: 高（各项数字均出自一手论文摘要/官方仓库）；92.2%为中高（出自模型卡与第三方复核论文）
- 用于: slide-09



<!-- ===== research/sources-models.md ===== -->

# 来源台账：模型版图与 Harness 生态（核实日期 2026-06-13）

体例：
## [键名] 一句话断言
- 断言细节: …
- 来源: <URL>（访问 2026-06-13）；第二来源: <URL2>
- 置信: 高/中
- 用于: slide-NN

---

## [model-landscape/anthropic] Anthropic 当前旗舰是 Claude Fable 5（2026-06-09 发布），首个面向公众的 Mythos 级模型
- 断言细节: 官方定位 "a Mythos-class model that we've made safe for general use"；能力超过 Anthropic 此前所有公开模型；定价 $10/$50 每百万 token；敏感请求自动降级由 Claude Opus 4.8 应答；姊妹模型 Claude Mythos 5 仅授权用户（Project Glasswing）。第三方佐证：Artificial Analysis Intelligence Index 榜首（65 分 / 401 模型，访问 2026-06-13）。
- 来源: https://www.anthropic.com/news/claude-fable-5-mythos-5 （访问 2026-06-13）；第二来源: https://www.cnbc.com/2026/06/09/anthropic-mythos-claude-fable-5.html ；第三方评测: https://artificialanalysis.ai/models
- 置信: 高（官方公告直接抓取核实）
- 用于: slide-06

## [model-landscape/openai] OpenAI 当前旗舰是 GPT-5.5（2026-04-23 发布）；ChatGPT 默认模型为 GPT-5.5 Instant（2026-05-05 起）
- 断言细节: 官方开发者文档定位 GPT-5.5 = "A new class of intelligence for coding and professional work"（1M 上下文）；发布博客称其为最强 agentic coding 模型（Terminal-Bench 2.0 82.7%）。GPT-5.5 Instant 取代 GPT-5.3 Instant 成为 ChatGPT 默认。第三方佐证：Artificial Analysis Intelligence Index GPT-5.5 (xhigh)=60，居 Fable 5、Opus 4.8 之后（访问 2026-06-13）。
- 来源: https://developers.openai.com/api/docs/models （官方，访问 2026-06-13）；第二来源: https://techcrunch.com/2026/05/05/openai-releases-gpt-5-5-instant-a-new-default-model-for-chatgpt/ ；发布博客（403 仅经搜索摘要）: https://openai.com/index/introducing-gpt-5-5/ ；第三方评测: https://artificialanalysis.ai/models
- 置信: 高（日期经 TechCrunch+官方文档双源；openai.com 正文未直抓）
- 用于: slide-06

## [model-landscape/google] Google 已发布的最新模型是 Gemini 3.5 Flash（I/O 2026，2026-05-20 GA）；Gemini 3.5 Pro 截至 2026-06-13 尚未正式发布
- 断言细节: 官方定位 3.5 Flash = "the first in our latest series of models combining frontier intelligence with action"，长程 agent 任务向，已是 Search AI Mode 全球默认；同场发布生成媒体模型 Gemini Omni（Omni Flash 经 Gemini app 推送订阅用户）。3.5 Pro 仅内部+有限 Vertex 预览，预计 6 月 GA（讲台原话 "Give us until next month"）——报告日 (06-15) 前可能变化，需讲前一天复查。第三方佐证：Arena 文本榜 gemini-3.1-pro-preview 1487（#7）、gemini-3.5-flash 在榜 Top15（访问 2026-06-13）。
- 来源: https://blog.google/innovation-and-ai/technology/ai/google-io-2026-all-our-announcements/ （访问 2026-06-13）；第二来源: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/ ；第三方评测: https://arena.ai/leaderboard
- 置信: 高（官方博客直接抓取；3.5 Pro 未发布状态另经 2026-06-06 TechTimes 佐证）
- 用于: slide-06

## [model-landscape/deepseek] DeepSeek 当前旗舰是 DeepSeek-V4（Preview，2026-04-24 发布），已驱动网页版
- 断言细节: 双型号 V4-Pro（1.6T/49B 激活）与 V4-Flash（284B/13B），1M 上下文，Thinking/Non-Thinking 双模式，权重开源（HF）；官方口号 "Welcome to the era of cost-effective 1M context length"；网页版 chat.deepseek.com 以 Expert/Instant 两档提供 V4；旧 deepseek-chat/reasoner 2026-07-24 退役。
- 来源: https://api-docs.deepseek.com/news/news260424 （官方，访问 2026-06-13）；第二来源: https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/ ；第三方评测: https://artificialanalysis.ai/models/deepseek-v4-pro （AA Intelligence Index：V4 Pro (Max)=52，开源权重第 2、仅次于 Kimi K2.6=54；较 V3.2 的 42 提升 10 分。访问 2026-06-13）
- 置信: 高（官方 API 新闻页直接抓取）
- 用于: slide-06

## [model-landscape/kimi] 月之暗面当前旗舰是 Kimi K2.6（2026-04-20 发布），开源权重模型中评分最高
- 断言细节: 官网定位 "natively multimodal model, powerful coding capabilities, and Agent performance"；1T MoE / 32B 激活、256K 上下文、MoonViT 原生视觉，Modified MIT 开源；上线 Kimi.com/App/API/Kimi Code CLI。第三方佐证：Artificial Analysis Intelligence Index = 54，为开源权重榜首（访问 2026-06-13）。
- 来源: https://www.moonshot.ai/ （访问 2026-06-13）；第二来源: https://developers.cloudflare.com/changelog/post/2026-04-20-kimi-k2-6-workers-ai/ ；第三方评测: https://artificialanalysis.ai/models
- 置信: 高（官网直接核实名称与定位；架构细节来自多家二手源一致）
- 用于: slide-06

## [model-landscape/minimax] MiniMax 当前旗舰是 MiniMax M3（2026-06-01 发布），主打 1M 上下文 + 原生多模态 + agentic coding
- 断言细节: 官方定位 "the latest M-series language model for agentic reasoning, tool use, coding, multimodal chat input, and long-context tasks"；MSA 稀疏注意力；开源权重承诺发布后约 10 天放出（截至 2026-06-13 尚未放出）。第三方佐证：Artificial Analysis Intelligence Index = 55，AA 称其为"权重放出后的开源领跑者"（访问 2026-06-13）。
- 来源: https://platform.minimax.io/docs/release-notes/models （官方，访问 2026-06-13）；第二来源: https://www.minimax.io/models/text/m3 ；第三方评测: https://artificialanalysis.ai/articles/minimax-m3
- 置信: 高（官方平台文档直接抓取）
- 用于: slide-06

## [model-landscape/qwen] 阿里通义当前旗舰是 Qwen3.7-Max（2026-05-20 阿里云峰会发布），闭源 API-only 的"Agent Frontier"
- 断言细节: 官方博客标题《Qwen3.7: The Agent Frontier》；1M 上下文；面向长程自主任务/代码/多步自动化；另有 Qwen3.7-Plus-Preview 多模态版；Qwen Chat 网页免费可用，API 走阿里云百炼。注意：阿里云国际站文档截至 2026-06-13 仅列到 Qwen3.5 系（国际站滞后）。第三方佐证：Artificial Analysis Intelligence Index 56.6，全球第 5、中国第 1（多家转引，访问 2026-06-13）。
- 来源: https://qwen.ai/blog?id=qwen3.7 （官方博客，JS 渲染正文未直抓，访问 2026-06-13）；第二来源: https://www.marktechpost.com/2026/05/21/qwen-introduces-qwen3-7-max-a-reasoning-agent-model-with-a-1m-token-context-window/ ；https://developer.aliyun.com/article/1738538 ；第三方评测: https://artificialanalysis.ai/models
- 置信: 高（名称/日期多源一致；唯官方博客正文未直抓，AA 56.6 为转引数字）
- 用于: slide-06

## [harness-lineup/codex] Codex 是 OpenAI 的编码 agent，随 ChatGPT 订阅（Plus/Pro/Business/Edu/Enterprise）附带
- 断言细节: 官方原文 "Codex is OpenAI's coding agent for software development"；形态覆盖 App/IDE 扩展/CLI/云端；"ChatGPT Plus, Pro, Business, Edu, and Enterprise plans include Codex"。价格量级（官方定价页核实）：Plus $20/月；Pro 自 $100/月起，分 5x/20x 两档；按 5 小时窗口限消息数，可购 ChatGPT credits 超额。
- 来源: https://developers.openai.com/codex （访问 2026-06-13）；定价: https://developers.openai.com/codex/pricing （访问 2026-06-13）
- 置信: 高
- 用于: slide-20/39

## [harness-lineup/openclaw] OpenClaw 是开源（MIT）的"跑在自己设备上的个人 AI 助理"，约 37.8 万 GitHub stars
- 断言细节: README 原文 "a personal AI assistant you run on your own devices. It answers you on the channels you already use"（WhatsApp/Telegram/Slack/Discord 等）；npm 一行安装（npm install -g openclaw@latest）或 Docker；软件免费、BYO 模型 API key。注意定位是通用个人助理/agent 网关而非专职编码工具。
- 来源: https://github.com/openclaw/openclaw （访问 2026-06-13）
- 置信: 高
- 用于: slide-20/39

## [harness-lineup/opencode] opencode 是开源的终端 AI 编码 agent（160k+ stars），免费+可接 75+ 模型源，另有付费 Zen 模型订阅
- 断言细节: 官网原文 "The open source AI coding agent"；curl 一行安装，桌面端 Beta；不存储用户代码/上下文；可接 75+ LLM 提供商（含本地模型）及 GitHub Copilot/ChatGPT/Claude 订阅；Zen = 官方精选并基准测试过的模型付费接入。
- 来源: https://opencode.ai/ （访问 2026-06-13）
- 置信: 高
- 用于: slide-20/39

## [harness-lineup/gemini-cli] Gemini CLI 是 Google 开源（Apache-2.0）终端 agent，个人 Google 账号免费 60 次/分、1000 次/天
- 断言细节: README 原文 "An open-source AI agent that brings the power of Gemini directly into your terminal"；npm/brew/npx 安装；免费档绑定个人 Google 账号（60 rpm / 1000 rpd），可升级 API key/Vertex 计费；支持 Gemini 3 系模型、1M 上下文。
- 来源: https://github.com/google-gemini/gemini-cli （访问 2026-06-13）
- 置信: 高
- 用于: slide-20/39

## [harness-lineup/claude-code] Claude Code 是 Anthropic 的 agentic 编码工具，覆盖终端/IDE/桌面/Web，需 Claude 订阅或 API
- 断言细节: 官方原文 "an agentic coding tool that reads your codebase, edits files, runs commands, and integrates with your development tools. Available in your terminal, IDE, desktop app, and browser"；一行安装（curl install.sh / PowerShell install.ps1 / brew / winget）；多数入口需 Claude 订阅（claude.com/pricing）或 Anthropic Console API 计费，CLI/VS Code 支持第三方模型。价格量级（claude.com/pricing 核实）：Free $0；Pro $20/月（年付折合 $17/月）；Max 自 $100/月起（5x/20x 两档）；或 Anthropic Console API 按用量（Fable 5 $10/$50 每百万 token）。
- 来源: https://code.claude.com/docs/en/overview （访问 2026-06-13）；定价: https://claude.com/pricing （访问 2026-06-13）
- 置信: 高
- 用于: slide-20/39

## [multimodal-pdf-support/chatgpt] ChatGPT 网页版支持 PDF/Office/图片上传与视觉理解
- 断言细节: 官方 File Uploads FAQ：回形针上传 PDF/DOCX/PPTX/CSV/XLSX/JPEG/PNG/TXT 等，文档单文件 2M tokens、图片 20MB；Visual Retrieval with PDFs 可读 PDF 内嵌图表（FAQ 明示 Enterprise；Plus/Pro 的 PDF 视觉检索范围未直接核实）；图片视觉理解原生支持，手机 App 可拍照。
- 来源: https://help.openai.com/en/articles/8555545-file-uploads-faq （官方，403 未直抓、经搜索摘要核实，访问 2026-06-13）；第二来源: https://help.openai.com/en/articles/10416312-visual-retrieval-with-pdfs-faq
- 置信: 中（官方页未能直抓；内容经多条搜索结果一致）
- 用于: slide-16

## [multimodal-pdf-support/claude] Claude 网页版支持 PDF（含视觉解析<100页）与图片上传
- 断言细节: 官方帮助原文：文档 PDF/DOCX/CSV/TXT/HTML/ODT/RTF/EPUB/JSON/XLSX，图片 JPEG/PNG/GIF/WebP；500MB/文件、20 文件/对话；PDF<100 页解析 "both text and visual elements (like images, charts, and graphics)"，>1000 页仅文本。
- 来源: https://support.claude.com/en/articles/8241126-upload-files-to-claude （官方直抓，访问 2026-06-13）
- 置信: 高
- 用于: slide-16

## [multimodal-pdf-support/gemini] Gemini 网页版支持 PDF/Office/图片上传（10 文件/提示、100MB/文件）
- 断言细节: 官方帮助（直抓）：图片 JPEG/JPG/PNG/WEBP/HEIF；文档 PDF/DOC/DOCX/RTF/TXT/PPTX/XLS/XLSX/CSV/TSV/Google 套件；"Up to 10 files (subject to availability) can be uploaded in the same prompt"；普通文件至 100MB、视频至 2GB/5 分钟；**免订阅即可用**，升级 Google AI 计划获更高限额（视频 1h/音频 3h）；Gemini 原生多模态可视觉理解图片。
- 来源: https://support.google.com/gemini/answer/14903178 （官方直抓，访问 2026-06-13）
- 置信: 高
- 用于: slide-16

## [multimodal-pdf-support/kimi] Kimi 网页版支持批量文件上传（约 50 个×100MB，含 PDF/图片），K2.6 原生视觉理解
- 断言细节: 官方 API 文件接口列出数十种格式（pdf/docx/xlsx/pptx/jpeg/png/webp…）；网页版批量上传 50 文件×100MB 为社区指南口径；K2.6 官网自述 natively multimodal（MoonViT），图片为真视觉理解，移动端可拍照。
- 来源: https://platform.moonshot.cn/docs/api/files （官方，访问 2026-06-13）；第二来源: https://zhuanlan.zhihu.com/p/703044009 ；https://www.moonshot.ai/
- 置信: 中（上限数字未经官方帮助页直抓）
- 用于: slide-16

## [multimodal-pdf-support/deepseek] DeepSeek 网页版：PDF/文档一直可传（提取文字）；图片旧版仅 OCR，2026-04-29 识图模式灰度、**2026-05-09 大范围开放**（真视觉理解，"正式跨入图文交互时代"）
- 断言细节: 旧版（V3/R1 至 2026-04）附件按钮标注"上传附件（只识别文字）"，图片仅 OCR 不懂内容；V4（2026-04-24）官方公告未提视觉；2026-04-29 网页端"识图模式"灰度（与快速/专家模式并列按钮，"彻底超越传统OCR范畴"）；**2026-05-09 结束灰度、大范围开放**——支持图片内容识别、联网增强问答、截图一键提问，可精准解析图中文字/表格/数学公式；尚未集成图像生成与视频理解。PDF 内图表是否视觉解析：未核实。讲给听众的一句话：**"你们用的 DeepSeek 网页版，5 月 9 日起图片终于从'只认字'升级到'真看懂'，但 PDF 仍以文字抽取为主"**。
- 来源: https://www.ithome.com/0/948/020.htm （IT之家，2026-05-09，访问 2026-06-13）；第二来源: https://finance.sina.com.cn/tech/digi/2026-05-09/doc-inhxhcxr3627755.shtml ；https://openaxo.com/innovation/deepseek-v4-vision-mode-analysis （灰度阶段）；https://blog.csdn.net/ZHY0091/article/details/147114315 （旧版 OCR 基线）；官方 V4 公告（无视觉表述）: https://api-docs.deepseek.com/news/news260424
- 置信: 高（全量开放经 IT之家/新浪/中关村在线等多家主流媒体一致报道；唯 DeepSeek 官方无公告页可直引）
- 用于: slide-16


<!-- ===== research/sources-boundaries.md ===== -->

# 台账：AI使用边界——翻车实例与学术政策（核实日 2026-06-13）

## [fake-citations-case] 美国律师用ChatGPT编造6个判例被联邦法院罚款5,000美元；编造引文已蔓延至学术文献并逐年加速
- 断言细节: Mata v. Avianca, Inc., No. 1:22-cv-01461 (PKC) (S.D.N.Y.)，制裁判决2023-06-22（678 F. Supp. 3d 443）；Castel法官对律师Schwartz、LoDuca及律所Levidow, Levidow & Oberman连带罚款$5,000，并责令致函被冒名的6位真实法官；6个假判例：Varghese/Shaboon/Petersen/Martinez/Durden/Miller；ChatGPT曾向律师保证假判例"indeed exist"；判词金句"Many harms flow from the submission of fake opinions."。量化研究：Topaz等（哥伦比亚大学）2026年5月Lancet通讯，审计PubMed约250万篇论文/9,700万条引文，发现约2,800篇论文含约4,000条编造引文，含编造引文论文比例 2023年1/2,828 → 2025年1/458 → 2026年前7周1/277；辅证：Zhao等 arXiv:2605.07723（250万篇/1.11亿条引文，仅2025年≈146,932条幻觉引文）；Chelli等 JMIR 2024;26:e53164（471条参考文献样本，幻觉率GPT-3.5 39.6%、GPT-4 28.6%、Bard 91.4%）
- 来源: https://en.wikipedia.org/wiki/Mata_v._Avianca,_Inc.（访问 2026-06-13）；第二来源: https://www.statnews.com/2026/05/07/lancet-study-finds-steep-rise-fraudulent-citations-academic-papers/ （另见 https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/575368/54/ 、https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(26)00603-3/fulltext 、https://arxiv.org/abs/2605.07723 、https://www.jmir.org/2024/1/e53164 、https://www.legaldive.com/news/lawyer-fake-chatgpt-cases-sanctions-fines-lawyers-chatgpt-fake-cases/653741/）
- 置信: 高
- 用于: slide-34

## [math-fail-example] LLM数学推导可"看起来步步有据实则出错"：GSM-NoOp猕猴桃题中模型把无关从句当运算依据，单句干扰可致SOTA模型性能跌65%
- 断言细节: 可展示原题（arXiv:2410.05229, Apple团队GSM-Symbolic论文）："Oliver picks 44 kiwis on Friday... On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average. How many kiwis does Oliver have?"；o1-mini与Llama3-8B均推理"88 − 5 = 83 ... 44 + 58 + 83 = 185"，而"略小的5个"与计数无关，正确答案190；论文摘要："Adding a single clause that seems relevant to the question causes significant performance drops (up to 65%)"。佐证：USAMO 2025整卷证明评测（arXiv:2503.21934）除Gemini-2.5-Pro得25%外其余模型均<5%；陶哲轩（Lex Fridman播客#472，2025-06）："the AI-generated proofs, they look superficially flawless... the errors are often really subtle and then when you spot them, they're really stupid"（措辞经X帖转录与The Decoder一致，官方逐字稿未完整核到，引用时注明"转录自访谈"）
- 来源: https://arxiv.org/abs/2410.05229（访问 2026-06-13）；第二来源: https://arxiv.org/abs/2503.21934 （陶哲轩引语 https://the-decoder.com/math-genius-terence-tao-says-that-ai-still-cant-smell-bad-math/ 、https://lexfridman.com/terence-tao-transcript/）
- 置信: 高（陶哲轩引语逐字措辞为中）
- 用于: slide-34

## [erdos-incident-flip] 2025年10月OpenAI高管宣称GPT-5"解决10个未解Erdős问题"，约一两天内被数据库维护者证伪为文献检索，删帖澄清
- 断言细节: 约2025-10-17，OpenAI VP Kevin Weil发帖（现已删除）："GPT-5 found solutions to 10 (!) previously unsolved Erdős problems and made progress on 11 others"；erdosproblems.com维护者Thomas Bloom回应称这是"a dramatic misrepresentation"，网站标"open"仅指"I personally am unaware of a paper which solves it"，实为"GPT-5 found references, which solved these problems, that I personally was unaware of"；DeepMind CEO Hassabis评论"this is embarrassing"，LeCun讥讽"Hoisted by their own GPTards"；Weil删帖，研究员Bubeck承认"only solutions in the literature were found"；媒体报道时间线：The Decoder 10-18、TechCrunch 10-19、Fortune 10-20、Futurism 10-21。注意：X原帖直链未核实（已删除），引用一律用媒体链接
- 来源: https://techcrunch.com/2025/10/19/openais-embarrassing-math/（访问 2026-06-13）；第二来源: https://the-decoder.com/leading-openai-researcher-announced-a-gpt-5-math-breakthrough-that-never-happened/ （另见 https://futurism.com/artificial-intelligence/openai-researcher-deletes-tweet 、https://fortune.com/2025/10/20/did-openais-latest-ai-model-solve-famously-difficult-math-problems-well/）
- 置信: 高（事件与原话经多家独立媒体一致转述；具体发帖钟点为中）
- 用于: slide-35

## [journal-policies] Nature/Elsevier/AMS均明文规定：AI不得列为作者、作者使用须声明、审稿人不得把稿件喂给AI
- 断言细节: Nature Portfolio："LLMs... do not currently satisfy our authorship criteria"（作者身份意味着问责），"Use of an LLM should be properly documented in the Methods section"，并要求"peer reviewers do not upload manuscripts into generative AI tools"（配套社论Nature 613, 612 (2023)："No LLM tool will be accepted as a credited author"）；Elsevier："Authors should disclose the use of AI tools for manuscript preparation in a separate AI declaration statement included in their manuscript upon submission"、"Authors should not list AI tools as an author or co-author"、"Reviewers should not upload a submitted manuscript or any part of it into an AI tool"；AMS有专门政策页"Use of Artificial Intelligence"（改编COPE 2023-02声明）："AI tools cannot be listed as an author of a paper"、用AI须在Materials and Methods等处披露所用工具及用法、"Editors and referees are not to upload papers under review to an LLM in any format, for any reason"
- 来源: https://www.nature.com/nature-portfolio/editorial-policies/ai（访问 2026-06-13）；第二来源: https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals （AMS官方政策页 https://www.ams.org/publications/journals/policies/UseofArtificialIntelligence 、Nature社论 https://www.nature.com/articles/d41586-023-00191-1）
- 置信: 高
- 用于: slide-36

## [data-security] 三家厂商条款：消费者网页对话OpenAI/DeepSeek默认可用于训练（可关闭），Anthropic须用户自选；OpenAI/Anthropic明文承诺API数据默认不训练，DeepSeek无此明确承诺
- 断言细节: OpenAI官方政策页："we may use your content to train our models"（个人版ChatGPT），"By default, we do not train on any inputs or outputs from our products for business users, including ChatGPT Team, ChatGPT Enterprise, and the API"；opt-out=隐私门户"do not train on my content"或设置中关闭"Improve the model for everyone"，临时对话不训练。Anthropic：2025-08-28条款更新，消费者须自选是否允许训练（老用户限2025-10-08前选择，可随时在Privacy Settings更改；允许则保留5年，否则30天）；商用/API："By default, we will not use your inputs or outputs from our commercial products (e.g. Claude for Work, Anthropic API, Claude Gov, etc.) to train our models"。DeepSeek隐私政策（Last Update 2026-02-10）："to train and improve our technology, such as our machine learning models and algorithms"，数据"store your Personal Data in People's Republic of China"；用户条款4.3提供opt-out：关闭"Improve the model for everyone"；API（开放平台）条款未见"不用于训练"承诺——标注"无明确承诺"，勿写成"不训练"
- 来源: https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/（访问 2026-06-13）；第二来源: https://www.anthropic.com/news/updates-to-our-consumer-terms （另见 https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training 、https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html 、https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html 、https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html）
- 置信: 高
- 用于: slide-36


<!-- ===== research/sources-zagier.md ===== -->

## [zagier-paper] Zagier (1990) 一句话证明原文
- 断言细节: D. Zagier, "A One-Sentence Proof That Every Prime p ≡ 1 (mod 4) Is a Sum of Two Squares", The American Mathematical Monthly, Vol. 97, No. 2 (Feb. 1990), p. 144。作者署名单位为 Department of Mathematics, University of Maryland（注意：不是马普所，他后来才任职 MPIM）。证明是对 Heath-Brown (1984, 受 Liouville 启发) 的简化；唯一不动点 (1,1,k)，仅此处用到 p = 4k+1；原文明确指出证明**非构造性**，并给出"有限集与其对合不动点集基数同奇偶"的组合原理表述
- 来源: https://people.mpim-bonn.mpg.de/zagier/files/doi/10.2307/2323918/fulltext.pdf （Zagier 本人主页托管的 JSTOR 原版扫描，访问 2026-06-13）；第二来源: https://www.tandfonline.com/doi/abs/10.1080/00029890.1990.11995565 （DOI 10.1080/00029890.1990.11995565）
- 置信: 高（原文 PDF 在手，已存 talk/demo/zagier.pdf）
- 用于: slide-24/25/26/27、demo 全程

## [windmill-viz] "风车"几何可视化的出处链
- 断言细节: Zagier 对合的几何"风车"解释由 Alexander Spivak 提出（2006/2007，不同来源记载年份不一）；大众化传播：Mathologer 视频 "Why was this visual proof missed for 400 years? (Fermat's two square theorem)"；算法化/形式化研究：Hing-Lun Chan, "Windmills of the minds: an algorithm for Fermat's Two Squares Theorem" (CPP 2022, arXiv:2112.02556)，扩展版发表于 J. Automated Reasoning 68(4), 2024
- 来源: https://www.youtube.com/watch?v=DjI1NICfjOk （访问 2026-06-13）；第二来源: https://arxiv.org/abs/2112.02556 ；https://en.wikipedia.org/wiki/Fermat%27s_theorem_on_sums_of_two_squares （Spivak 归属）
- 置信: 高（视频/论文直接可查）；Spivak 具体年份: 中
- 用于: slide-28
