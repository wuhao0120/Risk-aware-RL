# 调研笔记：2026-06 时点模型版图与 Harness 生态

为 2026-06-15 中国数学学院报告核实。除特别注明外，所有访问日期均为 2026-06-13。
本文件为原始摘录 + URL；结构化台账见 `sources-models.md`。
体例：每个条目 = 摘录（尽量原文）+ 来源 URL + 访问日期。查不到的标"未核实"。

---

## 1. 模型版图（slide-06）

### 1.1 Anthropic — Claude Fable 5（已核实，官方）
- 官方公告《Claude Fable 5 and Claude Mythos 5》：发布日期 **2026-06-09**。
- 原文摘录："a Mythos-class model that we've made safe for general use"；"Fable 5's capabilities exceed those of any model we've ever made generally available."
- 定价：$10 / 百万输入 token，$50 / 百万输出 token（Fable 5 与 Mythos 5 同价）。
- 订阅可用性：6/9–6/22 包含在 Pro/Max/Team/Enterprise 内，6/23 起需 usage credits。
- 与 Opus 4.8 关系：敏感请求（网络安全/生物化学/蒸馏）自动转由 Claude Opus 4.8 应答；Mythos 5 = 同一底层模型、对授权用户（Project Glasswing）解除部分安全限制。
- 来源: https://www.anthropic.com/news/claude-fable-5-mythos-5 （访问 2026-06-13）
- 第二来源（媒体）: https://www.cnbc.com/2026/06/09/anthropic-mythos-claude-fable-5.html ；https://techcrunch.com/2026/06/09/anthropic-released-claude-fable-5-its-most-powerful-model-publicly-days-after-warning-ai-is-getting-too-dangerous/ （访问 2026-06-13）

### 1.2 OpenAI — GPT-5.5（官网 403，以 Help/TechCrunch+官方开发者文档核实）
- openai.com 与 help.openai.com 对爬虫返回 403，已换源。
- TechCrunch（2026-05-05）：GPT-5.5 于 **2026-04-23** 发布；**GPT-5.5 Instant** 于 **2026-05-05** 发布并取代 GPT-5.3 Instant 成为 ChatGPT 默认模型；"reduces hallucination in sensitive areas such as law, medicine, and finance, while maintaining the low latency of its predecessor."
- 搜索摘要（openai.com/index/introducing-gpt-5-5/，正文未能直接抓取）：GPT-5.5 定位 "smartest and most intuitive to use model yet"，"strongest agentic coding model to date"，Terminal-Bench 2.0 82.7%，SWE-Bench Pro 58.6%。
- 来源: https://openai.com/index/introducing-gpt-5-5/ （官方，403 未直接抓取，标题与日期经搜索摘要核实，访问 2026-06-13）
- 第二来源: https://techcrunch.com/2026/05/05/openai-releases-gpt-5-5-instant-a-new-default-model-for-chatgpt/ （访问 2026-06-13）
- 官方开发者文档（developers.openai.com/api/docs/models，访问 2026-06-13）：GPT-5.5 = "A new class of intelligence for coding and professional work"，上下文 1M；GPT-5.4 = "A more affordable model"；GPT-5.4 mini = "strongest mini model yet for coding, computer use, and subagents"（400K）。
- ChatGPT 网页版当前默认模型：GPT-5.5 Instant（2026-05-05 起，替代 GPT-5.3 Instant）。

### 1.3 第三方评测站（用于全部 7 家的佐证）
- Artificial Analysis（https://artificialanalysis.ai/models ，访问 2026-06-13）Intelligence Index（401 个模型）：
  1. Claude Fable 5 (Adaptive Reasoning, Max Effort, Opus 4.8 Fallback) = 65（榜首）
  2. Claude Opus 4.8 (Max Effort) = 61
  3. GPT-5.5 (xhigh) = 60；GPT-5.5 (high) = 59
  5. Claude Opus 4.7 (Max Effort) = 57
  - 开源权重最高：Kimi K2.6 = 54
  - 原文："Claude Fable 5 ... currently leads the Artificial Analysis Intelligence Index with a score of 65, out of 401 models evaluated."
- LMArena：lmarena.ai/leaderboard 现 301 跳转至 arena.ai/leaderboard（品牌迁移，访问 2026-06-13），排名见下。
- Arena（原 LMArena，https://arena.ai/leaderboard ，访问 2026-06-13）文本榜 Top10 摘录：
  #1 claude-fable-5 (1510±11) > #2 claude-opus-4-6-thinking (1504) > #3 claude-opus-4-7-thinking (1502) > … #6 muse-spark (Meta, 1487) ≈ #7 gemini-3.1-pro-preview (1487) ≈ #8 gemini-3-pro (1486) > #9 claude-opus-4-8-thinking (1486) > #10 gpt-5.5-high (1481)。
  注：榜单含 gemini-3.5-flash、gpt-5.4-high、claude-opus-4-8 等（名次 10-15 区间）。

### 1.4 Google — Gemini 3.5 Flash + Gemini Omni（已核实，官方博客）
- Google I/O 2026 时间：**2026-05-20**（blog.google 官方汇总页）。
- **Gemini 3.5 Flash**：5/20 当日 GA。原文："the first in our latest series of models combining frontier intelligence with action"；"ideal for tackling long-horizon agentic tasks"；超越 Gemini 3.1 Pro 的 coding/agentic 基准；已是 Search AI Mode 全球默认模型。可经 Gemini API / AI Studio / Antigravity / Android Studio 获取。
- **Gemini 3.5 Pro**：截至 I/O 仍未发布——"already being used internally"，预计 "next month"（2026-06）推出。（截至 2026-06-13 是否已发布另行核实，见下）
- **Gemini Omni**：生成媒体统一模型，"turns any reference — image, text, video or audio — into a single, cohesive output"；Omni Flash 向 Google AI Plus/Pro/Ultra 订阅者经 Gemini app 推送。
- 来源: https://blog.google/innovation-and-ai/technology/ai/google-io-2026-all-our-announcements/ （访问 2026-06-13）
- 第三方佐证：Arena 榜 gemini-3.1-pro-preview 1487（#7）、gemini-3.5-flash 在榜（访问 2026-06-13）。
- 补充核实（访问 2026-06-13）：Gemini 3.5 官方页 https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/ ；3.5 Pro 截至 2026-06-13 **尚未正式发布**（仅内部使用+有限 Vertex 预览，Pichai 原话 "Give us until next month to get it to you"；预计 6 月 GA，2M 上下文 + Deep Think）。二手源: https://wavespeed.ai/blog/posts/gemini-3-5-pro-coming-next-month/ ；https://www.techtimes.com/articles/317919/20260606/google-gemini-35-pro-nears-june-launch-2-million-token-context-deep-think-reasoning.htm

### 1.5 DeepSeek — DeepSeek-V4（Preview）（已核实，官方 API 新闻页）
- 官方《DeepSeek V4 Preview Release》（api-docs.deepseek.com/news/news260424）：发布 **2026-04-24**。
- 模型：**DeepSeek-V4-Pro**（1.6T 总参 / 49B 激活）与 **DeepSeek-V4-Flash**（284B / 13B 激活），双模式（Thinking / Non-Thinking），默认 **1M 上下文**。
- 定位原文："Welcome to the era of cost-effective 1M context length."
- 技术点：Token-wise compression + DSA (DeepSeek Sparse Attention)；权重开源（Hugging Face）；兼容 OpenAI ChatCompletions 与 Anthropic API。
- 网页版：官方明示 "Try it now at chat.deepseek.com via Expert Mode / Instant Mode" —— 即 V4 已驱动 DeepSeek 网页版。
- 旧模型 deepseek-chat / deepseek-reasoner 别名 2026-07-24 后完全退役。
- 来源: https://api-docs.deepseek.com/news/news260424 （访问 2026-06-13）
- 第二来源（媒体）: https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/ （访问 2026-06-13）
- 第三方佐证（Artificial Analysis，访问 2026-06-13）：V4 Pro (Max) Intelligence Index = **52**（开源权重第 2，仅次于 Kimi K2.6 = 54；较 V3.2 的 42 提升 10 分）。AA 专文：《DeepSeek is back among the leading open weights models with V4 Pro and V4 Flash》 https://artificialanalysis.ai/articles/deepseek-is-back-among-the-leading-open-weights-models-with-v4-pro-and-v4-flash ；模型页 https://artificialanalysis.ai/models/deepseek-v4-pro

### 1.6 月之暗面 Moonshot AI — Kimi K2.6（已核实，官网）
- 官网 moonshot.ai 首页（访问 2026-06-13）：最新旗舰 **Kimi K2.6**，发布 **2026-04-20**，定位 "natively multimodal model, powerful coding capabilities, and Agent performance"。
- 架构（搜索摘要佐证）：1T 总参 MoE / 32B 激活，上下文扩至 256K，内置 400M MoonViT 视觉编码器（原生图像/视频输入）；权重 Modified MIT License 开源（HF）；上线 Kimi.com / App / API / Kimi Code CLI。
- 来源: https://www.moonshot.ai/ （访问 2026-06-13）
- 第二来源: https://developers.cloudflare.com/changelog/post/2026-04-20-kimi-k2-6-workers-ai/ （Cloudflare 官方 changelog，2026-04-20）
- 第三方佐证：Artificial Analysis Intelligence Index **K2.6 = 54，开源权重模型榜首**（https://artificialanalysis.ai/models ，访问 2026-06-13）。

### 1.7 MiniMax — MiniMax M3（已核实，官方平台文档）
- 官方 platform.minimax.io 模型发布说明（访问 2026-06-13）：**MiniMax M3**，发布 **2026-06-01**，定位原文 "the latest M-series language model for agentic reasoning, tool use, coding, multimodal chat input, and long-context tasks"。
- 官方产品页: https://www.minimax.io/models/text/m3 （"Coding & Agentic Frontier, 1M Context, Multimodal"）。
- 关键点（搜索摘要佐证）：MiniMax Sparse Attention (MSA)，1M 上下文，原生多模态；宣称 SWE-Bench Pro 59.0%；开源权重预计发布后约 10 天放出（即 6 月中旬）。
- 第三方佐证（Artificial Analysis，访问 2026-06-13）：**M3 = 55**，AA 专文标题《MiniMax-M3: Leading open weights model, once the weights are released》——即权重放出后将成开源榜首（截至访问日权重尚未放出，已放出的开源榜首仍为 Kimi K2.6=54）。 https://artificialanalysis.ai/articles/minimax-m3 ；https://artificialanalysis.ai/models/minimax-m3
- 来源: https://platform.minimax.io/docs/release-notes/models （访问 2026-06-13）
- 第二来源（媒体）: https://www.scmp.com/tech/tech-trends/article/3355529/minimax-debuts-ai-model-built-long-and-complex-coding-tasks （访问 2026-06-13）

### 1.8 阿里通义 Qwen — Qwen3.7-Max（已核实，多源交叉）
- 官方博客《Qwen3.7: The Agent Frontier》（https://qwen.ai/blog?id=qwen3.7 ，JS 渲染正文未能直接抓取，标题与存在性经搜索核实）；正式发布 **2026-05-20**（杭州阿里云峰会，博客 5/19-21 区间发出）。
- 型号：**Qwen3.7-Max**（文本/推理旗舰，闭源 API-only）+ **Qwen3.7-Plus-Preview**(多模态)。
- 定位（MarkTechPost 转述）："A reasoning agent model designed for long-horizon autonomous tasks, code generation, and multi-step automation"；1M 上下文（上代 Qwen3.6 Max Preview 为 256K）；闭权重。
- 可用性：Qwen Chat（chat.qwen.ai 免费）+ 阿里云百炼/Model Studio API（兼容 OpenAI/Anthropic 接口）。
- 注意：阿里云**国际站** Model Studio 文档（alibabacloud.com/help/en/model-studio/models，访问 2026-06-13）尚只列到 Qwen3-Max (qwen3-max-2026-01-23)、Qwen3.5-Plus (2026-02-15)、Qwen3.5-Flash (2026-02-23)，未列 3.7 ——国际站滞后于国内百炼，讲时勿用国际站截图。
- 国内佐证：阿里云开发者社区多篇《阿里云百炼Qwen3.7-Max》（https://developer.aliyun.com/article/1738538 等，访问 2026-06-13）。
- 第三方佐证：Artificial Analysis Intelligence Index **56.6，全球第 5、中国模型第 1**（经 developer.aliyun.com 与 digitalapplied 等多家转引 AA 榜单，访问 2026-06-13）。
- 第二来源: https://www.marktechpost.com/2026/05/21/qwen-introduces-qwen3-7-max-a-reasoning-agent-model-with-a-1m-token-context-window/ （访问 2026-06-13）

## 2. Harness 生态（slide-20/39）

### 2.1 Codex（OpenAI）（已核实，官方开发者站）
- 官方一句话："Codex is OpenAI's coding agent for software development."
- 形态：专用 App + IDE 扩展 + CLI + 云端 Web。
- 获取/价格："ChatGPT Plus, Pro, Business, Edu, and Enterprise plans include Codex."（即随 ChatGPT 订阅附带）
- 来源: https://developers.openai.com/codex （访问 2026-06-13）

### 2.2 OpenClaw（开源）（已核实，GitHub）
- 一句话（README 原文）："OpenClaw is a _personal AI assistant_ you run on your own devices. It answers you on the channels you already use."——跑在自己设备上的个人 AI 助理，接管 WhatsApp/Telegram/Slack/Discord 等消息渠道（注意：它是通用个人助理/agent 网关，不是纯写码工具）。
- 开源：MIT License；GitHub **约 37.8 万 stars**（访问 2026-06-13）。
- 安装：`npm install -g openclaw@latest`（Node 22.19+/24）或 Docker；配套 macOS/iOS/Android 应用。
- 价格：软件免费，自带模型 API key（BYO key），成本=所接模型的 API 费。
- 来源: https://github.com/openclaw/openclaw （访问 2026-06-13）

### 2.3 opencode（开源）（已核实，官网）
- 一句话（官网）："The open source AI coding agent"——终端/IDE/桌面端的开源写码 agent。
- 开源：160k+ stars、900 贡献者；隐私声明 "does not store any of your code or context data"。
- 安装：`curl -fsSL https://opencode.ai/install | bash`；亦有桌面 Beta（macOS/Win/Linux）。
- 价格：免费（含免费模型）；可接 75+ 模型供应商（Models.dev）、GitHub Copilot、ChatGPT Plus/Pro、Claude 等；增值服务 **Zen**：为写码 agent 精选/测过的模型集合（付费）。
- 来源: https://opencode.ai/ （访问 2026-06-13）

### 2.4 Gemini CLI（Google）（已核实，GitHub）
- 一句话（README）："An open-source AI agent that brings the power of Gemini directly into your terminal."
- 开源：Apache License 2.0。
- 安装：`npm install -g @google/gemini-cli` 或 `brew install gemini-cli` 或 npx。
- 价格：免费档（个人 Google 账号登录）= **60 次/分钟、1,000 次/天**；亦可用 Gemini API key / Vertex 计费升级。
- 来源: https://github.com/google-gemini/gemini-cli （访问 2026-06-13）

### 2.5 Claude Code（Anthropic）（已核实，官方文档）
- 一句话（官方 docs 原文）："Claude Code is an agentic coding tool that reads your codebase, edits files, runs commands, and integrates with your development tools. Available in your terminal, IDE, desktop app, and browser."
- 形态：终端 CLI、VS Code/JetBrains 插件、桌面 App（macOS/Win）、Web（claude.ai/code）、iOS。
- 安装：`curl -fsSL https://claude.ai/install.sh | bash`（Win: `irm https://claude.ai/install.ps1 | iex`）；或 brew / winget (Anthropic.ClaudeCode)。
- 获取/价格：多数入口需要 Claude 订阅（claude.com/pricing）或 Anthropic Console（API 计费）；CLI 与 VS Code 也支持第三方模型供应商。
- 来源: https://code.claude.com/docs/en/overview （访问 2026-06-13）

### 2.6 Codex 定价补充（官方定价页，访问 2026-06-13）
- ChatGPT **Plus $20/月**：每 5 小时窗口 GPT-5.5 本地消息 15–80 条等。
- ChatGPT **Pro 自 $100/月起**：分 5x / 20x 两档（Pro 5x = 80–400 条 GPT-5.5；Pro 20x = 300–1600 条 / 5h）；可购 ChatGPT credits 超额使用。
- 来源: https://developers.openai.com/codex/pricing

### 2.7 Claude 订阅价格（官方定价页，访问 2026-06-13）
- Free $0；Pro "$17 Per month with annual subscription discount ($200 billed up front). $20 if billed monthly."；Max "From $100 Per month"（5x / 20x 两档）。
- 来源: https://claude.com/pricing

## 3. 网页版多模态/PDF 支持（slide-16）

### 3.1 ChatGPT 网页版（官方 Help Center 经搜索核实；help.openai.com 直接抓取 403）
- 《File Uploads FAQ》（help.openai.com/en/articles/8555545）：支持上传 PDF/Word/PPT/CSV/XLSX/JPEG/PNG/TXT 等；入口=聊天框回形针；网页 chatgpt.com 与 iOS/Android 均可；文本/文档类单文件上限 2M tokens，图片单张 20MB。
- 《Visual Retrieval with PDFs FAQ》（help.openai.com/en/articles/10416312）：可读取 PDF 内嵌图片/图表（搜索摘要标注 Enterprise 支持视觉检索；Plus/Pro 的 PDF 视觉解析范围未直接核实）。
- 图片识别：支持图片上传+视觉理解（GPT 系多模态），拍照经手机 App。
- 来源: https://help.openai.com/en/articles/8555545-file-uploads-faq （403 未直抓，经搜索摘要核实，访问 2026-06-13）；https://help.openai.com/en/articles/10416312-visual-retrieval-with-pdfs-faq

### 3.2 DeepSeek 网页版（听众重点；详见 3.2.x 核实过程）
- 初步（中文媒体/社区，访问 2026-06-13）：V4 发布 5 天后（**2026-04-29**），DeepSeek **网页端"识图模式"灰度上线**——聊天框左下角功能区开启后出现回形针图标，可拖拽/粘贴图片，原生视觉理解（非旧版纯 OCR 提文字）。
- 待官方源进一步核实（见下）。

### 3.3 Claude 网页版（已核实，官方 Help Center 直抓）
- 《Upload files to Claude》（support.claude.com/en/articles/8241126）：文档 "PDF, DOCX, CSV, TXT, HTML, ODT, RTF, EPUB, JSON, XLSX"；图片 "JPEG, PNG, GIF, WebP"；"File size: 500MB per file"，"Up to 20 files per chat"；图片最大 8000x8000。
- PDF 视觉解析：**支持**——"both text and visual elements (like images, charts, and graphics) in PDFs that are under 100 pages"；超 1000 页仅文本。
- 来源: https://support.claude.com/en/articles/8241126-upload-files-to-claude （访问 2026-06-13）

### 3.4 Gemini 网页版（官方 Help 经搜索核实+直抓见下）
- 《Upload & analyze files in Gemini Apps》（support.google.com/gemini/answer/14903178）：图片 JPEG/JPG/PNG/WEBP/HEIF；文档 PDF/DOC/DOCX/RTF/TXT/Google Docs/PPTX/XLS/XLSX/CSV/TSV 等；单提示最多 10 个文件，单文件至 100MB。
- 来源: https://support.google.com/gemini/answer/14903178 （访问 2026-06-13）

### 3.5 Kimi 网页版（官方 API 文档 + 社区指南交叉）
- Kimi 网页版长期支持文件上传：**最多 50 个文件、单个 100MB**，含 pdf/doc/xlsx/ppt/txt/图片等（社区整理指南，访问 2026-06-13）。
- 官方 API 文件接口格式清单（platform.moonshot.cn/docs/api/files）：.pdf .txt .csv .doc .docx .xls .xlsx .ppt .pptx .md .jpeg .png .bmp .gif .webp …（数十种）。
- K2.6 为原生多模态（官网 "natively multimodal"，MoonViT 视觉编码器）→ 网页版图片为真视觉理解；移动端支持拍照。
- 来源: https://platform.moonshot.cn/docs/api/files （访问 2026-06-13）；社区: https://zhuanlan.zhihu.com/p/703044009 ；https://www.moonshot.ai/
- 注：网页版"50 个×100MB"为社区数字，官方帮助页未直抓 → 置信中。

### 3.2.x DeepSeek 网页版（听众重点，最终结论）
- **旧版基线（V3/R1 时代，至 2026-04）**：网页附件按钮明确标注"上传附件（只识别文字）"——图片仅 OCR 提取文字后喂给纯文本模型，**无真视觉理解**；API 连图片都不收。佐证: https://blog.csdn.net/ZHY0091/article/details/147114315 ；https://blog.csdn.net/qq_15963745/article/details/149326948 （访问 2026-06-13）
- **2026-04-24**：V4 发布（官方公告只讲文本/1M 上下文，不提视觉）。
- **2026-04-29**：网页端新增"识图模式"入口，**灰度测试**；定位"彻底超越了传统的OCR范畴……深度的内容理解与逻辑分析"，与快速/专家模式并列；实测可读医学 CT 图。来源: https://openaxo.com/innovation/deepseek-v4-vision-mode-analysis （访问 2026-06-13）
- 其后**大范围开放**（知乎热议《DeepSeek 大范围开放识图模式，你能用了吗？》 https://www.zhihu.com/question/2036432998422115875 ，访问 2026-06-13；具体全量日期待核）
- PDF 上传：一直支持（文本提取）；PDF 内图表是否走视觉解析：**未核实**。

### 3.2.y DeepSeek 识图模式——全量开放时间（已核实，主流媒体多源）
- **2026-05-09**：IT之家/新浪科技/中关村在线等一致报道《DeepSeek 大范围开放"识图模式"，正式跨入图文交互时代》——结束灰度、几乎所有账号可见入口；核心能力=图片内容识别+联网增强问答+截图一键提问，可精准解析图中文字/表格/数学公式；**尚未集成图像生成、视频理解**；短板=知识库更新滞后、反直觉视觉难题不稳。
- 来源: https://www.ithome.com/0/948/020.htm ；https://finance.sina.com.cn/tech/digi/2026-05-09/doc-inhxhcxr3627755.shtml ；https://ai.zol.com.cn/1177/11776185.html （均访问 2026-06-13）

### 3.4.x Gemini 帮助页直抓补充（访问 2026-06-13）
- 原文："Up to 10 files (subject to availability) can be uploaded in the same prompt."；普通文件 100MB、视频 2GB/5min、音频 10min；**免订阅可用**，升级 Google AI 计划获更高限额（视频 1h、音频 3h）。

---

## 4. 讲前复查清单（2026-06-14 晚建议复查的易变项）
1. **Gemini 3.5 Pro 是否在 6/14-15 突然 GA**（Google 明示"6 月推出"，撞期风险最高）→ 查 blog.google。
2. **MiniMax M3 开源权重是否已放出**（承诺 6/1 后约 10 天 ≈ 6/11 前后）→ 查 HF minimax-ai。
3. Claude Fable 5 订阅赠送窗口 6/22 截止（讲时仍在窗口内，可放心说"Pro/Max 现在就能用"）。
4. Arena/AA 名次每日波动，slide 截图当天重截。
5. 未核实项：DeepSeek 网页版 PDF 内图表是否走视觉解析（仅文字抽取为保守口径）；ChatGPT 免费档上传限额；Qwen3.7 官方博客正文细节（JS 渲染未直抓）。
