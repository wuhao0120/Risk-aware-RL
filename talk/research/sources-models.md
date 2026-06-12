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
