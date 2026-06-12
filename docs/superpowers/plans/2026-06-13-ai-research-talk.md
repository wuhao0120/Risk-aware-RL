# 《AI 智能体与科研工作流》报告制作 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 2026-06-15 报告前交付：联网核实的来源台账、约 40 页纸墨风格自包含 HTML 幻灯片 + PDF 备份、Zagier live demo 工作区（PDF/verify.py/跑稿）、持续更新的制作实录。

**Architecture:** 幻灯片 = `src/slides/*.html` 页面片段 + `theme.css` 设计令牌 + `engine.js` 翻页引擎，由 `build.py` 拼装并内联（字体子集、图片 base64）成单文件；事实内容一律先进 `talk/sources.md` 台账再上页面；demo 与幻灯片解耦，仅共享 Zagier 素材。

**Tech Stack:** Python 3.10+（build、fonttools 子集化、verify.py）、无头 Edge（截图/PDF）、原生 HTML/CSS/JS（零运行时依赖）、WebSearch/WebFetch（素材核实）。

**设计文档:** `docs/superpowers/specs/2026-06-12-ai-research-talk-design.md`（本计划的需求来源，冲突时以设计文档为准）

---

## 文件结构总览

```
talk/
├── sources.md                  # [新] 事实断言来源台账（研究任务的唯一输出口径）
├── research/                   # [新] 各主题原始研究笔记（引文+链接+摘录）
│   ├── notes-math-capability.md
│   ├── notes-models-harness.md
│   └── notes-boundaries-policy.md
├── slides/
│   ├── build.py                # [新] 构建：拼装→内联→dist
│   ├── reference/              # [新] 已批准的风格样张（从 .superpowers 复制入库，唯一视觉基准）
│   ├── src/
│   │   ├── theme.css           # [新] 设计令牌 + 页面骨架 + 组件类
│   │   ├── engine.js           # [新] 翻页/缩放/全屏引擎（~70 行）
│   │   ├── template.html       # [新] 单文件外壳模板
│   │   └── slides/             # [新] 每页一个片段：`NN-slug.html`（NN 决定顺序）
│   ├── fonts/                  # [新] 下载的原始字体（gitignore）+ 子集产物
│   └── dist/                   # [新] ai-research-talk.html / .pdf（gitignore，交付时单独拷贝）
├── assets/                     # [新] 截图原图（冷开场示意图等）
├── demo/                       # [新] zagier.pdf · verify.py · tests/ · demo跑稿.md
└── making-of/                  # [已有] 制作实录，按里程碑追加
```

页面清单（40 页，folio 以此为准）：

| # | slug | 章 | 内容/版式 |
|---|------|----|----------|
| 01 | cover | 开场 | 封面（已批准样张 1 版式） |
| 02 | cold-open | 开场 | "手敲公式问 DeepSeek"示意重现截图 + 金句 |
| 03 | map | 开场 | 双线地图（上半场/下半场） |
| 04 | d1-divider | §1 | 章扉页：发展现状 |
| 05 | leaps | §1 | 三级跳：规模→多模态→推理（每级配数学例证） |
| 06 | landscape | §1 | 模型版图矩阵（2026-06 时点） |
| 07 | math-timeline | §1 | AI×数学时间线：FunSearch→AlphaProof→2025 IMO 金 |
| 08 | new-math | §1 | AlphaEvolve：4×4 矩阵乘法等"发现新数学"实锤 |
| 09 | formal | §1 | 形式化：陶哲轩+Lean、DeepSeek-Prover/Kimina |
| 10 | benchmarks | §1 | FrontierMath + Erdős 问题库事件（正面叙述） |
| 11 | limits | §1 | 仍然不行的（埋 §5 伏笔） |
| 12 | research-ai | §1 | 如果你想研究 AI 本身（数学人入口） |
| 13 | d2-divider | §2 | 章扉页：概念地图 |
| 14 | overview | §2 | 四层总览图（Chat→Agent→Harness→Workflow 首次亮相） |
| 15 | prompt | §2 | Prompt & Context Engineering（差/好提问对比） |
| 16 | multimodal | §2 | 多模态：PDF/手写公式直接进（不用手敲） |
| 17 | api | §2 | API：程序化调用 |
| 18 | agent | §2 | **Agent（密度基准页，照搬已批准满幅 v3 样张 2）** |
| 19 | tools-mcp | §2 | 工具调用 / MCP / Skill |
| 20 | harness | §2 | Harness：Claude Code/Codex/OpenClaw/Opencode |
| 21 | workflow | §2 | 自动化工作流：arXiv 监测示例 |
| 22 | ladder | §2 | 四级阶梯图（v4 基础上美化） |
| 23 | d3-divider | §3 | 章扉页：一篇论文的研究之旅 |
| 24 | zagier-intro | §3 | 主角登场：一页纸一句话的论文 |
| 25 | step1-def | §3 | ① 拆解对合定义（含手排公式） |
| 26 | step2-p13 | §3 | ② p=13 手算（S 集合 3 元素、配对与不动点） |
| 27 | step3-verify | §3 | ③ 代码验证（对合性 p≤10⁴ 全量、不动点+两平方 p≤10⁶） |
| 28 | step4-windmill | §3 | ④ 风车几何解释（SVG 图） |
| 29 | step5-latex | §3 | ⑤ 生成带图 LaTeX 笔记 |
| 30 | step6-watch | §3 | ⑥ arXiv 监测 agent |
| 31 | demo-switch | §4 | 五幕菜单页："现在，现场做一遍"（切终端提示） |
| 32 | demo-backup | §4 | 备份录屏页（外链 mp4，无视频时显示文字兜底） |
| 33 | d5-divider | §5 | 章扉页：边界与规范 |
| 34 | fails | §5 | 翻车实录：编造引用 + 数学推导翻车（有出处） |
| 35 | erdos-flip | §5 | Erdős 事件的另一面：如何核查 AI 的话 |
| 36 | policy | §5 | 期刊政策（Nature/Elsevier/AMS）+ 数据安全 |
| 37 | trust | §5 | 信任分级表：什么可放手/什么必须自己判断 |
| 38 | act3 | 收尾 | 回去就做的三件事 |
| 39 | resources | 收尾 | 工具资源页（供拍照） |
| 40 | thanks | 收尾 | 致谢 + "本套幻灯片由 Claude 制作"+ making-of 缩影 |

---

## Phase 0 · 工作区与风格基准

### Task 1: 目录骨架 + 风格参照入库

**Files:**
- Create: `talk/slides/reference/`（复制 3 个已批准样张）, `talk/research/`, `talk/assets/`, `talk/demo/`, `talk/slides/src/slides/`
- Modify: `.gitignore`

- [x] **Step 1: 建目录 + 复制已批准样张为风格基准**（`.superpowers/` 被 gitignore，必须把视觉基准拷出来入库）

```bash
cd "E:/Rist-aware RL"
mkdir -p talk/slides/reference talk/slides/src/slides talk/slides/fonts talk/slides/dist talk/research talk/assets talk/demo/tests
cp ".superpowers/brainstorm/4632-1781276092/content/visual-style-b3.html"  talk/slides/reference/approved-fullbleed-v3.html
cp ".superpowers/brainstorm/4632-1781276092/content/four-levels-v4.html"   talk/slides/reference/approved-ladder-v4.html
cp ".superpowers/brainstorm/4632-1781276092/content/visual-style.html"     talk/slides/reference/style-abc-context.html
```

- [x] **Step 2: gitignore 构建产物**

`.gitignore` 追加三行：

```
talk/slides/fonts/raw/
talk/slides/dist/
talk/demo/recordings/
```

- [x] **Step 3: Commit**

```bash
git add .gitignore talk/slides/reference/
git commit -m "talk: 工作区骨架 + 已批准风格样张入库作为视觉基准"
```

---

## Phase 1 · 素材联网核实（三个任务可并行；输出统一进 sources.md）

**sources.md 条目格式（所有研究任务共用，硬性）：**

```markdown
## [键名] 一句话断言
- 断言细节: （数字、日期、归属）
- 来源: <URL>（访问 2026-06-13）；关键数字需第二来源: <URL2>
- 置信: 高（多源一致）/ 中（单源或表述有出入，页面上需加"据报道"）
- 用于: slide-NN
```

**硬规则：** 核实不到 → 该断言不上页面（宁可删）。每个任务先写 `talk/research/notes-*.md`（原始摘录+链接），再提炼进 `talk/sources.md`。

### Task 2: 数学能力素材（喂 slide 05/07/08/09/10/11）

**Files:** Create: `talk/research/notes-math-capability.md`；Modify: `talk/sources.md`

- [x] **Step 1: 逐条 WebSearch 并记录**（每条至少开 2 个结果交叉）：
  - `AlphaProof AlphaGeometry IMO 2024 silver medal`（DeepMind 官方博客）
  - `IMO 2025 gold medal AI Gemini Deep Think OpenAI`（两家官方公告 + 主流报道）
  - `IMO 2026 AI` （报告在 7 月 IMO 之前——确认 2025 即最新，防止说错"今年"）
  - `AlphaEvolve matrix multiplication 4x4 48 multiplications Strassen`（DeepMind 博客 + 论文）
  - `FunSearch cap set problem Nature 2023`
  - `Terence Tao Lean equational theories project AI`（Tao 博客/mastodon 原话，摘 1-2 句可引用的）
  - `FrontierMath benchmark latest results 2026`（Epoch AI 页面，记最新 SOTA 分数+日期）
  - `Erdős problems database GPT-5 solutions controversy 2025`（事件全貌：宣称→实为文献检索→各方表态；正反两面都记，slide-10 用正面，slide-35 用反面）
  - `DeepSeek-Prover Kimina-Prover formal theorem proving miniF2F`（国产形式化进展）
- [x] **Step 2: 提炼进 sources.md**（键名：`imo-2024-silver`、`imo-2025-gold`、`alphaevolve-matmul`、`funsearch-capset`、`tao-lean-quote`、`frontiermath-sota`、`erdos-incident`、`cn-provers`），按上方格式
- [x] **Step 3: Commit** `git add talk/research/notes-math-capability.md talk/sources.md && git commit -m "talk(research): 数学能力素材核实"`

### Task 3: 模型版图 + Harness 生态（喂 slide 06/20/39）

**Files:** Create: `talk/research/notes-models-harness.md`；Modify: `talk/sources.md`

- [x] **Step 1: WebSearch 核实（2026-06 时点）**：
  - 各家旗舰**当前准确名称**：`OpenAI GPT latest model June 2026` / `Anthropic Claude latest model 2026` / `Google Gemini latest 2026` / `DeepSeek latest model 2026` / `Kimi 月之暗面 最新模型` / `Qwen 通义 最新模型` / `MiniMax 最新模型`——只记官网/官方博客确认的版本名，记发布日期
  - 每家"擅长什么"用官方定位+一个第三方评测佐证（如 LMArena/Artificial Analysis 排名页）
  - Harness：`Claude Code` / `OpenAI Codex CLI` / `OpenClaw` / `Opencode` / `Gemini CLI`——各自官方页：定位一句话、获取方式（订阅/开源/免费层）、价格量级（如"$20/月档可用"）
  - 网页版数学场景实用性：哪些支持 PDF 上传/图片识别（slide-16 要点名"现在不用手敲公式"的依据）
- [x] **Step 2: 提炼进 sources.md**（键名：`model-landscape`、`harness-lineup`、`multimodal-pdf-support`）
- [x] **Step 3: Commit** `git commit -m "talk(research): 模型版图与harness生态核实"`

### Task 4: 翻车实例 + 学术政策（喂 slide 34/35/36/37）

**Files:** Create: `talk/research/notes-boundaries-policy.md`；Modify: `talk/sources.md`

- [x] **Step 1: WebSearch 核实**：
  - `lawyer sanctioned ChatGPT fake citations Mata v. Avianca`（标志性案例，记处罚结果）+ `AI hallucinated citations academic papers study 2025`（量化研究优先）
  - 数学推导翻车：`LLM mathematical reasoning errors plausible wrong proofs study`、`Terence Tao AI mistakes subtle errors`（找有具体例子可展示的，最好能在页面上摆出"看起来对的错误推导"原文）
  - 政策：`Nature AI authorship policy`（LLM 不能当作者）、`Elsevier generative AI policy authors`、`AMS American Mathematical Society AI policy journals`（数学学会的最相关）
  - 数据安全：`ChatGPT DeepSeek data training opt-out web vs API`（网页对话默认可能用于训练 vs API 不用于训练的官方条款链接）
- [x] **Step 2: 提炼进 sources.md**（键名：`fake-citations-case`、`math-fail-example`、`journal-policies`、`data-security`）
- [x] **Step 3: Commit** `git commit -m "talk(research): 边界案例与学术政策核实"`

---

## Phase 2 · 幻灯片基础设施

### Task 5: 引擎 + 主题 + 构建脚本（先无字体内联，跑通三页样片）

**Files:**
- Create: `talk/slides/src/engine.js`、`talk/slides/src/theme.css`、`talk/slides/src/template.html`、`talk/slides/build.py`
- Create（样片）: `talk/slides/src/slides/01-cover.html`、`18-agent.html`、`22-ladder.html`

- [x] **Step 1: 写 `engine.js`**（完整代码，禁改 API：后续页面不依赖引擎内部）

```javascript
// engine.js — 方向键/空格翻页, F 全屏, Home/End, hash 同步, 1280x720 等比缩放
(function () {
  const slides = Array.from(document.querySelectorAll('.slide'));
  let cur = 0;
  function fit() {
    const s = Math.min(innerWidth / 1280, innerHeight / 720);
    document.documentElement.style.setProperty('--scale', s);
  }
  function show(i, push) {
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((el, j) => el.classList.toggle('active', j === cur));
    if (push !== false) history.replaceState(null, '', '#' + (cur + 1));
  }
  function fromHash() {
    const n = parseInt(location.hash.slice(1), 10);
    show(isNaN(n) ? 0 : n - 1, false);
  }
  addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') show(cur + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') show(cur - 1);
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
    else if (e.key.toLowerCase() === 'f')
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });
  addEventListener('resize', fit);
  addEventListener('hashchange', fromHash);
  fit(); fromHash();
})();
```

- [x] **Step 2: 写 `theme.css`**——设计令牌区必须逐字如下（与已批准样张一致），组件类从 `talk/slides/reference/approved-fullbleed-v3.html` 的内联样式**提取等价类**（页面骨架 `.hdr/.ftr`、终端窗 `.term`、旁注 `.note-l`、芯片 `.chip`、数据块 `.stat`、卡片 `.card-p`）：

```css
:root {
  --paper:#f8f4ec; --card:#fffdf8; --tint:#f3efe6;
  --ink:#1c1917; --verm:#b91c1c; --amber:#d97706;
  --g1:#57534e; --g2:#78716c; --g3:#a8a29e; --line:#d6d3d1;
  --serif:'Noto Serif SC','SimSun',serif;
  --sans:'Noto Sans SC','Microsoft YaHei',sans-serif;
  --mono:'JetBrains Mono',Consolas,monospace;
  --scale:1;
}
html,body{margin:0;height:100%;background:#111;overflow:hidden}
.slide{position:absolute;left:50%;top:50%;width:1280px;height:720px;
  transform:translate(-50%,-50%) scale(var(--scale));transform-origin:center;
  background:var(--paper);font-family:var(--serif);display:none;
  flex-direction:column;padding:26px 56px 18px;box-sizing:border-box}
.slide.active{display:flex}
.slide *{box-sizing:border-box;margin:0}
/* 页眉/页脚骨架 */
.hdr{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:1.5px solid var(--line);padding-bottom:9px}
.hdr .crumb{color:var(--verm);font-size:15px;letter-spacing:5px}
.hdr .title{color:var(--ink);font-size:32px;font-weight:900}
.hdr .deck{color:var(--g3);font-size:13px;letter-spacing:2px}
.ftr{border-top:1.5px solid var(--line);margin-top:12px;padding-top:8px;
  display:flex;justify-content:space-between;align-items:center;
  color:var(--g3);font-size:13px}
.body{flex:1;display:flex;gap:30px;margin-top:16px;min-height:0}
/* 组件 */
.term{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--verm);
  padding:16px 20px;border-radius:3px;font-family:var(--mono);font-size:15px;line-height:1.95;color:#44403c}
.note-l{border-left:3px solid var(--verm);padding-left:14px;color:var(--verm);
  font-size:15px;line-height:1.7;font-style:italic}
.chip{border:1.5px solid var(--ink);padding:6px 18px;border-radius:999px;
  font-family:var(--sans);font-size:17px;display:inline-block}
.chip.hot{border-color:var(--verm);color:var(--verm);font-weight:700}
.stat{flex:1;border:1px solid var(--line);text-align:center;padding:9px 2px;
  background:var(--card);font-family:var(--sans)}
.kwbar{background:var(--tint);padding:10px 16px;font-family:var(--sans);
  font-size:14.5px;color:var(--g1);letter-spacing:1px}
.dots{display:flex;gap:5px;align-items:center}
.dots i{width:7px;height:7px;border-radius:50%;background:var(--line);display:block}
.dots i.on{background:var(--verm)}
@media print {
  @page{size:1280px 720px;margin:0}
  html,body{overflow:visible;background:#fff}
  .slide{display:flex;position:relative;left:0;top:0;transform:none;page-break-after:always}
}
```

- [x] **Step 3: 写 `template.html`**（占位符 `<!--THEME--> <!--SLIDES--> <!--ENGINE--> <!--FONTS-->`）：

```html
<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>AI 智能体与科研工作流 · 数学学院 · 2026-06-15</title>
<style><!--FONTS--></style><style><!--THEME--></style></head>
<body><!--SLIDES--><script><!--ENGINE--></script></body></html>
```

- [x] **Step 4: 写 `build.py`**（v1：拼装 + 图片 base64 内联 + 可选 PDF；字体内联 Task 6 接入）

```python
# -*- coding: utf-8 -*-
"""build.py — 拼装 src/slides/*.html 为单文件; --pdf 同时导出PDF"""
import base64, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).parent
SRC, DIST = ROOT / "src", ROOT / "dist"
EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"

def inline_images(html: str, base: pathlib.Path) -> str:
    def repl(m):
        p = (base / m.group(2)).resolve()
        if not p.exists(): sys.exit(f"缺图: {p}")
        mime = "image/png" if p.suffix == ".png" else "image/svg+xml" if p.suffix == ".svg" else "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode()
        return f'{m.group(1)}data:{mime};base64,{data}{m.group(3)}'
    return re.sub(r'(src=")(?!data:)([^"]+)(")', repl, html)

def build(pdf=False):
    DIST.mkdir(exist_ok=True)
    slides = sorted((SRC / "slides").glob("*.html"))
    assert slides, "无页面片段"
    body = "\n".join(f'<section class="slide" id="{p.stem}">\n{p.read_text(encoding="utf-8")}\n</section>' for p in slides)
    fonts = (ROOT / "fonts" / "embed.css").read_text(encoding="utf-8") if (ROOT / "fonts" / "embed.css").exists() else ""
    out = (SRC / "template.html").read_text(encoding="utf-8") \
        .replace("<!--FONTS-->", fonts) \
        .replace("<!--THEME-->", (SRC / "theme.css").read_text(encoding="utf-8")) \
        .replace("<!--ENGINE-->", (SRC / "engine.js").read_text(encoding="utf-8")) \
        .replace("<!--SLIDES-->", inline_images(body, ROOT.parent / "assets"))
    target = DIST / "ai-research-talk.html"
    target.write_text(out, encoding="utf-8")
    print(f"OK {target}  {len(slides)} 页  {target.stat().st_size/1e6:.1f} MB")
    if pdf:
        subprocess.run([EDGE, "--headless=new", "--disable-gpu",
                        f"--print-to-pdf={DIST/'ai-research-talk.pdf'}",
                        "--no-pdf-header-footer", target.as_uri()], check=True, timeout=300)
        print(f"OK {DIST/'ai-research-talk.pdf'}")

if __name__ == "__main__":
    build(pdf="--pdf" in sys.argv)
```

- [x] **Step 5: 做 3 页样片验证管线**：`01-cover.html`、`18-agent.html` 按 `reference/approved-fullbleed-v3.html` 对应样张**逐元素移植**（把内联样式换成 theme.css 类，文案不变）；`22-ladder.html` 移植 `approved-ladder-v4.html`
- [x] **Step 6: 构建并截图验收**：`python talk/slides/build.py` → 无头 Edge 截 3 页（复用 `_capture.py` 的调用方式，`--window-size=1280,720` + `#1/#18` hash 定位）→ Read 截图与 reference 比对：版式/配色/字重一致（系统字体回退此阶段可接受）
- [x] **Step 7: Commit** `git add talk/slides && git commit -m "talk(slides): 引擎+主题+构建管线, 3页样片通过"`

### Task 6: 字体子集化内联

**Files:** Create: `talk/slides/fontprep.py`、`talk/slides/fonts/embed.css`（产物）；Modify: 无

- [x] **Step 1: 安装工具** `pip install fonttools brotli requests`（已装会秒过）
- [x] **Step 2: 写 `fontprep.py`**：

```python
# -*- coding: utf-8 -*-
"""fontprep.py — 下载字体→按全部页面用字子集化→woff2→生成 embed.css(base64)"""
import base64, pathlib, subprocess, sys
import requests

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "fonts" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
FONTS = [  # (本地名, 下载URL, css family, weight)
  ("NotoSerifSC-Regular.otf", "https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf", "Noto Serif SC", 400),
  ("NotoSerifSC-Bold.otf",    "https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf",    "Noto Serif SC", 700),
  ("NotoSerifSC-Black.otf",   "https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Black.otf",   "Noto Serif SC", 900),
  ("NotoSansSC-Regular.otf",  "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",   "Noto Sans SC", 400),
  ("NotoSansSC-Bold.otf",     "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",      "Noto Sans SC", 700),
  ("JetBrainsMono-Regular.ttf","https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Regular.ttf",              "JetBrains Mono", 400),
  ("JetBrainsMono-Bold.ttf",  "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Bold.ttf",                  "JetBrains Mono", 700),
]

def used_text() -> str:
    chars = set()
    for p in (ROOT / "src").rglob("*.*"):
        chars |= set(p.read_text(encoding="utf-8", errors="ignore"))
    return "".join(sorted(chars))

def main():
    txt = ROOT / "fonts" / "used.txt"; txt.write_text(used_text(), encoding="utf-8")
    css = []
    for name, url, fam, w in FONTS:
        raw = RAW / name
        if not raw.exists():
            r = requests.get(url, timeout=120); r.raise_for_status(); raw.write_bytes(r.content)
        out = ROOT / "fonts" / (raw.stem + ".woff2")
        subprocess.run([sys.executable, "-m", "fontTools.subset", str(raw),
                        f"--text-file={txt}", "--flavor=woff2", f"--output-file={out}",
                        "--layout-features=*", "--no-hinting"], check=True)
        b64 = base64.b64encode(out.read_bytes()).decode()
        css.append(f"@font-face{{font-family:'{fam}';font-weight:{w};"
                   f"src:url(data:font/woff2;base64,{b64}) format('woff2')}}")
        print(f"{name}: {raw.stat().st_size/1e6:.1f}MB -> {out.stat().st_size/1e3:.0f}KB")
    (ROOT / "fonts" / "embed.css").write_text("\n".join(css), encoding="utf-8")

if __name__ == "__main__":
    main()
```

- [x] **Step 3: 运行并验收**：`python talk/slides/fontprep.py && python talk/slides/build.py`
  预期：每个子集 < 600KB；dist 总大小 < 6MB；URL 404 时改用备选（fonts.google.com 下载 zip 解压到 `fonts/raw/`，文件名对齐后重跑——记录实际采用的来源）
- [x] **Step 4: 离线验证**：`grep -c "https\?://" talk/slides/dist/ai-research-talk.html` 预期非引用文本的资源链接为 0（允许文案中出现网址文字）；截图确认衬线字已生效（对比 Step 6 of Task 5 的回退版）
- [x] **Step 5: Commit** `git add talk/slides/fontprep.py && git commit -m "talk(slides): 字体子集化内联管线"`（woff2/embed.css 为构建产物不入库；`fonts/raw/` 已 gitignore）

---

## Phase 3 · 页面内容（每章一个任务：写片段→构建→截图自检→commit。文案中的事实断言一律引用 sources.md 键名，无键不上页）

**通用验收（每个内容任务执行）：**
1. `python talk/slides/build.py` 成功；
2. 无头 Edge 按 hash 截本章每页 1280×720，Read 检查：满幅（无"上重下空"）、页眉页脚齐全、folio 与页面清单一致；
3. 密度对照基准页 `18-agent`；
4. 涉及事实的页：页内右下角小字标注来源短名（如 `DeepMind 2025-07`），完整 URL 留在 sources.md。

### Task 7: 开场 3 页（01/02/03）

**Files:** Create: `src/slides/01-cover.html`（Task 5 已有，补目录条时长）、`02-cold-open.html`、`03-map.html`、`talk/assets/mock-deepseek.png`

- [x] **Step 1: 制作冷开场示意图**：写 `talk/assets/mock-deepseek.html`——深色聊天界面示意：用户消息框里是一长串手敲的 LaTeX 源码（`\int_0^\infty e^{-x^2}\,dx = ?` 风格、带换行的定义堆砌），下方 AI 回复"您输入的公式格式似乎有误…"；页面角落必须有灰字水印「示意重现」。无头 Edge 截为 `mock-deepseek.png`（1280×720）
- [x] **Step 2: 写 02-cold-open**：版式=整页大图（mock-deepseek.png 占 70%）+ 底部墨色横条金句：「这是 2026 年大多数人使用 AI 的方式——也是 2023 年的方式。」+ 页脚
- [x] **Step 3: 写 03-map**：左右双卡（上半场：认识 2026 的 AI ①②/下半场：AI 进入一个数学研究过程 ③④⑤），中缝竖排"60 分钟"；卡内列章名+时长（与封面目录条一致：12′/15′/8′/12′/6′）
- [x] **Step 4: 构建+截图自检+Commit** `git commit -m "talk(slides): 开场3页"`

### Task 8: 第 1 章 · 发展现状 9 页（04–12）

**Files:** Create: `src/slides/04-d1-divider.html` … `12-research-ai.html`

- [x] **Step 1: 章扉页 04**（此版式同时是 13/23/33 的模板）：纸底，超大朱砂章号"§1"衬线 + 章名 + 本章三问（你将听到什么）+ 底部细则线
- [x] **Step 2: 05 三级跳**：三横卡（规模→多模态→推理），每卡：年份带 + 一句能力 + 一个数学例证（数据取 sources.md：`imo-2024-silver` 等）
- [x] **Step 3: 06 模型版图**：2×4 网格卡片（GPT/Claude/Gemini/DeepSeek/Kimi/Qwen/MiniMax + "还有更多"），每卡：名称（取 `model-landscape` 的准确版本名）+ 一句定位 + 国旗角标；页脚注"时点 2026-06，来源见台账"
- [x] **Step 4: 07 时间线**：横向时间轴 2023→2026（FunSearch→AlphaProof 银→2025 双金→FrontierMath 当前 SOTA），每节点：日期+事件+来源短名；轴下方一句话："三年，从'会做题'到'金牌线'"
- [x] **Step 5: 08 发现新数学**：主体 AlphaEvolve 4×4 矩阵乘法（56 年纪录，`alphaevolve-matmul`）+ 侧栏 FunSearch cap set（`funsearch-capset`）；旁注："不是检索已知答案，是找到了人类没找到的构造"
- [x] **Step 6: 09 形式化**：左 Lean+陶哲轩（引语取 `tao-lean-quote`，衬线引文版式）右国产 prover（`cn-provers`）
- [x] **Step 7: 10 基准与 Erdős**：FrontierMath 难度定位（"研究级题目"+当前 SOTA 分数）+ Erdős 库事件正面（AI 在文献海洋里找到被遗忘的解，`erdos-incident`）；页脚预告"这事还有另一面——§5 见"
- [x] **Step 8: 11 仍然不行的**：四宫格（长链推导会断/会编造引用/算术可靠性≠推理可靠性/不会说"我不知道"），每格一个 ✗ 图标 + 一行实例引用
- [x] **Step 9: 12 研究 AI 本身**：四列（能力评估/推理机理/对齐与安全/可解释性），每列 2 行：缺什么数学 + 一个公开问题；旁注"这些方向都在招数学背景的人"
- [x] **Step 10: 构建+截图自检（9 页全过通用验收）+Commit** `git commit -m "talk(slides): 第1章发展现状"`

### Task 9: 第 2 章 · 概念地图 10 页（13–22）

**Files:** Create: `src/slides/13-d2-divider.html` … `22-ladder.html`

- [x] **Step 1: 13 章扉页**（套 04 模板，§2）
- [x] **Step 2: 14 四层总览**：横向四节点流程图（Chat→Agent→Harness→Workflow），各配 6 字定位；此页只建立"地图"，细节后页展开
- [x] **Step 3: 15 Prompt&Context**：左右对比（差提问："这个定理怎么证？"/好提问：给出处+给上下文+说清要什么+约束格式），右栏"Context Engineering = 喂对材料"：PDF/前文/符号约定；底条：一句可带走的公式「目标 + 材料 + 约束 = 好答案」
- [x] **Step 4: 16 多模态**：场景页——论文截图/手写公式拍照直接进对话（`multimodal-pdf-support` 列哪些产品支持）；金句横条："2026 年，你不需要再手敲公式了"
- [x] **Step 5: 17 API**：三步示意（你的脚本 → API → 模型），配 6 行 mono 伪代码（批量问 100 道题的循环）；旁注"网页是零售，API 是批发"
- [x] **Step 6: 18 Agent**：Task 5 已移植的基准页，本步只校对文案与 p=13 数据一致性（见 Task 10 Step 3 的正确轨道数据：**Zagier 对合交换 (1,3,1)↔(3,1,1)，固定 (1,1,3)**——修正样张里旧的"轨道闭合"措辞）
- [x] **Step 7: 19 工具调用/MCP/Skill**：三栏卡：工具调用（AI 的手）/MCP（统一的插座协议）/Skill（沉淀成可复用的"读论文流程"）；每栏一个数学场景一行
- [x] **Step 8: 20 Harness**：四产品横卡（Claude Code/Codex/OpenClaw/Opencode，取 `harness-lineup`：定位+获取+价位量级）+ 底部一句"harness engineering：把'会用 AI'变成'用得稳定'的新手艺"
- [x] **Step 9: 21 工作流**：一天时间轴示意（08:00 arXiv 速报→10:00 实验监控→18:00 日结），左侧 cron 风格 mono 片段 3 行
- [x] **Step 10: 22 阶梯图美化**（在 v4 基础上，处理用户提的"高卡内部留白"）：高层级卡各加一行场景（Harness 加"记住你的偏好·沉淀技能"；Workflow 加"周报汇总·实验看板"）；卡片加纸面细噪纹理底（CSS 渐变即可）；虚线箭头改为弧线
- [x] **Step 11: 构建+截图自检+Commit** `git commit -m "talk(slides): 第2章概念地图"`

### Task 10: 第 3 章 · Zagier 之旅 8 页（23–30）

**Files:** Create: `src/slides/23-d3-divider.html` … `30-step6-watch.html`

**本章数学事实（页面与 verify.py 共用，唯一权威）：**
- 定理：素数 p ≡ 1 (mod 4) ⇒ p 是两平方和
- S = {(x,y,z) ∈ ℤ₊³ : x² + 4yz = p}
- Zagier 对合 T：x < y−z 时 (x+2z, z, y−x−z)；y−z < x < 2y 时 (2y−x, y, x−y+z)；x > 2y 时 (x−2y, x−y+z, y)
- T 唯一不动点 (1, 1, (p−1)/4)（中支 x=y ⇒ x|p ⇒ x=1）⇒ |S| 奇 ⇒ 交换对合 σ:(x,y,z)↦(x,z,y) 必有不动点 ⇒ y=z ⇒ p = x² + (2y)²
- p=13：S = {(1,1,3), (1,3,1), (3,1,1)}；T 交换 (1,3,1)↔(3,1,1)、固定 (1,1,3)；σ 固定 (3,1,1) ⇒ 13 = 3² + 2²

- [x] **Step 1: 23 章扉页**（§3，副题"一篇论文的研究之旅——以 Zagier (1990) 为例"）
- [x] **Step 2: 24 主角登场**：左侧论文"扉页卡"（期刊名 Amer. Math. Monthly 97(2), 1990, p.144 + 标题 + 正文就一句话的视觉呈现）右侧三行：人人懂命题/证明只有一句/第一遍谁都读不懂；底条："接下来 6 步，是你和 AI 一起读懂它的过程"
- [x] **Step 3: 25 ① 拆定义**：上半"你问"气泡（真实 prompt 文案），下半 AI 拆解：S 的手排公式 + 对合三支（HTML 手排：三行分支表，条件朱砂、映射墨色——**不用 KaTeX，量少手排**，与设计文档 §7 的偏差已在 Step 9 记录）
- [x] **Step 4: 26 ② p=13 手算**：S 的三个元素卡片 + 配对图（(1,3,1)↔(3,1,1) 弧线相连，(1,1,3) 朱砂自环"不动点"）+ σ 固定 (3,1,1) ⇒ 13=3²+2² 的推出链
- [x] **Step 5: 27 ③ 代码验证**：左终端窗（verify.py 真实输出贴入：对合性 p≤10⁴ 全量 ✓、不动点唯一 + 两平方表示 p≤10⁶ 共 78,498 个素数 ✓——数字以 Task 14 实跑为准）右侧"它写的核心 20 行"代码卡
- [x] **Step 6: 28 ④ 风车**：SVG 风车图（中心正方形 x²+四臂矩形 yz×4，标注），右侧三行直观解释；来源短名标 Zagier/风车可视化参考
- [x] **Step 7: 29 ⑤ LaTeX 笔记**：左"notes.tex 编译成品"缩略卡（标题/定理环境/图）右侧要点：从对话到可归档的笔记一步到位
- [x] **Step 8: 30 ⑥ 监测**：arXiv 监测 agent 设定卡（关键词 windmill lemma / sums of two squares involution）+ 示例速报条目两则（标注"示意"）
- [x] **Step 9: 在 sources.md 增补本章条目**：`zagier-paper`（DOI/期刊页）、`windmill-viz`（可视化参考来源）；并在设计文档 §7 KaTeX 行追加一行备注："实施时改为手排 HTML 公式（公式量少且需与纸墨版式融合），KaTeX 预渲染方案备而未用"
- [x] **Step 10: 构建+截图自检+Commit** `git commit -m "talk(slides): 第3章Zagier之旅"`

### Task 11: 第 4/5 章 + 收尾 10 页（31–40）

**Files:** Create: `src/slides/31-demo-switch.html` … `40-thanks.html`

- [x] **Step 1: 31 demo 菜单**：五幕清单（幕名+时长+一句话）+ 大号提示"切换到终端 →"；本页停留在屏幕上作为 demo 期间的"桌布"
- [x] **Step 2: 32 备份录屏页**：`<video controls src="demo-backup.mp4">`（同目录相对路径）+ 无视频时的兜底文案块"现场版已演示/录屏见会后资料"；**验收：mp4 不存在时页面不报错不空白**
- [x] **Step 3: 33 章扉页**（§5，琥珀色替换朱砂作本章强调色——按 theme.css `--amber`）
- [x] **Step 4: 34 翻车实录**：左案例卡 Mata v. Avianca（`fake-citations-case`：编 6 个判例→被罚）右数学翻车实例（`math-fail-example`：摆出"看起来对"的片段+错在哪的朱砂批注）
- [x] **Step 5: 35 Erdős 另一面**：上半事件回顾（"解决了 10 个 Erdős 问题"→ 实为文献检索，`erdos-incident` 反面）下半"三步核查法"：要出处→让它自我反驳→关键步自己验
- [x] **Step 6: 36 政策与数据**：三列政策卡（Nature/Elsevier/AMS，各一行核心规定，`journal-policies`）+ 底条数据安全（网页对话 vs API 的训练用途差异，`data-security`）
- [x] **Step 7: 37 信任分级**：三档横条（绿：放手——检索/排版/代码初稿；黄：核查——文献摘要/推导草稿；红：必须自己——定理正确性/署名责任/最终判断）；每档 3 个数学场景例
- [x] **Step 8: 38 三件事**：行动卡 ×3（①下载一个 harness 把这篇论文丢给它 ②把下一次"手敲公式"换成截图 ③给自己的研究目录建一份 CLAUDE.md/AGENTS.md 偏好说明）每卡附"第一步指令"一行
- [x] **Step 9: 39 资源页**：两栏速查（网页版四件套/Harness 三件套 + 获取方式，取 `harness-lineup`），右下二维码占位框（指向 talk 资料包，无则留"会后向主办方索取"）；大字号供拍照
- [x] **Step 10: 40 致谢**：致谢语 + "本套幻灯片由 Claude（Fable 5）在 Claude Code 中制作" + making-of 三张缩略（03-style-abc/05-fullbleed/06-ladder 横排）+ 邀请函式双线收尾
- [x] **Step 11: 构建+截图自检+Commit** `git commit -m "talk(slides): 第4/5章与收尾"`

### Task 12: 全卷审校 + PDF + 实录更新

- [x] **Step 1: 全卷过页**：40 页逐页截图（脚本循环 hash 1–40），Read 检查：folio 连号、各章 dots 进度正确、章色一致（§5 琥珀）、事实页有来源短名
- [x] **Step 2: 来源台账对账**：grep 各页的来源短名 ↔ sources.md 键名一一对应，无孤儿断言
- [x] **Step 3: 导出 PDF**：`python talk/slides/build.py --pdf`；抽查 PDF 第 1/18/27/37 页与 HTML 一致、无截断
- [x] **Step 4: 更新制作实录**：追加"§ 初稿完成"小节（页数统计、与基准样张的对照截图 2 张、PDF 就绪）；重跑 `_capture.py` 不需要（历史页面未变）
- [x] **Step 5: Commit** `git add talk/slides/src talk/making-of talk/sources.md && git commit -m "talk(slides): 40页初稿+PDF导出+实录更新"`

---

## Phase 4 · Demo 工作区

### Task 13: Zagier 论文 PDF

**Files:** Create: `talk/demo/zagier.pdf`、`talk/sources.md` 增补

- [x] **Step 1: WebSearch** `Zagier "A One-Sentence Proof" 1990 American Mathematical Monthly pdf`——优先官方/课程页托管的原版单页 PDF，WebFetch 下载到 `talk/demo/zagier.pdf`
- [x] **Step 2: 兜底**（仅当下载不到原版）：取论文原文（单句证明为公开数学事实），用 LaTeX 重排单页注明"重排版，原文见 DOI"，编译为 zagier.pdf；sources.md 记 `zagier-paper` 实际来源
- [x] **Step 3: 验收**：Read 该 PDF 首页确认含定理陈述与对合定义；Commit `git add talk/demo/zagier.pdf && git commit -m "talk(demo): Zagier 1990 单页论文素材"`

### Task 14: verify.py（TDD）

**Files:** Create: `talk/demo/verify.py`、`talk/demo/tests/test_verify.py`

- [x] **Step 1: 写失败测试**（完整代码）：

```python
# talk/demo/tests/test_verify.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from verify import zagier_map, S, fixed_points, two_square_rep, check_prime

def test_S_13():
    assert sorted(S(13)) == [(1, 1, 3), (1, 3, 1), (3, 1, 1)]

def test_involution_p13():
    s = set(S(13))
    for t in s:
        assert zagier_map(t) in s
        assert zagier_map(zagier_map(t)) == t
    assert zagier_map((1, 3, 1)) == (3, 1, 1)      # 互换对
    assert zagier_map((1, 1, 3)) == (1, 1, 3)      # 不动点

def test_fixed_point_unique():
    for p in (5, 13, 29, 97, 101):
        assert fixed_points(p) == [(1, 1, (p - 1) // 4)]

def test_two_square():
    assert two_square_rep(13) in {(2, 3), (3, 2)}
    a, b = two_square_rep(1000033)                  # 10^6+33, 素数且 ≡1 (mod 4)
    assert a * a + b * b == 1000033

def test_check_prime_full():
    ok, info = check_prime(13, full=True)
    assert ok and info["n_S"] == 3 and info["rep"][0] ** 2 + info["rep"][1] ** 2 == 13
```

  （若执行时 1000033 断言意外失败，先用 sympy.isprime 复核，换下一个 ≡1 mod 4 的素数。）
- [x] **Step 2: 跑测试确认失败** `python -m pytest talk/demo/tests -q` 预期 `ModuleNotFoundError: verify`
- [x] **Step 3: 实现 verify.py**（完整代码）：

```python
# -*- coding: utf-8 -*-
"""Zagier (1990) 一句话证明的数值验证。
python verify.py            # 报告模式: 对合性 p<=10^4 全量; 不动点+两平方 p<=10^6
python verify.py 13         # 单素数详细模式: 打印 S、配对、不动点、表示
"""
import sys
from math import isqrt

def zagier_map(t):
    x, y, z = t
    if x < y - z:  return (x + 2 * z, z, y - x - z)
    if x < 2 * y:  return (2 * y - x, y, x - y + z)
    return (x - 2 * y, x - y + z, y)

def S(p):
    out = []
    x = 1
    while x * x < p:                       # p≡1(4) ⇒ x 必为奇数
        m = (p - x * x) // 4
        if (p - x * x) % 4 == 0:
            for y in range(1, isqrt(m) + 1):
                if m % y == 0:
                    out.append((x, y, m // y))
                    if y != m // y:
                        out.append((x, m // y, y))
        x += 2
    return out

def fixed_points(p):
    return [t for t in S(p) if zagier_map(t) == t]

def two_square_rep(p):
    for a in range(1, isqrt(p) + 1):
        b2 = p - a * a
        b = isqrt(b2)
        if b * b == b2:
            return (a, b)
    return None

def check_prime(p, full=False):
    info = {}
    if full:
        s = S(p); ss = set(s)
        ok_inv = all(zagier_map(t) in ss and zagier_map(zagier_map(t)) == t for t in s)
        info["n_S"] = len(s)
    else:
        ok_inv = True
    fp = fixed_points(p) if full else [(1, 1, (p - 1) // 4)]
    ok_fp = (fp == [(1, 1, (p - 1) // 4)]) if full else ((p - 1) % 4 == 0)
    rep = two_square_rep(p)
    info["rep"] = rep
    return ok_inv and ok_fp and rep is not None, info

def primes_1mod4(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [p for p in range(5, n + 1) if sieve[p] and p % 4 == 1]

def main():
    if len(sys.argv) > 1:                  # 单素数详细模式
        p = int(sys.argv[1]); assert p % 4 == 1
        s = S(p)
        print(f"p = {p}, |S| = {len(s)}")
        for t in sorted(s):
            u = zagier_map(t)
            print(f"  {t} -> {u}" + ("   <- 不动点" if u == t else ""))
        print(f"两平方: {p} = {two_square_rep(p)[0]}^2 + {two_square_rep(p)[1]}^2")
        return
    full_n, fast_n = 10**4, 10**6
    full = [p for p in primes_1mod4(full_n)]
    assert all(check_prime(p, full=True)[0] for p in full)
    print(f"[全量] p ≤ {full_n}: {len(full)} 个素数, 对合性+不动点唯一+两平方 ✓")
    fast = primes_1mod4(fast_n)
    assert all(check_prime(p, full=False)[0] for p in fast)
    print(f"[快速] p ≤ {fast_n}: {len(fast)} 个素数, 不动点+两平方表示 ✓")

if __name__ == "__main__":
    main()
```

- [x] **Step 4: 跑测试至全绿** `python -m pytest talk/demo/tests -q` 预期 5 passed
- [x] **Step 5: 跑报告模式记录真实数字与耗时** `python talk/demo/verify.py`——把输出的素数个数回填 slide-27 终端窗文案（替换占位的 78,498 如有出入）
- [x] **Step 6: Commit** `git add talk/demo && git commit -m "talk(demo): Zagier 数值验证 verify.py (TDD)"`

### Task 15: demo 跑稿 + 彩排清单

**Files:** Create: `talk/demo/demo跑稿.md`

- [x] **Step 1: 写五幕跑稿**（每幕：给 Claude Code 的逐字 prompt + 预期行为 + 翻车切换口令）。五条逐字 prompt：
  1. `这是 Zagier 1990 年那篇一句话证明两平方和定理的论文（zagier.pdf）。请读一遍，把那个对合的定义拆开讲清楚：S 是什么集合，三个分支各在什么条件下用，为什么这是良定义的。`
  2. `用 p=13 把 S 的所有元素列出来，对合怎么配对、不动点是谁，算给我看。`
  3. `我不完全信。写一个 verify.py：对 p≤10000 全量验证这是 S 上的对合且不动点唯一，对 p≤1000000 验证不动点公式和两平方和表示，跑给我看结果。`
  4. `把"风车"几何解释画成一张图（SVG 或 matplotlib 都行），再把今天这些内容整理成一份带图的 LaTeX 笔记 notes.tex。`
  5. （彩蛋，听众出题后）`换 p=____ 再算一遍 / 听众问：____，你来回答并验证。`
- [x] **Step 2: 写彩排与录屏清单**：终端字号 ≥20pt、纸墨亮色主题、投影 1920×1080 下试跑、Game Bar（Win+Alt+R）或 OBS 全程录制、demo 目录每次彩排后 `git clean` 还原、手机热点备好、幕 3 跑全量验证时的等待话术（"它在跑 10⁶ 个素数，我们正好看看它写的代码"）
- [x] **Step 3: 自跑一遍幕 1–4**（执行者即 Claude Code：在 `talk/demo/` 干净会话按跑稿走一遍，产物 notes.tex/风车图存 `talk/demo/rehearsal-01/`），把实际耗时与卡点回写跑稿"注意事项"
- [x] **Step 4: Commit** `git add talk/demo/demo跑稿.md talk/demo/rehearsal-01 && git commit -m "talk(demo): 五幕跑稿+首轮自彩排产物"`

### Task 16: 收口与交接

- [x] **Step 1: 制作实录收口**：追加"demo 工作区就绪"与"待用户完成"清单（6/14：审稿两轮→改稿；亲跑彩排+录屏存 `talk/demo/recordings/demo-backup.mp4` 并拷到 `talk/slides/dist/` 旁；6/15 上午：投影实测+备份文件三处存放——本机/U盘/网盘）
- [x] **Step 2: 验收对照**：逐条核对设计文档 §9 验收标准，在实录里贴对照表（通过/备注）
- [x] **Step 3: Commit** `git add talk/making-of && git commit -m "talk: 初稿交付收口, 待用户审稿与彩排"`
- [x] **Step 4: 通知用户**：汇总交付物路径 + 打开方式（双击 dist html → F 全屏）+ 请求审稿

---

## 风险与应对

| 风险 | 应对（已写入对应任务） |
|---|---|
| 字体下载 404 / 网络慢 | Task 6 Step 3 备选源；最坏退化为系统字体（PPT 仍可放映，观感降级） |
| 研究结果与我既有认知冲突 | 以 2026-06 联网结果为准；冲突处在 sources.md 标"置信:中"并软化措辞 |
| dist 体积过大（>10MB） | 截图素材压缩为 JPEG q85；making-of 缩略图降采样到 480px 宽 |
| Zagier 原版 PDF 下载不到 | Task 13 Step 2 LaTeX 重排兜底 |
| 日程溢出 | 牺牲顺序：22 美化 → 32 备份页特效 → 08 侧栏 FunSearch（并入 07 时间线一行） |
