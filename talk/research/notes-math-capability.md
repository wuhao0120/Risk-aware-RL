# AI数学能力现状 — 原始摘录笔记

用途：2026-06-15 中国数学学院报告《AI智能体与科研工作流》。
说明：每条目含原始英文摘录 + URL + 访问日期。逐条增量写入。

---

## 1. imo-2024-silver — AlphaProof + AlphaGeometry 2 在 IMO 2024 达银牌标准

**来源1（一手）**: DeepMind官方博客，发布于 2024-07-25
URL: https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/ （访问 2026-06-13）

原文摘录：
> "Together, these systems solved four out of six problems from this year's International Mathematical Olympiad"
> "Our system achieved a final score of 28 points, earning a perfect score on each problem solved — equivalent to the top end of the silver-medal category"
> "This year, the gold-medal threshold starts at 29 points"（即差1分到金牌线）
> "Our solutions were scored according to the IMO's point-awarding rules by prominent mathematicians Prof Sir Timothy Gowers, an IMO gold medalist and Fields Medal winner, and Dr Joseph Myers, a two-time IMO gold medalist and Chair of the IMO 2024 Problem Selection Committee."
> "AlphaProof solved two algebra problems and one number theory problem by determining the answer and proving it was correct." / "AlphaGeometry 2 proved the geometry problem"

细节：AlphaProof解P1、P2、P6（P6是当年最难题，仅5名选手解出），AlphaGeometry 2解P4（几何）；两道组合题未解出。28/42=满分解出4题（每题7分）。

**来源2（同行评审）**: Nature论文 "Olympiad-level formal mathematical reasoning with reinforcement learning"（AlphaProof，2025年发表）
URL: https://www.nature.com/articles/s41586-025-09833-y （访问 2026-06-13）

---

## 2. imo-2025-gold — 2025年两家模型达IMO金牌线（35/42）

### 2a. Google DeepMind — Gemini Deep Think（官方认证）
**来源（一手）**: DeepMind官方博客，发布于 2025-07-21
URL: https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/ （访问 2026-06-13）

原文摘录：
> "solved five out of the six IMO problems perfectly, earning 35 total points, and achieving gold-medal level performance"
> "We can confirm that Google DeepMind has reached the much-desired milestone, earning 35 out of a possible 42 points — a gold medal score."（IMO官方对成绩的确认声明，引于DeepMind博客）
> "our advanced Gemini model operated end-to-end in natural language, producing rigorous mathematical proofs directly from the official problem descriptions"
> "all within the 4.5-hour competition time limit"

要点：与2024年AlphaProof需先形式化不同，2025年Gemini Deep Think端到端自然语言作答；成绩由IMO官方协调员认证（博客标题含"officially"）。

### 2b. OpenAI — 实验性推理模型（自行评分，非IMO官方认证）
**来源（一手）**: OpenAI官方X账号 + 研究员Alexander Wei的X线程，2025-07-19
URL: https://x.com/OpenAI/status/1946594928945148246 （访问 2026-06-13，经搜索快照确认）

原文摘录（OpenAI官方X）：
> "We achieved gold medal-level performance on the 2025 International Mathematical Olympiad with a general-purpose reasoning LLM! Our model solved world-class math problems—at the level of top human contestants. A major milestone for AI and mathematics."

**来源2**: Simon Willison博客（2025-07-19，转录Alexander Wei线程）
URL: https://simonwillison.net/2025/Jul/19/openai-gold-medal-math-olympiad/ （访问 2026-06-13）
> "the model solved 5 of the 6 problems on the 2025 IMO" / "The model earned 35/42 points in total, enough for gold!"
> "three former IMO medalists independently graded the model's submitted proof, with scores finalized after unanimous consensus"
> "the IMO gold LLM is an experimental research model. We don't plan to release anything with this level of math capability for several months"

要点（讲稿可用的差异）：OpenAI于2025-07-19先宣布，评分由三位前IMO奖牌得主独立评定（非IMO官方）；DeepMind于2025-07-21宣布且为IMO官方认证。两家均为35/42、解出5/6题、自然语言、4.5小时限时。

---

## 3. imo-2026-pending — 确认2026-06前无更新的IMO结果

**来源（一手）**: IMO官网 editions 页面
URL: https://www.imo-official.org/editions/2026/ （访问 2026-06-13）

原文摘录：
> Location: "Shanghai, People's Republic of China"; Dates: "July 10 - 21"（2026年，第67届）

结论：IMO 2026将于2026-07-10至07-21在上海举行（中国继1990年后第二次主办）。报告日期2026-06-15早于比赛，故"AI在IMO的最新成绩"截至报告日仍为2025年金牌线结果，无更新。

---

## 4. alphaevolve-matmul — AlphaEvolve破Strassen 1969纪录 + 其他数学发现

**来源1（一手）**: DeepMind官方博客，发布于 2025-05-14
URL: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ （访问 2026-06-13）

原文摘录：
> "AlphaEvolve's procedure found an algorithm to multiply 4x4 complex-valued matrices using 48 scalar multiplications, improving upon Strassen's 1969 algorithm"
> "we applied the system to over 50 open problems in mathematical analysis, geometry, combinatorics and number theory"
> "In roughly 75% of cases, it rediscovered state-of-the-art solutions" / "And in 20% of cases, AlphaEvolve improved the previously best known solutions"
> "AlphaEvolve discovered a configuration of 593 outer spheres and established a new lower bound in 11 dimensions"（kissing number问题，此前纪录592）

**来源2**: the-decoder报道（确认48 vs Strassen的49、11维kissing number 593打破此前592）
URL: https://the-decoder.com/alphaevolve-is-google-deepminds-new-ai-system-that-autonomously-creates-better-algorithms/ （访问 2026-06-13）

**延伸（讲稿可提）**: 数学家随后改进——arXiv:2506.13242 "A non-commutative algorithm for multiplying 4x4 matrices using 48 non-complex multiplications"（将AlphaEvolve的复数48次乘法改进为非复数48次），https://arxiv.org/pdf/2506.13242

注意事项（严谨表述）：AlphaEvolve的48次乘法算法是**复数域**上的；Strassen 1969递归方案为49次。这是该问题56年来首次改进。

---

## 5. funsearch-capset — FunSearch：LLM对开放数学问题的首个新发现

**来源1（一手）**: DeepMind官方博客，发布于 2023-12-14
URL: https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/ （访问 2026-06-13）

原文摘录：
> "This work represents the first time a new discovery has been made for challenging open problems in science or mathematics using LLMs."
> "FunSearch generated solutions - in the form of programs - that in some settings discovered the largest cap sets ever found. This represents the largest increase in the size of cap sets in the past 20 years."
> "Today, in a paper published in Nature... we introduce FunSearch"（Nature论文同日2023-12-14上线）
> 博客提及陶哲轩曾称cap set问题为他的"favorite open question"（出自陶2007年博文）。

**来源2（同行评审论文）**: Romera-Paredes et al., "Mathematical discoveries from program search with large language models", Nature 625, 468–475 (2024)，2023-12-14在线发表
URL: https://www.nature.com/articles/s41586-023-06924-6 （cookie墙，未能直接抓取；访问 2026-06-13）
作者版PDF（可公开访问）: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf

具体数字（经Wikipedia/作者版PDF/GitHub多方确认）：
- 8维cap set：FunSearch发现大小**512**的cap set，此前最大已知构造为**496**；
- FunSearch产出的是"生成该集合的程序"而非裸解，提升可解释性；
- 另在在线装箱(bin packing)问题上发现优于既有启发式的算法。
第二来源: https://en.wikipedia.org/wiki/FunSearch ；GitHub: https://github.com/google-deepmind/funsearch

---

## 6. tao-lean-quote — 陶哲轩：Equational Theories Project + AI辅助数学引语

### 6a. Equational Theories Project (ETP) 完成情况
**来源（一手）**: 陶哲轩博客 "The Equational Theories Project: Advancing Collaborative Mathematical Research at Scale"，2025-12-09
URL: https://terrytao.wordpress.com/2025/12/09/the-equational-theories-project-advancing-collaborative-mathematical-research-at-scale/ （访问 2026-06-13）

原文摘录：
> "there turn out to be 4694 equational laws that involve at most four invocations of the magma operation"，共 "22,028,942 implications of this type to settle"
> 项目 "resolved all the implications informally after two months, and have them completely formalized in Lean after a further five months."
> 关于AI的角色（诚实的限定，讲稿很有用）: "modern AI tools did not play a major role in this project (but it was largely completed in 2024, before the most recent advanced models became available); while they could resolve many implications, the older 'good old-fashioned AI' of automated theorem provers were far cheaper to run and already handled the overwhelming majority."

时间线：2024-09下旬启动 → 约2个月完成全部非形式判定 → 再约5个月（2025年4月，启动后200余天）全部2200余万条蕴涵在Lean中形式化完毕。成果论文 arXiv:2512.07087（2025-12），陶+33位合作者。
论文URL: https://arxiv.org/abs/2512.07087
项目导览（启动期）: https://terrytao.wordpress.com/2024/10/12/the-equational-theories-project-a-brief-tour/

### 6b. 可引用原话（英文原文）
**引语1 — o1"研究生"评价**（一手，Mastodon API原文确认）：
> "The experience seemed roughly on par with trying to advise a mediocre, but not completely incompetent, (static simulation of a) graduate student."
— Terence Tao, Mathstodon, 2024-09-13（评OpenAI o1）
URL: https://mathstodon.xyz/@tao/113132503432772494 （原帖页面两次超时，经API端点 https://mathstodon.xyz/api/v1/statuses/113132503432772494 取得原文；访问 2026-06-13）
注：陶随后发澄清帖（https://mathstodon.xyz/@tao/113145334235914812 ），强调不应把研究生简化为一维"能力值"，引用时建议一并提及。

**引语2 — "2026-level AI"预言**（一手）：
> "When integrated with tools such as formal proof verifiers, internet search, and symbolic math packages, I expect, say, 2026-level AI, when used properly, will be a trustworthy co-author in mathematical research, and in many other fields as well."
— Terence Tao, "Embracing change and resetting expectations", Microsoft Unlocked, 2023-06-12
URL: https://unlocked.microsoft.com/ai-anthology/terence-tao/ （访问 2026-06-13）
讲稿亮点：2023年预言"2026级AI"，恰可在2026年的报告里回看对照。

---

## 7. frontiermath-sota — FrontierMath定位与当前最高分

### 定位（一手，arXiv论文 2024-11-07提交）
**来源**: Glazer et al., "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI", arXiv:2411.04872
URL: https://arxiv.org/abs/2411.04872 （访问 2026-06-13）

原文摘录：
> "a benchmark of hundreds of original, exceptionally challenging mathematics problems crafted and vetted by expert mathematicians"
> 发布时（2024-11）: "Current state-of-the-art AI models solve under 2% of problems, revealing a vast gap between AI capabilities and the prowess of the mathematical community."

结构：Tier 1–3共300题（本科~高年级研究生探索性难度）+ Tier 4共50题（研究级，单题需研究者数小时至数天）；全对全错计分（最终答案对得1分，无过程分）。
Epoch页面: https://epoch.ai/frontiermath ；Tier 4: https://epoch.ai/benchmarks/frontiermath-tier-4

### 当前最高分（2026年中）
**来源1（Epoch AI数据，经Our World in Data CSV）**:
URL: https://ourworldindata.org/grapher/ai-frontiermath-over-time.csv （访问 2026-06-13；数据源标注 "Epoch AI (2026)"）
- GPT-5.5 Pro (high reasoning): **52.4%**（2026-04）← 当前最高
- GPT-5.5: 51.7%（2026-04）；GPT-5.4 Pro: 50%（2026-03）
- Claude Opus 4.x: 43.8%（2026-04）；Kimi K2.6: 39%（2026-04，国产模型）；Gemini 3.5 Flash: 39%（2026-05）

**来源2（聚合站，2026-06-09快照）**: https://benchlm.ai/benchmarks/frontierMath （访问 2026-06-13，数字一致：52.4/51.7/50/43.8）

### 重要警示（讲稿必须提，slide-10/35可用）
**Epoch AI官方X，2026-05-11**: https://x.com/EpochAIResearch/status/2053995435870892048 （访问 2026-06-13）
> "We are conducting an AI-assisted review of FrontierMath: Tiers 1-4. This has flagged fatal errors in about a third of problems, and we believe most of these flags to be valid. We will release updated scores on a corrected dataset after completing a thorough human review."

即：约1/3题目被（AI辅助审查）标记存在致命错误，官方将发布修订数据集上的更新分数——引用52.4%时需注明"修订前数据，Epoch正在复核"。耐人寻味的点：参与找错的正是前沿模型本身（基准的审计问题）。
Epoch简报佐证: https://epochai.substack.com/p/the-epoch-brief-may-15-2026

---

## 8. erdos-incident — 2025年10月Erdős问题库事件（过度宣称 vs 文献检索价值）

### 事件经过（2025-10中下旬）
**来源1**: TechCrunch "OpenAI's 'embarrassing' math"，2025-10-19
URL: https://techcrunch.com/2025/10/19/openais-embarrassing-math/ （访问 2026-06-13）
**来源2**: Futurism，2025-10-21
URL: https://futurism.com/artificial-intelligence/openai-researcher-deletes-tweet （访问 2026-06-13）

时间线与原话：
1) OpenAI副总裁 **Kevin Weil** 发帖（约2025-10-17/18，后删除）：
> "GPT-5 found solutions to 10 (!) previously unsolved Erdős problems and made progress on 11 others."
2) erdosproblems.com 维护者、数学家 **Thomas Bloom** 回应，称该帖是
> "a dramatic misrepresentation"
并解释其网站上"open"仅意味着 "I personally am unaware of a paper which solves it"；实情是：
> "GPT-5 found references, which solved these problems, that I personally was unaware of."（GPT-5找到的是**已有文献**，非新解）
3) **Demis Hassabis**（DeepMind CEO，2025-10-19）: "This is embarrassing"；**Yann LeCun**: "hoisted by their own GPTards"。
4) Weil删帖并澄清；OpenAI研究员 **Sebastien Bubeck**（最初演示GPT-5检索Erdős问题文献者）承认 "only solutions in the literature were found"，但辩护文献检索本身有价值：
> "I know how hard it is to search the literature"

### 正面价值与后续（讲稿平衡视角）
- 正面：GPT-5确实找到了连库维护者都不知道的既有文献——强大的文献检索能力对数学家是真实有用的（Bloom本人亦认可这些参考文献的价值）。
- 后续（2026-05-02，Futurism报道）：GPT-5.4对一个Erdős问题给出**此前研究者未尝试过的解法路径**（由Liam Price分享、陶哲轩验证、Stanford的Jared Lichtman分析）；Lichtman同时指出 "the raw output of ChatGPT's proof was actually quite poor"，需专家解读。即：从"误报已解"到"真实新进展"，但人类专家把关仍不可少。
URL: https://futurism.com/artificial-intelligence/mathematicians-claim-significant-discovery-using-chatgpt （访问 2026-06-13）

### 讲稿要点
反面教训：宣称"解决开放问题"前必须核查文献与问题库语义（"open"≠"unsolved"）；正面教训：AI文献检索已能超出单个专家的视野；事件双方（Weil删帖澄清、Bloom更新网站标注）都体现了纠错机制。

---

## 9. cn-provers — 国产形式化定理证明进展（Lean 4 / miniF2F）

### DeepSeek-Prover 系列（深度求索，开源）
**来源（一手）**: arXiv:2504.21801 "DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition"，2025-04-30
URL: https://arxiv.org/abs/2504.21801 ；GitHub: https://github.com/deepseek-ai/DeepSeek-Prover-V2 （访问 2026-06-13）
- DeepSeek-Prover-V2-671B：miniF2F-test **88.9%** pass ratio；PutnamBench解出**49/658**；提出新基准ProverBench
- 方法：DeepSeek-V3驱动的递归子目标分解 + RL；开源（671B与7B）
- 前作轨迹：V1.5（2024-08，arXiv:2408.08152）miniF2F-test 63.5% → V2 88.9%，一年内大幅跃升

### Kimina-Prover（Numina & Kimi团队 / 月之暗面）
**来源（一手）**: arXiv:2504.11354 "Kimina-Prover Preview"，2025-04-15提交
URL: https://arxiv.org/abs/2504.11354 ；GitHub: https://github.com/MoonshotAI/Kimina-Prover-Preview （访问 2026-06-13）
- 摘要原话：模型 "sets a new state-of-the-art on the miniF2F benchmark, reaching **80.7%** with pass@8192"（首个公开结果破80%）；基座Qwen2.5-72B，大规模RL训练
- 后续：Kimina-Prover-72B 于2025-07-10正式发布；配合测试时RL（TTRL）报告miniF2F **92.2%**（见对比来源 arXiv:2511.03108 "miniF2F-Lean Revisited" 及模型卡）

### Seed-Prover（字节跳动Seed团队）
**来源（一手）**: arXiv:2507.23726 "Seed-Prover"，2025-07-31提交
URL: https://arxiv.org/abs/2507.23726 （访问 2026-06-13）
- 摘要原话：在Lean中 "fully prove 5 out of 6 problems"（IMO 2025，赛后形式化证明）；"proves **78.1%** of formalized past IMO problems"；"saturates MiniF2F"（接近打满）；PutnamBench "over 50%"

### 讲稿要点
形式化证明赛道（Lean 4）2024–2025年的领先结果主要由中国团队产出：DeepSeek（开源671B）、月之暗面/Numina（Kimina）、字节Seed；miniF2F从2024年的~63%推进到2025年的接近饱和；与DeepMind AlphaProof（IMO 2024银牌、Nature 2025）同属"形式化路线"，与Gemini/GPT的自然语言路线互补。

