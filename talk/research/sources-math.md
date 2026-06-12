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

