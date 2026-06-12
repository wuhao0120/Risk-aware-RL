# AI使用边界：翻车实例与学术政策 —— 原始摘录与来源笔记

- 核实人: 研究助理（Claude）
- 核实日期: 2026-06-13（下列所有URL访问日期均为 2026-06-13）
- 用途: 2026-06-15 数学学院报告，slides 34–36
- 原则: 凡未能从可靠来源核实的点，明确标注【未核实】。本文所有英文摘录均为来源原文。

---

## 1. 编造引用标志案例：Mata v. Avianca（slide-34）

### 1.1 案件基本事实（已核实，高置信）

- **案件**: Mata v. Avianca, Inc., No. 1:22-cv-01461 (PKC) (S.D.N.Y.)；制裁判决书（Opinion & Order, Document 54）发布于 **2023年6月22日**，公开引用为 **678 F. Supp. 3d 443 (S.D.N.Y. 2023)**。
- **法官**: P. Kevin Castel（纽约南区联邦地区法院）。
- **事由**: 原告律师用 ChatGPT 做法律检索，向法院提交了包含 **6个不存在判例** 的书状。6个假判例名称：
  1. Varghese v. China Southern Airlines
  2. Shaboon v. Egyptair
  3. Petersen v. Iran Air
  4. Martinez v. Delta Airlines
  5. Estate of Durden v. KLM Royal Dutch Airlines
  6. Miller v. United Airlines
- **处罚**: 对律师 Steven A. Schwartz、Peter LoDuca 及其律所 Levidow, Levidow & Oberman P.C. **连带（jointly and severally）罚款 5,000 美元**；并责令向被假判例冒名的每位真实法官逐一寄送信函（附判决书、听证记录与含假判例的宣誓书）。假"Miller"判例甚至冒用了第二巡回上诉法院 Barrington D. Parker 法官之名。
- **细节（讽刺点，适合幻灯片）**: 律师曾问 ChatGPT 案例是否真实，ChatGPT 回答这些案例 "indeed exist"、"can be found in reputable legal databases such as LexisNexis and Westlaw"（确实存在，可在 LexisNexis 和 Westlaw 等权威法律数据库中找到）。

### 1.2 Castel 法官判词原文（英文摘录）

> "Many harms flow from the submission of fake opinions. The opposing party wastes time and money in exposing the deception. The Court's time is taken from other important endeavors. The client may be deprived of arguments based on authentic judicial precedents. There is potential harm to the reputation of judges and courts whose names are falsely invoked as authors of the bogus opinions and to the reputation of a party attributed with fictional conduct."
> （另一处）"...it promotes cynicism about the legal profession and the American judicial system. And a future litigant may be tempted to defy a judicial ruling by disingenuously claiming doubt about its authenticity."

平衡性引语（法官并未一概否定AI工具）：

> "technological advances are commonplace and there is nothing inherently improper about using a reliable artificial intelligence tool" —— 但律师必须履行 "gatekeeping role"，"ensure the accuracy of their filings"。

法官还认定律师存在 "subjective bad faith"（主观恶意），并称其中一个假判例的法律分析为 "gibberish"（不知所云）。

### 1.3 来源URL（访问 2026-06-13）

- Wikipedia 案件条目（案号、$5,000、时间线、ChatGPT保证"indeed exist"）: https://en.wikipedia.org/wiki/Mata_v._Avianca,_Inc.
- Justia 判决原文（Document 54, S.D.N.Y. 2023）: https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/575368/54/
- CourtListener 案卷: https://www.courtlistener.com/docket/63107798/mata-v-avianca-inc/
- 判决书PDF（678 F.Supp.3d 443, Berkeley Law镜像）: https://www.law.berkeley.edu/wp-content/uploads/archive/2025/12/Mata-v-Avianca-Inc.pdf
- Legal Dive（"Many harms…"全段引文出处）: https://www.legaldive.com/news/lawyer-fake-chatgpt-cases-sanctions-fines-lawyers-chatgpt-fake-cases/653741/
- ACC 案例分析（律师姓名、律所、法官引语）: https://www.acc.com/resource-library/practical-lessons-attorney-ai-missteps-mata-v-avianca
- LegalClarity（6个假判例完整名单、连带罚款）: https://legalclarity.org/what-happened-in-the-mata-v-avianca-case/

### 1.4 量化研究：学术论文中AI编造参考文献的比例（已核实，高置信）

**研究A（首选，规模最大、最新）: Lancet 通讯（2026年5月）**

- 出处: Topaz 等（哥伦比亚大学，Maxim Topaz 领衔），"Fabricated citations: an audit across 2·5 million biomedical papers"，发表于 The Lancet（2026年5月，通讯/letter形式）。
- 样本: PubMed 索引的 **200多万篇论文、9,700万条引文**（用AI工具辅助筛查，并区分真编造与格式差异）。
- 结果: 在约 **2,800篇论文中发现约4,000条编造引文**；含编造引文的论文比例逐年上升：
  - 2023年: **1/2,828**
  - 2025年: **1/458**（约6倍增长）
  - 2026年前7周: **1/277**
- STAT原文摘录: "In 2023, 1 in 2,828 papers contained one or more fabricated references, but in 2025 that number had reached 1 in 458 — a sixfold increase in frequency. During the first seven weeks of 2026, the rate reached 1 in 277 papers."
- URL（访问 2026-06-13）:
  - Lancet 原文: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(26)00603-3/fulltext
  - STAT News 报道（2026-05-07，数字与方法细节）: https://www.statnews.com/2026/05/07/lancet-study-finds-steep-rise-fraudulent-citations-academic-papers/
  - Retraction Watch（2026-05-07）: https://retractionwatch.com/2026/05/07/one-in-277-pubmed-indexed-papers-in-2026-shows-fabricated-references-says-analysis/
  - Nature 新闻报道: https://www.nature.com/articles/d41586-026-00748-w

**研究B（跨库大规模审计）: arXiv 预印本（2026-05-08）**

- 出处: Zhenyue Zhao, Yihe Wang, Toby Stuart, Mathijs De Vaan, Paul Ginsparg, Yian Yin, "LLM hallucinations in the wild: Large-scale evidence from non-existent citations", arXiv:2605.07723（提交于2026-05-08）。
- 样本: **250万篇论文中的1.11亿条参考文献**（arXiv、bioRxiv、SSRN、PubMed Central 四大库）。
- 结果: 仅2025年就发现约 **146,932条**"幻觉引文"（保守估计）；据 phys.org 报道，约 **78.8%** 的不存在引文通过了 arXiv 的moderation（编辑把关）；编造引文并非集中于少数论文，而是"广泛分布、每篇少量"。
- phys.org原文摘录: "The hallucinated citations were not limited to a handful of bad apples but appeared across many papers, each containing a small number of fake references." / "An estimated 78.8% of non-existent citations still passed through" (arXiv moderation)。
- URL（访问 2026-06-13）:
  - arXiv 摘要页: https://arxiv.org/abs/2605.07723
  - phys.org 报道（2026-05）: https://phys.org/news/2026-05-ai-generated-fake-citations-scientific.html

**研究C（模型层面对照实验，2024年）: JMIR**

- 出处: Chelli M, et al. "Hallucination Rates and Reference Accuracy of ChatGPT and Bard for Systematic Reviews: Comparative Analysis", Journal of Medical Internet Research 2024;26:e53164。
- 样本: 11篇系统综述场景、33个提示词、共分析 **471条** 模型生成的参考文献。
- 结果（参考文献幻觉率）: **GPT-3.5: 39.6% (55/139)；GPT-4: 28.6% (34/119)；Bard: 91.4% (95/104)**。
- URL（访问 2026-06-13）:
  - JMIR 原文: https://www.jmir.org/2024/1/e53164
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/38776130/

---

## 2. 数学推导翻车的可展示实例（slide-34）

### 2.1 首选展示例：GSM-NoOp"猕猴桃"题（最短、最直观，已逐字核实）

- 出处: Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio, Farajtabar（Apple团队）, "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models", arXiv:2410.05229（2024-10-07提交）。
- **题目原文（可直接上幻灯片）**:

> "Oliver picks 44 kiwis on Friday. Then he picks 58 kiwis on Saturday. On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average. How many kiwis does Oliver have?"

- **模型的错误推理（论文Figure原文，o1-mini与Llama3-8B同样犯错）**:

> Llama3-8B: "88 - 5 = 83 kiwis. Now, let's add up the total: 44 + 58 + 83 = 185 kiwis"
> o1-mini: "88 (Sunday's kiwis) - 5 (smaller kiwis) = 83 kiwis... 44 + 58 + 83 = 185"

- **错在哪（一句话）**: "five of them were a bit smaller than average"（其中5个略小）是与计数无关的干扰从句——小猕猴桃也是猕猴桃，正确答案是 44 + 58 + 88 = **190**；模型却"看起来很有条理地"扣掉了5个。论文原话: 这类No-Op语句 "carry no operational significance"，"does not contribute to the reasoning chain needed to reach the final answer"。
- **量化结论（论文摘要原文）**: "Adding a single clause that seems relevant to the question causes significant performance drops (**up to 65%**) across all state-of-the-art models, even though the clause doesn't contribute to the reasoning chain needed for the final answer."
- URL（访问 2026-06-13）:
  - 摘要页: https://arxiv.org/abs/2410.05229
  - HTML全文（含猕猴桃例原文）: https://arxiv.org/html/2410.05229

### 2.2 佐证数据：USAMO 2025 整卷证明评测"Proof or Bluff?"

- 出处: Petrov et al., "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad", arXiv:2503.21934（2025-03-27提交，2025年USAMO赛题发布数小时内由人类专家评卷）。
- 摘要原文: "Our results reveal that all tested models struggled significantly: **only Gemini-2.5-Pro achieves a non-trivial score of 25%, while all other models achieve less than 5%**."（注意：模型在只看最终数值答案的AIME类基准上得分很高，但要求严格写证明时近乎全军覆没——"看起来严谨、实则不及格"的整体量化证据。）
- URL（访问 2026-06-13）: https://arxiv.org/abs/2503.21934

### 2.3 权威背书引语：陶哲轩（Terence Tao）

- 场合: Lex Fridman Podcast #472（2025年6月发布）。
- 流传引文（经X帖与The Decoder报道一致转录）:

> "the AI-generated proofs, they look superficially flawless... the [RL] has actually trained them to produce text that *looks like* what is correct... the errors are often really subtle and then when you spot them, they're really stupid."

- 同场合相关表述（The Decoder报道转录）: "where the AI really struggles right now is knowing when it's made a wrong turn"；Tao称AI目前缺少数学家的"metaphorical mathematical smell"（隐喻意义上的数学嗅觉）。
- 【注意/部分未核实】: 官方文字稿页面抓取时被截断，未能在官方transcript中逐字定位该段；以上措辞综合自X帖视频转录与The Decoder报道，两者一致。幻灯片引用时建议标注"转录自播客访谈"。
- URL（访问 2026-06-13）:
  - 官方transcript页: https://lexfridman.com/terence-tao-transcript/
  - The Decoder 报道（2025-06-16）: https://the-decoder.com/math-genius-terence-tao-says-that-ai-still-cant-smell-bad-math/
  - X帖（含访谈片段视频）: https://x.com/nabeelqu/status/1934285488614945035

---

## 3. Erdős问题事件："过度宣称 → 撤回澄清"时间线（slide-35）

> 分工说明：本节只核实"翻车—澄清"时间线与各方原话；正面价值（文献检索能力、2026年后续真实成果）由同事负责。

### 3.1 时间线（已核实，高置信；具体到日的精度见注）

1. **2025年10月17日前后**: OpenAI研究员（Mark Sellke、Sebastien Bubeck）在X上发帖称GPT-5在Erdős问题上取得进展；时任OpenAI VP **Kevin Weil** 转发并宣称（原帖已删除，措辞经多家媒体存档转述一致）:
   > "GPT-5 found solutions to 10 (!) previously unsolved Erdős problems and made progress on 11 others."
2. **随即（约24–48小时内）**: 维护 erdosproblems.com 的数学家 **Thomas Bloom** 回应，称该说法是
   > "a dramatic misrepresentation"
   并澄清: 网站上标"open"只意味着 "**I personally am unaware of a paper which solves it**"（我个人不知道有论文解决了它）；实际情况是
   > "GPT-5 found references, which solved these problems, that I personally was unaware of."
   （GPT-5找到的是早已发表、只是他本人不知道的文献——是文献检索，不是新数学。）
3. **同期反应**: Google DeepMind CEO **Demis Hassabis**: "**this is embarrassing**"；Meta首席AI科学家 **Yann LeCun** 嘲讽: "Hoisted by their own GPTards."
4. **撤回与澄清**: Weil 删除原帖；Bubeck 承认 "**only solutions in the literature were found**"（只找到了文献中已有的解法），但辩称这仍有价值: "I know how hard it is to search the literature."
5. **媒体报道时间**: The Decoder 2025-10-18；TechCrunch 2025-10-19（标题即 "OpenAI's 'embarrassing' math"）；Fortune 2025-10-20；Futurism 2025-10-21。

【注/未核实】: Weil原帖与Bloom、Hassabis回应的X原始status链接未能直接核实（Weil原帖已删除；其余原帖链接未在可抓取来源中找到稳定URL），以上原话均经下列多家独立媒体一致转述，可放心引用媒体链接。

### 3.2 来源URL（访问 2026-06-13）

- TechCrunch（2025-10-19，Weil原帖措辞、Bloom原话、Hassabis/LeCun/Bubeck原话）: https://techcrunch.com/2025/10/19/openais-embarrassing-math/
- The Decoder（2025-10-18，时间线最早的整理）: https://the-decoder.com/leading-openai-researcher-announced-a-gpt-5-math-breakthrough-that-never-happened/
- Futurism（2025-10-21）: https://futurism.com/artificial-intelligence/openai-researcher-deletes-tweet
- Fortune（2025-10-20）: https://fortune.com/2025/10/20/did-openais-latest-ai-model-solve-famously-difficult-math-problems-well/

---

## 4. 期刊政策（slide-36）—— 全部来自官方政策页，逐字核实

### 4.1 Nature / Nature Portfolio

- **核心规定一句话**: LLM不满足作者资格（因无法承担责任），使用LLM必须在Methods部分声明。
- 政策页原文（英文）:
  > "Large Language Models (LLMs), such as ChatGPT, do not currently satisfy our authorship criteria. Notably an attribution of authorship carries with it accountability for the work, which cannot be effectively applied to LLMs."
  > "Use of an LLM should be properly documented in the Methods section (and if a Methods section is not available, in a suitable alternative part) of the manuscript."
- 审稿环节（同页）:
  > "...peer reviewers do not upload manuscripts into generative AI tools."
- 配套社论（2023-01-24, Nature 613, 612）原文:
  > "No LLM tool will be accepted as a credited author on a research paper." / "Researchers using LLM tools should document this use in the methods or acknowledgements sections."
- URL（访问 2026-06-13）:
  - 政策页: https://www.nature.com/nature-portfolio/editorial-policies/ai
  - 社论: https://www.nature.com/articles/d41586-023-00191-1

### 4.2 Elsevier

- **核心规定一句话**: 作者用AI须在投稿稿件中以单独声明披露；AI不得列为作者，审稿人不得把稿件上传给AI工具。
- 政策页原文（英文）:
  > "Authors should disclose the use of AI tools for manuscript preparation in a separate AI declaration statement included in their manuscript upon submission."
  > "Authors should not list AI tools as an author or co-author, nor cite AI tools as an author."
  > "Reviewers should not upload a submitted manuscript or any part of it into an AI tool as this may violate the authors' confidentiality and proprietary rights."
  > （图像）"AI tools must not be used to create or alter images that represent primary observed or experimental data that were not directly obtained in the research."
- 声明的标准位置（作者指引）: 文末参考文献之前，标题为 "Declaration of Generative AI and AI-assisted technologies in the writing process"；仅语法/拼写检查及Mendeley/EndNote/Zotero等文献管理工具无须声明。
- URL（访问 2026-06-13）:
  - 期刊生成式AI政策: https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals
  - 写作中使用AI政策: https://www.elsevier.com/about/policies-and-standards/the-use-of-generative-ai-and-ai-assisted-technologies-in-writing-for-elsevier
  - 审稿中使用AI政策: https://www.elsevier.com/about/policies-and-standards/the-use-of-generative-ai-and-ai-assisted-technologies-in-the-review-process

### 4.3 AMS（美国数学会）—— 有专门政策（很多人以为没有）

- **核心规定一句话**: AMS期刊政策（改编自COPE 2023年2月立场声明）规定AI不能列为作者、用AI须在论文中披露、编辑与审稿人不得以任何形式上传审稿论文给LLM。
- 官方政策页 "Use of Artificial Intelligence"（AMS Committee on Publications 改编采纳）原文（英文）:
  > "AI tools cannot be listed as an author of a paper. AI tools cannot meet the requirements for authorship as they cannot take responsibility for the submitted work."
  > "Authors who use AI tools in the writing of a manuscript, production of images or graphical elements of the paper, or in the collection and analysis of data, must be transparent in disclosing in the Materials and Methods (or similar section) of the paper how the AI tool was used and which tool was used."
  > "Authors are fully responsible for the content of their manuscript, even those parts produced by an AI tool, and are thus liable for any breach of publication ethics."
  > "**Editors and referees are not to upload papers under review to an LLM in any format, for any reason.**"
- 背景: AMS还设有 "Advisory Group on Artificial Intelligence and the Mathematical Community"（2023年7月由时任主席Bryna Kra授权成立），其出版专题白皮书（Harington & Silverman, 2024-01-30）确认 "the AMS is implementing guidelines and best practices for using AI in writing manuscripts"。
- URL（访问 2026-06-13）:
  - AMS官方政策页: https://www.ams.org/publications/journals/policies/UseofArtificialIntelligence
  - AMS AI顾问组: https://www.ams.org/about-us/governance/committees/artificial-intelligence
  - 出版白皮书PDF: https://www.ams.org/about-us/CPub_AI-WhitePaper.pdf
- 顺带（可作对比口头提及，来源为其投稿页转述）: Annals of Mathematics 更严格——"The Annals of Mathematics does not consider papers generated using AI products."【该句转引自搜索结果，原页面未直接抓取，引用前建议再点开 annals.math.princeton.edu 确认 → 标注：部分未核实】

---

## 5. 数据安全条款：OpenAI / Anthropic / DeepSeek（slide-36）

### 5.1 OpenAI（官方政策页逐字核实）

- **网页对话（消费者版ChatGPT）默认可用于训练**:
  > "When you use our services for individuals such as ChatGPT, Sora, or Operator, **we may use your content to train our models**."
- **API与企业版默认不训练**:
  > "**By default, we do not train on any inputs or outputs from our products for business users, including ChatGPT Team, ChatGPT Enterprise, and the API.** We offer API customers a way to opt-in to share data with us..."
- **Opt-out路径**:
  > "You can opt out of training through our privacy portal by clicking on 'do not train on my content.'"
  以及 ChatGPT 内: Settings → Data Controls → 关闭 "Improve the model for everyone"（OpenAI帮助中心Data Controls FAQ）。临时对话不用于训练: "Chats from Temporary Chat won't appear in history, use or create memories, or be used to train our models."
- URL（访问 2026-06-13）:
  - 官方政策页: https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/
  - 帮助中心同文: https://help.openai.com/en/articles/5722486-how-your-data-is-used-to-improve-model-performance
  - Data Controls FAQ: https://help.openai.com/en/articles/7730893-data-controls-faq

### 5.2 Anthropic（官方新闻页+隐私中心逐字核实）

- **消费者版（Claude Free/Pro/Max）**: 2025年8月28日条款更新后，用户**必须自行选择**是否允许数据用于训练（2025-09-28后新用户注册时选择；老用户须在**2025-10-08**前作出选择），并可随时在 Privacy Settings 更改:
  > "We're now giving users the choice to allow their data to be used to improve Claude..."
  选择允许则数据保留期延长至**5年**；不允许则维持**30天**保留期。隐私中心（2026-03-16更新）: 仅当 (1)用户选择允许、(2)对话被安全审查标记、(3)显式opt-in（如反馈/受信测试计划）时才用于训练。
- **API与商用版不训练（默认）**:
  > "**By default, we will not use your inputs or outputs from our commercial products (e.g. Claude for Work, Anthropic API, Claude Gov, etc.) to train our models.**"
  新闻页同样明确: "These updates do **not** apply to services under our Commercial Terms, including Claude for Work, Claude for Government, Claude for Education, or API use, including via third parties such as Amazon Bedrock and Google Cloud's Vertex AI."
- URL（访问 2026-06-13）:
  - 官方公告: https://www.anthropic.com/news/updates-to-our-consumer-terms
  - 隐私中心（消费者）: https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training
  - 隐私中心（商用/API）: https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training

### 5.3 DeepSeek（官方条款页逐字核实）

- **网页/App对话默认可用于训练**。Privacy Policy（Last Update: **Feb 10, 2026**）"How We Use Your Personal Data"列明:
  > "To improve and develop the Services and **to train and improve our technology, such as our machine learning models and algorithms**. Including by monitoring interactions and usage across your devices, analyzing how people are using it, and training and improving our technology."
- **Opt-out路径**（Terms of Use 第4.3条）:
  > "...we may, to a minimal extent, use Inputs and Outputs to provide, maintain, operate, develop or improve the Services or the underlying technologies supporting the Services. **If you refuse to allow us to process the data in the manner described above, you can opt out by turning off 'Improve the model for everyone'.**"
- **数据存储地**（Privacy Policy）:
  > "To provide you with our services, we directly collect, process and store your Personal Data in **People's Republic of China**."
- **API数据**:【未核实/无明确条款】DeepSeek开放平台（API）服务条款将个人信息处理指向其Privacy Policy（第5.5条），**未找到**类似OpenAI/Anthropic的"API数据默认不用于训练"的明确承诺；条款仅赋予用户对Inputs/Outputs的使用权（含用于蒸馏训练自己的模型, 4.2(3)）。幻灯片上请勿声称"DeepSeek API不用于训练"——应表述为"未作出明确承诺"。
- URL（访问 2026-06-13）:
  - Privacy Policy: https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html
  - Terms of Use（4.3条opt-out）: https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html
  - 开放平台（API）服务条款: https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html

---

## 6. 未核实点汇总（务必在讲稿中规避或加限定）

1. Erdős事件中 Weil/Bloom/Hassabis 的 X 原帖直链 —— 未核实（Weil原帖已删除）；引用请用 TechCrunch/The Decoder/Futurism 链接。
2. 陶哲轩引语的逐字版本 —— 官方transcript抓取被截断，措辞以X帖转录+The Decoder为准（两者一致）；幻灯片注明"转录自播客访谈"。
3. Annals of Mathematics 的"不收AI生成论文"原句 —— 转引自搜索结果，原页面未直接抓取。
4. DeepSeek API 数据是否用于训练 —— 官方条款无明确排除承诺，标"无明确承诺"，不要写"不用于训练"。
5. Lancet研究的"12-fold in two years"说法（个别报道使用）—— 与STAT给出的1/2,828→1/458（约6倍）口径不同，建议只用STAT/Retraction Watch的原始比例数字。
