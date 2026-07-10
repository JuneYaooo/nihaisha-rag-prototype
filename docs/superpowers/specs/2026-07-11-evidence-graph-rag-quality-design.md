# Nihaisha 分层证据图谱与高质量 RAG 设计

**日期：** 2026-07-11  
**状态：** 待用户书面审阅  
**首期范围：** 以现有 11 份 PDF 为基线，提升准确率、知识图谱质量、查询性能和可理解性；支持后续增量加入 PDF，并为音频、视频、字幕、网页和扫描图片预留统一入口。

## 1. 背景与结论

当前项目已经具备原文段落、页码、FTS、BAAI/bge-m3 dense 向量、FAISS、规则知识单元和引用输出，但仍属于可追溯原型，尚不能对“全面且准确”作出可验证承诺。

已确认的主要问题如下：

- 中文自然问句经常被全文检索当成一个完整长词，导致词法召回为空。
- `method` 和 `clinical` 回答逻辑包含针对少数示例的固定文案，能产生与问题无关的结论和追问。
- hybrid 直接累加不同量纲的分数，没有通道级融合校准和最终 reranker。
- 默认安装没有 FAISS 时会扫描全部 dense 向量，查询延迟不可接受。
- 当前“知识图谱”主要是规则抽取的知识单元集合，没有规范实体、稳定边、别名解析、多跳遍历和边级证据。
- PDF 段落存在断裂、超长、页眉污染和方剂名称噪声。
- 54 个单元测试验证了函数行为，但没有黄金问题集，也没有 Recall@K、MRR、nDCG、引用准确率和回答忠实度基线。
- README 解释了概念，但缺少可复制安装命令、快速开始、架构说明、运行模式、延迟预期和故障排查。

因此首期采用“来源可追溯的分层证据图谱”，不直接引入重型社区聚类或全局 GraphRAG。先把当前 11 份 PDF 做准、做快、做清楚，再通过稳定的增量流程扩充资料。

## 2. 目标与非目标

### 2.1 目标

1. 对当前 11 份 PDF 建立可量化的检索和引用质量基线。
2. 消除固定示例答案，所有回答只从实际检索证据生成。
3. 建立规范实体、显式关系和边级 provenance，使知识图谱可查询、可核验、可增量更新。
4. 将课程原文、古籍原文、第二手参考和派生内容严格分层。
5. 使用 dense、词法和图谱三路召回，通过 RRF 或等价的秩融合合并，再用可配置 reranker 精排。
6. FAISS 缺失时快速失败或明确降级到 text/knowledge，而不是静默进行全库 dense 扫描。
7. 支持后续按资料批次增量加入 PDF；同一模型支持将来接入音视频、字幕、网页和扫描件。
8. 用清晰文档让新用户在五分钟内完成安装、健康检查和第一次带引用查询。

### 2.2 非目标

- 首期不实现社区发现、全库主题报告或重型 GraphRAG 全局搜索。
- 首期不实现音视频转写、网页抓取和图片 OCR，只定义它们必须输出的统一文档接口。
- 首期不把未经审核的互联网内容当作答案事实。
- 首期不提供个人诊断、处方、剂量决定或自我治疗建议。
- 首期不为了追求节点数量而保留低质量实体或无证据关系。

## 3. 仓库职责

### 3.1 Runtime 仓库

`nihaisha-rag-prototype` 负责：

- 查询规范化和意图/任务识别；
- dense、词法、图谱召回；
- 融合、rerank、证据选择；
- 图谱邻居查询和原始段落回查；
- 分层引用、回答草稿和安全边界；
- 运行时诊断、统计和用户文档。

### 3.2 Builder 仓库

`nihaisha-rag-builder` 负责：

- PDF 解析、清洗、分块和质量报告；
- 文档注册、内容哈希和稳定 ID；
- 实体、关系、古籍引用候选的结构化抽取；
- 别名归一、实体消歧和证据边验证；
- embedding、FTS、FAISS 和图谱索引构建；
- staging、增量合并、回滚和发布校验；
- 生成 runtime 仓库使用的最终数据库资产。

运行时代码不得隐式修改生产数据库。所有数据库更新先经过 builder 的 staging 和验收门。

## 4. 四层证据模型

所有文档、段落、古籍条文和关系必须携带 `source_layer`：

| 层级 | 标识 | 用途 | 是否默认成为答案证据 |
|---|---|---|---|
| 课程原文 | `course_primary` | 说明倪海厦课程如何表述 | 是 |
| 古籍原文 | `classic_primary` | 对照讲课引用、转述与经典原文 | 是，单独分区展示 |
| 第二手参考 | `reference_secondary` | 术语规范、别名、校勘和关系校验 | 否，用户明确要求时才展示 |
| 派生内容 | `derived` | 实体、边、摘要、问题单元和导图 | 否，只做导航 |

任何 `derived` 结论必须能够沿 `evidence_id` 回到 `course_primary` 或 `classic_primary`。如果只由第二手参考支持，它只能标为校验线索，不能伪装成课程或古籍原文。

## 5. 统一内容模型

未来所有输入格式先转换为同一中间表示：

```text
SourceDocument
  document_id
  content_hash
  source_layer
  media_type
  canonical_title
  edition
  source_uri
  rights_note
  ingestion_version

SourceSegment
  segment_id
  document_id
  locator
  section_path
  text
  previous_segment_id
  next_segment_id
  quality_flags
```

PDF 的 `locator` 使用页码和页内顺序；音视频使用时间码；字幕使用 cue 范围；网页使用标题路径与稳定锚点；扫描图片使用页码和 OCR 区域。首期只实现 PDF 适配器，但检索、图谱和引用不得依赖 PDF 专有字段。

`document_id` 基于规范来源标识生成；`content_hash` 用于检测版本变化。数据库和 manifest 不保存本机绝对路径，发布资产只保存可移植的来源标识和显示名称。

## 6. 图谱模型

### 6.1 实体类型

首期支持以下规范类型：

- `formula`：方剂；
- `herb`：药物及炮制形式；
- `symptom`：症状；
- `sign`：体征；
- `pulse`：脉象；
- `tongue`：舌象；
- `pattern`：证候和方证；
- `six_channel`：六经；
- `organ`：脏腑；
- `pathogenesis`：病机；
- `treatment_method`：治法；
- `acupoint`：穴位；
- `meridian`：经络；
- `disease_term`：资料中的病名；
- `classic_work`、`classic_passage`：古籍和条文；
- `course_segment`：课程证据段落。

### 6.2 关系类型

首期使用受控谓词：

- `mentions`、`quotes`、`paraphrases`、`explains`；
- `indicates_pattern`、`has_symptom`、`has_sign`、`has_pulse`、`has_tongue`；
- `treated_by`、`contains_herb`、`uses_method`；
- `modifies_with`、`contraindicated_for`、`cautions_against`；
- `differentiates_from`、`belongs_to`；
- `same_as`、`alias_of`、`variant_of`、`derived_from`；
- `sourced_from`、`discussed_in`、`supported_by`；
- `agrees_with`、`differs_from`、`unverified_against`。

自由文本关系不能直接进入生产图谱。无法映射到受控谓词的候选保存在抽取 trace 中，供后续扩展 ontology。

### 6.3 边级证据

每条关系必须保存：

```text
edge_id
subject_entity_id
predicate
object_entity_id or literal_value
source_layer
evidence_segment_id
evidence_quote
extraction_method
extractor_version
confidence
review_status
created_at
```

`confidence` 表示抽取器信心，不表示事实真伪。它必须由具体信号计算或由模型输出后校准，不能继续按知识类型写死同一个值。`review_status` 取 `auto_accepted`、`needs_review`、`reviewed`、`rejected`。

### 6.4 实体归一与别名

- 保留原文 surface form，同时映射到 canonical entity。
- 繁简体、异体字和明确简称可自动归一。
- 同名异义实体必须结合类型、邻居和证据上下文消歧。
- 第二手参考可提供 alias 候选，但不能单独证明课程关系。
- 自动合并必须保留 merge trace；低置信候选保持分离并进入审核队列。

## 7. 古籍对照

古籍作为独立文档进入 `classic_primary`，必须记录书名、篇章、条文号、版本/底本和来源。讲课段落与古籍条文之间允许四种状态：

1. `quotes`：文字高度一致的直接引用；
2. `paraphrases`：含义对应但文字不同；
3. `explains`：课程对条文的解释发挥；
4. `unverified_against`：疑似关联但没有可靠匹配。

系统不得用模型记忆补全古文。无法确定版本或条文时，保留候选和待核验状态。不同版本文字不自动覆盖，而以 `variant_of` 连接并分别引用。

古籍来源采用白名单注册。引入前检查来源可靠性、版本信息、文本完整性和使用权说明。泛网页搜索只用于发现候选来源，不直接写入生产证据层。

## 8. 解析与质量门

PDF 首期构建增加以下质量检查：

- 页眉、页脚、公众号标记和重复水印检测；
- 目录页和索引页识别；
- 断裂句合并和跨页连续性处理；
- 超长段落二次切分；
- 极短残片与相邻段落合并或标记；
- OCR 重复、乱码和异常字符比例；
- 标题、页码、章节路径和前后段落链接；
- 精确重复与近重复检测；
- 方剂、药物、古文引用的抽样报告。

质量门不静默删除原文。清洗后的检索文本与原始文本分开保存；最终引用优先展示原始文本，必要时同时展示清洗说明。

## 9. 检索架构

### 9.1 查询理解

查询规范化包括繁简映射、标点与单位标准化、领域别名解析、方剂/药物/穴位识别和任务分类。任务类型包括：

- 精确原文或出处；
- 实体事实；
- 方证或概念比较；
- 跨段落关联；
- 课程与古籍对照；
- 全库主题问题；
- 临床情境下的课程资料整理。

任务分类只决定检索策略，不允许选择固定答案文案。

### 9.2 三路召回

1. **Dense**：SiliconFlow `BAAI/bge-m3`，FAISS 为运行时必需的 dense 索引。
2. **Lexical**：中文友好的 trigram/领域词切分与短语查询，保留精确短语加权。
3. **Graph**：从识别到的规范实体出发，按受控关系和证据层扩展一至两跳，再回查原始段落。

BGE-M3 的 sparse/ColBERT 能力作为后续实验项，只有在黄金集上显著提升时才进入默认路径。

### 9.3 融合与精排

- 各召回通道先独立排名，不直接相加不可比的原始分数。
- 使用 RRF 进行第一阶段融合，并记录每个结果的通道排名和命中理由。
- 对融合后的候选使用可配置 reranker；默认可接 SiliconFlow 支持的重排服务，若不可用则使用确定性的短语、实体覆盖和证据层特征重排。
- 精排后执行近重复折叠和证据多样性选择；多样性不能把明显不相关结果抬入引用区。
- 出处类问题必须要求关键实体在证据中直接出现，不能只因泛化词如“热熨”而进入主要引用。

### 9.4 降级与性能

- 启动健康检查验证 SQLite、FTS、FAISS、ID 映射和向量维度一致。
- FAISS 文件存在但 Python 模块缺失时，hybrid/vector 明确报出安装命令，不进行全库 dense 扫描。
- SiliconFlow 不可用时，允许显式降级为 lexical + graph，并在返回 trace 中标明。
- 索引加载在进程内缓存；同一查询计划共享 embedding 和索引对象。
- 首期性能目标：已安装 FAISS 且网络正常时，单次 search 的本地检索部分 P95 小于 1 秒；完整 answer 的 P95 由外部 API 延迟主导，但不得因重复构造查询而产生无界调用。

## 10. 回答与引用契约

答案按问题需要输出以下部分：

1. 总结；
2. 课程原文依据；
3. 古籍原文对照；
4. 关联知识与关系路径；
5. 尚未确认或版本差异；
6. 安全边界。

每个事实声明必须绑定引用编号。引用记录包含 `source_layer`、显示书名、页码或条文定位、原文摘录和 segment ID。关系路径示例：

```text
课程 p68 ─explains→ 桂枝汤证 ─differentiates_from→ 麻黄汤证
           └quotes→ 《伤寒论》某篇某条
```

图谱路径只帮助解释为什么检索到这些证据。若路径中的任一边没有原始证据，答案必须标为未确认，不得写成确定事实。

## 11. 增量更新

新资料不要求一次补齐。更新流程为：

1. 注册文档和内容哈希；
2. 在 staging 解析并生成质量报告；
3. 生成新段落、实体候选、关系候选和向量；
4. 对已有实体做别名解析与消歧；
5. 抽样审核新增关系和古籍关联；
6. 运行黄金集与性能回归；
7. 更新 FTS、FAISS、manifest 和数据库版本；
8. 原子发布完整资产。

删除或替换文档时，按 `document_id` 清除或重建由该文档派生的 segments、retrieval units、edges 和索引行。相同讲义、视频稿和古籍版本之间使用 `variant_of`、`derived_from` 或重复讲解关系，不进行无来源的破坏性去重。

当解析规则、ontology、抽取模型、embedding 模型或稳定 ID 算法改变时执行全量 staging 重建，而不是增量混用不同版本。

## 12. 评测与验收

### 12.1 黄金集

首期建立版本化黄金集，覆盖：

- 精确原文与页码；
- 方剂、药物、穴位和术语；
- 方证鉴别；
- 古今单位和相互矛盾表述；
- 课程与古籍对应；
- 跨文档比较；
- 无答案问题；
- 容易产生噪声的泛化词；
- 安全边界和临床情境。

每条样例记录 query、任务类型、相关 segment IDs、期望实体、允许的经典版本、应拒绝的错误证据和参考回答要点。

评测集按交付阶段启用：阶段一先覆盖现有课程库；阶段二加入图谱实体和关系标注；阶段三在古籍资料注册后加入课程—古籍对照样例。未进入当前数据版本的来源不作为发布门要求。

### 12.2 指标

- Retrieval：Recall@5、Recall@10、MRR@10、nDCG@10、Context Precision@K。
- Graph：实体精确率/召回率、关系精确率、别名合并准确率、证据覆盖率、无证据边比例。
- Answer：引用正确率、声明支持率/faithfulness、拒答正确率、课程与古籍分层准确率。
- Quality：页眉污染率、短残片率、异常长段率、明显方剂噪声率。
- Performance：search P50/P95、answer P50/P95、embedding/rerank 调用次数、FAISS 健康状态。

首期发布门要求所有已知固定答案错误归零；黄金集不能只包含当前代码已有的“一钱、木香饼、下利恶心”示例。

## 13. 错误处理与可观测性

每次 JSON 查询返回可选 trace：

```text
normalized_query
task_type
recognized_entities
retrieval_channels
channel_ranks
fusion_score
rerank_score
selected_evidence
rejected_evidence_and_reason
degraded_features
latency_breakdown
```

API 密钥、Authorization header 和完整敏感环境变量不得进入 trace。外部服务失败时返回明确的降级状态和恢复建议。数据库 schema、manifest、FAISS 或映射不一致时停止 vector 查询，避免返回悄然错误的结果。

## 14. 代码边界

当前 4,000 行以上的单文件需要按职责渐进拆分，目标边界如下：

```text
nihaisha_kg/
  models.py          # 文档、段落、实体、边、证据类型
  normalization.py   # 中文、别名、单位和查询规范化
  retrieval/
    lexical.py
    vector.py
    graph.py
    fusion.py
    rerank.py
  graph/
    schema.py
    store.py
    traversal.py
  answers/
    intent.py
    evidence.py
    synthesis.py
    safety.py
  diagnostics.py
  cli.py
```

拆分按测试驱动渐进完成，不一次性重写整个模块。公共 CLI 保持兼容；新增 `doctor`、`evaluate` 和可选 `--trace` 接口。

## 15. 文档与易用性

README 重写为用户路径：

1. Git LFS 与安装；
2. SiliconFlow 配置；
3. 推荐安装 `.[faiss]`；
4. `doctor` 健康检查；
5. 三个可复制查询示例；
6. search、answer、trace 的输出解释；
7. 证据层级和安全边界；
8. 常见错误与性能预期；
9. 如何分批补充新资料。

维护者文档说明 runtime/builder 边界、staging、质量门、黄金集、数据库发布和回滚。

## 16. 分阶段交付

### 阶段一：正确性、性能和评测地基

- 删除固定示例回答；
- 修正中文词法召回；
- 通道级 RRF、可配置 reranker 和严格证据过滤；
- FAISS 健康检查与快速失败；
- 建立黄金集、评测 CLI 和查询 trace；
- 改进 README 与快速开始。

### 阶段二：高质量证据图谱

- 新 schema、规范实体、受控关系和边级证据；
- 别名解析、实体消歧、图谱召回和关系路径；
- builder 质量门和现有 11 份 PDF 的 staging 重建；
- 清除低质量 guide/formula 节点。

### 阶段三：古籍原文对照

- 注册首批权威古籍来源和版本；
- 建立引用、转述、解释与版本差异关系；
- 分层检索、分层引用和古籍对照评测。

### 阶段四：持续增量与其他媒介

- 分批加入新 PDF；
- 在确有资料时实现字幕、音视频、网页和 OCR 适配器；
- 数据规模和问题类型达到需要时，再评估全局主题层与社区摘要。

每个阶段都产生可运行、可评测的版本。阶段二和阶段三不得绕过阶段一的黄金集与发布门。

## 17. 设计决策摘要

- 选择分层证据图谱，不把派生节点当作事实来源。
- 当前 11 份 PDF 先做质量基线，不等待未来资料一次补齐。
- PDF 为首期唯一完整实现的输入，其他媒介共享统一内容接口。
- SiliconFlow 继续用于 BGE-M3 query embedding，并允许可配置 rerank。
- FAISS 是默认 hybrid/vector 路径的运行依赖；缺失时不再静默全库扫描。
- 先做局部证据图谱和跨来源关联，暂缓重型全局 GraphRAG。
- 古籍原文与课程讲解分别引用，冲突和版本差异显式保留。
- 所有新增资料经过 staging、质量报告、黄金集回归和原子发布。
