# Nihaisha 知识结构二期设计

**日期：** 2026-07-11

**状态：** 待用户书面审阅

**范围：** 以现有 11 份 PDF 和已发布的可移植 runtime 资产为输入，建设证据分层、规范实体、类型化关系和可量化图谱质量；安全测试只保留通用边界，不成为项目主体。

## 1. 目标与原则

本阶段的核心交付不是增加更多查询特判，而是把现有段落、知识单元和导览节点迁移成可验证的知识结构。系统必须回答四个问题：一条知识来自哪份资料和哪个原文段落、它描述了什么规范实体、实体之间是什么受控关系、该关系经过何种抽取和审核。

设计遵循以下原则：

- 原文证据是一等数据，派生知识只负责组织和导航。
- 每条可展示关系必须能沿证据 ID 回到原始段落。
- 课程讲解、古籍原文、二手参考和派生内容在 schema 中显式分层。
- 实体规范化保留原文写法，不以合并为代价丢失来源差异。
- 运行时只读已发布资产；构建、迁移和审核在 builder staging 中完成。
- 不以节点或边的数量衡量质量，优先降低无证据、重复和冲突关系。

## 2. 非目标

- 不在本阶段引入全局社区发现、自动主题报告或重型 GraphRAG。
- 不通过新增方名黑名单、固定问句、固定答案或单案例规则提高表面命中率。
- 不把模型记忆、派生三元组或二手资料伪装成课程原文。
- 不承诺一次迁移后图谱完整；质量由版本化指标和审核状态表达。
- 不在 runtime 仓库原地修改生产 SQLite。

## 3. 数据模型

### 3.1 文档与证据

新增 `documents`：

```text
document_id
canonical_title
source_layer
media_type
edition
content_hash
logical_source_path
ingestion_version
rights_note
```

`source_layer` 只允许：

- `course_primary`：课程讲义、同步文稿和课程转录；
- `classic_primary`：已注册版本的古籍原文；
- `reference_secondary`：校勘、术语或学术参考；
- `derived`：不能独立作为事实依据的派生内容。

新增 `evidence_records`：

```text
evidence_id
document_id
paragraph_id
locator
section_path
original_text
normalized_text
previous_evidence_id
next_evidence_id
quality_flags
```

`original_text` 用于引用展示，`normalized_text` 用于检索和匹配。清洗不得覆盖原文。前后证据链接用于展开同页或相邻段落上下文。

### 3.2 规范实体

新增 `entities`：

```text
entity_id
entity_type
canonical_name
normalized_key
definition_status
created_by_version
```

新增 `entity_aliases`：

```text
alias_id
entity_id
surface_form
normalized_form
alias_type
evidence_id
confidence
review_status
```

首期实体类型包括方剂、药物、症状、体征、脉象、舌象、证候、六经、脏腑、病机、治法、穴位、经络、病名、古籍和古籍条文。繁简体与明确异体字可自动归一；简称、同名异义和疑似 OCR 变体进入审核队列，不自动破坏性合并。

### 3.3 类型化关系

新增 `relations`：

```text
relation_id
subject_entity_id
predicate
object_entity_id
literal_value
evidence_id
source_layer
confidence
extraction_method
extractor_version
review_status
created_at
```

`object_entity_id` 与 `literal_value` 二选一。首批谓词采用受控词表，包括 `has_symptom`、`has_sign`、`has_pulse`、`has_tongue`、`indicates_pattern`、`treated_by`、`contains_herb`、`uses_method`、`contraindicated_for`、`differentiates_from`、`belongs_to`、`alias_of`、`variant_of`、`quotes`、`paraphrases`、`explains` 和 `supported_by`。

无法映射到受控谓词的候选只进入构建报告，不进入生产图谱。`confidence` 表示抽取置信度而非医学事实真值；`review_status` 只允许 `auto_accepted`、`needs_review`、`reviewed`、`rejected`。

## 4. 构建与迁移流程

迁移在 builder 的 staging 副本中完成：

1. 为 11 份文档生成稳定 `document_id` 并人工确认证据层。
2. 将现有 `paragraphs` 映射到 `evidence_records`，保留页码、原文和前后文顺序。
3. 从现有 `knowledge_units` 生成实体和关系候选，不直接认定为生产事实。
4. 按规范名称、繁简体和显式别名合并高置信实体；含糊候选保持分离。
5. 验证每条关系的证据摘录确实存在于对应原文；失败项拒绝发布。
6. 生成质量报告和人工抽样队列。
7. 同时构建 SQLite、FTS、FAISS、映射文件和 manifest，并运行跨资产一致性检查。
8. 通过检索、图谱和引用回归后，以目录级原子发布替换 runtime 资产。

迁移过程不得改变原始 PDF。旧表首个版本保持只读兼容，runtime 优先读取新结构；完成等价性验证后再决定是否移除旧表。

## 5. Runtime 使用方式

Runtime 查询先识别规范实体，再从关系表扩展最多两跳，并通过 `evidence_id` 回查原文。图谱分数只影响候选导航，最终答案仍由原始证据支持。

引用输出分为三层：

1. 默认展示短证据摘录；
2. 可展开完整 `original_text`；
3. 可继续获取 `previous_evidence_id` 和 `next_evidence_id` 对应的上下文。

关系路径必须同时返回实体类型、谓词、关系审核状态和证据定位。任何缺少有效证据、处于 `rejected` 状态或仅由 `derived` 支持的关系不得写成确定性事实。

## 6. 安全测试数据规则

安全测试验证通用属性，不保存固定的泄露内容：

- 每次测试动态生成不可预测的 synthetic sentinel。
- 用户名、目录段、凭据值、签名参数和 URI authority 均由测试 fixture 组合生成。
- 断言只验证 sentinel 或输入路径片段没有出现在公开输出中。
- 测试源码、快照、黄金集和文档不得包含真实用户名、真实机器目录、真实密钥或可复用凭据格式。
- 路径平台类型可以参数化表达为 POSIX、Windows、UNC 和 URI，但不固化个人身份或私密叙事内容。
- 保留少量属性测试和跨平台边界测试，不继续扩张与知识结构无关的对抗样例矩阵。

生产代码继续执行输出规范化与 trace 脱敏；测试数据简化不降低公开边界要求。

## 7. 图谱质量指标

每次 staging 构建至少报告：

- 文档证据层覆盖率；
- 实体证据覆盖率；
- 关系证据覆盖率和无证据关系数；
- 重复实体候选率和别名合并准确率；
- 孤立实体率；
- 每种实体和关系的数量、覆盖文档数与抽样准确率；
- 冲突关系数及其证据来源；
- `needs_review`、`reviewed` 和 `rejected` 比例；
- 原文摘录定位成功率；
- 旧知识单元到新实体/关系的迁移覆盖率。

首个发布门要求所有生产关系具有有效证据和 extractor provenance；不存在绝对路径；抽样中不得把派生字段当作原文。实体和关系召回率先建立基线，不使用未经测量的“完整”或“完美”表述。

## 8. 测试与验收

测试分为四组：

- Schema：约束、外键、受控枚举和迁移幂等性。
- Provenance：实体、别名和关系均可回到正确文档与原文。
- Graph：别名解析、一至两跳遍历、重复折叠、冲突保留和审核状态过滤。
- Retrieval/Answer：图谱导航不得降低文本基线；引用必须展示原文，展开后保持段落与页码一致。

评测集从当前 7 条回归种子扩展为按实体类型、关系类型和问题类型分层的集合。新增案例必须来自已注册资料，记录期望 evidence IDs；不得以针对单个问题的生产代码分支换取通过。

发布前必须验证旧 CLI 的兼容行为、SQLite/FTS/FAISS 一致性、LFS 对象完整性、路径可移植性和 builder staging 原子发布。

## 9. 交付顺序

1. 将固定泄露样例重构为动态 synthetic fixtures，收缩安全测试矩阵。
2. 在 builder 增加文档、证据、实体、别名和关系 schema。
3. 实现旧资产到新结构的 staging 迁移器与质量报告。
4. 为 runtime 增加新 schema 的只读查询、原文展开和图谱路径输出。
5. 扩展图谱评测集，完成现有 11 份 PDF 的候选资产重建。
6. 经独立规格、质量和发布门审查后发布新的 LFS 资产。

每一步保持可回滚，不把 schema 迁移、抽取规则替换和检索算法重写合并成一次不可验证的大改动。

## 10. 成功标准

本阶段完成时，用户应能从任一公开关系回到完整原文段落和相邻上下文；维护者能量化图谱的证据覆盖、重复、孤立、冲突和审核状态；新增文档能通过同一 staging 流程进入明确证据层。项目以可验证的知识结构质量为核心，而不是以硬编码案例或知识数量作为完成证明。
