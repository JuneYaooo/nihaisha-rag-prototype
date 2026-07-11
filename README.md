# 倪海厦课程资料本地 RAG

本仓库把 11 份课程 PDF 的原文段落、全文索引、知识图谱、BAAI/bge-m3 dense 向量和 FAISS 索引打包成一个可追溯的本地检索工具。它适合课程学习、原文与页码查找、方证比较和证据整理；不是医疗诊断或处方系统。

## 五分钟开始

需要 Python 3.11+ 和 Git LFS。在仓库根目录执行：

```bash
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[runtime]"
cp .env.example .env
```

Windows PowerShell 用 `.venv\Scripts\Activate.ps1` 激活环境。后续命令都在已激活的虚拟环境中执行；退出时运行 `deactivate`。

编辑 `.env`，只在本机填写 `SILICONFLOW_API_KEY`，不要提交它。运行需要 SiliconFlow 的命令时，程序会查找当前目录、父目录和模块根目录中最近的 `.env`，按 `KEY=VALUE` 解析（不会作为 shell 脚本执行）；已经导出的同名环境变量优先。也可不把 key 写入文件，改由密码管理器/环境管理工具注入；在 zsh 中可用下面的无回显方式，仅导出到当前 shell，避免写进命令历史：

```zsh
read -r -s "SILICONFLOW_API_KEY?SiliconFlow API Key: "
echo
export SILICONFLOW_API_KEY
```

然后依次运行：

```bash
python3 -m nihaisha_kg doctor
python3 -m nihaisha_kg search "桂枝汤和麻黄汤的方证如何鉴别？" --mode hybrid --limit 8
python3 -m nihaisha_kg answer "木香饼热熨法来自哪一本书哪一段？" --json --trace
python3 -m nihaisha_kg evaluate --cases evals/golden_v1.jsonl --mode hybrid
```

`doctor` 应返回顶层 `"status": "ok"`。若只需精确词句或知识图谱，可不配置 key，改用 `--mode text` 或 `--mode knowledge`。

## 运行规则

- 当前生产库包含 159,286 个 dense 检索单元。`vector` 和 `hybrid` 必须有 FAISS；不会静默退化成 SQLite 全量向量扫描。缺少模块或索引时先按 `doctor` 的诊断修复。
- `text` 和 `knowledge` 不调用 embedding API，不需要 key。
- `vector` 和 `hybrid` 同时需要查询 embedding 后端与 FAISS。推荐 SiliconFlow `BAAI/bge-m3`；离线可选本地后端：`python3 -m pip install -e ".[local]"`，并传 `--embedding local-bge-m3`。
- CLI 的 `--reranker auto` 仅在存在 `SILICONFLOW_API_KEY` 时调用 SiliconFlow reranker；`--reranker none` 明确关闭。服务失败时会保留召回结果，并在 trace 中给出已脱敏的降级状态。
- 普通文本输出保持兼容，但 `search` 和 `answer` 的纯文本输出都不显示 trace；必须同时传 `--json --trace`。搜索 trace 给出召回渠道/各渠道排名、最终选中段落 ID、reranker 模型/降级状态和搜索延迟；回答 trace 还给出分组的初始/追问计划、各计划观察结果的排名与渠道，以及各阶段延迟。
- trace 采用诊断字段白名单，不主动复制 provider 凭据、请求头、环境变量转储、向量或完整证据正文；已识别的凭据模式和 reranker 错误会脱敏。但 `normalized_query` 有意保留用户查询文本，脱敏不是保密保证：**绝不要把 API key、密码、患者隐私或其他私密文本写进查询**。

当引用可疑、结果串题或需要复核排序时：

```bash
python3 -m nihaisha_kg search "问题" --mode hybrid --limit 8 --json --trace
python3 -m nihaisha_kg answer "问题" --mode hybrid --limit 8 --json --trace
```

先核对 `selected_paragraph_ids`、渠道/排名和降级状态，再回到 citation 中的 PDF 名称、页码与原文摘录。trace 是诊断信息，不是证据。

## 当前数据与质量基线

`python3 -m nihaisha_kg stats` 当前实测为：11 份文档、5,375 个原文段落、159,286 个检索单元、10,436 个知识图谱单元、19,072 个 guide nodes；dense 向量为 1024 维，FAISS 映射同为 159,286 条。

仓库提交了 7 条回归种子用例 `evals/golden_v1.jsonl` 和一次实测结果 `evals/baseline_v1.json`。复现命令：

```bash
python3 -m nihaisha_kg evaluate --cases evals/golden_v1.jsonl --mode hybrid --limit 10
```

以下小数是从 `evals/baseline_v1.json` 中的原始浮点值四舍五入后展示，并非声明精确相等：Recall@5 ≈ **0.4761904762**，Recall@10 ≈ **0.5238095238**，MRR（`reciprocal_rank`）≈ **0.4047619048**，nDCG@10 ≈ **0.3908372076**，context precision@10 ≈ **0.3190476190**，forbidden hits@10 ≈ **0.0**。

同一语料、同一 7 条种子、`limit=10` 的新鲜对比结果如下（均为四舍五入显示）：

| 模式 | Recall@5 | Recall@10 | MRR | nDCG@10 | context precision@10 | forbidden hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.5714285714 | 0.5714285714 | 0.4285714286 | 0.4659799296 | 0.4285714286 | 0.0 |
| hybrid | 0.4761904762 | 0.5238095238 | 0.4047619048 | 0.3908372076 | 0.3190476190 | 0.0 |

`hybrid` 一般仍适合用户表达与原文措辞不一致的语义召回，但它在这组很小的种子上**没有胜过 text**；这是当前已知质量缺口。模式选择必须按任务和评测实测，不能假定 hybrid 更好。精确术语、书名或原句查询应优先同时测 `text`。

- Recall@k：前 k 条覆盖了多少标注相关段落。
- MRR：第一个相关段落排名倒数的平均值，越高表示首次命中越靠前。
- nDCG@10：考虑相关结果排序位置的折损收益。
- context precision@k：先按段落 ID 对排名去重；对前 k 条中每次相关命中累加该名次的 precision@rank，再除以 `min(标注相关 ID 数量, k)`，即 AP@k 风格指标。
- forbidden hits@10：前 10 条命中明确禁止段落的平均次数，越低越好。

这 7 条只是防止明显回归的种子，不是全面评测，更不能证明回答“全面”或“准确”。固定追问清单污染已移除，但小样本指标有限，检索仍可能带回无关证据；应检查 trace、PDF 页码和 citation 原文。其他限制包括 OCR 错误与课程来源覆盖不全、实体/关系规范化仍需改进，以及尚无独立版本化、核验并可结构化识别的权威古籍层或外部学术来源层。

`baseline_v1.json` 顶层 `provenance` 固定记录生成命令、模式/limit、运行时代码提交、embedding、Python/SQLite/运行依赖版本及 golden/DB/FAISS/vector IDs 哈希，用于发现用例、代码或资产变化后的陈旧基线。`evaluate` 没有 reranker 阶段，因此 provenance 记为 `none`；查询 embedding 按 DB metadata 自动解析，最终解析到的 provider/model/dim 另行固定记录。原始 CLI 命令只重现 `cases`、`aggregate`、`results`；`provenance` 是审核后稳定附加的元数据，不由 CLI 输出。复现时应比较这三个原始字段，重新计算 golden/vector IDs SHA256，并通过 Git LFS pointer 核对 DB/FAISS OID；不能把旧 provenance 复制到新结果。`tests.test_evaluation` 会持续检查 provenance schema 以及可直接计算的哈希。

## 证据分层

回答时必须区分以下层级，不能把不同来源悄悄合并：

| 层级 | 当前状态 | 用法 |
| --- | --- | --- |
| `course_primary` | 运行时策略，尚无结构化字段 | 当前运行时答案以检索到的 PDF 原文段落、文件名、页码和摘录为主证据 |
| `classic_primary` | 尚未实现独立版本化层 | 未来保存经版本核验的权威古籍原文，并与课程转述分别引用 |
| `reference_secondary` | 未来入库，权威级别较低 | 学术研究、参考书和网页材料，只作补充 |
| `derived` | 已有派生数据但无 `evidence_layer` 字段 | 图谱三元组、guide nodes、查询扩展仅用于导航；答案合成不把其 subject/object 当作事实或方名 |

当前 SQLite 没有结构化 `evidence_layer` 列；11 份 PDF（包括 `黄帝内经原文和翻译.pdf`）仍是同一个 legacy bundled corpus，因此运行时目前不能从结构上强制区分 `course_primary` 与 `classic_primary`。答案权威来自检索到的 PDF 原文段落，而不是派生图谱字段；但这不等于库中完全没有古籍原文或翻译材料。尚未实现的是**单独版本化、核验并可结构化识别的权威古籍层**。第二阶段 builder migration 将按文档分类并写入证据层。

当前运行时没有外部网页检索路径。原文未在 bundled corpus 中检索到时，绝不能用模型记忆补写。只有当用户另行提供外部来源，或明确授权在本运行时之外研究时，才可把所得材料标为“外部材料（本库未检索）”，单独核验和引用；它不能共用 bundled citation 编号或获得 PDF 原文段落的证据权威。

未来古籍或网页证据记录至少要保存：题名与版本/网站、卷/章/行或稳定定位、稳定 URL/来源、访问日期、许可与版本、逐字引文、对应课程提及的链接、置信度和审核状态。课程主张与古籍原文必须分别引用；不得把倪海厦的转述无标记地写成古籍原句。只有未来正式摄取、版本化并审核后，材料才可进入 `classic_primary` 或 `reference_secondary` 层。

## 新增资料与发布

以后可在独立的 `nihaisha-rag-builder` 仓库/Skill 中增量加入 PDF；若它是同级 clone，路径通常为 `../nihaisha-rag-builder`，也可使用实际配置路径。用户不必一次提供全部资料。构建端负责抽取、规范化、生成向量与索引、验证，并以 staging → 校验 → 原子发布的方式替换成套产物；本运行时仓库不直接修改生产数据库。新增资料仍受 OCR、版权/许可和人工复核质量约束。

## 安全边界

本项目只用于课程资料学习和来源核对。不要据此进行个人诊断、开方、剂量决策、自行购药、针灸或外治操作。

涉及剂量、方药或处方线索时必须谨慎：不同人的体质不同，病情阶段、兼证、年龄、基础病和用药史都不同；现代药材来源、炮制、浓度和药效也和以前差很多。建议去线下正规中医渠道面诊辨证，不要私自购药有风险。
