# nihaisha-rag-prototype

**把倪海厦 PDF 课程资料整理成可检索、可引用、带安全边界的本地 RAG Skill。**

这是 `nihaisha` 的轻量 RAG 版本：不保留旧截图资产、旧 references、旧 UI 和部署文件，只保留最终可用的 SQLite 知识图谱、BAAI/bge-m3 向量库、搜索/回答逻辑和 Skill 入口。

边界：本项目用于倪海厦课程资料学习、原文检索、出处定位和中医理论整理；不提供个人诊断、处方、剂量或自我用药建议。

## 核心能力

- **知识图谱检索**：从 PDF 段落中抽取 `subject-predicate-object` 知识单元，覆盖症状、方证、剂量、方法、比较、注意事项。
- **向量语义召回**：使用 `BAAI/bge-m3` dense embedding 构建多层召回单元，支持句子、拼句窗口、整段、问题化检索。
- **全文精确搜索**：SQLite FTS 检索原文段落和知识单元，适合查原句、术语、书名页码。
- **多路 hybrid 召回**：合并 vector + text + knowledge 结果，最终按原文段落去重返回。
- **引用溯源**：每条结果回到 PDF 名称、页码、原文段落和 evidence quote。
- **安全提示**：涉及剂量、方药、处方线索、外治法或真实病情时，自动加入风险提示。

## Skill 结构

```text
SKILL.md          Agent Skill 入口
agents/openai.yaml
README.md         人类使用说明
pyproject.toml

data/pdf_rag_bge_m3/
  rag.sqlite      最终 SQLite 知识图谱 + 向量库
  manifest.json   数据库构建摘要
  vectors.faiss   FAISS 向量索引
  vector_ids.jsonl

nihaisha_kg/
  pdf_vector.py   RAG、知识图谱、向量召回、答案生成
  cli.py          精简 CLI

tests/
  test_pdf_vector.py
```

## 数据库内容

```text
documents: 11
paragraphs: 5,375
retrieval_units: 159,286
knowledge_units: 10,436
embedding: SiliconFlow BAAI/bge-m3 dense vectors
vector_dim: 1024
text_index: fts5_trigram
faiss_index: vectors.faiss
```

知识图谱表：

```text
knowledge_units
  knowledge_unit_id
  paragraph_id
  subject
  predicate
  object
  unit_type
  evidence_quote
  source_path
  page_start/page_end
```

知识单元类型：

```text
symptom: 7,121
caution: 2,115
formula_pattern: 884
comparison: 303
dosage: 11
method: 2
```

向量库表：

```text
retrieval_units
  sentence   单句召回
  window     拼句窗口召回
  paragraph  整段召回
  question   段落问题化召回
```

最终输出不直接停留在向量单元或知识单元，而是映射回原文段落，便于引用和核对。

FAISS 索引：

```text
vectors.faiss     向量近邻索引，用于快速语义召回
vector_ids.jsonl  FAISS row -> retrieval_unit_id 映射
rag.sqlite        仍然是主库，保存原文、知识图谱、FTS、metadata
```

## 安装为 Skill

推荐把整个目录作为 Skill 使用，因为数据库也在目录内。

本仓库使用 Git LFS 保存大文件：

```text
data/pdf_rag_bge_m3/rag.sqlite
data/pdf_rag_bge_m3/vectors.faiss
```

clone 后如果没有看到完整数据库文件，先执行：

```bash
git lfs pull
```

本地开发时可以用软链接：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s /Users/june/code/github/nihaisha-rag-prototype \
  "${CODEX_HOME:-$HOME/.codex}/skills/nihaisha-rag-prototype"
```

如果是复制安装：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R /Users/june/code/github/nihaisha-rag-prototype \
  "${CODEX_HOME:-$HOME/.codex}/skills/nihaisha-rag-prototype"
```

安装后重启 Codex/Agent，让 Skill 元数据重新加载。之后可以用：

```text
用 nihaisha-rag-prototype 查木香饼热熨法的出处。
```

## API 与本地模型

推荐使用 SiliconFlow API 的 `BAAI/bge-m3` 做向量查询。

SiliconFlow embedding 文档：

```text
https://api-docs.siliconflow.cn/docs/api/embeddings-post
```

创建 `.env`：

```bash
cp .env.example .env
```

填写：

```text
SILICONFLOW_API_KEY=...
SILICONFLOW_CHAT_MODEL=Qwen/Qwen3-32B
```

不要提交 `.env`。

本地 `BAAI/bge-m3` 默认不安装。如果需要无 API 或离线完整向量召回，再安装可选依赖：

```bash
python3 -m pip install -e ".[local]"
```

第一次运行本地模型会下载 `BAAI/bge-m3` 到 HuggingFace 缓存。

FAISS 用于加速本地向量近邻检索。若需要重建索引：

```bash
python3 -m pip install -e ".[faiss]"
python3 -m nihaisha_kg build-faiss
```

如果当前系统 Python 装不上 `faiss-cpu`，建议使用 Python 3.12 创建虚拟环境后再安装。

## 命令使用

查看数据库：

```bash
python3 -m nihaisha_kg stats
```

知识图谱搜索，不需要 embedding：

```bash
python3 -m nihaisha_kg search "一钱是多少克" --mode knowledge --limit 5
python3 -m nihaisha_kg search "木香饼热熨法 出处" --mode knowledge --limit 5
```

原文精确搜索，不需要 embedding：

```bash
python3 -m nihaisha_kg search "猪肤汤 桔梗汤" --mode text --limit 5
```

Hybrid 搜索，推荐配置 SiliconFlow API key 后使用：

```bash
python3 -m nihaisha_kg search "古时候一钱是多少克" --mode hybrid --limit 5
```

强制使用 SiliconFlow：

```bash
python3 -m nihaisha_kg search "下利 恶心 黄芩加半夏生姜汤" \
  --mode hybrid \
  --embedding siliconflow \
  --limit 8
```

强制使用本地 bge-m3：

```bash
python3 -m nihaisha_kg search "下利 恶心 黄芩加半夏生姜汤" \
  --mode hybrid \
  --embedding local-bge-m3 \
  --limit 8
```

模板回答，稳定、可控、无需 LLM：

```bash
python3 -m nihaisha_kg answer "木香餅熱熨法是來自哪一本書的哪一段？" \
  --mode hybrid \
  --limit 8
```

LLM 组织答案，只能基于检索到的引用：

```bash
python3 -m nihaisha_kg answer "古時候的一錢，是現代的多少克？" \
  --mode hybrid \
  --composer llm \
  --llm-model Qwen/Qwen3-32B \
  --limit 8
```

JSON 输出：

```bash
python3 -m nihaisha_kg answer "木香饼热熨法 出处" --mode hybrid --json
```

## 使用示例

```text
用 nihaisha-rag-prototype 查古时候一钱是多少克，要给出处。
```

```text
用 nihaisha-rag-prototype 查木香饼热熨法来自哪一本书哪一段。
```

```text
用 nihaisha-rag-prototype 检索下利、恶心、黄臭相关的课程方证线索，只要证据，不要给个人处方。
```

```text
用 nihaisha-rag-prototype 对比猪肤汤和桔梗汤在资料里的相关段落。
```

## 推荐回答规范

回答尽量使用：

```text
1. 直接结论
2. 证据依据
3. 需要鉴别或注意的条件
4. 安全边界
5. 引用
```

所有关键事实都应来自检索证据。如果证据不足，直接说明“不足以支持结论”。

## 安全硬规则

凡涉及剂量、方药、处方线索、外治法或真实病情，必须提示：

```text
涉及剂量、方药或处方线索时必须谨慎：不同人的体质不同，病情阶段、兼证、年龄、基础病和用药史都不同；现代药材来源、炮制、浓度和药效也和以前差很多。建议去线下正规中医渠道面诊辨证，不要私自购药有风险。
```

临床类问题只整理课程证据和辨证线索，不直接给个人处方。

## 与原 nihaisha 的区别

`/Users/june/code/github/nihaisha` 是完整课程 Skill，包含大量 references、截图证据、学习路线和课程蒸馏资料。

本项目是 RAG prototype，只保留：

```text
知识图谱
向量库
原文段落
搜索方法
引用回答逻辑
安全边界
```

也就是说，它更适合做“原文搜索增强”和“可溯源问答”，而不是完整课程网站或长篇学习资料库。

## 验证

```bash
python3 -m unittest tests.test_pdf_vector -v
python3 -m py_compile nihaisha_kg/pdf_vector.py nihaisha_kg/cli.py
python3 -m nihaisha_kg stats
```
