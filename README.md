# 父子分块 RAG 检索系统（Parent-Child Chunking RAG）

> 当前版本 **1.2.0** ｜ 77 个测试通过 ｜ `ruff check` 全绿 ｜ 变更记录见 [CHANGELOG.md](CHANGELOG.md)
> 代码审查报告（问题 / 更新 / 遗留项）见 [docs/07-代码审查报告.md](docs/07-代码审查报告.md)

一个**可直接运行**的三存储协同 RAG 检索项目：**MySQL 存父文档 + Redis 缓存父文档内容 +
Milvus 存子块稀疏/稠密双向量**，实现「用子块精准检索，用父块完整生成」。

> 本项目依据《父子分块的练习 面试加实战》整理需求并工程化落地：

| 文档原始需求 | 本项目落点 |
| --- | --- |
| 设计 MySQL 库表保存父文档：`id`、`content` | `parent_document_t`（自增 `id` + `content` + 章节/哈希/幂等约束），见 `app/models/parent_document.py` |
| 设计 Milvus 表，含 `parent_id`、稀疏与稠密向量 | `child_chunk_t`（`parent_id` INT64 + `dense_vector` FLOAT_VECTOR + `sparse_vector` SPARSE_FLOAT_VECTOR），见 `app/clients/milvus_client.py` |
| 同步 `parent_document_t` 到 Redis：`key = id`，`value = content` | 入库时写 `psd:parent:{id}` → 父块全文，见 `app/clients/redis_client.py` |
| 查询：子块 → `parent_id` → Redis 取父文档 → 交给大模型 | `RetrievalService.retrieve()` + `RagService.answer()`，见 `app/services/retriever.py`、`app/services/rag.py` |

---

## 一、项目背景

朴素 RAG 把整篇文档按固定字数切块，会遇到一个结构性矛盾：

- **块切小（200 字）**：单块语义集中，向量表示精准，**检索准**；但上下文残缺，**生成差**（断章取义）。
- **块切大（2000 字）**：上下文完整，**生成好**；但向量被无关信息稀释，**检索差**，且 Prompt token 暴涨。

这就是「检索粒度」与「生成粒度」的矛盾。父子分块用**两级结构**同时满足两者：

```
                 ┌──────────── 一次入库 ────────────┐
   原始文档 ──▶  父块(1200字) ──▶ MySQL(parent_document_t, 自增 id)
                  │                    │
                  │                    └──▶ Redis (key = parent_id, value = 父块全文)
                  │
                  └──▶ 子块(280字) ──▶ Milvus(child_chunk_t, 稠密+稀疏向量, parent_id)

                 ┌──────────── 一次查询 ────────────┐
   用户问题 ──▶ 稠密/稀疏混合检索 ──▶ 命中「子块」 ──▶ 取 parent_id 去重
                                                          │
                                                          ▼
                        Redis MGET 批量取父块全文（未命中回表 MySQL 并回填）
                                                          │
                                                          ▼
                                              父块全文 ──▶ 大模型生成答案
```

## 二、项目目标

1. **跑通完整链路**：入库（文档 → 父块 → 子块 → 三存储）与查询（子块 → parent_id → 父块 → 答案）。
2. **覆盖文档关键技术点**：MySQL 幂等设计、Redis 缓存策略、Milvus 稀疏/稠密混合检索与 RRF 融合、`parent_id` 回表。
3. **可观测、可对比、可调参**：三路检索对比、阶段耗时、缓存命中率、分块质量统计，全部通过接口/页面暴露。
4. **开箱即用**：无 GPU、无 API Key 也能跑通全流程（`hash` 向量 + 抽取式回答），随时可切换真实模型。

## 三、技术选型

| 层次 | 选型 | 版本 | 选型理由 |
| --- | --- | --- | --- |
| 父文档持久化 | MySQL + SQLAlchemy 2.0 | 8.4 | 需要**自增主键**作为跨存储锚点，事务与唯一索引保证幂等；ORM 便于换库 |
| 父块内容缓存 | Redis + connection pool | 8 | 父块全文「读多写少」，`MGET` 把 N 次往返压成 1 次，延迟亚毫秒级 |
| 向量库 | Milvus | 3.0.0 | 原生支持 **SPARSE_FLOAT_VECTOR**，稀疏/稠密**同库同集合**，`hybrid_search` 内置 RRF/加权融合 |
| 稠密向量 | 可插拔 Provider：`hash` / `api` / `local` | — | `hash` 零依赖离线跑通；`api` 接 OpenAI 兼容服务；`local` 接 BGE-M3 |
| 稀疏向量 | 自实现 BM25 + jieba | — | 不依赖 Milvus 内置 BM25 函数，词表/IDF 全流程可控、可讲解、可调试 |
| Web 框架 | FastAPI + Uvicorn | — | Pydantic v2 强校验、自动 OpenAPI 文档、异步友好 |
| 调试前端 | 原生 HTML/CSS/JS | — | 无构建步骤，`FileResponse` 直接托管，专注展示链路而非前端工程 |

### 为什么稀疏向量要自己算 BM25？

复用 Milvus 内置 BM25 函数更省事，但自己实现有三个收益：
1. **词表持久化**（`data/index/vocab.json`）这个工程细节必须显式面对 —— 下标不一致会导致检索**静默返回空**，这是最容易踩的坑；
2. IDF 依赖全库 `df / N / avgdl` 统计量，需要落盘复用，这是 BM25 的**真正难点**；
3. 可以直观解释「为什么稀疏分支擅长错误码/专有名词，稠密分支擅长语义泛化」。

## 四、功能模块划分

| 模块 | 职责 | 代码位置 |
| --- | --- | --- |
| 配置层 | 环境变量 → 强类型 Settings，集中管理分块/检索/存储参数 | `app/config.py` |
| 契约层 | API 与内部流转共用的 Pydantic 模型 | `app/schemas.py` |
| 存储客户端 | MySQL 连接池与建表、Redis 缓存封装、Milvus 集合与检索封装 | `app/clients/` |
| ORM 模型 | 父文档表结构定义 | `app/models/parent_document.py` |
| 分块器 | 两级切分（父块 → 子块）+ 章节标签 + 无标点兜底硬切 | `app/services/chunker.py` |
| 向量化 | 稠密向量 Provider 工厂、BM25 稀疏编码器 | `app/services/embedder.py`、`app/services/sparse.py` |
| 入库编排 | 一次写入三处，幂等 upsert，自增 id 回填 `parent_id` | `app/services/ingest.py` |
| 检索编排 | 混合检索 → `parent_id` 聚合去重 → `MGET` 取父块 → 回表兜底 | `app/services/retriever.py` |
| 生成层 | 可插拔大模型 + 无密钥抽取式降级 + 强制引用标注 | `app/services/llm.py` |
| 问答编排 | 检索与生成串联，产出答案与引用 | `app/services/rag.py` |
| 接口层 | 15 个 REST 接口（入库/检索/问答/对比/统计/管理） | `app/api/` |
| 调试前端 | 五个功能页签的可视化调试台 | `web/index.html` |
| 脚本 | 初始化、灌数据、端到端演示、重置 | `scripts/` |
| 测试 | 24 个单元 + 集成 + HTTP 测试 | `tests/` |

## 五、代码结构说明

```
par_sub_split/
├── app/
│   ├── config.py                 # Settings：MySQL/Redis/Milvus/分块/检索/LLM 全部参数
│   ├── schemas.py                # DocumentIn / SearchRequest / ParentHit / ChatResponse ...
│   ├── clients/
│   │   ├── mysql.py              # Engine + session_scope + 自动建库建表
│   │   ├── redis_client.py       # ParentCache：set/get/mget/delete + 跨进程命中率统计
│   │   └── milvus_client.py      # ChildChunkStore：建集合、upsert、hybrid_search、count
│   ├── models/
│   │   └── parent_document.py    # parent_document_t（LONGTEXT content + 唯一索引幂等）
│   ├── services/
│   │   ├── chunker.py            # ★ split_document()：两级切分，子块绝不跨父块边界
│   │   │                         #   含 Markdown 表格 / 围栏代码块的"原子块"保护
│   │   ├── embedder.py           # ★ HashEmbedder / ApiEmbedder / LocalEmbedder + 工厂
│   │   │                         #   LocalEmbedder 自动解析 ModelScope 本地快照，支持 cpu/cuda + fp16
│   │   ├── sparse.py             # ★ BM25SparseEncoder：词表+IDF 落盘，输出 {idx: weight}
│   │   ├── reranker.py           # 可选 Cross-Encoder 精排（作用在子块上），依赖缺失自动降级
│   │   ├── ingest.py             # ★ IngestService：三存储写入 + parent_id 回填
│   │   ├── retriever.py          # ★ RetrievalService：混合检索 →（可选重排）→ 去重 → Redis 回表
│   │   ├── llm.py                # ★ AnswerGenerator：Prompt 模板 + 抽取式降级
│   │   └── rag.py                # RagService：检索 + 生成编排
│   └── api/
│       ├── main.py               # FastAPI app，lifespan 自动初始化表/集合
│       └── routes.py             # /api/v1/* 全部接口
├── data/samples/                 # 3 篇可灌入的中文样例文档（讲的就是本项目技术栈）
├── docs/
│   ├── 01-架构与原理.md           # 三存储协同、混合检索、分块策略、缓存设计的原理与取舍
│   ├── 02-学习路径.md             # 6 阶段 + 12 个动手实验 + 自查清单
│   ├── 03-面试题库.md             # 45 道面试题含参考答案
│   ├── 04-优化建议（基于本机配置）.md  # 本机配置评估、优化建议与实测验证记录
│   ├── 05-API.md                 # 接口参考（请求/响应/错误码/安全约定）
│   └── 06-配置说明.md             # 全量配置项参考与典型场景速查
├── scripts/
│   ├── init_db.py                # 初始化三存储 + 连通性自检
│   ├── check_local_models.py     # 本地模型/依赖/显存/LLM 的 preflight 自检
│   ├── ingest_samples.py         # 灌样例数据 + 打印分块结构预览
│   ├── demo_query.py             # 端到端演示（三路对比 + 问答 + 耗时）
│   ├── reset_data.py             # 重置演示数据（破坏性操作，需 --yes）
│   └── run_server.py             # 启动 HTTP 服务
├── tests/                        # 77 个测试
│   ├── test_chunker.py           #   分块器（含原子块保护）
│   ├── test_vectors.py           #   稠密/稀疏编码器
│   ├── test_e2e_pipeline.py      #   三存储集成（幂等/缓存/回表/幽灵缓存/删除）
│   ├── test_api_http.py          #   HTTP 接口（含上传防护、元数据一致性）
│   ├── test_api_security.py      #   安全与运维回归（令牌/上传/CORS/预热）
│   ├── test_optimizations.py     #   本机适配回归（路径解析/LLM 判定/重排）
│   └── test_parent_child_demo.py # ★ 父子分块效果演示（可独立运行，含预期输出说明）
├── web/index.html                # 可视化调试台
├── docker-compose.yml            # ★ 全部 5 个依赖服务的统一编排（项目名 par_sub_split）
├── ruff.toml                     # 静态检查配置（python -m ruff check app scripts tests）
├── pytest.ini                    # 测试配置
├── CHANGELOG.md                  # 变更日志
├── .env.example                  # 配置样例（复制为 .env）
└── requirements.txt
```

### 核心数据结构流转

```python
# ① 分块：ParentChunk → ChildChunk（父子绑定在内存中完成）
parents = split_document(text, parent_size=1200, child_size=280)
#  ParentChunk(index=0, heading_path="手册 > 安装", content="…1200字…")
#    └── ChildChunk(index=0, parent_index=0, content="…280字…")

# ② 落库：MySQL 自增 id 回填到每个子块
parent_ids = self._persist_parents(parents, doc_id, title, source)
#  child.parent_id = 12   ← 关键：这就是 Milvus 的 parent_id 来源

# ③ 向量化：稠密矩阵 + 稀疏字典
dense_matrix  = embedder.encode_documents(texts)      # shape=(n, 1024)
sparse_dicts  = encoder.encode_batch(texts)           # [{idx: weight}, ...]

# ④ 检索：子块命中 → parent_id → Redis
hits = store.hybrid_search(dense_vector, sparse_vector, limit=20)
grouped = {parent_id: [子块命中列表]}                  # 按 parent_id 聚合去重
contents = cache.mget(target_pids)                    # 一次 RTT 取回全部父块全文
```

## 六、快速开始

### 0. 前置：一条命令启动全部依赖服务

本项目的**全部 5 个依赖服务**（MySQL / Redis / etcd / MinIO / Milvus）统一由仓库根目录的
`docker-compose.yml` 编排，Compose 项目名为 `par_sub_split`：

```bash
docker compose up -d          # 启动全部服务
docker compose ps             # 查看状态（5 个服务应为 running / healthy）
docker compose stop           # 停止（保留数据卷）
docker compose down           # 移除容器（保留数据卷）
```

启动后的容器与端口：

| 服务 | 容器名 | 宿主机端口 | 作用 |
| --- | --- | --- | --- |
| MySQL | `par_sub_mysql` | **9999** → 3306 | 父文档持久化 |
| Redis | `par_sub_redis` | 6379 | 父文档内容缓存 |
| etcd | `par_sub_etcd` | 仅容器网络 | Milvus 元数据 |
| MinIO | `par_sub_minio` | 9000（API）/ 9001（控制台） | Milvus 对象存储 |
| Milvus | `par_sub_milvus` | **19530**（gRPC）/ 9091（healthz） | 子块稀疏+稠密向量 |

> **端口为什么用 9999 而不是 3306**：避免与本机可能存在的其他 MySQL 抢占默认端口。
> 这三个端口（9999 / 6379 / 19530）必须与 `.env` 中的 `MYSQL_PORT` / `REDIS_PORT` / `MILVUS_URI` 一致。
>
> **首次在新机器部署**：`docker compose up -d` 会自动创建数据卷并初始化；
> Milvus 首启较慢（健康检查有 90s 的 `start_period`），耐心等待 `docker compose ps` 变为 healthy。
>
> **一键脚本**：`python scripts/start.py` 会自动完成「依赖 → 容器 → 建表 → 灌数据 → 起服务 → 健康检查」，
> 并在容器缺失时回退调用上面的 `docker compose up -d`。
>
> **切换编排后数据是空的？** 统一编排使用新的命名卷（`par_sub_split_*`），
> 旧容器/旧卷里的数据**不会自动迁移**。重新执行 `python scripts/init_db.py` 与
> `python scripts/ingest_samples.py` 即可；确需保留旧数据请先用 `mysqldump` 导出再导入。

<details>
<summary>可选：清理早期独立部署遗留的容器（不清理也不影响本项目）</summary>

早期这些服务是各自 `docker run` 独立启动的，现已全部由本项目 compose 接管。
它们都处于 `Exited` 状态，确认不再需要后可释放磁盘空间：

```bash
# 先确认没有其他项目在依赖它们
docker ps -a --format "{{.Names}}\t{{.Status}}" | Select-String "mysql84|milvus-standalone|milvus-etcd|milvus-minio"

# 按需逐个移除（⚠ 会连同其匿名卷一起删除容器内的数据）
docker rm mysql84 milvus-standalone milvus-etcd milvus-minio
```

> 注意：`cmbk_*`、`nacos-standalone`、`mongodb` 属于**其他项目**的容器，
> 本项目的文档与脚本都不会触碰它们，请自行判断是否清理。

</details>

### 1. 安装依赖并配置

```bash
python -m pip install -r requirements.txt
copy .env.example .env                     # Linux/macOS: cp .env.example .env
```

> `.env` 已按 compose 编排填好：MySQL `127.0.0.1:9999`、Redis `6379`、
> Milvus `http://localhost:19530`（Milvus 开启了鉴权，`MILVUS_TOKEN=root:123456`）。

### 1.5 接入本机已有模型（推荐）

本机已具备本地嵌入模型与大模型，建议直接切换（零成本、数据不出机）：

```bash
# ① 本地嵌入：BGE-M3（ModelScope 已下载，1024 维与集合定义一致，无需改表）
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers transformers
# .env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
MODELSCOPE_CACHE=D:/modelscope/hub/models   # 注意用正斜杠：dotenv 会吞掉反斜杠
EMBEDDING_DEVICE=cpu                        # 8GB 显存留给 7B LLM

# ② 本地大模型：Ollama（无需 API Key）
# .env
LLM_PROVIDER=ollama
LLM_OLLAMA_BASE=http://127.0.0.1:11434/v1
LLM_OLLAMA_MODEL=qwen2.5:7b

# ③ 上线前跑一次自检（模型文件 / 依赖 / 显存预算 / LLM 连通性）
python scripts/check_local_models.py --load --llm "你好"

# ④ 换了嵌入模型必须重灌数据（语义空间变了）
python scripts/reset_data.py --yes --reset-ids --reset-vocab
python scripts/ingest_samples.py
```

可选开启精排（本机已下载 `bge-reranker-v2-m3`）：

```bash
RERANK_ENABLED=true
RERANK_DEVICE=cpu        # 与 7B LLM 共用 8GB 显存会 OOM，默认走 CPU
RERANK_CANDIDATES=20     # 对混合检索的 top-20 子块重排
RERANK_KEEP=10
```

### 2. 初始化三存储

```bash
python scripts/init_db.py
```

预期输出：MySQL 表就绪、Redis 连通、Milvus 集合创建成功。

### 3. 灌入样例数据

```bash
python scripts/ingest_samples.py
```

会打印每篇文档的分块结构预览，例如：

```
[03-父子分块检索优化手册.md] 1608 字
   分块预览: 父块 2 | 子块 7 | 父块均长 726.5 | 子块均长 235.0
     ├─ 父块#0 [父子分块检索优化手册 > 一、固定分块的困境] 1301字 -> 6 个子块
```

### 4. 端到端演示

```bash
python scripts/demo_query.py                          # 内置 4 个问题
python scripts/demo_query.py --query "你的问题"        # 自定义
python scripts/demo_query.py --no-llm                 # 只看检索链路
```

### 5. 启动 Web 服务与调试台

```bash
python scripts/run_server.py               # http://127.0.0.1:8000/
```

- 可视化调试台：<http://127.0.0.1:8000/>
- 自动接口文档：<http://127.0.0.1:8000/docs>

### 6. 运行测试与效果演示

```bash
python -m pytest tests -q                  # 52 passed

# 父子分块效果演示（独立运行，带完整预期输出说明与量化对照）
python tests/test_parent_child_demo.py
python tests/test_parent_child_demo.py --no-llm    # 只看检索链路
python tests/test_parent_child_demo.py --keep      # 保留数据便于手工核查
```

## 七、接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/health` | MySQL / Redis / Milvus 连通性 + 当前向量与 LLM 模式 |
| GET | `/api/v1/stats` | 父块数、子块数、文档数、Redis 命中率 |
| GET | `/api/v1/documents` | 已入库文档列表 |
| POST | `/api/v1/documents` | 文本入库（`doc_id` 可选，`force_reindex` 支持重建） |
| POST | `/api/v1/documents/upload` | 文件入库（`.md` / `.txt` / `.pdf`） |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档（MySQL + Redis + Milvus 三处同步） |
| POST | `/api/v1/search` | 只检索：返回命中子块 + 回表父块 + 阶段耗时 |
| POST | `/api/v1/chat` | 完整 RAG 问答，返回答案 + 引用 |
| POST | `/api/v1/compare` | 稠密 / 稀疏 / 混合 三路召回对比 |
| GET | `/api/v1/parents/{id}` | 按父文档 id 取内容（`from_redis` 标识是否命中缓存） |
| POST | `/api/v1/admin/reindex` | 重建 Milvus 集合（维度变更后必须执行） |
| POST | `/api/v1/admin/reset-cache-stats` | 重置缓存命中率统计 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"父子分块为什么比固定分块好？","top_k":20,"final_top_k":4}'
```

## 八、关键配置说明

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `PARENT_CHUNK_SIZE` | 1200 | 父块目标字数，**决定送模型的上下文粒度** |
| `CHILD_CHUNK_SIZE` | 280 | 子块目标字数，**决定检索精度** |
| `RETRIEVE_TOP_K` | 20 | Milvus 召回子块数 |
| `FINAL_TOP_K` | 4 | 去重后送入模型的父块数（不是越多越好） |
| `HYBRID_RANKER` | rrf | `rrf` 只依赖排名、无需归一化，是安全默认；`weighted` 需调参 |
| `SCORE_RATIO_THRESHOLD` | 0.2 | 父块分数低于「最高分 × 该比例」时丢弃，防止陪跑内容污染 Prompt |
| `EMBEDDING_PROVIDER` | hash | `hash` 离线跑通 / `api` 接 OpenAI 兼容 / `local` 接 BGE-M3 |
| `SPARSE_ENABLED` | true | 关闭后集合不建稀疏字段，仅稠密检索（用于对比实验） |
| `LLM_PROVIDER` | auto | 有 `LLM_API_KEY` 走真实模型，否则自动降级为抽取式回答 |

### 切换到真实模型（可选）

```bash
# .env
EMBEDDING_PROVIDER=api
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_DIM=1024

LLM_PROVIDER=openai
LLM_API_BASE=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat
```

切换 Embedding 模型后**必须重建集合**（维度/语义空间都会变）：

```bash
python scripts/init_db.py --recreate
python scripts/reset_data.py --yes --reset-ids --reset-vocab
python scripts/ingest_samples.py
```

## 九、实现过程中的关键工程决策

这些是本项目最值得复盘的取舍，也是面试高频追问点：

1. **Milvus `flush` 必须去掉**：集合配置 `consistency_level="Strong"` 后写入即可检索；
   而 `flush` 会强制刷盘重建段，实测单篇入库从 **11.6 秒降到 81 毫秒**（约 140 倍）。
2. **`get_collection_stats().row_count` 不能当准确值**：不 flush 时长期显示 0，
   改用 `query(count(*))` 走查询链路，结果实时准确。
3. **父块加成必须用乘法而非加法**：RRF 融合分数只有 `0.016~0.033` 量级，
   而稠密余弦分数在 `0.3~0.9` 量级；固定加法偏置会在 RRF 场景下压过真实排序信号。
4. **父块必须比子块显著大**：只按 Markdown 标题切父块时，小节往往只有 200~300 字，
   父块退化到子块粒度、父子分块失去意义；因此加入「跨小节合并到目标大小」的步骤
   （实测父块均长 726~1314 字，子块 235~257 字）。
5. **必须防止超长块**：整段没有句末标点（代码块/表格/清单）时，固定窗口切分会产出
   2900 字的巨型块；增加「次级分隔符 → 字符硬切」两级兜底。
6. **Redis 统计写进 Redis 而非进程内存**：多实例部署时命中率可天然聚合，重启不丢数据。
7. **缓存未命中必须回表兜底**：Redis 只是派生数据，`MGET` 未命中的父块要回表 MySQL
   并顺手回填（Cache-Aside），否则一次 Redis 抖动就会导致答案缺上下文。

## 十、安全与运维

### 10.1 已内置的安全措施

| 项 | 措施 |
| --- | --- |
| 上传防护 | 扩展名白名单（`UPLOAD_ALLOW_EXT`）+ **流式限量读取**（超限 413），不会把整文件读进内存 |
| 错误信息 | 500 响应只返回通用文案，内部异常细节（连接串/表名/堆栈）仅进服务端日志 |
| CORS | 来源白名单 + **显式关闭凭证携带**（不使用 `*`，避免无效且危险的组合） |
| 管理接口 | `/api/v1/admin/*` 支持 `ADMIN_TOKEN` 校验（请求头 `X-Admin-Token`），未配置时打印告警日志 |
| SQL | 全部走参数化查询；唯一的动态 SQL（`CREATE DATABASE`）先做标识符白名单正则校验 |
| 向量过滤表达式 | `doc_id` 在拼接 filter 前做引号/反斜杠转义 |
| 凭据管理 | 凭据只在 `.env`（已 gitignore），代码中仅为本机开发默认值；`/health` 不回显任何凭据 |

### 10.2 与"生产可用"的差距（诚实说明）

| 项 | 当前实现 | 生产建议 |
| --- | --- | --- |
| 业务接口鉴权 | **无**（设计为内网服务） | 在网关接入统一认证与限流 |
| 稠密向量 | 默认 `hash` 特征哈希（**仅供跑通链路，检索质量不代表生产水平**） | 换 BGE-M3 / Qwen3-Embedding 等真实模型 |
| 稀疏 IDF | 全量增量统计，未做重建 | 定期重建词表；超大语料改用 Milvus 内置 BM25 |
| 缓存一致性 | 删除时同步清理 + 幽灵缓存保护 | 引入消息队列 + 延迟双删，或订阅 binlog |
| 统计接口 | `redis_parent_keys` 依赖 `SCAN` 遍历（O(N)），已有 5 秒记忆化 | 超大规模改用维护计数器或 `DBSIZE` 分片统计 |
| 并发写入 | 数据库唯一索引保证幂等 | 增加分布式锁 / 任务队列，避免重复计算向量 |
| 限流 | 无 | 网关层按 IP/Key 限流，尤其是 `/chat`（LLM 成本高） |
| 评估 | 人工观察 + 三路对比 + 演示测试中的覆盖率断言 | 建立标注问答对的召回率/端到端准确率回归 |

## 十一、常见问题排查

| 现象 | 原因与解决 |
| --- | --- |
| `Fail connecting to server ... UNAUTHENTICATED` | Milvus 开了鉴权，`.env` 里补 `MILVUS_TOKEN=root:123456` |
| `Unknown database 'par_sub_split'` | 首次运行 `scripts/init_db.py` 会自动建库；若仍失败检查 MySQL 端口是否为 `9999` |
| 检索结果为空 | ① 是否已执行 `ingest_samples.py`；② 是否换过 Embedding 维度但没重建集合；③ BM25 词表与入库时是否一致（`data/index/vocab.json`） |
| 入库很慢（10s+） | 检查是否误开了 `flush`；本项目默认 `consistency_level="Strong"` 无需 flush |
| `子块 0 个 / 分块结果为空` | 文档正文为空或全是标题，`chunk_stats` 可确认 |
| RAG 答案是「降级模式」 | 未配置 `LLM_API_KEY`，属预期行为；配置后即为真实模型答案 |
| Redis 命中率低 | 参见 `docs/01-架构与原理.md` 第 5 节「缓存命中率上不去怎么办」 |

## 十二、命令速查

```bash
python scripts/init_db.py                    # 初始化（幂等）
python scripts/init_db.py --recreate         # 重建 Milvus 集合
python scripts/check_local_models.py --load --llm "你好"   # 本地模型/显存/LLM 自检
python scripts/ingest_samples.py             # 灌样例数据
python scripts/ingest_samples.py --rebuild   # 先删同 doc_id 再灌
python scripts/demo_query.py                 # 端到端演示
python scripts/run_server.py --reload        # 开发模式启动
python scripts/reset_data.py --yes --reset-ids --reset-vocab   # 重置演示数据
python tests/test_parent_child_demo.py       # 父子分块效果演示
python -m pytest tests -q                    # 全部测试（65 passed）
python -m ruff check app scripts tests       # 静态检查（All checks passed!）
```

`pytest` 全部通过（77 个用例）：

| 测试文件 | 用例数 | 覆盖重点 |
| --- | --- | --- |
| `tests/test_chunker.py` | 8 | 子块不跨父块边界、序号唯一、父块显著大于子块、`chunk_id` 确定性、章节标签、空文档、超长无标点段落兜底、**表格/代码块原子保护** |
| `tests/test_vectors.py` | 9 | 哈希向量确定性与归一化、相似文本得分更高、BM25 权重归一化、词表落盘、空查询占位、IDF 惩罚高频词 |
| `tests/test_e2e_pipeline.py` | 2 | 三存储集成：入库三处写入、检索回表、缓存命中、**元数据完整性**、**幽灵缓存不返回脏数据**、幂等重复入库、删除三处同步 |
| `tests/test_api_http.py` | 11 | 健康检查、首页、入库→检索→问答→对比→删除全流程、404/422 校验、**上传白名单拦截**、**上传正常入库**、**元数据跨缓存状态一致**、**stats 记忆化**、**错误响应不泄漏细节** |
| `tests/test_api_security.py` | 19 | 管理令牌校验、上传体积上限（413）、扩展名白名单、CORS 白名单与凭证关闭、统一文件解析、**监控接口不触发模型加载**、**启动预热容错** |
| `tests/test_optimizations.py` | 22 | 本机适配回归：ModelScope 路径解析（含 dotenv 反斜杠坑）、原子块保护、LLM 接入判定（本地服务无需 Key）、重排空实现语义、请求默认值 |
| `tests/test_parent_child_demo.py` | 6 | **★ 父子分块效果演示**。核心断言 —— 小块 RAG 的 top-1 上下文覆盖处置步骤 **0/5**，父子分块的父块覆盖 **5/5** |

## 十三、延伸阅读

| 文档 | 内容 |
| --- | --- |
| `docs/01-架构与原理.md` | 三存储协同、混合检索、分块策略、缓存设计的原理与取舍 |
| `docs/02-学习路径.md` | 6 个阶段 + 12 个动手实验 + 自查清单 |
| `docs/03-面试题库.md` | 45 道面试题（基础概念 / 核心原理 / 应用场景 / 高频考点）含参考答案 |
| `docs/04-优化建议（基于本机配置）.md` | 结合本机算力/模型/显存的评估与优化清单，含实测数据与 7B 模型「假拒绝」问题定位 |
| `docs/05-API.md` | 接口参考：请求/响应字段、错误码、安全约定、调用示例 |
| `docs/06-配置说明.md` | 全量配置项、取值范围、调参建议与三个典型场景速查 |
| `docs/07-代码审查报告.md` | 全面审查：已发现问题（按严重级）/ 已做的更新 / 遗留事项 |
| `CHANGELOG.md` | 版本变更记录 |
| `tests/test_parent_child_demo.py` | 可直接运行的父子分块效果演示（含预期输出说明） |
