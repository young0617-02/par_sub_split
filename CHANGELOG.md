# 变更日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

---

## [1.2.0] - 2026-09-22

本版本把**全部依赖服务统一到一份 Docker 编排**，并同步更新代码与文档，使
「哪些容器、叫什么名字、映射什么端口」只有一处定义。

### 新增

- **统一 Docker 编排**：`docker-compose.yml` 成为全部 5 个依赖服务
  （MySQL / Redis / etcd / MinIO / Milvus）的**唯一事实来源**，
  一条 `docker compose up -d` 拉起全部；通过顶层 `name: par_sub_split` 固定 Compose 项目名，
  不再依赖目录名。

### 变更（部署）

- **容器命名统一为 `par_sub_*`**：Redis 容器由 `redis8` 改为 **`par_sub_redis`**，
  与 `par_sub_mysql` / `par_sub_etcd` / `par_sub_minio` / `par_sub_milvus` 保持一致。
  迁移步骤：先 `docker rm par_sub_redis`（移除早期遗留的同名容器），再 `docker compose up -d`。
- **MinIO 环境变量现代化**：`MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`（已废弃）
  → `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`。凭据值保持不变（`minioadmin/minioadmin`），
  因此 Milvus 的 `MINIO_ADDRESS=minio:9000` 连接不受影响。
- **移除 profile 分支**：原编排把 Milvus 及其依赖标记在 `milvus` profile 下，
  但实际已全部随默认 profile 启动，故删除该分支与注释，
  `scripts/start.py` 的 `compose_up()` 相应简化为单组调用（超时放宽到 600s，适配 Milvus 首启）。
- **`scripts/start.py` 容器名优先级反转**：优先探测本项目 compose 的 `par_sub_*` 容器名，
  早期遗留容器名（`mysql84` / `milvus-standalone` / `milvus-etcd` / `milvus-minio` / `redis8`）
  降级为兼容回退；失败提示中的容器名同步修正为 `par_sub_milvus`。

### 变更（数据）

- ⚠️ **统一编排引入了新的命名卷（`par_sub_split_*`），旧容器中的数据不会自动迁移**。
  切换后需重新初始化并灌数据：`python scripts/init_db.py` → `python scripts/ingest_samples.py`。
  该现象与处置方式已写入 `docs/06-配置说明.md` 排查表。

### 变更（文档）

- `README.md`：「前置：启动依赖服务」重写为统一编排流程，附**服务 / 容器 / 端口**对照表，
  并说明「MySQL 为什么用 9999」与首次部署注意事项；新增遗留容器可选清理指引。
- `docs/02-学习路径.md`：阶段 1/阶段 2 的动手命令全部改为 `docker compose` 与 `par_sub_*` 容器名。
- `docs/06-配置说明.md`：MySQL 端口说明改为引用 compose 服务名；排查表新增
  「换用统一编排后数据『消失』」条目，并修正 Docker 重启后的恢复方式。
- `docs/07-代码审查报告.md`：环境问题表格的处置方式改为 `docker compose up -d`。
- `.env` / `.env.example`：新增说明段，明确三个存储服务由 `docker-compose.yml` 统一编排，
  容器名一律为 `par_sub_*`，端口须与 compose 映射一致。
- `scripts/init_db.py`：失败提示改为给出 `docker compose up -d <service>` 与容器名两种方式。
- `tests/test_parent_child_demo.py`：环境不可用时的提示改为 `docker compose up -d`。

---

## [1.1.0] - 2026-09-21

本版本为**全面代码审查 + 本机配置适配**的合并发布，包含 8 项安全/正确性修复、
4 项本机模型接入能力、3 个新文档与 2 个新测试文件。

### 新增（本机模型与推理服务适配）

- **本地嵌入模型接入**：`LocalEmbedder` 支持 `cpu/cuda` + `fp16` + 批大小 + 查询指令前缀；
  自动解析 ModelScope 本地快照（本机 `BAAI/bge-m3`，1024 维与集合定义一致，**无需改表**）。
- **本地大模型接入**：`LLM_PROVIDER=ollama` 走本机 `http://127.0.0.1:11434/v1`，
  **无需 API Key**（此前"Key 为空即判为降级"会误伤本地服务）。
- **可选 Rerank 精排**：新增 `app/services/reranker.py`（Cross-Encoder，作用在**子块**上），
  依赖缺失或加载失败时自动降级为空实现，不影响主链路。
- **本地环境自检脚本**：`scripts/check_local_models.py`，一次性检查模型文件、Python 依赖、
  显存预算与 LLM 连通性。
- **父子分块效果演示**：`tests/test_parent_child_demo.py`，可独立运行，
  内置「小块 RAG vs 父子分块」的量化对照（答案覆盖率 0/5 vs 5/5）。
- **新文档**：`docs/05-API.md`（接口参考）、`docs/06-配置说明.md`（配置参考）、
  `docs/04-优化建议（基于本机配置）.md`（本机评估与优化清单）。
- **代码质量门禁**：新增 `ruff.toml`，`python -m ruff check app scripts tests` 全绿。

### 修复（安全）

- **上传接口缺少防护**（中危）：原实现 `await file.read()` 一次性读入内存且不校验类型，
  超大文件或任意文件可造成内存耗尽与解析告警。现改为**流式限量读取**（超限返回 413）
  + **扩展名白名单**（`UPLOAD_ALLOW_EXT`）。
- **错误响应泄漏内部信息**（中危）：`HTTPException` 与全局异常处理器此前把
  `str(exc)` 直接返回给调用方，会暴露连接串、表名、第三方 SDK 报文等。
  现响应体统一为通用文案，完整堆栈仅进服务端日志。
- **CORS 配置错误**（中危）：原为 `allow_origins=["*"]` + `allow_credentials=True`，
  这是浏览器会拒绝的无效组合，且等价于对所有站点开放凭证请求。
  现改为**来源白名单 + 关闭凭证**，来源通过 `CORS_ALLOW_ORIGINS` 配置。
- **管理接口无保护**（中危）：`/admin/reindex` 可清空整个向量集合却无需任何凭据。
  现支持 `ADMIN_TOKEN`，配置后必须携带 `X-Admin-Token`；未配置时保持本机免校验并打印告警日志。

### 修复（可用性）

- **健康检查被模型加载阻塞**（高）：`/health` 与 `/stats` 内部会访问检索服务，
  而检索服务是懒加载的 —— 首次调用会连带加载嵌入模型（本机 BGE-M3 CPU 约 6~10s）
  与 jieba 词典，导致健康检查**超时**，上游会误判服务不可用，第一个业务请求也会
  额外等待这段冷启动。修复方式：
  1. **启动预热**：`WARMUP_ON_STARTUP=true`（默认开启），启动阶段一次性加载
     BM25 词表 / 嵌入模型 / 重排模型；单个组件失败只记日志、不阻断启动（自动走降级路径）。
  2. **监控接口不触碰懒加载资源**：未就绪时返回配置预期值并标注 `detail.retrieval_ready=false`。
  3. `/health` 增加兜底：Milvus 元数据读取失败也只记录日志并返回错误标记，接口自身永不抛异常。

  实测 `/health` 响应时间从**超时（>10s）降到 0.061s**。
- **缓存超时过短**：`REDIS_SOCKET_TIMEOUT` 由 5s 放宽到 **10s**。
  高负载（如同时加载本地嵌入模型）时 5s 容易误判超时，使入库跳过缓存写入、
  白白退化为回表。缓存故障本就不影响正确性，超时阈值应留足余量。

### 修复（正确性）

- **`/api/v1/parents/{id}` 响应随缓存状态抖动**：缓存命中时只回正文、不返回元数据，
  导致 `doc_id/title/heading_path/child_count` 变空。现**元数据始终从 MySQL 读取**
  （只 SELECT 少数字段，不加载 LONGTEXT 正文），两条路径响应结构完全一致。
- **幽灵缓存**：Redis 中存在残留 key 但 MySQL 行已删除时，此前会返回脏数据。
  现以 MySQL 为准返回 404。
- **统计接口被轮询放大**：`/stats` 每次调用都对 Redis 做 `SCAN` 遍历，
  而调试台每 20 秒轮询一次。现增加 5 秒记忆化并支持 `?force=true` 强制刷新。
- **`group_ms` 指标口径错误**：该阶段耗时把 Rerank 耗时也算在内，与指标名不符。
  现改为从 Rerank 结束时刻起算。

### 修复（代码质量）

- 清除重复实现：路由层与 `services/ingest.py` 各有一份 PDF 解析，现统一为
  `extract_text_from_bytes()` / `extract_pdf_bytes()`。
- 清除死代码与未使用项：`UPLOAD_DIR` 常量（从未使用却在导入时创建目录）、
  `app/config.py` 的 `Field` 导入、`scripts/init_db.py` 的无用导入、
  `scripts/demo_query.py` 的未使用变量。
- 修复 `app/services/llm.py` 的循环变量被覆盖（`PLW2901`）、未使用循环变量（`B007`），
  以及降级日志应使用 `logger.exception`（`TRY400`）。
- 消除 `setattr` 常量赋值、无占位符 f-string、多余中间变量等 7 处写法问题。
- `.gitignore` 补充 `.pytest_cache/`、`.ruff_cache/`、`build/`、`dist/`、`*.egg-info/`。
- `requirements.txt` 补齐 `pytest`、`ruff`，并把本地模型依赖明确标注为可选。

### 变更（行为）

- `ParentHit` / `ChildHit` 接口新增 `rerank_score` 字段（未启用重排时为 `null`）。
- `/api/v1/health` 的 `detail` 新增 `retrieval_ready` 字段，用于区分
  「组件已初始化」与「仅返回配置预期值」。
- `SearchRequest` 与 `ChatRequest` 新增 `explain` 字段（默认 `false`）：
  此前 `/search` 硬编码开启单分支对照检索，会让检索延迟翻倍。
- `StatsResponse` 新增 `rerank_provider` 字段。
- `LLM_MODEL` 在 `LLM_PROVIDER=ollama` 且仍为 `gpt-4o-mini` 时，自动改用 `LLM_OLLAMA_MODEL`。

### 变更（文档）

- `README.md`：新增「接入本机已有模型」步骤、依赖与安全说明、更新测试数量与文件清单。
- `docs/01-架构与原理.md`：补充本次实测数据与修复记录。
- `docs/04-优化建议（基于本机配置）.md`：新增实测验证记录与代码改动清单。
- 新增 `docs/05-API.md`、`docs/06-配置说明.md`。

---

## [1.0.0] - 2026-09-21

首个可运行版本，实现《父子分块的练习 面试加实战》中的完整架构。

### 新增

- **三存储协同**：MySQL（父块持久化 + 自增 id）、Redis（`psd:parent:{id}` 缓存父块正文）、
  Milvus（子块稠密 + 稀疏向量，携带 `parent_id`）。
- **父子分块器**：Markdown 章节感知的两级切分，跨小节合并保证父块粒度，
  `chunk_id` 内容哈希保证入库幂等；子块严格不跨越父块边界。
- **BM25 稀疏向量**：jieba 分词 + 词表与 IDF 统计落盘，输出 Milvus 稀疏向量；
  含空向量占位、L2 归一化等工程处理。
- **混合检索**：Milvus `hybrid_search` + `RRFRanker`，按 `parent_id` 聚合去重，
  `max(子块分) × (1 + 0.15 × 额外命中数)` 计算父块分数，低分父块过滤。
- **Cache-Aside 回表**：Redis `MGET` 批量取父块正文，未命中回表 MySQL 并回填。
- **RAG 问答**：OpenAI 兼容协议调用大模型，强制标注 `[父文档ID: x]` 引用；
  无可用模型时降级为抽取式回答并明确标识。
- **11 个 REST 接口** + 可视化调试台（`web/index.html`，五个功能页签）。
- **脚本**：初始化、样例灌入、端到端演示、数据重置、服务启动。
- **文档**：`docs/01-架构与原理.md`、`docs/02-学习路径.md`、`docs/03-面试题库.md`（45 题）。
- **测试**：分块器、向量编码器、三存储集成、HTTP 接口。

[1.2.0]: #120---2026-09-22
[1.1.0]: #110---2026-09-21
[1.0.0]: #100---2026-09-21
