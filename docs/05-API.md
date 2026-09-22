# API 参考

- **Base URL**：`http://127.0.0.1:8000`
- **前缀**：所有业务接口位于 `/api/v1`
- **交互式文档**：`GET /docs`（Swagger UI）、`GET /openapi.json`
- **数据格式**：请求与响应均为 `application/json; charset=utf-8`
  （文件上传接口除外，使用 `multipart/form-data`）
- **版本**：与 `app/__init__.py:__version__` 保持一致（当前 `1.1.0`）

---

## 0. 通用约定

### 0.1 错误响应

所有错误统一为如下结构：

```json
{ "detail": "错误说明" }
```

| 状态码 | 含义 | 典型场景 |
| --- | --- | --- |
| 400 | 请求内容不合法 | 正文为空、不支持的文件类型、文件无法解析 |
| 401 | 未通过管理令牌校验 | 调用 `/admin/*` 未带或带错 `X-Admin-Token` |
| 404 | 资源不存在 | `parent_id` 在 MySQL 中不存在 |
| 413 | 上传文件过大 | 超过 `UPLOAD_MAX_MB` |
| 422 | 参数校验失败 | 缺少必填字段、类型错误、超出取值范围（由 Pydantic 产生） |
| 500 | 服务内部错误 | **响应体只给通用文案，完整堆栈仅记录在服务端日志** |

> 安全说明：500 响应**不会**回显异常原文，避免泄漏连接串、表名、文件路径或
> 第三方 SDK 报文。排查请查看服务端日志（`logger.exception` 已记录 traceback）。

### 0.2 鉴权

- 业务接口（入库 / 检索 / 问答 / 统计）默认**不鉴权**，设计上作为内网服务，
  生产部署需在网关层补充认证。
- 管理接口 `/api/v1/admin/*` 受 `ADMIN_TOKEN` 控制：
  - `ADMIN_TOKEN` 为空 → 不校验（**仅限本机开发**）；
  - `ADMIN_TOKEN` 非空 → 必须携带请求头 `X-Admin-Token: <token>`。

### 0.3 CORS

来源白名单由 `CORS_ALLOW_ORIGINS` 控制（逗号分隔，默认 `http://127.0.0.1,http://localhost`）。
`allow_credentials` **显式关闭**，不使用 `*` 通配来源。

---

## 1. 运维接口

### GET `/api/v1/health`

返回三存储连通性与各组件接线情况。**不返回任何凭据**。

```json
{
  "mysql": true,
  "redis": true,
  "milvus": true,
  "detail": {
    "milvus": {
      "exists": true,
      "collection": "child_chunk_t",
      "fields": ["chunk_id", "parent_id", "doc_id", "title", "heading_path",
                 "chunk_index", "content", "dense_vector", "sparse_vector"],
      "indexes": ["sparse_vector_index", "dense_vector_index"],
      "row_count": 20
    },
    "llm": {
      "provider": "ollama", "mode": "openai",
      "base_url": "http://127.0.0.1:11434/v1", "model": "qwen2.5:7b",
      "has_api_key": true
    },
    "embedding": {
      "provider": "local", "dim": 1024, "device": "cpu", "batch_size": 16,
      "model_path": "D:/modelscope/hub/models/BAAI--bge-m3/snapshots/master",
      "query_instruction": false
    },
    "sparse": { "enabled": true, "vocab_size": 1081, "n_docs": 20, "avgdl": 96.15,
                "k1": 1.5, "b": 0.75 },
    "rerank": { "provider": "none", "available": false },
    "retrieval_ready": true
  }
}
```

> `detail.llm.mode` 为 `extractive` 时说明未接通真实模型（会降级为抽取式回答）；
> `detail.embedding.provider` 为 `hash` 时说明用的是离线占位向量，检索质量不代表生产水平。

**`retrieval_ready` 的含义（重要）**

该接口**不会触发**重资源懒加载（否则会用模型加载时间拖垮健康检查）。
因此：

| 值 | 含义 |
| --- | --- |
| `true` | 检索服务已初始化（启动预热已完成），`embedding` / `sparse` / `rerank` 是真实运行态 |
| `false` | 尚未就绪（预热未完成或失败），此时 `embedding` 返回**配置预期值**并带 `note` 字段说明 |

正常启动后（默认 `WARMUP_ON_STARTUP=true`）应为 `true`。若长期为 `false`，
说明预热失败 —— 查看服务端日志中「预热 … 失败」记录，通常是缺少依赖或模型路径不对。

本接口承诺**永不抛出异常**：Milvus 元数据读取失败时返回 `detail.milvus_error`
（只给通用文案，不回显内部异常原文）。实测响应时间约 **60ms**。

### GET `/api/v1/stats`

全链路统计。**结果有 5 秒记忆化**（因为内部包含一次 Redis `SCAN` 遍历）。

| 查询参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `force` | bool | `false` | 忽略记忆化，强制重新统计 |

```json
{
  "mysql_parent_count": 5,
  "mysql_doc_count": 3,
  "milvus_child_count": 20,
  "milvus_collection": "child_chunk_t",
  "redis_parent_keys": 5,
  "cache_hit_rate": 1.0,
  "cache_requests": 121,
  "cache_hits": 121,
  "embedding_provider": "local",
  "embedding_dim": 1024,
  "sparse_enabled": true,
  "llm_mode": "openai",
  "rerank_provider": "none"
}
```

---

## 2. 入库接口

### POST `/api/v1/documents`

文本入库（JSON）。

| 字段 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `title` | string | 否 | ≤255 | 文档标题，默认空 |
| `content` | string | **是** | 非空 | 文档正文，支持 Markdown |
| `doc_id` | string \| null | 否 | 落库时截断到 64 字符 | 不传则按 `sha256(title::content)` 生成 |
| `source` | string | 否 | ≤512 | 来源标识，默认 `api` |
| `force_reindex` | bool | 否 | 默认 `false` | `true` 时先删除同 `doc_id` 的旧父块/向量/缓存再重建 |

```json
{
  "doc_id": "doc_1f74ac59e9acc50a1689072f",
  "parent_count": 2,
  "child_count": 7,
  "parent_ids": [1, 2],
  "elapsed_ms": 3592.5,
  "detail": {
    "chunk_stats": {
      "parent_count": 2, "child_count": 7,
      "avg_parent_chars": 802.0, "avg_child_chars": 257.0,
      "max_parent_chars": 1343, "max_child_chars": 296
    },
    "redis_written": 2,
    "milvus_written": 7,
    "timing_ms": { "split": 0.27, "mysql": 88.98, "redis": 8.12, "milvus": 3495.12 },
    "embedding_provider": { "provider": "local", "dim": 1024, "device": "cpu" },
    "sparse": { "enabled": true, "vocab_size": 1081 }
  }
}
```

**幂等性**：同一 `doc_id` + 相同父块内容重复入库，会复用已有 `parent_id`，
不产生重复父块；子块按内容哈希 `upsert`，不产生重复向量。

### POST `/api/v1/documents/upload`

文件入库（`multipart/form-data`）。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | **是** | 扩展名必须在 `UPLOAD_ALLOW_EXT` 白名单内，体积 ≤ `UPLOAD_MAX_MB` |
| `doc_id` | string | 否 | 同文本入库 |
| `force_reindex` | bool | 否 | 同文本入库 |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
     -F "file=@data/samples/01-Milvus混合检索实践.md"
```

返回值与 `POST /documents` 相同。错误：非白名单扩展名 → `400`；超出体积 → `413`；
无法解析 → `400`。

### GET `/api/v1/documents`

文档列表（按入库时间倒序聚合）。

| 查询参数 | 类型 | 默认 | 范围 |
| --- | --- | --- | --- |
| `limit` | int | 100 | 1 ~ 1000 |

```json
[
  { "doc_id": "doc_a7ce...", "title": "03-父子分块检索优化手册",
    "parent_count": 2, "child_count": 7, "created_at": "2026-09-21 20:41:03" }
]
```

### DELETE `/api/v1/documents/{doc_id}`

删除文档，**三处同步清理**：MySQL 父块 → Redis 缓存 → Milvus 子块。

```json
{ "doc_id": "doc_a7ce...", "deleted_parents": 2, "deleted_children": 7 }
```

> 幂等：`doc_id` 不存在时返回 `deleted_parents=0`，不报错。

---

## 3. 检索与问答

### 请求体（`/search`、`/chat`、`/compare` 共用）

| 字段 | 类型 | 默认 | 范围 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | string | — | 非空 | 用户问题 |
| `top_k` | int | 20 | 1 ~ 200 | Milvus 召回的**子块**数量 |
| `final_top_k` | int | 4 | 1 ~ 50 | 去重后送入模型的**父块**数量 |
| `use_sparse` | bool | `true` | — | 是否启用稀疏（BM25）分支 |
| `explain` | bool | `false` | — | 是否额外跑单分支对照检索（**会多 2 次 Milvus 查询**） |
| `max_parent_chars` | int \| null | 全局配置 | — | 仅 `/chat`：单个父块截断长度 |

### POST `/api/v1/search`

只检索、不生成。

```json
{
  "query": "父子分块为什么比固定分块好？",
  "parents": [
    {
      "parent_id": 4,
      "doc_id": "doc_a7ce...",
      "title": "03-父子分块检索优化手册",
      "heading_path": "父子分块检索优化手册 > 一、固定分块的困境 ~ 六、参数调优建议",
      "content": "最朴素的 RAG 做法是把文档按固定字数切成 500 字的块……",
      "score": 0.056466,
      "cache_hit": true,
      "child_hits": [
        {
          "chunk_id": "9f44767f9b2c...",
          "parent_id": 4,
          "doc_id": "doc_a7ce...",
          "chunk_index": 0,
          "content": "最朴素的 RAG 做法是把文档按固定字数切成 500 字的块，然后向量化入库。",
          "score": 0.032018,
          "rerank_score": null,
          "dense_rank": 1,
          "sparse_rank": 4
        }
      ]
    }
  ],
  "timings": {
    "embed_ms": 97.88, "milvus_ms": 7.28, "milvus_hits": 20,
    "rerank_ms": 0.01, "rerank_enabled": 0,
    "group_ms": 0.07, "redis_ms": 6.56, "mysql_fallback_ms": 0.0,
    "total_ms": 124.28, "parent_count": 4
  }
}
```

**`timings` 字段含义**

| 字段 | 含义 |
| --- | --- |
| `embed_ms` | 查询向量化（稠密 + 稀疏） |
| `milvus_ms` | Milvus 混合检索（含 RRF 融合） |
| `milvus_hits` | 命中的子块数量 |
| `rerank_ms` | Cross-Encoder 重排耗时（未启用时约 0） |
| `rerank_enabled` | `1` 已启用重排 / `0` 未启用 |
| `group_ms` | 按 `parent_id` 聚合去重 + 父块打分 + 低分过滤 |
| `redis_ms` | `MGET` 批量取父块正文 |
| `mysql_fallback_ms` | 缓存未命中时回表 MySQL 的耗时 |
| `total_ms` | 检索阶段总耗时（不含 LLM） |
| `parent_count` | 最终返回的父块数量 |

> `dense_rank` / `sparse_rank` 仅在 `explain=true` 时填充，表示该子块在单分支中的排名；
> `rerank_score` 仅在启用重排时填充（sigmoid 归一化到 0~1）。

### POST `/api/v1/chat`

完整 RAG 问答：检索 → 父块装配 → 大模型生成。

```json
{
  "query": "报警代码 E-1042 的处理方法是什么？",
  "answer": "根据报警代码 E-1042 的处理方法如下：\n1. 立即停机并切断主轴电源……",
  "citations": [
    { "parent_id": 6, "doc_id": "demo_fault_manual",
      "title": "fault_manual", "heading_path": "数控加工中心故障诊断手册 > E-1042 主轴轴承过热" }
  ],
  "llm_mode": "openai",
  "parents": [ "…结构同 /search…" ],
  "timings": { "…同上…", "llm_ms": 6405.85, "total_ms": 6939.75 }
}
```

**`llm_mode` 取值**

| 值 | 含义 |
| --- | --- |
| `openai` | 成功调用真实大模型（含本地 Ollama） |
| `extractive` | 未接通模型，降级为抽取式回答（答案由原文句子拼成并标注出处） |
| `extractive(fallback)` | 模型调用失败后降级（详情见服务端日志） |

### POST `/api/v1/compare`

同一 query 的稠密 / 稀疏 / 混合三路召回对比（用于调参与可解释性分析）。

```json
{
  "query": "SPARSE_FLOAT_VECTOR 的下标要注意什么？",
  "sparse_query_tokens": 5,
  "dense_only":  [ { "chunk_id": "…", "parent_id": 1, "score": 0.52, "content": "…" } ],
  "sparse_only": [ { "chunk_id": "…", "parent_id": 1, "score": 3.14, "content": "…" } ],
  "hybrid":      [ { "chunk_id": "…", "parent_id": 1, "score": 0.033, "content": "…" } ],
  "analysis": {
    "dense_parents": 3, "sparse_parents": 3, "overlap_parents": 3,
    "only_in_sparse": [], "only_in_dense": [2, 4]
  },
  "timings": { "vectorize_ms": 138.12, "search_ms": 12.98 }
}
```

> 注意：三种路径的 `score` **量纲不同**，不可跨列直接比较。
> 稠密分支是余弦相似度，稀疏分支是 BM25 内积，混合分支是 RRF 融合分。

### GET `/api/v1/parents/{parent_id}`

按父文档 id 直取正文（演示 Redis 缓存价值）。

```json
{
  "id": 4,
  "doc_id": "doc_a7ce...",
  "title": "03-父子分块检索优化手册",
  "heading_path": "父子分块检索优化手册 > 一、固定分块的困境 ~ 六、参数调优建议",
  "child_count": 6,
  "content": "最朴素的 RAG 做法是把文档按固定字数切成 500 字的块……",
  "created_at": "2026-09-21 20:41:03",
  "from_redis": true,
  "elapsed_ms": 0.42
}
```

> **v1.1.0 行为变更**：元数据（`doc_id/title/heading_path/child_count/created_at`）
> **始终**从 MySQL 读取，因此无论 `from_redis` 为何都为完整值。
> `from_redis` 仅标识**正文**来源；`elapsed_ms` 为正文获取耗时。
> 若 Redis 有残留但 MySQL 已无该行，返回 `404`（避免"幽灵"数据）。

---

## 4. 管理接口

以下接口在配置 `ADMIN_TOKEN` 后需要请求头 `X-Admin-Token`。

### POST `/api/v1/admin/reindex`

重建 Milvus 集合（**会清空全部向量**，之后需重新灌数据）。

| 查询参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `drop` | bool | `true` | 是否先删除已有集合 |

```json
{
  "milvus": { "created": true, "collection": "child_chunk_t",
              "dim": 1024, "sparse_enabled": true },
  "note": "重建集合后需重新执行 scripts/ingest_samples.py 灌数据"
}
```

**何时必须重建**：切换 Embedding 模型导致向量维度变化时；
或修改 `SPARSE_ENABLED` 改变集合字段结构时。仅更换模型但维度不变（如
`hash` → `BGE-M3`，均为 1024 维）时集合结构无需变更，但**数据必须重灌**。

### POST `/api/v1/admin/reset-cache-stats`

重置 Redis 缓存命中率统计，并清空 `/stats` 的记忆化。

```json
{ "ok": true }
```

---

## 5. 静态页面

| 路径 | 说明 |
| --- | --- |
| `GET /` | 可视化调试台（`web/index.html`），含问答 / 检索链路 / 三路对比 / 入库 / 文档管理五个页签 |
| `GET /docs` | Swagger UI |
| `GET /openapi.json` | OpenAPI 3 规范 |
| `GET /favicon.ico` | 返回 204 |

---

## 6. 调用示例

```bash
# 文本入库
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"title":"示例","content":"# 标题\n\n正文内容……"}'

# 检索（开启单分支对照）
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"父子分块好处","top_k":20,"final_top_k":4,"explain":true}'

# 问答
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"父子分块为什么比固定分块好？"}'

# 按父文档 id 取正文
curl http://127.0.0.1:8000/api/v1/parents/1

# 管理接口（配置了 ADMIN_TOKEN 时）
curl -X POST "http://127.0.0.1:8000/api/v1/admin/reindex?drop=true" \
  -H "X-Admin-Token: <your-token>"
```
