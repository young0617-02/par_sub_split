"""API 与内部流转共用的 Pydantic 数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# 入库
# --------------------------------------------------------------------------
class DocumentIn(BaseModel):
    """单篇文档入库请求。"""

    doc_id: str | None = Field(
        default=None,
        description="业务文档唯一标识；不传则根据 title+content 自动生成",
    )
    title: str = Field(default="", max_length=255, description="文档标题")
    content: str = Field(min_length=1, description="文档正文")
    source: str = Field(default="api", max_length=512, description="来源标识/文件名")
    force_reindex: bool = Field(
        default=False,
        description="为 true 时，同一 doc_id 的旧父块/向量会被先删除再重建",
    )


class IngestResult(BaseModel):
    doc_id: str
    parent_count: int
    child_count: int
    parent_ids: list[int]
    elapsed_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    top_k: int = Field(default=20, ge=1, le=200, description="Milvus 子块召回数量")
    final_top_k: int = Field(default=4, ge=1, le=50, description="去重后保留的父块数量")
    use_sparse: bool = Field(default=True, description="是否参与稀疏(BM25)分支检索")
    explain: bool = Field(
        default=False,
        description="是否额外跑稠密/稀疏单分支检索用于对照（会多 2 次 Milvus 查询，默认关闭）",
    )


class ChildHit(BaseModel):
    """命中的子块。"""

    chunk_id: str
    parent_id: int
    doc_id: str
    chunk_index: int
    content: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = Field(
        default=None, description="Cross-Encoder 重排后的归一化分数（未启用重排时为 None）"
    )


class ParentHit(BaseModel):
    """子块回表得到的父文档。"""

    parent_id: int
    doc_id: str
    title: str
    heading_path: str
    content: str
    score: float
    child_hits: list[ChildHit] = Field(default_factory=list)
    cache_hit: bool = Field(default=False, description="父文档正文是否命中 Redis 缓存")
    # 说明：父块正文是批量（MGET）取回的，无法按单个父块拆分耗时，
    # 因此这里不提供 per-parent 耗时字段；阶段耗时统一见 SearchResponse.timings。


class SearchResponse(BaseModel):
    query: str
    parents: list[ParentHit]
    timings: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 问答
# --------------------------------------------------------------------------
class ChatRequest(SearchRequest):
    max_parent_chars: int | None = Field(
        default=None, description="单个父块送模型的字符上限，默认取全局配置"
    )


class Citation(BaseModel):
    parent_id: int
    doc_id: str
    title: str
    heading_path: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    llm_mode: str = Field(description="openai=真实大模型调用 / extractive=无密钥降级模式")
    parents: list[ParentHit]
    timings: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 运维
# --------------------------------------------------------------------------
class ParentDocumentOut(BaseModel):
    id: int
    doc_id: str
    title: str
    heading_path: str
    child_count: int
    content: str
    created_at: str | None = None
    from_redis: bool = False
    elapsed_ms: float = 0.0


class StatsResponse(BaseModel):
    mysql_parent_count: int
    mysql_doc_count: int
    milvus_child_count: int
    milvus_collection: str
    redis_parent_keys: int
    cache_hit_rate: float
    cache_requests: int
    cache_hits: int
    embedding_provider: str
    embedding_dim: int
    sparse_enabled: bool
    llm_mode: str
    rerank_provider: str = "none"


class HealthResponse(BaseModel):
    mysql: bool
    redis: bool
    milvus: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class DeleteResult(BaseModel):
    doc_id: str
    deleted_parents: int
    deleted_children: int
