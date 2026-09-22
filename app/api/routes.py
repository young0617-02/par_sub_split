"""REST 接口定义。

接口一览：
    GET    /api/v1/health              三件套连通性
    GET    /api/v1/stats               全链路统计（含缓存命中率）
    GET    /api/v1/documents           已入库文档列表
    POST   /api/v1/documents           文本入库
    POST   /api/v1/documents/upload    文件入库（md/txt/pdf）
    DELETE /api/v1/documents/{doc_id}  删除文档（三处同步清理）
    POST   /api/v1/search              只检索，返回「子块 + 父块」
    POST   /api/v1/chat                完整 RAG 问答
    POST   /api/v1/compare             稠密 / 稀疏 / 混合 三路对比实验
    GET    /api/v1/parents/{id}        按父文档 id 直取内容（Redis 优先）
    POST   /api/v1/admin/reindex       重建 Milvus 集合（维度变更时必须执行）
    POST   /api/v1/admin/reset-cache-stats  重置缓存命中率统计

安全约定（详见 docs/05-API.md）：
    · 错误响应只返回通用文案，内部异常细节（连接串/表名/堆栈）仅进日志；
    · 上传接口有扩展名白名单与体积上限，避免任意文件与内存耗尽；
    · ``/admin/*`` 在配置 ``ADMIN_TOKEN`` 后需要请求头 ``X-Admin-Token`` 才能调用。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from sqlalchemy import func, select

from app.clients import mysql as mysql_client
from app.clients import redis_client
from app.clients.milvus_client import get_chunk_store
from app.clients.mysql import session_scope
from app.config import settings
from app.models import ParentDocument
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResult,
    DocumentIn,
    HealthResponse,
    IngestResult,
    ParentDocumentOut,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from app.services.ingest import extract_text_from_bytes, get_ingest_service
from app.services.llm import get_answer_generator
from app.services.rag import get_rag_service
from app.services.retriever import get_retrieval_service, is_retrieval_service_ready

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["parent-child-rag"])


# ==========================================================================
# 通用依赖
# ==========================================================================
def require_admin(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    """管理接口保护。

    ``ADMIN_TOKEN`` 为空时不校验（仅适合本机开发）；
    一旦配置为非空值，``/admin/*`` 必须携带一致的请求头。
    """
    expected = settings.admin_token.strip()
    if not expected:
        return
    if (x_admin_token or "").strip() != expected:
        raise HTTPException(status_code=401, detail="管理接口需要有效的 X-Admin-Token")


AdminGuard = Depends(require_admin)


# ==========================================================================
# 运维
# ==========================================================================
@router.get("/health", response_model=HealthResponse, summary="连通性检查")
def health() -> HealthResponse:
    """返回三存储连通性与各组件接线情况（不含任何凭据）。

    设计约束：健康检查**绝不触发**重资源懒加载，也**绝不抛异常**。
    - 组件接线信息只在检索服务已初始化（启动预热完成）时读取；
      未就绪时报告配置中的预期值，避免监控请求被模型加载阻塞。
    - LLM 客户端不加载本地模型，可以安全地随时读取。
    """
    store = get_chunk_store()
    detail: dict[str, object] = {}
    mysql_ok = mysql_client.ping()
    redis_ok = redis_client.ping()
    milvus_ok = store.ping()
    try:
        detail["milvus"] = store.describe()
    except Exception as exc:  # noqa: BLE001  健康检查本身不应抛异常
        logger.warning("Milvus describe 失败：%s", exc)
        detail["milvus_error"] = "Milvus 元数据读取失败，详见服务端日志"

    detail["llm"] = get_answer_generator().describe()
    ready = is_retrieval_service_ready()
    detail["retrieval_ready"] = ready
    if ready:
        service = get_retrieval_service()
        detail["embedding"] = service.embedder.describe()
        detail["sparse"] = service.encoder.describe()
        detail["rerank"] = service.reranker.describe()
    else:
        # 预热尚未完成（或失败）：给出配置中的期望值，并明确标注未就绪
        detail["embedding"] = {
            "provider": settings.embedding_provider,
            "dim": settings.milvus_dim_resolved,
            "note": "检索服务未就绪，展示的是配置值",
        }
        detail["rerank"] = {
            "provider": "local-cross-encoder" if settings.rerank_enabled else "none",
            "available": False,
        }
    return HealthResponse(mysql=mysql_ok, redis=redis_ok, milvus=milvus_ok, detail=detail)


# /stats 里的 redis 父块数量依赖 SCAN 遍历，key 数量大时成本随规模线性增长。
# 前端调试台会周期性轮询该接口，因此加一个短 TTL 的记忆化，避免放大开销。
_STATS_MEMO_TTL_SECONDS = 5.0
_stats_memo: tuple[float, StatsResponse] | None = None


@router.get("/stats", response_model=StatsResponse, summary="全链路统计")
def stats(
    force: bool = Query(default=False, description="忽略 5 秒记忆化，强制重新统计"),
) -> StatsResponse:
    global _stats_memo
    now = time.monotonic()
    if not force and _stats_memo is not None and now - _stats_memo[0] < _STATS_MEMO_TTL_SECONDS:
        return _stats_memo[1]

    store = get_chunk_store()
    cache = redis_client.get_parent_cache()

    with session_scope() as session:
        parent_count = int(session.execute(select(func.count(ParentDocument.id))).scalar() or 0)
        doc_count = int(
            session.execute(select(func.count(func.distinct(ParentDocument.doc_id)))).scalar() or 0
        )

    # 与 /health 同样的约束：统计接口不应触发嵌入模型懒加载
    if is_retrieval_service_ready():
        service = get_retrieval_service()
        embedding_provider, embedding_dim = service.embedder.name, service.embedder.dim
        rerank_provider = service.reranker.name
    else:
        embedding_provider = settings.embedding_provider
        embedding_dim = settings.milvus_dim_resolved
        rerank_provider = "local-cross-encoder" if settings.rerank_enabled else "none"

    cache_stats = cache.stats()
    response = StatsResponse(
        mysql_parent_count=parent_count,
        mysql_doc_count=doc_count,
        milvus_child_count=store.count(),
        milvus_collection=store.collection_name,
        redis_parent_keys=cache.count_keys(),
        cache_hit_rate=cache_stats["hit_rate"],
        cache_requests=cache_stats["requests"],
        cache_hits=cache_stats["hits"],
        embedding_provider=embedding_provider,
        embedding_dim=embedding_dim,
        sparse_enabled=settings.sparse_enabled,
        llm_mode=get_answer_generator().mode,
        rerank_provider=rerank_provider,
    )
    _stats_memo = (now, response)
    return response


# ==========================================================================
# 入库
# ==========================================================================
async def read_upload_limited(file: UploadFile, limit_bytes: int) -> bytes:
    """分块读取上传内容，超过上限立刻中断。

    必要性：直接 ``await file.read()`` 会把整个文件读进内存，
    超大文件（或恶意构造）可轻易打爆进程内存。这里改成流式累计并提前拒绝。
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过上限 {settings.upload_max_mb}MB，请拆分后再上传",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/documents", response_model=IngestResult, summary="文本入库")
def create_document(doc: DocumentIn) -> IngestResult:
    try:
        return get_ingest_service().ingest_text(
            content=doc.content,
            title=doc.title,
            doc_id=doc.doc_id,
            source=doc.source,
            force_reindex=doc.force_reindex,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("文本入库失败")
        raise HTTPException(
            status_code=500, detail="入库失败，请查看服务端日志（traceback 已记录）"
        ) from exc


@router.post("/documents/upload", response_model=IngestResult, summary="文件入库")
async def upload_document(
    file: Annotated[UploadFile, File(description="待入库文件")],
    doc_id: Annotated[str | None, Form()] = None,
    force_reindex: Annotated[bool, Form()] = False,
) -> IngestResult:
    """上传 md / txt / pdf 并入库。

    两道防线：扩展名白名单（避免任意文件被当文本解析）+ 体积上限（避免内存耗尽）。
    """
    name = file.filename or "untitled"
    suffix = Path(name).suffix.lower()
    allowed = settings.upload_allow_ext_set
    if allowed and suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix or '(无扩展名)'}，允许：{', '.join(sorted(allowed))}",
        )

    raw = await read_upload_limited(file, settings.upload_max_bytes)
    try:
        text = extract_text_from_bytes(raw, name)
    except Exception as exc:
        logger.warning("文件解析失败：%s", exc)
        raise HTTPException(status_code=400, detail="文件解析失败，请确认格式是否正确") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析出文本")

    try:
        return get_ingest_service().ingest_text(
            content=text,
            title=Path(name).stem,
            doc_id=doc_id,
            source=name[:512],
            force_reindex=force_reindex,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("文件入库失败")
        raise HTTPException(
            status_code=500, detail="入库失败，请查看服务端日志（traceback 已记录）"
        ) from exc


@router.get("/documents", summary="文档列表")
def list_documents(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    with session_scope() as session:
        rows = session.execute(
            select(
                ParentDocument.doc_id,
                func.min(ParentDocument.title).label("title"),
                func.count(ParentDocument.id).label("parent_count"),
                func.sum(ParentDocument.child_count).label("child_count"),
                func.min(ParentDocument.created_at).label("created_at"),
            )
            .group_by(ParentDocument.doc_id)
            .order_by(func.min(ParentDocument.created_at).desc())
            .limit(limit)
        ).all()
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "parent_count": int(r.parent_count or 0),
            "child_count": int(r.child_count or 0),
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
        }
        for r in rows
    ]


@router.delete("/documents/{doc_id}", response_model=DeleteResult, summary="删除文档")
def delete_document(doc_id: str) -> DeleteResult:
    result = get_ingest_service().delete_document(doc_id)
    return DeleteResult(
        doc_id=doc_id,
        deleted_parents=result["deleted_parents"],
        deleted_children=result["deleted_children"],
    )


# ==========================================================================
# 检索 / 问答
# ==========================================================================
@router.post("/search", response_model=SearchResponse, summary="父子分块检索")
def search(req: SearchRequest) -> SearchResponse:
    return get_retrieval_service().search(
        query=req.query,
        top_k=req.top_k,
        final_top_k=req.final_top_k,
        use_sparse=req.use_sparse,
        # 单分支对照检索会额外发起 2 次 Milvus 查询（延迟约翻倍），
        # 因此默认关闭，只在需要分析召回来源时显式打开
        explain=req.explain,
    )


@router.post("/chat", response_model=ChatResponse, summary="RAG 问答")
def chat(req: ChatRequest) -> ChatResponse:
    return get_rag_service().answer(
        query=req.query,
        top_k=req.top_k,
        final_top_k=req.final_top_k,
        use_sparse=req.use_sparse,
        max_parent_chars=req.max_parent_chars,
    )


@router.post("/compare", summary="稠密 / 稀疏 / 混合 三路对比")
def compare(req: SearchRequest) -> dict:
    """对比实验：同一 query 分别走稠密、稀疏、混合三条路径，观察召回差异。"""
    svc = get_retrieval_service()
    t0 = time.perf_counter()
    dense_vector = svc.embedder.encode_query(req.query)
    sparse_vector = svc.encoder.encode(req.query) if req.use_sparse else None
    t_vec = time.perf_counter()

    hybrid = svc.store.hybrid_search(dense_vector, sparse_vector, limit=req.top_k)
    dense_only = svc.store.search_single_branch("dense", dense_vector=dense_vector, limit=req.top_k)
    sparse_only = (
        svc.store.search_single_branch("sparse", sparse_vector=sparse_vector, limit=req.top_k)
        if sparse_vector
        else []
    )
    t_search = time.perf_counter()

    def compact(hits: list[dict]) -> list[dict]:
        """按 parent_id 去重并压缩字段，便于对比展示。"""
        seen: set[int] = set()
        out: list[dict] = []
        for h in hits:
            entity = h["entity"]
            pid = int(entity.get("parent_id") or 0)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(
                {
                    "chunk_id": h["chunk_id"],
                    "parent_id": pid,
                    "score": round(float(h["score"]), 6),
                    "content": str(entity.get("content", ""))[:120],
                }
            )
        return out

    # 每路只压缩一次，避免同一列表被重复遍历（历史实现调用了两次 compact）
    dense_view = compact(dense_only)
    sparse_view = compact(sparse_only)
    hybrid_view = compact(hybrid)
    dense_pids = {item["parent_id"] for item in dense_view}
    sparse_pids = {item["parent_id"] for item in sparse_view}

    return {
        "query": req.query,
        "sparse_query_tokens": len(sparse_vector or {}),
        "dense_only": dense_view,
        "sparse_only": sparse_view,
        "hybrid": hybrid_view,
        "analysis": {
            "dense_parents": len(dense_pids),
            "sparse_parents": len(sparse_pids),
            "overlap_parents": len(dense_pids & sparse_pids),
            "only_in_sparse": sorted(sparse_pids - dense_pids),
            "only_in_dense": sorted(dense_pids - sparse_pids),
        },
        "timings": {
            "vectorize_ms": round((t_vec - t0) * 1000, 2),
            "search_ms": round((t_search - t_vec) * 1000, 2),
        },
    }


@router.get("/parents/{parent_id}", response_model=ParentDocumentOut, summary="按父文档 id 取内容")
def get_parent(parent_id: int) -> ParentDocumentOut:
    """按父文档 id 取正文。

    无论正文来自 Redis 还是 MySQL，响应字段都保持完整一致
    （``from_redis`` 仅标识正文来源，``elapsed_ms`` 为正文获取耗时）。
    """
    result = get_retrieval_service().get_parent_document(parent_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"父文档 {parent_id} 不存在")

    meta: dict = result.get("meta") or {}
    created_at = meta.get("created_at")
    return ParentDocumentOut(
        id=parent_id,
        doc_id=str(meta.get("doc_id") or ""),
        title=str(meta.get("title") or ""),
        heading_path=str(meta.get("heading_path") or ""),
        child_count=int(meta.get("child_count") or 0),
        content=result["content"],
        created_at=created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else None,
        from_redis=bool(result["from_redis"]),
        elapsed_ms=float(result["elapsed_ms"]),
    )


# ==========================================================================
# 管理（配置 ADMIN_TOKEN 后需携带 X-Admin-Token）
# ==========================================================================
@router.post("/admin/reindex", summary="重建 Milvus 集合", dependencies=[AdminGuard])
def reindex(drop: bool = Query(default=True, description="是否先删除已有集合")) -> dict:
    """重建集合会清空全部向量，需重新灌数据。"""
    logger.warning("收到管理请求：重建 Milvus 集合（drop=%s）", drop)
    store = get_chunk_store()
    result = store.ensure_collection(recreate=drop)
    return {
        "milvus": result,
        "note": "重建集合后需重新执行 scripts/ingest_samples.py 灌数据",
    }


@router.post("/admin/reset-cache-stats", summary="重置缓存统计", dependencies=[AdminGuard])
def reset_cache_stats() -> dict:
    redis_client.get_parent_cache().reset_stats()
    _stats_memo_clear()
    return {"ok": True}


def _stats_memo_clear() -> None:
    """清空 /stats 记忆化，保证计数类接口的改动立即可见。"""
    global _stats_memo
    _stats_memo = None
