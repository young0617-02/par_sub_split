"""入库编排 —— 一次写入，三处落库。

    ┌──────────────┐   ①父块全文 + 自增 id
    │  原始文档     │ ─────────────────────────▶  MySQL.parent_document_t
    │  (md/txt/pdf)│   ②父块内容镜像 key=id
    │              │ ─────────────────────────▶  Redis  psd:parent:{id}
    └──────────────┘   ③子块稠密+稀疏向量 + parent_id
                        ─────────────────────────▶  Milvus child_chunk_t

一致性策略（先写持久层，再写缓存，最后写索引）：
    - MySQL 是唯一事实来源（Source of Truth），必须先落库拿到 ``parent_id``；
    - Redis 只是派生缓存，写入失败可容忍（下次回表自动回填）；
    - Milvus 写入按 ``chunk_id`` upsert，天然幂等，重复入库不会产生重复向量。
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from sqlalchemy import delete, select

from app.clients.milvus_client import (
    DENSE_FIELD,
    PRIMARY_FIELD,
    SPARSE_FIELD,
    get_chunk_store,
)
from app.clients.mysql import session_scope
from app.clients.redis_client import get_parent_cache
from app.config import settings
from app.models import ParentDocument
from app.schemas import IngestResult
from app.services.chunker import ParentChunk, chunk_stats, split_document
from app.services.embedder import get_embedder
from app.services.sparse import get_sparse_encoder

logger = logging.getLogger(__name__)

# Milvus VARCHAR 长度限制，超长截断防御
MAX_CONTENT_CHARS = 20000
MAX_META_CHARS = 480


def make_doc_id(title: str, content: str) -> str:
    """根据标题 + 正文生成稳定的业务文档 id。"""
    raw = f"{title.strip()}::{content.strip()}"
    return "doc_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def extract_pdf_bytes(raw: bytes) -> str:
    """从 PDF 字节流中抽取文本。

    单独抽成函数是为了让「本地文件入库」与「HTTP 上传入库」共用同一份实现，
    避免两处各写一遍 Pypdf 调用（历史上这两条路径确实各有一份副本）。
    """
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def extract_text_from_bytes(raw: bytes, filename: str = "") -> str:
    """把上传内容解析为文本（按扩展名分派，非 PDF 一律按 UTF-8 文本处理）。"""
    if filename.lower().endswith(".pdf"):
        return extract_pdf_bytes(raw)
    return raw.decode("utf-8", errors="ignore")


def load_text_file(path: str | Path) -> str:
    """读取本地 md / txt / pdf 文件。"""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return extract_pdf_bytes(p.read_bytes())
    return p.read_text(encoding="utf-8", errors="ignore")


class IngestService:
    """文档入库服务。"""

    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.encoder = get_sparse_encoder()
        self.store = get_chunk_store()
        self.cache = get_parent_cache()

    # ==================================================================
    def ingest_text(
        self,
        content: str,
        title: str = "",
        doc_id: str | None = None,
        source: str = "api",
        force_reindex: bool = False,
    ) -> IngestResult:
        """把一篇文档切分成父子块并写入 MySQL / Redis / Milvus。"""
        t0 = time.perf_counter()
        content = content.strip()
        if not content:
            raise ValueError("文档内容为空")

        doc_id = (doc_id or make_doc_id(title, content))[:64]

        if force_reindex:
            self.delete_document(doc_id)

        parents = split_document(
            content,
            parent_size=settings.parent_chunk_size,
            parent_overlap=settings.parent_chunk_overlap,
            child_size=settings.child_chunk_size,
            child_overlap=settings.child_chunk_overlap,
        )
        if not parents:
            raise ValueError("分块结果为空，请检查文档内容")

        t_split = time.perf_counter()

        # ---------- ① MySQL：父块落库，拿到自增 parent_id ----------
        parent_ids = self._persist_parents(parents, doc_id=doc_id, title=title, source=source)
        t_mysql = time.perf_counter()

        # ---------- ② Redis：父块内容镜像（key = parent_id） ----------
        redis_written = 0
        try:
            redis_written = self.cache.set_many(
                {pid: p.content for pid, p in zip(parent_ids, parents, strict=True)}
            )
        except Exception as exc:  # noqa: BLE001  缓存失败不影响主流程
            logger.warning("父块内容写入 Redis 失败（可回表补偿）：%s", exc)
        t_redis = time.perf_counter()

        # ---------- ③ Milvus：子块向量化 + 落库 ----------
        child_rows = self._build_child_rows(parents, doc_id=doc_id, title=title)
        milvus_written = self.store.upsert(child_rows)
        t_milvus = time.perf_counter()

        stats = chunk_stats(parents)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "入库完成 doc_id=%s 父块=%s 子块=%s 耗时=%.1fms",
            doc_id, stats["parent_count"], stats["child_count"], elapsed_ms,
        )
        return IngestResult(
            doc_id=doc_id,
            parent_count=int(stats["parent_count"]),
            child_count=int(stats["child_count"]),
            parent_ids=parent_ids,
            elapsed_ms=elapsed_ms,
            detail={
                "chunk_stats": stats,
                "redis_written": redis_written,
                "milvus_written": milvus_written,
                "timing_ms": {
                    "split": round((t_split - t0) * 1000, 2),
                    "mysql": round((t_mysql - t_split) * 1000, 2),
                    "redis": round((t_redis - t_mysql) * 1000, 2),
                    "milvus": round((t_milvus - t_redis) * 1000, 2),
                },
                "embedding_provider": self.embedder.describe(),
                "sparse": self.encoder.describe(),
            },
        )

    def ingest_file(
        self,
        path: str | Path,
        doc_id: str | None = None,
        title: str | None = None,
        force_reindex: bool = False,
    ) -> IngestResult:
        """从本地文件入库（.md / .txt / .pdf）。"""
        p = Path(path)
        text = load_text_file(p)
        return self.ingest_text(
            content=text,
            title=title or p.stem,
            doc_id=doc_id,
            source=str(p.name),
            force_reindex=force_reindex,
        )

    # ==================================================================
    def _persist_parents(
        self, parents: list[ParentChunk], doc_id: str, title: str, source: str
    ) -> list[int]:
        """写 MySQL 并返回与 ``parents`` 顺序一致的 parent_id 列表。

        幂等：同一 ``doc_id`` + ``content_hash`` 已存在时复用原 id，不重复插入。
        """
        hashes = [p.content_hash for p in parents]
        parent_ids: list[int] = []

        with session_scope() as session:
            existing_rows = (
                session.execute(
                    select(ParentDocument).where(
                        ParentDocument.doc_id == doc_id,
                        ParentDocument.content_hash.in_(hashes),
                    )
                )
                .scalars()
                .all()
            )
            by_hash = {row.content_hash: row for row in existing_rows}

            for parent in parents:
                row = by_hash.get(parent.content_hash)
                if row is not None:
                    # 复用已存在的父块，但刷新元数据，保证与最新入库一致
                    row.title = title[:255]
                    row.heading_path = parent.heading_path[:MAX_META_CHARS]
                    row.chunk_index = parent.index
                    row.child_count = len(parent.children)
                    row.source = source[:512]
                    parent_id = row.id
                else:
                    obj = ParentDocument(
                        doc_id=doc_id,
                        title=title[:255],
                        heading_path=parent.heading_path[:MAX_META_CHARS],
                        content=parent.content,
                        source=source[:512],
                        chunk_index=parent.index,
                        child_count=len(parent.children),
                        char_count=parent.char_count,
                        content_hash=parent.content_hash,
                    )
                    session.add(obj)
                    session.flush()  # 立刻拿到自增主键
                    parent_id = obj.id
                    by_hash[parent.content_hash] = obj

                # 关键一步：把 MySQL 主键回填到每个子块，作为 Milvus 的 parent_id
                for child in parent.children:
                    child.parent_id = parent_id
                parent_ids.append(int(parent_id))

        return parent_ids

    def _build_child_rows(
        self, parents: list[ParentChunk], doc_id: str, title: str
    ) -> list[dict]:
        """子块批量向量化并组装成 Milvus 行。"""
        children = [c for p in parents for c in p.children]
        if not children:
            return []

        texts = [c.embed_text for c in children]

        # 稀疏：先让词表/df 统计吃下本批文档，再编码，避免新词被全部丢弃
        sparse_vectors = None
        if self.encoder.enabled:
            self.encoder.fit(texts)
            sparse_vectors = self.encoder.encode_batch(texts)

        dense_matrix = self.embedder.encode_documents(texts)
        if dense_matrix.shape[1] != self.store.dim:
            raise ValueError(
                f"Embedding 维度({dense_matrix.shape[1]}) 与 Milvus 集合维度"
                f"({self.store.dim}) 不一致，请修改 EMBEDDING_DIM 后重建集合"
            )

        rows: list[dict] = []
        for i, child in enumerate(children):
            row: dict = {
                PRIMARY_FIELD: child.chunk_id,
                "parent_id": int(child.parent_id or 0),
                "doc_id": doc_id[:64],
                "title": title[:MAX_META_CHARS],
                "heading_path": child.heading_path[:MAX_META_CHARS],
                "chunk_index": int(child.index),
                "content": child.content[:MAX_CONTENT_CHARS],
                DENSE_FIELD: dense_matrix[i].tolist(),
            }
            if sparse_vectors is not None:
                row[SPARSE_FIELD] = sparse_vectors[i]
            rows.append(row)
        return rows

    # ==================================================================
    def delete_document(self, doc_id: str) -> dict[str, int]:
        """删除一篇文档：MySQL 父块 + Redis 缓存 + Milvus 子块。"""
        with session_scope() as session:
            ids = (
                session.execute(
                    select(ParentDocument.id).where(ParentDocument.doc_id == doc_id)
                )
                .scalars()
                .all()
            )
            session.execute(delete(ParentDocument).where(ParentDocument.doc_id == doc_id))

        deleted_cache = self.cache.delete([int(i) for i in ids]) if ids else 0
        deleted_children = self.store.delete_by_doc_id(doc_id)
        logger.info(
            "删除文档 doc_id=%s 父块=%s 子块=%s 缓存=%s",
            doc_id, len(ids), deleted_children, deleted_cache,
        )
        return {
            "deleted_parents": len(ids),
            "deleted_children": deleted_children,
            "deleted_cache_keys": deleted_cache,
        }


_ingest_service: IngestService | None = None


def get_ingest_service() -> IngestService:
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService()
    return _ingest_service
