"""检索编排 —— 「命中子块 → 取 parent_id → Redis 取父块」的核心实现。

完整链路（也是本题的知识主线）：

    用户问题
      ├─① 稠密向量 (语义)  ─┐
      └─② 稀疏向量 (关键词) ─┴─▶ Milvus 混合检索 + RRF 融合
                                   │
                                   ▼
                             命中若干「子块」
                                   │  取 parent_id 去重（关键：N 个子块可能同父）
                                   ▼
                         Redis MGET 批量取父块全文（1 次 RTT）
                                   │  未命中 -> 回表 MySQL 并回填缓存
                                   ▼
                         父块全文 → 大模型上下文
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

from sqlalchemy import select

from app.clients.milvus_client import get_chunk_store
from app.clients.mysql import session_scope
from app.clients.redis_client import get_parent_cache
from app.config import settings
from app.models import ParentDocument
from app.schemas import ChildHit, ParentHit, SearchResponse
from app.services.embedder import get_embedder
from app.services.reranker import get_reranker
from app.services.sparse import get_sparse_encoder

logger = logging.getLogger(__name__)

# 同一父块命中多个子块时的相对加权系数。
# 注意必须用「乘法」而非「加法」：RRF 融合分数只有 0.016~0.033 量级，
# 而稠密分支的余弦分数在 0.3~0.9 量级，一个固定加法偏置会在 RRF 场景下
# 直接压过真实的排序信号，导致"命中子块多但都不相关"的父块排到第一。
MULTI_HIT_BONUS = 0.15


class RetrievalService:
    """父子分块检索服务。"""

    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.encoder = get_sparse_encoder()
        self.store = get_chunk_store()
        self.cache = get_parent_cache()
        self.reranker = get_reranker()

    # ==================================================================
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        final_top_k: int | None = None,
        use_sparse: bool = True,
        explain: bool = False,
    ) -> tuple[list[ParentHit], dict[str, float]]:
        """执行检索，返回 (父块列表, 各阶段耗时)。"""
        top_k = int(top_k or settings.retrieve_top_k)
        final_top_k = int(final_top_k or settings.final_top_k)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()

        # ---------- ① 查询向量化（稠密 + 稀疏） ----------
        dense_vector = self.embedder.encode_query(query)
        sparse_vector = None
        if use_sparse and self.encoder.enabled:
            sparse_vector = self.encoder.encode(query)
        t_embed = time.perf_counter()
        timings["embed_ms"] = round((t_embed - t0) * 1000, 2)

        # ---------- ② Milvus 混合检索（仍然是"子块"粒度） ----------
        raw_hits = self.store.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=top_k,
        )
        t_milvus = time.perf_counter()
        timings["milvus_ms"] = round((t_milvus - t_embed) * 1000, 2)
        timings["milvus_hits"] = float(len(raw_hits))

        if not raw_hits:
            return [], timings

        # ---------- ②.5 Rerank 精排（可选；作用于子块，见 reranker.py 的设计说明） ----------
        raw_hits = self._apply_rerank(query, raw_hits)
        t_rerank = time.perf_counter()
        timings["rerank_ms"] = round((t_rerank - t_milvus) * 1000, 2)
        timings["rerank_enabled"] = 1.0 if self.reranker.available else 0.0

        # ---------- ③ 按 parent_id 聚合去重（父子分块的关键一步） ----------
        grouped: OrderedDict[int, list[dict]] = OrderedDict()
        for hit in raw_hits:
            pid = int(hit["entity"].get("parent_id") or 0)
            if pid <= 0:
                continue
            grouped.setdefault(pid, []).append(hit)

        if not grouped:
            return [], timings

        # 父块得分 = 最佳子块分数 × (1 + 系数 × 额外命中数)
        # 启用重排后，这里改用重排分数（已归一化到 0~1，与阈值逻辑兼容）
        use_rerank = any(h.get("rerank_score") is not None for h in raw_hits)
        if use_rerank:
            def _child_score(hit: dict) -> float:
                return float(hit.get("rerank_score") or 0.0)
        else:
            def _child_score(hit: dict) -> float:
                return float(hit["score"])

        scores = {
            pid: max(_child_score(h) for h in hits) * (1 + MULTI_HIT_BONUS * (len(hits) - 1))
            for pid, hits in grouped.items()
        }

        ordered_pids = sorted(scores, key=lambda p: scores[p], reverse=True)

        # 低分过滤：只保留与最佳父块分数量级接近的结果，避免把"陪跑"父块塞给大模型
        best_score = scores[ordered_pids[0]]
        ratio = settings.score_ratio_threshold
        if ratio > 0 and best_score > 0:
            cutoff = best_score * ratio
            filtered = [p for p in ordered_pids if scores[p] >= cutoff]
            filtered = filtered or ordered_pids[:1]
        else:
            filtered = ordered_pids

        target_pids = filtered[:final_top_k]
        t_group = time.perf_counter()
        # 注意：这里要用 t_rerank 而非 t_milvus 作为起点，
        # 否则 group_ms 会把重排耗时也算进去，指标名与含义不符。
        timings["group_ms"] = round((t_group - t_rerank) * 1000, 2)

        # ---------- ④ Redis 批量取父块全文（核心提速点） ----------
        contents = {}
        try:
            contents = self.cache.mget(target_pids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 Redis 失败，回表 MySQL：%s", exc)
        t_redis = time.perf_counter()

        # 缓存未命中 -> 回表 MySQL，并顺手回填缓存（Cache-Aside 模式）
        missed = [p for p in target_pids if p not in contents]
        if missed:
            fallback = self._fetch_from_mysql(missed)
            contents.update(fallback)
            if fallback:
                try:
                    self.cache.set_many(fallback)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("回填 Redis 失败：%s", exc)
        t_fetch = time.perf_counter()
        timings["redis_ms"] = round((t_redis - t_group) * 1000, 2)
        timings["mysql_fallback_ms"] = round((t_fetch - t_redis) * 1000, 2)

        # ---------- ⑤ 组装父块结果 ----------
        rank_maps = self._build_rank_maps(dense_vector, sparse_vector, top_k) if explain else None

        parents: list[ParentHit] = []
        for pid in target_pids:
            hits = grouped[pid]
            best = hits[0]["entity"]
            children: list[ChildHit] = []
            for h in hits:
                entity = h["entity"]
                children.append(
                    ChildHit(
                        chunk_id=str(h["chunk_id"]),
                        parent_id=pid,
                        doc_id=str(entity.get("doc_id", "")),
                        chunk_index=int(entity.get("chunk_index", 0)),
                        content=str(entity.get("content", "")),
                        score=round(float(h["score"]), 6),
                        rerank_score=(
                            round(float(h["rerank_score"]), 6)
                            if h.get("rerank_score") is not None
                            else None
                        ),
                        dense_rank=rank_maps[0].get(str(h["chunk_id"])) if rank_maps else None,
                        sparse_rank=rank_maps[1].get(str(h["chunk_id"])) if rank_maps else None,
                    )
                )
            if use_rerank:
                children.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
            else:
                children.sort(key=lambda c: c.score, reverse=True)
            content = contents.get(pid)
            parents.append(
                ParentHit(
                    parent_id=pid,
                    doc_id=str(best.get("doc_id", "")),
                    title=str(best.get("title", "")),
                    heading_path=str(best.get("heading_path", "")),
                    content=content or "",
                    score=round(scores[pid], 6),
                    child_hits=children,
                    cache_hit=pid in contents and pid not in missed,
                )
            )

        t_end = time.perf_counter()
        timings["total_ms"] = round((t_end - t0) * 1000, 2)
        timings["parent_count"] = float(len(parents))
        logger.info(
            "检索完成 query=%.30s 子块命中=%s 去重父块=%s 耗时=%.1fms",
            query, len(raw_hits), len(parents), timings["total_ms"],
        )
        return parents, timings

    # ==================================================================
    def search(
        self,
        query: str,
        top_k: int | None = None,
        final_top_k: int | None = None,
        use_sparse: bool = True,
        explain: bool = False,
    ) -> SearchResponse:
        parents, timings = self.retrieve(
            query, top_k=top_k, final_top_k=final_top_k, use_sparse=use_sparse, explain=explain
        )
        return SearchResponse(query=query, parents=parents, timings=timings)

    # ==================================================================
    def get_parent_document(self, parent_id: int) -> dict | None:
        """按 parent_id 取父文档：优先 Redis 取正文，未命中回表 MySQL 并回填。

        元数据（doc_id / 标题 / 章节 / 子块数）**始终**从 MySQL 读取，原因：

        1. Redis 的 value 按设计只存父块正文（``key=id, value=content``），不含元数据；
        2. 若"缓存命中就返回空元数据"，接口响应会随缓存状态抖动，调用方拿不到
           doc_id/title —— 这是修复前存在的真实缺陷。

        元数据查询只 SELECT 少数字段，**不会加载 LONGTEXT 正文**，开销可忽略。

        Returns:
            含 ``content`` / ``from_redis`` / ``elapsed_ms`` / ``meta`` 的字典；
            父文档不存在（含"缓存有残留但库已删除"）时返回 None。
        """
        t0 = time.perf_counter()
        try:
            content = self.cache.get(parent_id)
        except Exception as exc:  # noqa: BLE001  缓存不可用时回表兜底，不影响正确性
            logger.warning("读取 Redis 失败，回表 MySQL：%s", exc)
            content = None

        from_redis = content is not None
        if not from_redis:
            row = self._fetch_row(parent_id)
            if row is None:
                return None
            content = row.content
            try:
                self.cache.set(parent_id, content)
            except Exception as exc:  # noqa: BLE001
                logger.warning("回填 Redis 失败：%s", exc)

        meta = self._fetch_meta(parent_id)
        if meta is None:
            # MySQL 已删除但缓存仍有残留：以事实来源为准，避免返回"幽灵"数据
            return None

        return {
            "content": content or "",
            "from_redis": from_redis,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
            "meta": meta,
        }

    # ==================================================================
    def _apply_rerank(self, query: str, hits: list[dict]) -> list[dict]:
        """对召回的**子块**做 Cross-Encoder 重排。

        未启用重排器（或依赖缺失）时原样返回，调用方无需分支判断。
        """
        if not self.reranker.available or len(hits) <= 1:
            return hits

        candidate_count = int(settings.rerank_candidates or len(hits))
        candidate_count = max(1, min(candidate_count, len(hits)))
        candidates = hits[:candidate_count]
        documents = [str(h["entity"].get("content") or "") for h in candidates]

        try:
            ranked = self.reranker.rerank(query, documents, settings.rerank_keep)
        except Exception as exc:  # noqa: BLE001  重排失败不能影响主链路
            logger.warning("重排失败，保持原顺序：%s", exc)
            return hits

        ordered: list[dict] = []
        for index, score in ranked:
            item = dict(hits[index])
            item["rerank_score"] = score
            ordered.append(item)
        # 未进入重排窗口的候选保留在尾部（不带重排分数，仅作为兜底候选）
        ordered.extend(hits[candidate_count:])
        return ordered

    def _fetch_from_mysql(self, parent_ids: list[int]) -> dict[int, str]:
        """回表取父块全文（Cache-Aside 的兜底路径）。"""
        if not parent_ids:
            return {}
        with session_scope() as session:
            rows = (
                session.execute(
                    select(ParentDocument.id, ParentDocument.content).where(
                        ParentDocument.id.in_(parent_ids)
                    )
                )
                .all()
            )
        return {int(pid): content for pid, content in rows}

    def _fetch_row(self, parent_id: int) -> ParentDocument | None:
        """取完整父文档行（含 LONGTEXT 正文），仅缓存未命中时使用。"""
        with session_scope() as session:
            return session.execute(
                select(ParentDocument).where(ParentDocument.id == parent_id)
            ).scalar_one_or_none()

    def _fetch_meta(self, parent_id: int) -> dict | None:
        """只取元数据列（不加载正文），供接口在缓存命中时补齐响应字段。"""
        with session_scope() as session:
            row = session.execute(
                select(
                    ParentDocument.id,
                    ParentDocument.doc_id,
                    ParentDocument.title,
                    ParentDocument.heading_path,
                    ParentDocument.child_count,
                    ParentDocument.created_at,
                ).where(ParentDocument.id == parent_id)
            ).first()
        return dict(row._mapping) if row is not None else None

    def _build_rank_maps(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None,
        top_k: int,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """对比实验：分别跑单分支，拿到每个子块在稠密/稀疏分支中的排名。"""
        try:
            dense_hits = self.store.search_single_branch(
                "dense", dense_vector=dense_vector, limit=top_k
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("dense 分支对照检索失败：%s", exc)
            dense_hits = []
        try:
            sparse_hits = self.store.search_single_branch(
                "sparse", sparse_vector=sparse_vector, limit=top_k
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sparse 分支对照检索失败：%s", exc)
            sparse_hits = []

        dense_rank = {str(h["chunk_id"]): i + 1 for i, h in enumerate(dense_hits)}
        sparse_rank = {str(h["chunk_id"]): i + 1 for i, h in enumerate(sparse_hits)}
        return dense_rank, sparse_rank


_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


def is_retrieval_service_ready() -> bool:
    """检索服务是否已初始化完成。

    供健康检查 / 统计接口使用：这两个接口**不应触发**检索服务的懒加载
    （会连带加载嵌入模型，本机 BGE-M3 在 CPU 上要 6~10s），
    否则监控请求会阻塞甚至超时。未就绪时改报配置中的预期值。
    """
    return _retrieval_service is not None
