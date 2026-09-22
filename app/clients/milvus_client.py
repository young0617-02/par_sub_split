"""Milvus 子块集合（稀疏 + 稠密双向量）—— 检索链路的第一跳。

集合 ``child_chunk_t`` 字段设计：
    ┌──────────────┬──────────────────┬───────────────────────────────────┐
    │ chunk_id     │ VARCHAR(64) PK   │ 子块内容哈希，天然幂等，upsert 去重 │
    │ parent_id    │ INT64 (标量)     │ ★ 回表 MySQL / Redis 的唯一锚点    │
    │ doc_id       │ VARCHAR(64)      │ 归属文档，用于按文档删除           │
    │ content      │ VARCHAR(65535)   │ 子块原文，用于展示与高亮           │
    │ dense_vector │ FLOAT_VECTOR     │ 语义召回（COSINE / HNSW）          │
    │ sparse_vector│ SPARSE_FLOAT_VECTOR│ 关键词召回（IP / 倒排索引）      │
    └──────────────┴──────────────────┴───────────────────────────────────┘

融合策略：
    RRF（Reciprocal Rank Fusion）只依赖排名不依赖分数尺度，是混合检索的安全默认；
    加权融合（WeightedRanker）需要先做分数归一化，调参空间大但更依赖数据。
"""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

from app.config import settings

logger = logging.getLogger(__name__)

DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
PRIMARY_FIELD = "chunk_id"

DEFAULT_OUTPUT_FIELDS = [
    "parent_id",
    "doc_id",
    "title",
    "heading_path",
    "chunk_index",
    "content",
]


class ChildChunkStore:
    """子块向量库读写封装。"""

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        collection_name: str | None = None,
        dim: int | None = None,
        sparse_enabled: bool | None = None,
        timeout: int | None = None,
    ) -> None:
        self.uri = uri or settings.milvus_uri
        self.token = token if token is not None else settings.milvus_token
        self.collection_name = collection_name or settings.milvus_collection
        self.dim = int(dim or settings.milvus_dim_resolved)
        self.sparse_enabled = (
            settings.sparse_enabled if sparse_enabled is None else sparse_enabled
        )
        self._timeout = int(timeout or settings.milvus_timeout)
        self._client = MilvusClient(
            uri=self.uri,
            token=self.token or None,
            timeout=self._timeout,
        )

    # ==================================================================
    # 元信息
    # ==================================================================
    @property
    def client(self) -> MilvusClient:
        return self._client

    def has_collection(self) -> bool:
        return bool(self._client.has_collection(self.collection_name))

    # ==================================================================
    # 建集合
    # ==================================================================
    def ensure_collection(self, recreate: bool = False) -> dict[str, Any]:
        """幂等建集合；``recreate=True`` 时先删后建（维度变更必须重建）。"""
        if self.has_collection():
            if not recreate:
                return {"created": False, "collection": self.collection_name}
            logger.warning("重建 Milvus 集合：%s", self.collection_name)
            self._client.drop_collection(self.collection_name)

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(PRIMARY_FIELD, DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("parent_id", DataType.INT64, description="父文档主键，回表锚点")
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("heading_path", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=self.dim)
        if self.sparse_enabled:
            schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)

        index_params = self._client.prepare_index_params()
        # 稠密：HNSW + COSINE（向量已 L2 归一化，内积与余弦等价，用余弦更直观）
        index_params.add_index(
            field_name=DENSE_FIELD,
            index_type="HNSW",
            index_name="dense_vector_index",
            metric_type="COSINE",
            params={"M": 64, "efConstruction": 200},
        )
        if self.sparse_enabled:
            # 稀疏：倒排索引 + 内积（BM25 权重已 L2 归一化）
            index_params.add_index(
                field_name=SPARSE_FIELD,
                index_type="SPARSE_INVERTED_INDEX",
                index_name="sparse_vector_index",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            # Strong 一致性：写入后立即可被检索到，无需依赖代价高昂的 flush
            consistency_level="Strong",
        )
        logger.info("Milvus 集合已创建：%s (dim=%s, sparse=%s)", self.collection_name, self.dim, self.sparse_enabled)
        return {
            "created": True,
            "collection": self.collection_name,
            "dim": self.dim,
            "sparse_enabled": self.sparse_enabled,
        }

    def flush(self) -> None:
        """显式落盘。集合使用 Strong 一致性时通常不需要调用。"""
        self._client.flush(self.collection_name)

    # ==================================================================
    # 写入
    # ==================================================================
    def upsert(self, rows: list[dict[str, Any]], flush: bool = False) -> int:
        """按主键 upsert 子块，重复入库不会产生脏数据。

        ``flush`` 默认关闭：flush 会强制刷盘并重建段，实测单次可达 10s 以上，
        而集合已配置 Strong 一致性，写入后即可被检索到，无需 flush。
        """
        if not rows:
            return 0
        payload = []
        for row in rows:
            item = dict(row)
            if not self.sparse_enabled:
                item.pop(SPARSE_FIELD, None)
            # Milvus 稀疏向量不接受空字典
            if self.sparse_enabled and not item.get(SPARSE_FIELD):
                item[SPARSE_FIELD] = {0: 1.0}
            payload.append(item)

        res = self._client.upsert(collection_name=self.collection_name, data=payload)
        count = int(res.get("upsert_count", res.get("insert_count", len(payload))))
        if flush:
            self._client.flush(self.collection_name)
        return count

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按业务文档删除其全部子块（重建索引时使用）。"""
        safe = doc_id.replace("\\", "\\\\").replace('"', '\\"')
        res = self._client.delete(
            collection_name=self.collection_name, filter=f'doc_id == "{safe}"'
        )
        return int(res.get("delete_count", 0))

    def delete_by_parent_ids(self, parent_ids: list[int]) -> int:
        if not parent_ids:
            return 0
        expr = f"parent_id in [{','.join(str(int(p)) for p in parent_ids)}]"
        res = self._client.delete(collection_name=self.collection_name, filter=expr)
        return int(res.get("delete_count", 0))

    # ==================================================================
    # 检索
    # ==================================================================
    def _dense_request(
        self, dense_vector: list[float], limit: int, expr: str | None
    ) -> AnnSearchRequest:
        return AnnSearchRequest(
            data=[dense_vector],
            anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"ef": max(64, limit * 4)}},
            limit=limit,
            expr=expr or None,
        )

    def _sparse_request(
        self, sparse_vector: dict[int, float], limit: int, expr: str | None
    ) -> AnnSearchRequest:
        return AnnSearchRequest(
            data=[sparse_vector],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "IP", "params": {}},
            limit=limit,
            expr=expr or None,
        )

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        limit: int = 20,
        expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """稀疏 + 稠密混合检索，返回统一结构。

        Returns:
            ``[{"chunk_id", "score", "entity": {...}}, ...]``，按融合分数降序。
        """
        output_fields = output_fields or DEFAULT_OUTPUT_FIELDS
        use_sparse = self.sparse_enabled and bool(sparse_vector)

        if not use_sparse:
            hits = self._client.search(
                collection_name=self.collection_name,
                data=[dense_vector],
                anns_field=DENSE_FIELD,
                limit=limit,
                filter=expr or "",
                search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 4)}},
                output_fields=output_fields,
            )
            return [self._normalize(h) for h in hits[0]]

        reqs = [self._dense_request(dense_vector, limit, expr)]
        reqs.append(self._sparse_request(sparse_vector or {}, limit, expr))

        ranker = self._build_ranker()
        hits = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields,
        )
        return [self._normalize(h) for h in hits[0]]

    def search_single_branch(
        self,
        branch: str,
        dense_vector: list[float] | None = None,
        sparse_vector: dict[int, float] | None = None,
        limit: int = 20,
        expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """只跑单分支（dense 或 sparse），用于对比实验与可解释性分析。"""
        output_fields = output_fields or DEFAULT_OUTPUT_FIELDS
        if branch == "dense":
            if dense_vector is None:
                raise ValueError("dense 分支需要 dense_vector")
            hits = self._client.search(
                collection_name=self.collection_name,
                data=[dense_vector],
                anns_field=DENSE_FIELD,
                limit=limit,
                filter=expr or "",
                search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 4)}},
                output_fields=output_fields,
            )
        elif branch == "sparse":
            if not sparse_vector or not self.sparse_enabled:
                return []
            hits = self._client.search(
                collection_name=self.collection_name,
                data=[sparse_vector],
                anns_field=SPARSE_FIELD,
                limit=limit,
                filter=expr or "",
                search_params={"metric_type": "IP", "params": {}},
                output_fields=output_fields,
            )
        else:
            raise ValueError("branch 只能是 dense / sparse")
        return [self._normalize(h) for h in hits[0]]

    def _build_ranker(self):
        if settings.hybrid_ranker == "weighted":
            return WeightedRanker(settings.dense_weight, settings.sparse_weight)
        return RRFRanker(60)

    @staticmethod
    def _normalize(hit: Any) -> dict[str, Any]:
        """把 pymilvus 的 Hit 对象拍平成普通 dict，隔离上层对 SDK 的依赖。"""
        return {
            "chunk_id": hit.get(PRIMARY_FIELD) or getattr(hit, "id", None),
            "score": float(hit.get("distance", 0.0)),
            "entity": dict(hit.get("entity") or {}),
        }

    # ==================================================================
    # 运维
    # ==================================================================
    def count(self) -> int:
        """准确统计实体数。

        ``get_collection_stats()`` 的 row_count 依赖已落盘的 segment，
        不 flush 时会长期显示 0；``query(count(*))`` 走查询链路，结果实时准确。
        """
        if not self.has_collection():
            return 0
        try:
            res = self._client.query(
                collection_name=self.collection_name,
                filter="chunk_index >= 0",
                output_fields=["count(*)"],
            )
            if res and res[0].get("count(*)") is not None:
                return int(res[0]["count(*)"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("count(*) 查询失败，回退 stats：%s", exc)
        stats = self._client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def describe(self) -> dict[str, Any]:
        if not self.has_collection():
            return {"exists": False, "collection": self.collection_name}
        info = self._client.describe_collection(self.collection_name)
        return {
            "exists": True,
            "collection": self.collection_name,
            "fields": [f["name"] for f in info.get("fields", [])],
            "indexes": list(self._client.list_indexes(self.collection_name)),
            "row_count": self.count(),
        }

    def drop(self) -> None:
        if self.has_collection():
            self._client.drop_collection(self.collection_name)

    def ping(self) -> bool:
        try:
            self._client.get_server_version()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus 连通性检查失败：%s", exc)
            return False


_store: ChildChunkStore | None = None


def get_chunk_store() -> ChildChunkStore:
    global _store
    if _store is None:
        _store = ChildChunkStore()
    return _store
