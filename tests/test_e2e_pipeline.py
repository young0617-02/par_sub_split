"""端到端集成测试：需要 MySQL / Redis / Milvus 均已启动。

运行：
    pytest tests/test_e2e_pipeline.py -v
若任一存储不可用，测试会自动 skip 而不是失败。
"""

from __future__ import annotations

import uuid

import pytest

from app.clients.milvus_client import get_chunk_store
from app.clients.mysql import init_schema
from app.clients.mysql import ping as mysql_ping
from app.clients.redis_client import get_client, get_parent_cache
from app.clients.redis_client import ping as redis_ping

DOC_TEXT = """# 集成测试文档

## 检索链路

父子分块的核心链路是先用子块做向量检索，命中后取出 parent_id，
再从缓存中取回父块全文交给大模型。

## 唯一标记

本段包含唯一检索标记 E2EQX7413，用于在共享语料中稳定验证检索链路。

## 存储分工

MySQL 负责持久化父块并生成自增主键，Redis 负责缓存父块内容，
Milvus 负责存放子块的稀疏向量与稠密向量。
"""

# 用文档内唯一标记发起查询，避免被共享语料里语义更接近的文档挤出结果集
UNIQUE_QUERY = "唯一检索标记 E2EQX7413 在哪个父块里？"


@pytest.fixture(scope="module")
def services():
    if not mysql_ping():
        pytest.skip("MySQL 不可用")
    if not redis_ping():
        pytest.skip("Redis 不可用")
    store = get_chunk_store()
    if not store.ping():
        pytest.skip("Milvus 不可用")
    init_schema()
    store.ensure_collection()
    return store


def test_full_ingest_search_delete(services) -> None:
    from app.services.ingest import IngestService
    from app.services.retriever import get_retrieval_service

    doc_id = f"test_{uuid.uuid4().hex[:12]}"
    ingest = IngestService()
    retriever = get_retrieval_service()

    # ---------- 入库 ----------
    result = ingest.ingest_text(content=DOC_TEXT, title="集成测试文档", doc_id=doc_id, source="pytest")
    assert result.parent_count >= 1
    assert result.child_count >= result.parent_count
    assert all(isinstance(pid, int) and pid > 0 for pid in result.parent_ids)

    # ---------- 三处写入核对 ----------
    assert services.count() >= result.child_count

    cache = get_parent_cache()
    keys = [cache.key(pid) for pid in result.parent_ids]
    assert all(get_client().exists(k) for k in keys), "父块内容必须已写入 Redis"

    # ---------- 检索：命中子块 -> 取 parent_id -> Redis 取父块 ----------
    parents, timings = retriever.retrieve(UNIQUE_QUERY, top_k=20, final_top_k=5)
    assert parents, "检索不应为空"
    hit_ids = {p.parent_id for p in parents}
    assert hit_ids & set(result.parent_ids), "应当命中刚入库文档的父块"
    assert any(p.content for p in parents), "父块全文必须被取回"
    assert any(p.child_hits for p in parents), "必须能看到命中的子块"
    assert "total_ms" in timings

    # ---------- 按 parent_id 直取（缓存命中路径） ----------
    pid = result.parent_ids[0]
    got = retriever.get_parent_document(pid)
    assert got is not None
    assert got["from_redis"] is True, "第二次读取应命中 Redis 缓存"
    assert got["content"]

    # 回归：缓存命中时元数据也必须完整（历史缺陷：命中缓存会返回空 doc_id/title）
    meta = got["meta"]
    assert meta is not None, "缓存命中时也必须返回元数据"
    assert meta["doc_id"] == doc_id
    assert meta["title"] == "集成测试文档"
    assert int(meta["child_count"]) == result.child_count

    # ---------- 缓存未命中回表兜底 ----------
    cache.delete([pid])
    miss = retriever.get_parent_document(pid)
    assert miss is not None and miss["from_redis"] is False
    assert miss["meta"]["doc_id"] == doc_id, "回表路径元数据同样应完整"
    assert cache.exists(pid), "回表后应自动回填缓存"

    # ---------- 幽灵缓存：Redis 有残留但 MySQL 已无该行 ----------
    ghost_id = 999_999_999
    cache.set(ghost_id, "这是一条来自已删除文档的缓存残留")
    try:
        assert retriever.get_parent_document(ghost_id) is None, (
            "MySQL 已无该父块时不得返回缓存中的脏数据"
        )
    finally:
        cache.delete([ghost_id])

    # ---------- 幂等：同内容重复入库不产生新父块 ----------
    again = ingest.ingest_text(content=DOC_TEXT, title="集成测试文档", doc_id=doc_id, source="pytest")
    assert again.parent_ids == result.parent_ids, "重复入库必须复用父块 id（幂等）"

    # ---------- 删除：三处同步清理 ----------
    children_before = services.count()
    deleted = ingest.delete_document(doc_id)
    assert deleted["deleted_parents"] == len(result.parent_ids)
    assert deleted["deleted_children"] == result.child_count, "Milvus 子块必须被同步删除"
    assert services.count() == children_before - result.child_count, "删除后向量总数应下降"
    assert not get_client().exists(cache.key(pid))
    remaining, _ = retriever.retrieve(UNIQUE_QUERY, top_k=20, final_top_k=5)
    assert not ({p.parent_id for p in remaining} & set(result.parent_ids)), "已删除文档不应再被检索到"


def test_rag_answer_returns_citations(services) -> None:
    from app.services.rag import get_rag_service

    doc_id = f"test_{uuid.uuid4().hex[:12]}"
    from app.services.ingest import IngestService

    ingest = IngestService()
    ingest.ingest_text(content=DOC_TEXT, title="集成测试文档", doc_id=doc_id, source="pytest")
    try:
        resp = get_rag_service().answer("MySQL 和 Redis 分别负责什么？", top_k=10, final_top_k=3)
        assert resp.answer
        assert resp.llm_mode.startswith(("openai", "extractive"))
        assert resp.citations, "必须返回引用来源"
        assert resp.parents
    finally:
        ingest.delete_document(doc_id)
