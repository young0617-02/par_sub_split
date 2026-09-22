"""清空全部数据（MySQL 父块 + Redis 缓存 + Milvus 向量），用于重置演示环境。

用法：
    python scripts/reset_data.py --yes                 # 清空数据，保留集合结构
    python scripts/reset_data.py --yes --drop-collection  # 连 Milvus 集合一起删掉重建
    python scripts/reset_data.py --yes --reset-vocab      # 同时清空 BM25 词表
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, text

from app.clients.milvus_client import get_chunk_store
from app.clients.mysql import get_engine, session_scope
from app.clients.redis_client import get_client, get_parent_cache
from app.models import ParentDocument


def main() -> int:
    parser = argparse.ArgumentParser(description="重置演示数据")
    parser.add_argument("--yes", action="store_true", help="确认执行（必须显式指定）")
    parser.add_argument("--reset-ids", action="store_true", help="TRUNCATE 表，自增 id 归零")
    parser.add_argument("--reset-vocab", action="store_true", help="清空 BM25 词表与统计")
    args = parser.parse_args()

    if not args.yes:
        print("这是破坏性操作，请加 --yes 确认执行。")
        return 1

    # ---------- MySQL ----------
    with session_scope() as session:
        before = session.execute(text("SELECT COUNT(*) FROM parent_document_t")).scalar() or 0
        session.execute(delete(ParentDocument))
    print(f"[OK] MySQL 已清空父块 {before} 条")
    if args.reset_ids:
        with get_engine().connect() as conn:
            conn.execute(text("TRUNCATE TABLE parent_document_t"))
            conn.commit()
        print("     已 TRUNCATE，自增 id 归零（后续父块 id 从 1 重新开始）")

    # ---------- Redis ----------
    cache = get_parent_cache()
    cleared = cache.clear_all()
    get_client().delete("psd:cache:hits", "psd:cache:misses", "psd:cache:requests")
    print(f"[OK] Redis 已清理父块缓存 {cleared} 个 key，命中率统计已重置")

    # ---------- Milvus ----------
    store = get_chunk_store()
    store.ensure_collection(recreate=True)
    print(f"[OK] Milvus 集合 {store.collection_name} 已重建（向量已清空）")

    # ---------- BM25 词表 ----------
    if args.reset_vocab:
        from app.services.sparse import get_sparse_encoder

        get_sparse_encoder().reset()
        print("[OK] BM25 词表与 df 统计已清空")

    print("\n重置完成。重新灌数据： python scripts/ingest_samples.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
