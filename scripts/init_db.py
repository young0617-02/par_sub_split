"""一键初始化：MySQL 建库建表 + Milvus 建集合 + 三件套连通性自检。

用法：
    python scripts/init_db.py            # 幂等初始化
    python scripts/init_db.py --recreate # 重建 Milvus 集合（维度变更时使用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.milvus_client import get_chunk_store
from app.clients.mysql import init_schema
from app.clients.redis_client import ping as redis_ping
from app.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化三存储")
    parser.add_argument("--recreate", action="store_true", help="重建 Milvus 集合（会清空向量）")
    args = parser.parse_args()

    print("=" * 68)
    print("父子分块 RAG 检索系统 —— 环境初始化")
    print("=" * 68)
    print(f"MySQL : {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")
    print(f"Redis : {settings.redis_host}:{settings.redis_port}/db{settings.redis_db}")
    print(f"Milvus: {settings.milvus_uri} -> {settings.milvus_collection}")
    print(f"向量维度: {settings.milvus_dim_resolved} | Embedding: {settings.embedding_provider}")
    print(f"稀疏向量: {'开启' if settings.sparse_enabled else '关闭'}")
    print("-" * 68)

    ok = True

    # ---------- MySQL ----------
    try:
        info = init_schema(create_database=True)
        print(f"[OK]   MySQL 表已就绪: {', '.join(info['tables'])}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] MySQL 初始化失败: {exc}")
        print("       请确认容器已启动： docker compose up -d mysql"
              "（或 docker start par_sub_mysql）")

    # ---------- Redis ----------
    if redis_ping():
        print("[OK]   Redis 连通正常")
    else:
        ok = False
        print("[FAIL] Redis 连接失败，请确认容器已启动： docker compose up -d redis"
              "（或 docker start par_sub_redis）")

    # ---------- Milvus ----------
    try:
        store = get_chunk_store()
        result = store.ensure_collection(recreate=args.recreate)
        status = "重建" if result.get("created") else "已存在，跳过"
        print(f"[OK]   Milvus 集合 [{result['collection']}] {status}")
        print(f"       字段: {', '.join(store.describe().get('fields', []))}")
        print(f"       索引: {', '.join(store.describe().get('indexes', []))}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] Milvus 初始化失败: {exc}")
        print("       注意本地 Milvus 开启了鉴权，请确认 .env 中 MILVUS_TOKEN 正确")

    print("-" * 68)
    print("初始化完成" if ok else "初始化未全部成功，请按上面提示修复")
    print("下一步： python scripts/ingest_samples.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
