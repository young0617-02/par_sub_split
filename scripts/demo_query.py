"""端到端演示：混合检索对比 + 父子回表 + RAG 问答。

用法：
    python scripts/demo_query.py                     # 跑内置问题集
    python scripts/demo_query.py --query "你的问题"   # 自定义问题
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.redis_client import get_parent_cache
from app.services.rag import get_rag_service
from app.services.retriever import get_retrieval_service

DEFAULT_QUERIES = [
    "父子分块为什么比固定分块好？",
    "SPARSE_FLOAT_VECTOR 的下标要注意什么？",
    "父块缓存的 key 应该怎么设计？",
    "RRF 和加权融合有什么区别？",
]


def show_retrieval(query: str) -> None:
    svc = get_retrieval_service()
    store = svc.store
    dense = svc.embedder.encode_query(query)
    sparse = svc.encoder.encode(query)
    dense_only = store.search_single_branch("dense", dense_vector=dense, limit=8)
    sparse_only = store.search_single_branch("sparse", sparse_vector=sparse, limit=8)

    resp = svc.search(query, top_k=20, final_top_k=4)
    print(f"\n{'=' * 68}\n问题: {query}\n{'=' * 68}")

    def pids(hits: list[dict]) -> list[int]:
        out, seen = [], set()
        for h in hits:
            pid = int(h["entity"].get("parent_id") or 0)
            if pid and pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out

    print(f"稠密分支命中父块: {pids(dense_only)}")
    print(f"稀疏分支命中父块: {pids(sparse_only)}")
    print(f"混合 RRF 命中父块: {[p.parent_id for p in resp.parents]}")
    print(f"耗时明细(ms): {resp.timings}")
    for p in resp.parents:
        cache = "Redis命中" if p.cache_hit else "回表MySQL"
        print(f"  ├─ 父块#{p.parent_id} [{p.title} > {p.heading_path}] "
              f"score={p.score} 子块命中 {len(p.child_hits)} 个 ({cache})")
        print(f"  │    父块内容 {len(p.content)} 字: {p.content[:60].replace(chr(10), ' ')}…")


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端演示")
    parser.add_argument("--query", action="append", default=None, help="自定义问题，可多次指定")
    parser.add_argument("--no-llm", action="store_true", help="只演示检索，不调用大模型")
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES

    for q in queries:
        show_retrieval(q)
        if args.no_llm:
            continue
        resp = get_rag_service().answer(q, top_k=20, final_top_k=4)
        print(f"\n  当前 LLM 模式: {resp.llm_mode}")
        print("  答案:")
        for line in resp.answer.splitlines():
            print(f"    {line}")
        print(f"  引用: {[c.parent_id for c in resp.citations]}")
        print(f"  全链路耗时(ms): {resp.timings}")

    cache = get_parent_cache()
    print(f"\n{'=' * 68}")
    print(f"缓存统计: {cache.stats()}")
    print("提示：先跑一次 demo 再跑第二次，可观察到 cache_hit_rate 明显上升。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
