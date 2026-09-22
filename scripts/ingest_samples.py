"""把 data/samples/*.md 灌入系统，并对分块结果做一次质量体检。

用法：
    python scripts/ingest_samples.py            # 增量入库（幂等）
    python scripts/ingest_samples.py --rebuild  # 先清空该文档再入库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.milvus_client import get_chunk_store
from app.clients.redis_client import get_parent_cache
from app.config import SAMPLE_DIR, settings
from app.services.chunker import chunk_stats, split_document
from app.services.ingest import IngestService


def preview_chunks(text: str) -> None:
    """打印分块结构，帮助理解父子两级切分。"""
    parents = split_document(
        text,
        parent_size=settings.parent_chunk_size,
        parent_overlap=settings.parent_chunk_overlap,
        child_size=settings.child_chunk_size,
        child_overlap=settings.child_chunk_overlap,
    )
    stats = chunk_stats(parents)
    print(f"   分块预览: 父块 {stats['parent_count']} | 子块 {stats['child_count']} "
          f"| 父块均长 {stats['avg_parent_chars']} | 子块均长 {stats['avg_child_chars']}")
    for p in parents[:2]:
        head = p.heading_path or "(无标题)"
        print(f"     ├─ 父块#{p.index} [{head}] {p.char_count}字 -> {len(p.children)} 个子块")
        for c in p.children[:2]:
            snippet = c.content[:36].replace("\n", " ")
            print(f"     │    └─ 子块 {c.chunk_id[:8]}… {len(c.content)}字: {snippet}…")


def main() -> int:
    parser = argparse.ArgumentParser(description="灌入样例文档")
    parser.add_argument("--rebuild", action="store_true", help="入库前先删除同 doc_id 旧数据")
    args = parser.parse_args()

    files = sorted(SAMPLE_DIR.glob("*.md"))
    if not files:
        print(f"未找到样例文件：{SAMPLE_DIR}")
        return 1

    service = IngestService()
    total_parents = total_children = 0

    print("=" * 68)
    print(f"开始灌入 {len(files)} 篇样例文档")
    print("=" * 68)

    for path in files:
        text = path.read_text(encoding="utf-8")
        print(f"\n[{path.name}] {len(text)} 字")
        if args.rebuild:
            print("   将先清理同 doc_id 的旧数据（force_reindex）")
        try:
            preview_chunks(text)
            result = service.ingest_text(
                content=text,
                title=path.stem,
                source=path.name,
                force_reindex=args.rebuild,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   [FAIL] 入库失败: {exc}")
            continue
        total_parents += result.parent_count
        total_children += result.child_count
        timing = result.detail.get("timing_ms", {})
        print(f"   [OK] doc_id={result.doc_id} parent_ids={result.parent_ids}")
        print(f"        父块 {result.parent_count} | 子块 {result.child_count} "
              f"| 总耗时 {result.elapsed_ms}ms (切分{timing.get('split')} "
              f"MySQL{timing.get('mysql')} Redis{timing.get('redis')} Milvus{timing.get('milvus')})")

    store = get_chunk_store()
    cache = get_parent_cache()
    print("\n" + "=" * 68)
    print(f"入库汇总：父块 {total_parents} | 子块 {total_children}")
    print(f"存储核对：MySQL 已由接口统计 | Milvus row_count = {store.count()} "
          f"| Redis keys = {cache.count_keys()}")
    print("下一步： python scripts/demo_query.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
