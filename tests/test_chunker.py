"""父子分块器单元测试（不依赖任何外部服务）。"""

from app.services.chunker import chunk_stats, split_document

SAMPLE = """# 产品手册

## 安装说明

第一步是下载安装包，注意选择与操作系统匹配的版本。第二步是校验安装包的哈希值，
确保文件完整。第三步是双击运行安装程序。

## 环境要求

操作系统需要 64 位版本，内存至少 8GB，磁盘剩余空间不少于 20GB。
如果需要使用 GPU 加速，还需要提前安装对应的驱动与运行时。

## 常见问题

安装失败最常见的原因是杀毒软件拦截，请先关闭后重试。
如果提示端口被占用，请修改配置文件中的端口号。
"""


def test_children_never_cross_parent_boundary() -> None:
    """子块必须严格归属于唯一的父块，且 parent_index 与父块一致。"""
    parents = split_document(SAMPLE, parent_size=300, parent_overlap=30, child_size=80, child_overlap=20)
    assert parents, "分块结果不应为空"

    for parent in parents:
        assert parent.children, "每个父块都应有子块"
        for child in parent.children:
            assert child.parent_index == parent.index
            # 子块内容必须能在父块内容中找到（允许空白差异）
            assert child.content[:20] in parent.content


def test_child_index_is_globally_unique() -> None:
    parents = split_document(SAMPLE, parent_size=300, child_size=80)
    indexes = [c.index for p in parents for c in p.children]
    assert len(indexes) == len(set(indexes)), "子块序号必须全局唯一"


def test_parent_is_larger_than_child() -> None:
    """父块粒度必须显著大于子块，否则父子分块失去意义。"""
    parents = split_document(SAMPLE, parent_size=400, parent_overlap=40, child_size=100, child_overlap=20)
    stats = chunk_stats(parents)
    assert stats["avg_parent_chars"] > stats["avg_child_chars"] * 2


def test_chunk_id_is_deterministic() -> None:
    """同一内容两次切分必须得到相同的 chunk_id，才能保证 upsert 幂等。"""
    a = [c.chunk_id for p in split_document(SAMPLE, 300, 30, 80, 20) for c in p.children]
    b = [c.chunk_id for p in split_document(SAMPLE, 300, 30, 80, 20) for c in p.children]
    assert a == b


def test_heading_path_preserved() -> None:
    parents = split_document(SAMPLE, parent_size=300, child_size=80)
    paths = {p.heading_path for p in parents}
    assert any("安装说明" in p for p in paths)
    assert any("环境要求" in p for p in paths)


def test_content_hash_changes_with_content() -> None:
    parents = split_document(SAMPLE, parent_size=300, child_size=80)
    h1 = parents[0].content_hash
    parents2 = split_document(SAMPLE + "\n补充一段全新的内容用于制造差异。\n", 300, 30, 80, 20)
    assert {p.content_hash for p in parents2} != {p.content_hash for p in parents}
    assert isinstance(h1, str) and len(h1) == 64


def test_empty_document_returns_empty() -> None:
    assert split_document("   \n\n  ") == []


def test_long_single_paragraph_is_split() -> None:
    """超长无标点段落也必须能被子块切分，不能出现单块吞掉全文。"""
    text = "、".join(f"片段{i}" for i in range(400))
    parents = split_document(text, parent_size=300, parent_overlap=0, child_size=100, child_overlap=0)
    assert len(parents) > 1
    assert all(len(p.content) <= 300 * 1.5 for p in parents)
