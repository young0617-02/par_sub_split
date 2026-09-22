"""稠密/稀疏向量编码器单元测试（不依赖外部服务）。"""

import math

import numpy as np

from app.services.embedder import HashEmbedder
from app.services.sparse import EMPTY_TOKEN_ID, BM25SparseEncoder


# ---------------------------------------------------------------- 稠密
def test_hash_embedder_shape_and_norm() -> None:
    emb = HashEmbedder(dim=256)
    mat = emb.encode_documents(["父子分块检索", "Redis 缓存"])
    assert mat.shape == (2, 256)
    assert mat.dtype == np.float32
    # L2 归一化后模长应为 1
    for row in mat:
        assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-5


def test_hash_embedder_deterministic_across_calls() -> None:
    """必须使用 crc32 这类稳定哈希，不能依赖 Python 内置 hash 的随机种子。"""
    a = HashEmbedder(128).encode_query("稳定哈希")
    b = HashEmbedder(128).encode_query("稳定哈希")
    assert a == b


def test_similar_text_scores_higher() -> None:
    emb = HashEmbedder(dim=512)
    vecs = emb.encode_documents(["父子分块检索方案", "父子分块检索方案说明", "今天天气很好"])
    sim_close = float(np.dot(vecs[0], vecs[1]))
    sim_far = float(np.dot(vecs[0], vecs[2]))
    assert sim_close > sim_far


def test_empty_text_returns_zero_vector() -> None:
    emb = HashEmbedder(dim=64)
    vec = emb.encode_query("   ")
    assert float(np.linalg.norm(vec)) == 0.0


# ---------------------------------------------------------------- 稀疏
def test_sparse_index_weight_shape(tmp_path) -> None:
    enc = BM25SparseEncoder(tmp_path)
    docs = ["Milvus 稀疏向量检索", "Redis 缓存父块内容", "MySQL 存储父文档"]
    enc.fit(docs)
    vec = enc.encode(docs[0])
    assert vec, "编码结果不应为空"
    assert set(vec) <= set(enc.vocab.values())
    # 权重经 L2 归一化
    assert abs(math.sqrt(sum(v * v for v in vec.values())) - 1.0) < 1e-6


def test_sparse_vocab_persisted(tmp_path) -> None:
    """词表必须落盘，否则进程重启后入库与查询的下标不一致，检索必然为空。"""
    enc = BM25SparseEncoder(tmp_path)
    enc.fit(["父子分块与向量检索"])
    vocab_size = enc.vocab_size

    reloaded = BM25SparseEncoder(tmp_path)
    assert reloaded.vocab_size == vocab_size
    assert reloaded.vocab == enc.vocab


def test_sparse_unknown_query_falls_back_to_placeholder(tmp_path) -> None:
    """查询词全不在词表中时不能返回空字典，Milvus 会直接报错。"""
    enc = BM25SparseEncoder(tmp_path)
    enc.fit(["只包含这一句话"])
    vec = enc.encode("完全不存在的词汇组合xyz")
    assert vec == {EMPTY_TOKEN_ID: 1.0}


def test_idf_penalizes_common_terms(tmp_path) -> None:
    """高频词（df 大）的权重应低于稀有词。

    注意：必须从分词结果里取真实 token，不能直接拿原文词串查词表
    （jieba 会把"通用词"切成"通用"+"词"，直接查询会命中不存在的 token）。
    """
    enc = BM25SparseEncoder(tmp_path)
    docs = ["安装说明 " + "通用词 " * 5 for _ in range(5)] + ["安装说明 稀有专有名词ERR1001"]
    enc.fit(docs)

    common_tokens = set(enc.tokenize(docs[0]))
    rare_tokens = set(enc.tokenize(docs[-1])) - {t for d in docs[:-1] for t in enc.tokenize(d)}
    assert common_tokens and rare_tokens

    common_tok = max(common_tokens, key=lambda t: enc.df.get(t, 0))
    rare_tok = next(iter(rare_tokens))
    assert enc.df[common_tok] > enc.df[rare_tok]
    assert enc._idf(rare_tok) > enc._idf(common_tok)


def test_sparse_disabled_returns_empty(tmp_path) -> None:
    enc = BM25SparseEncoder(tmp_path, enabled=False)
    assert enc.encode("任意文本") == {}
