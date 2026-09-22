"""BM25 稀疏向量编码器 —— 混合检索的"关键词"分支。

为什么需要稀疏向量？
    稠密向量擅长语义泛化（"怎么装" ↔ "安装步骤"），但对**专有名词、错误码、
    型号、人名、生僻词**不敏感；BM25 恰好相反。两者并行召回再做融合（RRF /
    加权），是当前工业界性价比最高的检索方案。

工程要点：
    1. Milvus 稀疏向量要求 ``{dim_index: weight}``，所以必须把"词"映射为整数下标。
       这里维护一份**持久化词表**（vocab.json），保证入库与查询阶段下标一致。
    2. IDF 依赖全库统计量（df / N / avgdl），同样需要落盘复用。
    3. 权重做 L2 归一化后，内积(IP)打分与余弦相似度等价，便于与稠密分支融合。
    4. Milvus 不接受空稀疏向量，因此保留一个"空文档占位 token"（下标 0）。
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path

import jieba

# 空向量占位下标（Milvus 拒绝空稀疏向量）
EMPTY_TOKEN_ID = 0
_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# 极简停用词表：只去掉高频且无区分度的虚词，避免误伤专业术语
_STOPWORDS = {
    "的", "了", "和", "与", "是", "在", "有", "就", "都", "而", "及", "或",
    "一个", "我们", "你们", "他们", "这个", "那个", "可以", "进行", "以及",
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "in", "on", "for",
}


class BM25SparseEncoder:
    """BM25 词袋编码器（带持久化词表与全局统计量）。"""

    def __init__(
        self,
        index_dir: str | Path,
        k1: float = 1.5,
        b: float = 0.75,
        enabled: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_path = self.index_dir / "vocab.json"
        self.stats_path = self.index_dir / "bm25_stats.json"
        self.k1 = k1
        self.b = b
        self.enabled = enabled

        self._lock = threading.RLock()
        self.vocab: dict[str, int] = {}
        self.df: dict[str, int] = {}
        self.n_docs: int = 0
        self.total_len: int = 0
        self._load()

    # ------------------------------------------------------------------
    # 基础分词
    # ------------------------------------------------------------------
    @staticmethod
    def tokenize(text: str) -> list[str]:
        """中英混合分词：jieba 精确模式 + 规范化为小写。"""
        if not text:
            return []
        tokens: list[str] = []
        for raw in jieba.lcut(text):
            tok = raw.strip().lower()
            if not tok or len(tok) > 32:
                continue
            if _PUNCT_RE.match(tok):
                continue
            if tok in _STOPWORDS:
                continue
            tokens.append(tok)
        return tokens

    # ------------------------------------------------------------------
    # 统计量
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def avgdl(self) -> float:
        return (self.total_len / self.n_docs) if self.n_docs else 1.0

    def _idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        # BM25 的 +0.5 平滑：避免 idf 出现负值
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    # ------------------------------------------------------------------
    # 词表 / 统计量维护
    # ------------------------------------------------------------------
    def fit(self, texts: list[str], persist: bool = True) -> dict[str, int]:
        """把一批文本纳入词表与 df 统计（入库时调用）。

        Returns:
            本次新增的 token 数量等统计信息。
        """
        added_vocab = 0
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                if not tokens:
                    continue
                self.n_docs += 1
                self.total_len += len(tokens)
                for tok in set(tokens):
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab) + 1  # 0 留给占位
                        added_vocab += 1
                    self.df[tok] = self.df.get(tok, 0) + 1
            if persist:
                self.save()
        return {"added_vocab": added_vocab, "vocab_size": self.vocab_size, "n_docs": self.n_docs}

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode(self, text: str) -> dict[int, float]:
        """把文本编码为 Milvus 稀疏向量 ``{index: weight}``。"""
        if not self.enabled:
            return {}
        tokens = self.tokenize(text)
        if not tokens:
            return {EMPTY_TOKEN_ID: 1.0}

        with self._lock:
            counts = Counter(tokens)
            dl = len(tokens)
            avgdl = self.avgdl
            vocab = self.vocab
            weights: dict[int, float] = {}

            for tok, tf in counts.items():
                idx = vocab.get(tok)
                if idx is None:
                    # 词表外的词直接丢弃：稀疏分支只负责"精确词命中"
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score = self._idf(tok) * (tf * (self.k1 + 1)) / denom
                if score > 0:
                    weights[idx] = weights.get(idx, 0.0) + score

        if not weights:
            return {EMPTY_TOKEN_ID: 1.0}
        return self._l2_normalize(weights)

    def encode_batch(self, texts: list[str]) -> list[dict[int, float]]:
        return [self.encode(t) for t in texts]

    @staticmethod
    def _l2_normalize(weights: dict[int, float]) -> dict[int, float]:
        norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        return {k: v / norm for k, v in weights.items()}

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self) -> None:
        with self._lock:
            self._atomic_dump(self.vocab_path, self.vocab)
            self._atomic_dump(
                self.stats_path,
                {"df": self.df, "n_docs": self.n_docs, "total_len": self.total_len},
            )

    def _load(self) -> None:
        if self.vocab_path.exists():
            self.vocab = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        if self.stats_path.exists():
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
            self.df = data.get("df", {})
            self.n_docs = int(data.get("n_docs", 0))
            self.total_len = int(data.get("total_len", 0))

    @staticmethod
    def _atomic_dump(path: Path, payload: object) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def reset(self) -> None:
        """清空词表与统计量（重建索引时使用）。"""
        with self._lock:
            self.vocab, self.df, self.n_docs, self.total_len = {}, {}, 0, 0
            self.save()

    def describe(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "vocab_size": self.vocab_size,
            "n_docs": self.n_docs,
            "avgdl": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }


_encoder: BM25SparseEncoder | None = None


def get_sparse_encoder() -> BM25SparseEncoder:
    """进程内单例，保证词表在多请求间共享。"""
    global _encoder
    if _encoder is None:
        from app.config import INDEX_DIR, settings

        _encoder = BM25SparseEncoder(
            index_dir=INDEX_DIR,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
            enabled=settings.sparse_enabled,
        )
    return _encoder
