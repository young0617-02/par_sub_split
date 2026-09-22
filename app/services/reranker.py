"""Rerank 精排层（可选）—— 混合检索之后的第二道排序。

为什么需要 Rerank？
    向量检索是 **Bi-Encoder**：查询与文档分别编码，只在最后做一次点积/余弦，
    快但精度有限（两者从未"互相看过"）。
    重排是 **Cross-Encoder**：把 (query, document) 拼接后一起过模型，精度显著更高，
    代价是必须对每个候选做一次前向 —— 所以只能用在**小候选集**上。

本项目的落点（关键设计决策）：
    重排作用在 **子块**上，而不是父块。原因有三：

    1. **粒度一致**：子块才是检索单元（280 字），用父块（1200 字）重排相当于
       换了一个粒度重新排序，与"子块检索"的设计前提相冲突；
    2. **成本可控**：cross-encoder 计算量随文本长度线性增长，280 字 vs 1200 字
       差 4 倍以上，且长文本容易被截断而丢失关键信号；
    3. **分数可比**：同一父块内的多个子块分数可以在同一尺度上比较，便于
       保留"多子块命中"这一强信号。

    重排完成后仍按 `parent_id` 聚合，父块分数取"最佳子块的重排分数"。

显存提示（本机 RTX 4060 8GB）：
    bge-reranker-v2-m3 权重 fp32 约 2.1GB、fp16 约 1.1GB。7B LLM 常驻 GPU 后余量
    通常不足 2GB，因此**默认 rerank 走 CPU**；若要上 GPU，请先把 LLM 的
    `keep_alive` 调小或分时调度，避免 OOM。
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from app.config import settings

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """把 cross-encoder 的 logit 压到 (0, 1)。

    必要性：下游的父块低分过滤用的是「相对比例」（score ≥ best × 0.2），
    当重排分数为负数时该比例判断会失效，因此统一转到正区间。
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class BaseReranker(ABC):
    """重排器接口。"""

    name: str = "base"
    available: bool = False

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回 ``[(原下标, 归一化分数), ...]``，按分数降序，长度不超过 ``top_n``。"""

    def describe(self) -> dict[str, object]:
        return {"provider": self.name, "available": self.available}


class NoopReranker(BaseReranker):
    """未启用/不可用时的空实现：保持原顺序，不打分。"""

    name = "none"
    available = False

    def rerank(
        self, query: str, documents: list[str], top_n: int  # noqa: ARG002  签名需与抽象基类一致
    ) -> list[tuple[int, float]]:
        return [(i, 0.0) for i in range(min(top_n, len(documents)))]


class LocalCrossEncoderReranker(BaseReranker):
    """基于 sentence-transformers ``CrossEncoder`` 的本地重排器。"""

    name = "local-cross-encoder"

    def __init__(
        self,
        model_name: str,
        model_path: str = "",
        device: str = "cpu",
        fp16: bool = False,
        max_length: int = 1024,
    ) -> None:
        from sentence_transformers import CrossEncoder

        target = model_path or model_name
        if not _looks_like_local_dir(target):
            resolved = settings.resolve_local_model(target)
            if resolved:
                logger.info("命中 ModelScope 本地重排模型快照：%s", resolved)
                target = resolved

        kwargs: dict = {"max_length": max_length}
        if device:
            kwargs["device"] = device
        if fp16 and device == "cuda":
            try:
                import torch

                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            except ImportError:  # pragma: no cover
                pass

        self._model = CrossEncoder(target, **kwargs)
        self.model_path = target
        self.device = device
        self.available = True

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        logits = self._model.predict(pairs, show_progress_bar=False)
        scored = [(i, _sigmoid(float(score))) for i, score in enumerate(logits)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "available": True,
            "device": self.device,
            "model_path": self.model_path,
        }


def _looks_like_local_dir(value: str) -> bool:
    from pathlib import Path

    return bool(value) and Path(value).exists()


_reranker: BaseReranker | None = None


def build_reranker() -> BaseReranker:
    if not settings.rerank_enabled:
        return NoopReranker()
    try:
        return LocalCrossEncoderReranker(
            model_name=settings.rerank_model,
            model_path=settings.rerank_model_path,
            device=settings.rerank_device,
            fp16=settings.rerank_fp16,
        )
    except ImportError as exc:
        logger.warning(
            "RERANK_ENABLED=true 但缺少依赖（%s），本次运行将跳过重排。"
            "安装：pip install sentence-transformers transformers",
            exc,
        )
        return NoopReranker()
    except Exception as exc:  # noqa: BLE001
        logger.warning("重排模型加载失败（%s），本次运行将跳过重排", exc)
        return NoopReranker()


def get_reranker() -> BaseReranker:
    """进程内单例：模型加载昂贵。"""
    global _reranker
    if _reranker is None:
        _reranker = build_reranker()
    return _reranker


def reset_reranker() -> None:
    global _reranker
    _reranker = None
