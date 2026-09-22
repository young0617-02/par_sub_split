"""稠密向量 Provider —— 用统一接口屏蔽"本地模型 / 外部 API / 离线降级"三种实现。

生产/本机推荐用法（对齐本机已有资源）：

    EMBEDDING_PROVIDER=local
    EMBEDDING_MODEL=BAAI/bge-m3          # 本机已下载到 ModelScope 缓存
    EMBEDDING_DIM=1024                   # bge-m3 稠密维度，与本项目集合定义一致
    EMBEDDING_DEVICE=cpu                 # 8GB 显存建议留给 7B LLM
    EMBEDDING_FP16=false                 # 仅 device=cuda 时生效

``HashEmbedder`` 只是**零依赖的确定性替身**，用于在没有 GPU / 没有模型的环境下把
整条链路跑通（检索质量不代表生产水平）。
"""

from __future__ import annotations

import logging
import re
import zlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


class BaseEmbedder(ABC):
    """稠密向量编码器接口。"""

    name: str = "base"

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """批量编码文档，返回 shape=(n, dim) 的 float32 数组。"""

    def encode_query(self, text: str) -> list[float]:
        """编码查询（默认与文档同路，子类可覆写以适配 query 指令前缀）。"""
        return self.encode_documents([text])[0].tolist()

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (mat / norm).astype(np.float32)

    def describe(self) -> dict[str, object]:
        return {"provider": self.name, "dim": self.dim}


class HashEmbedder(BaseEmbedder):
    """基于特征哈希的离线稠密向量（char n-gram + 符号哈希）。

    确定性：使用 crc32 而非 Python 内置 ``hash()``，跨进程结果一致。
    """

    name = "hash"

    def __init__(self, dim: int = 1024, ngram_range: tuple[int, ...] = (1, 2, 3)) -> None:
        super().__init__(dim)
        self.ngram_range = ngram_range

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = _WS_RE.sub("", (text or "").lower())
        if not normalized:
            return vec

        for n in self.ngram_range:
            if len(normalized) < n:
                continue
            for i in range(len(normalized) - n + 1):
                gram = normalized[i : i + n]
                data = gram.encode("utf-8")
                h1 = zlib.crc32(data)
                h2 = zlib.crc32(data + b"\x01hash-salt")
                idx = h1 % self.dim
                sign = 1.0 if (h2 & 1) else -1.0
                vec[idx] += sign

        # 次线性压缩 + L2 归一化：抑制高频片段主导
        vec = np.sign(vec) * np.sqrt(np.abs(vec))
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode_one(t) for t in texts])


class ApiEmbedder(BaseEmbedder):
    """OpenAI 兼容协议的 Embedding 服务（OpenAI / 通义 / SiliconFlow / vLLM 均可）。"""

    name = "api"

    def __init__(
        self,
        dim: int,
        model: str,
        api_base: str,
        api_key: str,
        batch_size: int = 16,
        timeout: int = 60,
    ) -> None:
        super().__init__(dim)
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY 为空，无法使用 api 模式")
        from openai import OpenAI  # 延迟导入，避免无该依赖时影响其他模式

        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(base_url=api_base or None, api_key=api_key, timeout=timeout)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend(item.embedding for item in ordered)
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.shape[1] != self.dim:
            raise ValueError(
                f"Embedding 维度不匹配：模型返回 {mat.shape[1]}，配置为 {self.dim}。"
                "请同步修改 EMBEDDING_DIM 并重建 Milvus 集合。"
            )
        return self._l2_normalize(mat)


class LocalEmbedder(BaseEmbedder):
    """本地 sentence-transformers 模型（本机推荐 ``BAAI/bge-m3``）。

    三个针对本机配置的设计点：

    1. **自动解析 ModelScope 本地快照**：``D:\\modelscope\\hub\\models\\BAAI--bge-m3\\snapshots\\master``，
       命中后直接离线加载，不触发联网下载。
    2. **设备与精度可配**：8GB 显存要被 7B LLM 常驻占用，因此默认 ``device=cpu``；
       若确认显存有余量可切 ``cuda`` + ``fp16``（bge-m3 权重 fp32 约 2.1GB，fp16 约 1.1GB）。
    3. **查询指令前缀可选**：``bge-m3`` 无需指令前缀；若换成 ``bge-large-zh`` 这类
       非对称检索模型，可通过 ``EMBEDDING_QUERY_INSTRUCTION`` 配上「为这个句子生成表示…」。
    """

    name = "local"

    def __init__(
        self,
        dim: int,
        model_name: str,
        model_path: str = "",
        device: str = "cpu",
        fp16: bool = False,
        batch_size: int = 16,
        query_instruction: str = "",
    ) -> None:
        super().__init__(dim)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，无法使用 local 模式。请在目标 Python 环境执行：\n"
                "  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  python -m pip install sentence-transformers transformers\n"
                "若希望与主环境隔离，可装到专用环境（本机预留了 D:\\py312-torch，目前为空）。"
            ) from exc

        target = model_path or model_name
        if not Path(target).exists():
            resolved = self._resolve_from_modelscope(target)
            if resolved:
                logger.info("命中 ModelScope 本地快照，离线加载：%s", resolved)
                target = resolved
            else:
                logger.warning("本地未找到模型 %s，将尝试联网从 HuggingFace 下载", target)

        st_kwargs: dict = {}
        if device:
            st_kwargs["device"] = device
        if fp16:
            if device != "cuda":
                logger.warning("EMBEDDING_FP16 仅在 cuda 设备上有意义，当前 device=%s，已忽略", device)
            else:
                try:
                    import torch

                    st_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
                except ImportError:  # pragma: no cover
                    logger.warning("无法导入 torch，忽略 fp16 设置")

        self._model = SentenceTransformer(target, **st_kwargs)
        self.batch_size = int(batch_size)
        self.device = device
        self.query_instruction = (query_instruction or "").strip()
        self.model_path = target

        # sentence-transformers 6.x 把该方法改名为 get_embedding_dimension，做双兼容
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        real_dim = get_dim() if callable(get_dim) else None
        if real_dim and int(real_dim) != self.dim:
            # 以模型真实维度为准，避免与 Milvus 集合定义冲突却静默写错数据
            logger.warning("模型真实维度 %s 与配置 EMBEDDING_DIM=%s 不一致，以模型为准", real_dim, self.dim)
            self.dim = int(real_dim)

    @staticmethod
    def _resolve_from_modelscope(repo_id: str) -> str:
        from app.config import settings

        return settings.resolve_local_model(repo_id)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_query(self, text: str) -> list[float]:
        payload = f"{self.query_instruction}{text}" if self.query_instruction else text
        vec = self._model.encode(
            [payload],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32).tolist()

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "dim": self.dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "model_path": self.model_path,
            "query_instruction": bool(self.query_instruction),
        }


_embedder: BaseEmbedder | None = None


def build_embedder() -> BaseEmbedder:
    from app.config import settings

    provider = settings.embedding_provider
    dim = settings.milvus_dim_resolved

    if provider == "api":
        return ApiEmbedder(
            dim=dim,
            model=settings.embedding_model,
            api_base=settings.embedding_api_base,
            api_key=settings.embedding_api_key,
            batch_size=settings.embedding_batch_size,
        )
    if provider == "local":
        return LocalEmbedder(
            dim=dim,
            model_name=settings.embedding_model,
            model_path=settings.local_model_path,
            device=settings.embedding_device,
            fp16=settings.embedding_fp16,
            batch_size=settings.embedding_batch_size,
            query_instruction=settings.embedding_query_instruction,
        )
    return HashEmbedder(dim=dim)


def get_embedder() -> BaseEmbedder:
    """进程内单例：模型加载昂贵，必须复用。"""
    global _embedder
    if _embedder is None:
        _embedder = build_embedder()
    return _embedder


def reset_embedder() -> None:
    """切换 Embedding 配置后重建单例（供脚本 / 测试使用）。"""
    global _embedder
    _embedder = None
