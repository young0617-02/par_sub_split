"""全局配置：统一由环境变量 / .env 驱动，避免配置散落在业务代码里。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
SAMPLE_DIR = DATA_DIR / "samples"

for _d in (DATA_DIR, INDEX_DIR, SAMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 本机/内网地址：本地推理服务（Ollama 等）通常不校验 API Key
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "::1", "host.docker.internal")


def is_local_base(url: str | None) -> bool:
    """判断 base_url 是否指向本机/内网（用于判定「无需 API Key」）。"""
    lowered = (url or "").lower()
    return any(host in lowered for host in _LOCAL_HOSTS)


class Settings(BaseSettings):
    """项目配置。

    字段名小写，环境变量名自动映射为大写，例如 ``mysql_port`` -> ``MYSQL_PORT``。
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "parent-child-rag"
    debug: bool = False

    # ---------------------- 服务与安全 ----------------------
    # CORS 允许来源，逗号分隔。默认仅本机；生产环境必须显式配置为真实域名前缀列表。
    cors_allow_origins: str = "http://127.0.0.1,http://localhost"
    # 管理接口（/api/v1/admin/*）的访问令牌。
    # 为空表示不校验（仅建议本机开发使用）；非空时要求请求头 X-Admin-Token 一致。
    admin_token: str = ""
    # 上传文件大小上限（MB）。防止超大文件耗尽内存。
    upload_max_mb: int = 20
    # 允许上传的扩展名（逗号分隔）。白名单机制，避免任意文件被当作文本解析。
    upload_allow_ext: str = ".md,.markdown,.txt,.pdf"
    # 启动时预热本地模型（嵌入 / 重排）与 BM25 词表。
    # 为什么必须预热：这些组件是懒加载的，若不预热，**第一次 /health 或 /search 请求**
    # 会阻塞在模型加载上（本机 BGE-M3 CPU 加载约 6~10s），导致健康检查超时、
    # 上游误判服务不可用。预热后启动稍慢，但首个业务请求的延迟是可预期的。
    warmup_on_startup: bool = True

    # ------------------------- MySQL -------------------------
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 9999
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "par_sub_split"
    mysql_charset: str = "utf8mb4"
    mysql_pool_size: int = 10
    mysql_max_overflow: int = 20
    mysql_echo: bool = False

    # ------------------------- Redis -------------------------
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_parent_prefix: str = "psd:parent:"
    redis_parent_ttl: int = 0  # 0 = 永不过期
    # 读写超时（秒）。设为 5s 在高负载（如同时加载本地嵌入模型）下容易误判超时，
    # 因此放宽到 10s；缓存故障本就不应影响正确性，超时会自动回表 MySQL。
    redis_socket_timeout: int = 10

    # ------------------------- Milvus -------------------------
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "child_chunk_t"
    milvus_dim: int = 1024
    milvus_timeout: int = 30

    # ------------------------- 分块 -------------------------
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 120
    child_chunk_size: int = 280
    child_chunk_overlap: int = 60
    max_parent_chars: int = 4000

    # ------------------------- 检索 -------------------------
    retrieve_top_k: int = 20
    final_top_k: int = 4
    hybrid_ranker: str = "rrf"  # rrf | weighted
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    # 低分父块过滤：父块分数低于「最高分 × 该比例」时丢弃，0 表示不过滤
    score_ratio_threshold: float = 0.2
    # 说明：单分支对照检索（explain）由请求体参数控制，默认 false，
    # 不提供全局开关 —— 之前存在的 RETRIEVAL_EXPLAIN_DEFAULT 设置项从未被读取，已移除。

    # ---------------------- 稀疏向量 BM25 ----------------------
    sparse_enabled: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---------------------- 稠密向量 Embedding ----------------------
    embedding_provider: str = "hash"  # hash | api | local
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_batch_size: int = 16
    local_model_path: str = ""
    # 本地推理设备与精度。8GB 显存的机器上，LLM 常驻 GPU 后余量很小，
    # 建议 embedding 走 CPU（i7-13650HX 14 核足够），把显存留给 7B 模型。
    embedding_device: str = "cpu"  # cpu | cuda
    embedding_fp16: bool = False
    # 查询指令前缀：bge-m3 留空；bge-large-zh 等非对称模型可填
    # "为这个句子生成表示以用于检索相关文章："
    embedding_query_instruction: str = ""

    # -------- ModelScope 本地模型缓存（自动解析本地快照路径） --------
    # 形如 D:\modelscope\hub\models\BAAI--bge-m3\snapshots\master
    modelscope_cache: str = r"D:\modelscope\hub\models"

    # ---------------------- Rerank 精排（可选） ----------------------
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_model_path: str = ""
    rerank_device: str = "cpu"  # cpu | cuda
    rerank_fp16: bool = False
    # 参与重排的候选子块数（0 表示沿用 RETRIEVE_TOP_K）
    rerank_candidates: int = 0
    # 重排后保留的子块数（随后仍会按 parent_id 聚合）
    rerank_keep: int = 10

    # ------------------------- 大模型 -------------------------
    # auto   = 有 API Key、或 base_url 指向本机时走 OpenAI 兼容协议
    # ollama = 本地 Ollama（无需 Key，默认 http://127.0.0.1:11434/v1）
    # openai = 远程 OpenAI 兼容服务（必须有 Key）
    # mock   = 强制走抽取式降级
    llm_provider: str = "auto"  # auto | ollama | openai | mock
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # 本地 Ollama（provider=ollama 时生效，无需 API Key）
    llm_ollama_base: str = "http://127.0.0.1:11434/v1"
    llm_ollama_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout: int = 60

    # ========================= 派生属性 =========================

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS 允许来源列表（已去空、去重）。"""
        items = [o.strip() for o in (self.cors_allow_origins or "").split(",")]
        return [o for o in dict.fromkeys(items) if o]

    @property
    def upload_max_bytes(self) -> int:
        return max(1, int(self.upload_max_mb)) * 1024 * 1024

    @property
    def upload_allow_ext_set(self) -> set[str]:
        """允许的扩展名集合（统一小写、带点）。"""
        items = [e.strip().lower() for e in (self.upload_allow_ext or "").split(",")]
        return {e if e.startswith(".") else f".{e}" for e in items if e}

    @property
    def admin_enabled(self) -> bool:
        """是否启用管理接口令牌校验。"""
        return bool(self.admin_token.strip())

    @property
    def mysql_url(self) -> URL:
        return URL.create(
            drivername="mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={"charset": self.mysql_charset},
        )

    @property
    def milvus_dim_resolved(self) -> int:
        """真实生效的向量维度：EMBEDDING_DIM 优先，未设置时回退 MILVUS_DIM。"""
        return int(self.embedding_dim or self.milvus_dim)

    @property
    def llm_base_is_local(self) -> bool:
        return is_local_base(self.llm_api_base)

    @property
    def llm_base_resolved(self) -> str:
        """实际使用的 base_url。provider=ollama 时指向本地 Ollama。"""
        if self.llm_provider == "ollama":
            return self.llm_ollama_base
        return self.llm_api_base

    @property
    def llm_model_resolved(self) -> str:
        """实际使用的模型名。

        provider=ollama 但 LLM_MODEL 仍停留在 OpenAI 默认值时，
        自动改用 LLM_OLLAMA_MODEL，避免出现"用 Ollama 请求 gpt-4o-mini"的误配。
        """
        if self.llm_provider == "ollama" and self.llm_model == "gpt-4o-mini":
            return self.llm_ollama_model
        return self.llm_model

    @property
    def llm_enabled(self) -> bool:
        """是否具备调用真实大模型的条件。

        **本地推理服务（Ollama 等）不校验 API Key**，因此不能再用
        「Key 非空」作为唯一判据，否则本地模型会被误判为不可用而静默降级。
        """
        if self.llm_provider == "mock":
            return False
        if self.llm_provider == "ollama":
            return True
        if self.llm_provider == "openai":
            return bool(self.llm_api_key.strip())
        # auto：有 Key，或 base_url 明确指向本机/内网（本地服务通常无鉴权）
        return bool(self.llm_api_key.strip()) or self.llm_base_is_local

    def model_cache_roots(self) -> list[Path]:
        """候选的模型缓存根目录（兼容多种书写形式）。

        坑点：Windows 路径写在 ``.env`` 里时，反斜杠会被 dotenv 的转义规则吞掉，
        实测 ``D:\\modelscope\\hub\\models`` 会被解析成 ``D:\\modelscope\\hub``，
        导致「模型明明在本地却找不到」。

        因此这里不依赖环境文件的转义细节：统一把反斜杠归一化，
        并同时尝试「原值」「原值/models」「原值/hub/models」三种可能。
        """
        raw = (self.modelscope_cache or "").strip().replace("\\", "/")
        if not raw:
            return []
        base = Path(raw)
        candidates = [base]
        if base.name.lower() != "models":
            candidates.append(base / "models")
        candidates.append(base / "hub" / "models")

        roots: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                roots.append(candidate)
        return roots

    def resolve_local_model(self, repo_id: str) -> str:
        """把 ``BAAI/bge-m3`` 解析为 ModelScope 本地快照目录。

        本机模型缓存形如::

            D:/modelscope/hub/models/BAAI--bge-m3/snapshots/master

        解析成功返回目录绝对路径，失败返回空串（由调用方决定回退策略）。
        """
        repo_dir_name = repo_id.replace("/", "--")
        for root in self.model_cache_roots():
            repo_dir = root / repo_dir_name / "snapshots"
            if not repo_dir.is_dir():
                continue
            for snapshot in sorted(repo_dir.iterdir(), reverse=True):
                if snapshot.is_dir() and (snapshot / "config.json").exists():
                    return str(snapshot)
        return ""

    # ========================= 校验 =========================

    @field_validator("hybrid_ranker")
    @classmethod
    def _check_ranker(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"rrf", "weighted"}:
            raise ValueError("HYBRID_RANKER 只能是 rrf 或 weighted")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def _check_embedding_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"hash", "api", "local"}:
            raise ValueError("EMBEDDING_PROVIDER 只能是 hash / api / local")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例，避免反复读取 .env。"""
    return Settings()


settings = get_settings()
