"""接口层安全与运维行为的回归测试（纯单元测试，不需要外部服务）。

覆盖本次审查修复的几个风险点：
    1. 管理接口令牌校验（ADMIN_TOKEN）
    2. 上传体积上限（防内存耗尽）
    3. 上传扩展名白名单（防任意文件被当文本解析）
    4. CORS 与上传相关配置解析
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app.api.routes import read_upload_limited, require_admin
from app.config import Settings, settings
from app.services.ingest import extract_pdf_bytes, extract_text_from_bytes


# ==============================================================================
# 一、管理接口令牌
# ==============================================================================
def test_require_admin_is_noop_when_token_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """未配置 ADMIN_TOKEN 时不校验（本机开发模式）。"""
    monkeypatch.setattr(settings, "admin_token", "")
    require_admin(None)          # 不应抛异常
    require_admin("anything")


def test_require_admin_rejects_missing_or_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "admin_token", "s3cret-token")
    for bad in (None, "", "wrong", "s3cret-token-x"):
        with pytest.raises(HTTPException) as exc:
            require_admin(bad)
        assert exc.value.status_code == 401


def test_require_admin_accepts_correct_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "admin_token", "s3cret-token")
    require_admin("s3cret-token")   # 不应抛异常
    require_admin("  s3cret-token  ")  # 允许首尾空白


# ==============================================================================
# 二、上传体积上限
# ==============================================================================
class _FakeUpload:
    """模拟 UploadFile 的分块读取行为。"""

    def __init__(self, data: bytes) -> None:
        self._remaining = data

    async def read(self, size: int = -1) -> bytes:
        if not self._remaining:
            return b""
        take = len(self._remaining) if size < 0 else min(size, len(self._remaining))
        chunk, self._remaining = self._remaining[:take], self._remaining[take:]
        return chunk


def test_read_upload_limited_passes_small_payload() -> None:
    data = asyncio.run(read_upload_limited(_FakeUpload(b"x" * 100), 1000))  # type: ignore[arg-type]
    assert data == b"x" * 100


def test_read_upload_limited_rejects_oversize_payload() -> None:
    """超限必须立刻中断并返回 413，而不是把整个文件读进内存。"""
    payload = b"x" * (3 * 1024 * 1024)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(read_upload_limited(_FakeUpload(payload), 1024 * 1024))  # type: ignore[arg-type]
    assert exc.value.status_code == 413
    assert "MB" in str(exc.value.detail)


# ==============================================================================
# 三、扩展名白名单与配置解析
# ==============================================================================
def test_upload_allow_ext_set_normalizes() -> None:
    cfg = Settings(upload_allow_ext=".MD, txt ,PDF,,")
    assert cfg.upload_allow_ext_set == {".md", ".txt", ".pdf"}


def test_upload_max_bytes_converts_mb() -> None:
    assert Settings(upload_max_mb=5).upload_max_bytes == 5 * 1024 * 1024
    # 防御性下限：配置成 0 或负数也不能让上限变成 0（否则任何上传都会被拒）
    assert Settings(upload_max_mb=0).upload_max_bytes >= 1024 * 1024


def test_cors_origin_list_parsing_and_defaults() -> None:
    cfg = Settings(cors_allow_origins=" http://a.com , http://b.com ,, ")
    assert cfg.cors_origin_list == ["http://a.com", "http://b.com"]
    # 默认不放开为 *
    assert "*" not in Settings().cors_origin_list


def test_cors_middleware_is_restricted() -> None:
    """CORS 必须使用来源白名单且关闭凭证携带。

    ``allow_origins=["*"] + allow_credentials=True`` 是无效组合（浏览器会直接拒绝），
    且等价于对所有站点开放凭证请求，因此本项目显式禁用凭证。
    """
    from starlette.middleware.cors import CORSMiddleware

    from app.api.main import app

    entries = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    assert entries, "未找到 CORSMiddleware，请检查 app/api/main.py 的中间件注册"

    kwargs = entries[0].kwargs
    assert kwargs["allow_credentials"] is False
    assert kwargs["allow_origins"], "CORS 白名单不应为空"
    assert "*" not in kwargs["allow_origins"], "不应放开为通配来源"


# ==============================================================================
# 四、文件解析（统一实现）
# ==============================================================================
def test_extract_text_from_bytes_decodes_utf8() -> None:
    raw = "父子分块：用子块检索，用父块生成。".encode()
    assert extract_text_from_bytes(raw, "a.md") == "父子分块：用子块检索，用父块生成。"


def test_extract_text_from_bytes_tolerates_invalid_utf8() -> None:
    """非 UTF-8 字节不应抛异常（errors=ignore），避免上传接口 500。"""
    assert extract_text_from_bytes(b"\xff\xfeabc", "a.txt") == "abc"


def test_extract_pdf_bytes_rejects_garbage() -> None:
    """伪造的 PDF 应抛错，由路由层转成 400 而不是 500。"""
    with pytest.raises(Exception):  # noqa: B017  pypdf 异常类型随版本变化
        extract_pdf_bytes(b"not-a-pdf")


def test_routes_do_not_duplicate_pdf_parsing() -> None:
    """路由层必须复用 ingest 的解析实现，不能各写一份。"""
    from app.api import routes

    assert not hasattr(routes, "_parse_pdf_bytes"), "PDF 解析应统一放在 services/ingest.py"
    assert routes.extract_text_from_bytes is extract_text_from_bytes


# ==============================================================================
# 五、监控接口不得触发重资源懒加载
# ==============================================================================
class _StubMilvusStore:
    """替换真实 Milvus 客户端，让健康检查测试无需外部服务。"""

    collection_name = "stub_collection"

    def ping(self) -> bool:
        return True

    def describe(self) -> dict:
        return {"exists": True, "collection": self.collection_name}


def test_health_reports_config_values_without_loading_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检索服务未就绪时，/health 必须返回配置预期值而不是去加载模型。

    回归背景：`/health` 内部访问检索服务会触发嵌入模型懒加载（本机 BGE-M3 CPU 约 6~10s），
    导致健康检查超时、上游误判服务不可用。
    """
    from app.api import routes

    monkeypatch.setattr(routes, "get_chunk_store", _StubMilvusStore)
    monkeypatch.setattr(routes, "is_retrieval_service_ready", lambda: False)
    monkeypatch.setattr(routes.mysql_client, "ping", lambda: True)
    monkeypatch.setattr(routes.redis_client, "ping", lambda: True)
    monkeypatch.setattr(routes, "get_retrieval_service", _raise_if_called)

    resp = routes.health()
    assert resp.mysql is True and resp.redis is True and resp.milvus is True
    assert resp.detail["retrieval_ready"] is False
    assert resp.detail["embedding"]["provider"] == settings.embedding_provider
    assert resp.detail["embedding"]["dim"] == settings.milvus_dim_resolved
    assert "note" in resp.detail["embedding"]
    assert resp.detail["rerank"]["available"] is False
    # LLM 接线信息不依赖本地模型，任何情况下都应可用
    assert resp.detail["llm"]["mode"] in {"openai", "extractive"}


def _raise_if_called():
    raise AssertionError("健康检查不应初始化检索服务（会连带加载嵌入模型）")


def test_health_survives_milvus_describe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """/health 自身绝不能抛异常：Milvus 元数据读取失败也要返回可用的探测结果。"""

    class _BrokenStore(_StubMilvusStore):
        def describe(self) -> dict:
            raise RuntimeError("milvus unavailable")

    from app.api import routes

    monkeypatch.setattr(routes, "get_chunk_store", _BrokenStore)
    monkeypatch.setattr(routes, "is_retrieval_service_ready", lambda: False)
    monkeypatch.setattr(routes.mysql_client, "ping", lambda: False)
    monkeypatch.setattr(routes.redis_client, "ping", lambda: True)

    resp = routes.health()
    assert resp.milvus is True
    assert "milvus_error" in resp.detail
    assert "milvus unavailable" not in str(resp.detail["milvus_error"]), "不应回显内部异常原文"


# ==============================================================================
# 六、启动预热
# ==============================================================================
def test_warmup_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import main

    monkeypatch.setattr(main.settings, "warmup_on_startup", False)
    monkeypatch.setattr(
        main, "get_retrieval_service", _raise_if_called
    )
    main.warmup()   # 关闭预热时不应触碰检索服务


def test_warmup_swallows_component_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """预热失败必须只记录日志：嵌入不可用会退化为 hash，主链路仍要能启动。"""
    from app.api import main

    class _BrokenService:
        @property
        def encoder(self):
            raise RuntimeError("bm25 unavailable")

    monkeypatch.setattr(main.settings, "warmup_on_startup", True)
    monkeypatch.setattr(main, "get_retrieval_service", lambda: _BrokenService())
    main.warmup()   # 不应抛出异常


def test_warmup_default_enabled() -> None:
    """默认必须预热，否则首个请求/健康检查会被模型加载阻塞。"""
    assert Settings().warmup_on_startup is True


def test_redis_socket_timeout_is_generous() -> None:
    """缓存超时过短会在高负载（模型加载占满 CPU）时误判失败。"""
    assert Settings().redis_socket_timeout >= 5
