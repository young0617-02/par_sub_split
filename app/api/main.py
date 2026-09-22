"""FastAPI 应用入口。

启动：
    python -m uvicorn app.api.main:app --reload --port 8000
或：
    python scripts/run_server.py
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.api.routes import router
from app.clients.milvus_client import get_chunk_store
from app.clients.mysql import init_schema
from app.config import PROJECT_ROOT, settings
from app.services.retriever import get_retrieval_service

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

WEB_DIR = PROJECT_ROOT / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时自动建表 / 建集合，避免"忘记初始化"导致接口 500。"""
    logger.info(
        "启动 %s v%s（embedding=%s, sparse=%s, llm=%s, admin_token=%s）",
        settings.app_name, __version__, settings.embedding_provider,
        settings.sparse_enabled, settings.llm_provider,
        "已启用" if settings.admin_enabled else "未启用",
    )
    try:
        init_schema()
    except Exception:
        logger.exception("MySQL 初始化失败（请检查容器是否启动）")
    try:
        get_chunk_store().ensure_collection()
    except Exception:
        logger.exception("Milvus 集合初始化失败（请检查容器与 token）")
    warmup()
    yield
    logger.info("应用关闭")


def warmup() -> None:
    """预热懒加载组件，让首个业务请求的延迟可预期。

    不预热会带来两个真实问题：

    1. **`/health` 与 `/stats` 会触发模型加载并阻塞**。这两个接口内部要访问检索服务，
       而检索服务是懒加载的；本机 BGE-M3（CPU）加载约 6~10s，
       足以让健康检查超时，使上游误判服务不可用。
    2. 第一个业务请求要额外等待同样的冷启动时间。

    单个组件预热失败只记录日志、不阻断启动：嵌入不可用会退化为 hash 向量，
    重排不可用会退化为空实现，主链路仍可运行。
    """
    if not settings.warmup_on_startup:
        logger.info("WARMUP_ON_STARTUP=false，跳过组件预热")
        return

    t0 = time.perf_counter()
    targets = (
        ("BM25 词表", lambda: get_retrieval_service().encoder.describe()),
        ("嵌入模型", lambda: get_retrieval_service().embedder),
        ("重排模型", lambda: get_retrieval_service().reranker),
    )
    for label, factory in targets:
        start = time.perf_counter()
        try:
            factory()
        except Exception:
            logger.exception("预热 %s 失败，相关功能将走降级路径", label)
        else:
            logger.info("预热 %s 完成（%.2fs）", label, time.perf_counter() - start)
    logger.info("组件预热结束，共 %.2fs", time.perf_counter() - t0)


app = FastAPI(
    title="父子分块 RAG 检索系统",
    description="MySQL（父块）+ Redis（内容缓存）+ Milvus（稀疏/稠密子块）三存储协同的 RAG 检索实现",
    version=__version__,
    lifespan=lifespan,
)

# CORS：默认仅允许本机来源，且关闭凭证携带。
# 说明：`allow_origins=["*"] + allow_credentials=True` 是无效且危险的组合
# （浏览器会拒绝，且等于对所有站点开放凭证请求），因此这里改为白名单 + 关闭凭证。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
    """兜底异常处理：完整堆栈进日志，响应体只给通用信息。

    安全考虑：直接把 ``str(exc)`` 返回给调用方会泄漏内部实现细节
    （连接串、表名、文件路径、第三方 SDK 报文等），因此对外统一为通用文案。
    """
    logger.exception("未处理异常：%s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务内部错误，请查看服务端日志（traceback 已记录）"},
    )



@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """返回可视化调试页。"""
    return FileResponse(Path(WEB_DIR) / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> JSONResponse:
    return JSONResponse(status_code=204, content=None)
