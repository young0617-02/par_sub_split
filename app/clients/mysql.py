"""MySQL 连接与建表（父文档持久层）。"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """进程内单例 Engine（自带连接池）。"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.mysql_url,
            pool_size=settings.mysql_pool_size,
            max_overflow=settings.mysql_max_overflow,
            pool_pre_ping=True,  # 自动剔除被 MySQL 8h 空闲断开的连接
            pool_recycle=3600,
            echo=settings.mysql_echo,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务上下文：正常提交、异常回滚、最终关闭。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_database() -> None:
    """确保目标数据库存在（首次部署时自动创建）。"""
    db_name = settings.mysql_database
    if not _IDENT_RE.match(db_name):
        raise ValueError(f"非法的数据库名：{db_name}")

    # 注意：URL.set(database=None) 会忽略 None 值（保留原库名），
    # 必须用 _replace 才能真正构造出"不带库名"的连接串，否则无法执行 CREATE DATABASE
    bootstrap_url = settings.mysql_url._replace(database=None)
    tmp_engine = create_engine(bootstrap_url, pool_pre_ping=True, future=True)
    try:
        with tmp_engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            conn.commit()
    finally:
        tmp_engine.dispose()


def init_schema(create_database: bool = True) -> dict[str, object]:
    """初始化表结构（幂等）。"""
    if create_database:
        ensure_database()
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("MySQL 表结构就绪：%s", ", ".join(Base.metadata.tables))
    return {"tables": list(Base.metadata.tables.keys())}


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MySQL 连通性检查失败：%s", exc)
        return False


def dispose() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine, _session_factory = None, None
