"""MySQL 父文档表 ``parent_document_t``。

设计要点：
1. ``id`` 自增主键 —— 该 id 会一路传递到 Milvus 的 ``parent_id`` 和 Redis 的 key，
   是整条"父块回表"链路的唯一锚点。
2. ``content_hash`` 唯一索引 —— 保证同一份父块内容不会重复入库（幂等）。
3. ``content`` 是喂给大模型的最终上下文，因此字段类型用 ``LONGTEXT``。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类。"""


class ParentDocument(Base):
    __tablename__ = "parent_document_t"
    __table_args__ = (
        Index("idx_pdt_doc_id", "doc_id"),
        # 同一文档内父块内容唯一 -> 重复入库幂等；不同文档允许存在相同内容
        Index("uk_pdt_doc_hash", "doc_id", "content_hash", unique=True),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "mysql_collate": "utf8mb4_unicode_ci",
            "comment": "父文档表：保存父块全文，供大模型作为上下文使用",
        },
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="父文档主键，同时作为 Milvus.parent_id 与 Redis key",
    )
    doc_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="业务文档标识（同一文档的父块共享）"
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="", comment="文档标题")
    heading_path: Mapped[str] = mapped_column(
        String(512), nullable=False, default="", comment="父块所在章节路径，如 '指南 > 安装 > 环境要求'"
    )
    content: Mapped[str] = mapped_column(
        LONGTEXT().with_variant(Text(), "sqlite"), nullable=False, comment="父块全文"
    )
    source: Mapped[str] = mapped_column(
        String(512), nullable=False, default="", comment="来源，如文件名/URL"
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="父块在文档内的顺序号"
    )
    child_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="该父块切分出的子块数量"
    )
    char_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="父块字符数"
    )
    content_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="父块内容 sha256，用于幂等去重"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<ParentDocument id={self.id} doc_id={self.doc_id!r} "
            f"heading={self.heading_path!r} chars={self.char_count}>"
        )
