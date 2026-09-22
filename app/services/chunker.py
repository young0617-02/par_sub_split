"""父子分块器（Parent-Child Chunking）—— 本项目的算法核心之一。

为什么要"父子分块"？
    小粒度切块（子块）向量语义更聚焦，召回准确率高，但上下文残缺；
    大粒度切块（父块）语义完整，适合喂给大模型，但向量被稀释、召回差。
    父子分块让两者各司其职：*用子块去检索，用父块去生成*。

实现思路（两级切分）：
    1. 一级切分（父块）：先按 Markdown 标题拆"章节"，再把章节内段落打包到
       ``parent_chunk_size`` 大小，相邻父块之间保留 ``parent_chunk_overlap`` 重叠。
    2. 二级切分（子块）：在**每个父块内部**按句子边界打包成 ``child_chunk_size``
       的子块，并记录 ``parent_index`` 用于回填 MySQL 自增主键。

硬边界约束：子块绝不跨越父块边界，否则 parent_id 无法唯一确定。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# 句子切分：以中文/英文句末标点或换行收尾，保留标点
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]*[。！？!?；;\n]+|[^。！？!?；;\n]+")
# Markdown 标题
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# 次级分隔符（顿号/逗号/分号/冒号/右括号）：句子内部再切一层
_SOFT_SEP_RE = re.compile(r"(?<=[、，,；;：:）)])")

# 原子块（表格 / 代码块）允许的溢出倍数：在此倍数内保持整体不拆。
# 宁可块略微超长，也不要把一行故障码或一行表格从中间切断。
ATOMIC_OVERFLOW_RATIO = 2.5


@dataclass
class ChildChunk:
    """子块：写入 Milvus，参与向量检索。"""

    index: int  # 全文子块序号
    parent_index: int  # 所属父块序号（0-based，落库后换成 MySQL 自增 id）
    heading_path: str
    content: str
    parent_id: int | None = None  # 入库后回填

    @property
    def chunk_id(self) -> str:
        """确定性主键：同内容 + 同位置 -> 同 id，天然幂等（重复入库执行 upsert）。"""
        raw = f"{self.parent_index}|{self.index}|{self.content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    @property
    def embed_text(self) -> str:
        """送入 Embedding 模型的文本（带章节标题做上下文增强）。"""
        if self.heading_path:
            return f"{self.heading_path}\n{self.content}"
        return self.content


@dataclass
class ParentChunk:
    """父块：写入 MySQL，并在 Redis 中做内容镜像。"""

    index: int
    heading_path: str
    content: str
    children: list[ChildChunk] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def char_count(self) -> int:
        return len(self.content)


def _normalize(text: str) -> str:
    """统一换行、压缩空白，保证切分结果稳定。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """按句末标点切分，返回非空句子列表。"""
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def _hard_wrap(text: str, size: int, overlap: int = 0) -> list[str]:
    """兜底切分：完全没有可切分标点时（代码块、表格、长清单），按字符窗口硬切。"""
    if size <= 0 or len(text) <= size:
        return [text]
    step = max(1, size - max(0, overlap))
    out: list[str] = []
    pos = 0
    while pos < len(text):
        out.append(text[pos : pos + size])
        if pos + size >= len(text):
            break
        pos += step
    return out


def _wrap_lines(text: str, size: int) -> list[str]:
    """按**行边界**打包，保证不会把一行代码或一行表格从中间切断。"""
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in text.split("\n"):
        if len(line) > size:
            if buf:
                out.append("\n".join(buf))
                buf, buf_len = [], 0
            out.extend(_hard_wrap(line, size))
            continue
        if buf_len + len(line) + 1 > size and buf:
            out.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        out.append("\n".join(buf))
    return [c for c in out if c.strip()]


def _explode(text: str, size: int, atomic: bool = False) -> list[str]:
    """把可能超长的文本拆成不超过 ``size`` 的单元。

    优先在次级分隔符处断开（保留语义），实在没有分隔符才退化为字符硬切。
    这是"不能产出超长块"的硬保证 —— 否则一段没有句号的文本会一次性吃掉整篇文档。

    Args:
        atomic: 该文本是**不可分割的语义单元**（Markdown 表格 / 围栏代码块）。
            此时：① 在 ``size × ATOMIC_OVERFLOW_RATIO`` 以内保持整体不拆，
            因为把表格行或代码行切断造成的损失远大于块略微超长；
            ② 确实超限时只按行边界拆分，绝不按标点拆。
    """
    if len(text) <= size:
        return [text]

    if atomic:
        if len(text) <= size * ATOMIC_OVERFLOW_RATIO:
            return [text]
        return _wrap_lines(text, size)

    parts = [p for p in _SOFT_SEP_RE.split(text) if p.strip()]
    if len(parts) > 1:
        out: list[str] = []
        for part in parts:
            out.extend([part] if len(part) <= size else _hard_wrap(part, size))
        return out
    return _hard_wrap(text, size)


def _iter_paragraphs(body: str) -> list[tuple[str, bool]]:
    """把章节正文切分为段落，并标记哪些是「不可分割单元」。

    识别两类原子块：
      · 围栏代码块（``` / ~~~ 包裹，可能是工控报文的 JSON 片段）；
      · Markdown 表格（连续以 ``|`` 开头的行，例如故障码对照表）。

    Returns:
        ``[(段落文本, 是否原子块), ...]``
    """
    lines = body.split("\n")
    blocks: list[tuple[str, bool]] = []
    buf: list[str] = []
    fence: str | None = None

    def flush_normal() -> None:
        text = "\n".join(buf).strip()
        if text:
            blocks.append((text, False))
        buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---------- 围栏代码块 ----------
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            flush_normal()
            fence = stripped[:3]
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            blocks.append(("\n".join(block).strip(), True))
            fence = None
            continue

        # ---------- Markdown 表格 ----------
        if stripped.startswith("|"):
            flush_normal()
            table = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i])
                i += 1
            blocks.append(("\n".join(table).strip(), True))
            continue

        # ---------- 普通段落 ----------
        if not stripped:
            flush_normal()
        else:
            buf.append(line)
        i += 1

    flush_normal()
    return blocks


def _join_headings(headings: list[str]) -> str:
    """为跨小节合并而成的父块生成章节标签，如 ``产品手册 > 安装说明 ~ 环境要求``。

    只保留一次公共前缀，避免文档名在标签里重复出现。
    """
    uniq = [h for i, h in enumerate(headings) if h and h not in headings[:i]]
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0]

    parts = [h.split(" > ") for h in uniq]
    common = 0
    # 至少给每段保留最后一级，避免把标签压缩成空
    while all(len(p) > common + 1 for p in parts) and len({p[common] for p in parts}) == 1:
        common += 1

    if common:
        prefix = " > ".join(parts[0][:common])
        tails = [" > ".join(p[common:]) for p in parts]
    else:
        prefix, tails = "", uniq

    # 覆盖小节过多时只保留首尾，否则标签会长到无法阅读
    if len(tails) > 3:
        tails = [tails[0], tails[-1]]

    label = f"{prefix} > " + " ~ ".join(tails) if prefix else " ~ ".join(tails)
    return label if len(label) <= 400 else f"{uniq[0]} ~ {uniq[-1]}"


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切章节。

    Returns:
        [(heading_path, section_body), ...]；无标题时返回 [("", 全文)]。
    """
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            path = " > ".join(t for _, t in stack)
            sections.append((path, body))
        buf.clear()

    has_heading = any(_HEADING_RE.match(ln) for ln in lines)
    if not has_heading:
        return [("", text)]

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()

    if not sections:  # 只有标题没有正文
        return [("", text)]
    return sections


def _pack_units(units: list[str], size: int, overlap: int) -> list[str]:
    """把文本单元按 ``size`` 打包，块间保留 ``overlap`` 字符重叠。

    这是固定窗口切分的"语义友好版"：切割点永远落在单元边界（句子/段落）上。
    """
    if not units:
        return []
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for unit in units:
        # 保险丝：正常调用方已用 _explode 控制单元长度，这里兜底防止死循环与超长块
        if len(unit) > size and not cur:
            chunks.extend(_hard_wrap(unit, size, overlap))
            continue

        if cur_len + len(unit) > size and cur:
            chunks.append("".join(cur))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(cur):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            # 安全阀：重叠区不能等于整块，否则无法前进
            if tail_len >= size:
                tail, tail_len = [], 0
            cur, cur_len = tail, tail_len

        cur.append(unit)
        cur_len += len(unit)

    if cur:
        chunks.append("".join(cur))
    return [c.strip() for c in chunks if c.strip()]


def split_document(
    text: str,
    parent_size: int = 1200,
    parent_overlap: int = 120,
    child_size: int = 280,
    child_overlap: int = 60,
) -> list[ParentChunk]:
    """把一篇原始文档切成"父块 + 子块"两级结构。

    Args:
        text: 原始文档全文。
        parent_size: 父块目标字符数。
        parent_overlap: 父块间重叠字符数。
        child_size: 子块目标字符数。
        child_overlap: 子块间重叠字符数。

    Returns:
        父块列表，每个父块内含其子块列表。
    """
    text = _normalize(text)
    if not text:
        return []

    # ---------- 一级：按章节切，再把碎片合并成"够大"的父块 ----------
    # 文档里小节常常只有两三百字，若直接一小节一个父块，父块会退化成子块粒度，
    # 父子分块就失去了"父块给足上下文"的意义。因此在章节内打包后，
    # 还要把相邻的短块继续合并，直到接近 parent_size。
    packs: list[tuple[str, str]] = []
    for heading_path, body in _split_sections(text):
        # 段落级识别原子块（表格 / 代码块），避免把故障码对照表切断
        blocks = _iter_paragraphs(body)
        if not blocks:
            continue
        # 整段可能没有句末标点（代码块/表格/清单），先炸成不超长的小单元
        units: list[str] = []
        for block_text, is_atomic in blocks:
            units.extend(_explode(block_text, parent_size, atomic=is_atomic))
        packs.extend(
            (heading_path, p_text)
            for p_text in _pack_units(units, parent_size, parent_overlap)
        )

    merged: list[tuple[list[str], str]] = []
    soft_limit = int(parent_size * 1.15)  # 允许 15% 的溢出，避免频繁切断
    for heading_path, p_text in packs:
        if merged and len(merged[-1][1]) + len(p_text) + 2 <= soft_limit:
            prev_headings, prev_text = merged[-1]
            headings = list(prev_headings)
            if heading_path and heading_path not in headings:
                headings.append(heading_path)
            merged[-1] = (headings, f"{prev_text}\n\n{p_text}")
        else:
            merged.append(([heading_path], p_text))

    # ---------- 二级：父块内部按句子切子块（子块绝不跨越父块边界） ----------
    parents: list[ParentChunk] = []
    global_child_index = 0

    for headings, p_text in merged:
        heading_path = _join_headings(headings)
        parent = ParentChunk(index=len(parents), heading_path=heading_path, content=p_text)

        # 子块同样按段落处理：表格/代码块整体保留，普通段落再按句子切
        child_units: list[str] = []
        for block_text, is_atomic in _iter_paragraphs(p_text):
            if is_atomic:
                child_units.extend(_explode(block_text, child_size, atomic=True))
                continue
            for sent in _split_sentences(block_text) or [block_text]:
                child_units.extend(_explode(sent, child_size))
        for c_text in _pack_units(child_units, child_size, child_overlap):
            parent.children.append(
                ChildChunk(
                    index=global_child_index,
                    parent_index=parent.index,
                    heading_path=heading_path,
                    content=c_text,
                )
            )
            global_child_index += 1
        parents.append(parent)

    return parents


def chunk_stats(parents: list[ParentChunk]) -> dict[str, float]:
    """分块质量统计，便于调参。"""
    if not parents:
        return {"parent_count": 0, "child_count": 0, "avg_parent_chars": 0, "avg_child_chars": 0}
    children = [c for p in parents for c in p.children]
    return {
        "parent_count": len(parents),
        "child_count": len(children),
        "avg_parent_chars": round(sum(p.char_count for p in parents) / len(parents), 1),
        "avg_child_chars": (
            round(sum(len(c.content) for c in children) / len(children), 1) if children else 0.0
        ),
        "max_parent_chars": max(p.char_count for p in parents),
        "max_child_chars": max((len(c.content) for c in children), default=0),
    }
