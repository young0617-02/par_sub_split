"""大模型生成层 —— 只负责"拿着父块全文回答问题"。

设计要点：
    1. **可插拔**：任何 OpenAI 兼容协议的模型服务（OpenAI / 通义 / DeepSeek /
       本地 vLLM）都只需改 ``LLM_API_BASE`` + ``LLM_MODEL``。
    2. **可降级**：没有 API Key 时自动切换为"抽取式回答"，保证项目离线可跑通，
       并明确告诉调用方当前是降级模式（``llm_mode=extractive``），避免误以为
       这就是生产答案质量。
    3. **可溯源**：Prompt 强制模型标注 ``[父文档ID: x]``，答案可回查原文。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from app.config import is_local_base, settings
from app.services.sparse import BM25SparseEncoder

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")

SYSTEM_PROMPT = (
    "你是一名严谨的企业知识库问答助手。"
    "你的回答必须完全基于用户提供的【父文档片段】，禁止使用外部知识或自行编造。"
)

USER_TEMPLATE = """请依据下面提供的【父文档片段】回答【用户问题】。

【回答要求】
1. **先逐条检查每个片段**是否包含与问题相关的信息（故障码、型号、操作步骤、参数值、
   原因、注意事项等都算相关信息）。只要存在相关信息，就必须据此作答。
2. **不要轻易回答"无法回答"**：只有当所有片段都与问题完全无关时，才允许回答
   "根据已检索到的资料无法回答该问题"。片段中包含答案却回答无法回答，属于严重错误。
3. 严格使用片段中出现的信息，不得编造或推测片段之外的内容。若信息不完整，
   先给出片段能支持的部分，再说明还缺少什么。
4. 引用来源时，在相应句子末尾标注 [父文档ID: 12]。
5. 使用简体中文，先给结论，再给依据，条理清晰、不要啰嗦。

【父文档片段】
{context}

【用户问题】
{question}

【回答】"""


class AnswerGenerator:
    """答案生成器。"""

    def __init__(
        self,
        provider: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = (provider or settings.llm_provider).lower()
        self.api_base = api_base or settings.llm_base_resolved
        self.model = model or settings.llm_model_resolved
        # Ollama 等本地服务不校验 Key，但 OpenAI SDK 要求 api_key 非空
        self.api_key = (
            api_key if api_key is not None else settings.llm_api_key
        ) or ("ollama" if self.provider == "ollama" or settings.llm_base_is_local else "")
        self._client = None
        self._tokenizer = BM25SparseEncoder.tokenize

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """当前实际生效的模式：openai=真实大模型 / extractive=无密钥且无本地服务时降级。

        判定必须基于**实例自身的 base_url 与 key**，而不是全局 settings：
        否则用 `AnswerGenerator(api_base="http://127.0.0.1:11434/v1")` 构造的实例，
        会因全局配置指向远程而误报为降级模式。
        """
        if self.provider == "mock":
            return "extractive"
        if self.provider == "ollama":
            return "openai"
        if self.provider == "openai":
            return "openai" if self.api_key.strip() else "extractive"
        # auto：有 Key，或 base_url 指向本机/内网（本地服务通常无鉴权）
        return "openai" if (self.api_key.strip() or is_local_base(self.api_base)) else "extractive"

    def describe(self) -> dict[str, object]:
        """对外暴露当前的真实接线方式，便于排查"为什么走了降级"。"""
        return {
            "provider": self.provider,
            "mode": self.mode,
            "base_url": self.api_base,
            "model": self.model,
            "has_api_key": bool(self.api_key.strip()),
        }

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.api_base or None,
                api_key=self.api_key or "not-needed",
                timeout=settings.llm_timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    def generate(
        self, question: str, contexts: list[dict], max_parent_chars: int | None = None
    ) -> tuple[str, str]:
        """生成答案。

        Args:
            question: 用户问题。
            contexts: 父块列表，每项含 ``parent_id`` / ``title`` / ``heading_path`` / ``content``。
            max_parent_chars: 单个父块截断长度，控制 token 成本。

        Returns:
            ``(answer, mode)``。
        """
        if not contexts:
            return "未检索到相关文档，无法回答该问题。", self.mode

        limit = int(max_parent_chars or settings.max_parent_chars)
        context_block = self._build_context_block(contexts, limit)

        if self.mode == "openai":
            try:
                answer = self._call_openai(question, context_block)
                return answer, "openai"
            except Exception:
                logger.exception("大模型调用失败，降级为抽取式回答")
                return self._extractive(question, contexts, limit), "extractive(fallback)"

        return self._extractive(question, contexts, limit), "extractive"

    # ------------------------------------------------------------------
    @staticmethod
    def _build_context_block(contexts: list[dict], limit: int) -> str:
        blocks = []
        for ctx in contexts:
            content = (ctx.get("content") or "")[:limit]
            title = ctx.get("title") or "未命名"
            heading = ctx.get("heading_path") or ""
            blocks.append(
                f"[父文档ID: {ctx.get('parent_id')} | 标题：{title}"
                f"{' | 章节：' + heading if heading else ''}]\n{content}"
            )
        return "\n\n---\n\n".join(blocks)

    def _call_openai(self, question: str, context_block: str) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(context=context_block, question=question),
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    def _extractive(self, question: str, contexts: list[dict], limit: int) -> str:
        """无密钥降级策略：按与问题的词重合度抽取原文句子，并标注出处。"""
        q_tokens = Counter(self._tokenizer(question))
        if not q_tokens:
            q_tokens = Counter(list(question))

        scored: list[tuple[float, int, str]] = []
        for ctx in contexts:
            pid = ctx.get("parent_id")
            text = (ctx.get("content") or "")[:limit]
            for raw_sentence in _SENTENCE_SPLIT_RE.split(text):
                sentence = raw_sentence.strip()
                if len(sentence) < 8:
                    continue
                tokens = self._tokenizer(sentence) or list(sentence)
                overlap = sum(min(q_tokens[t], 1) for t in set(tokens) if t in q_tokens)
                if overlap == 0:
                    continue
                # 归一化：避免长句单纯因为长而得分高
                score = overlap / math.sqrt(len(tokens) or 1)
                scored.append((score, int(pid or 0), sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            head = (contexts[0].get("content") or "")[:300]
            return (
                "【降级模式】未配置大模型 API Key，以下为检索到的原文片段（与问题关键词重合度较低）：\n\n"
                f"{head}\n\n[父文档ID: {contexts[0].get('parent_id')}]"
            )

        lines = ["【降级模式】未配置大模型 API Key，以下为按相关度抽取的原文要点：", ""]
        for _score, pid, sentence in scored[:6]:
            lines.append(f"- {sentence} [父文档ID: {pid}]")
        lines.append("")
        lines.append("提示：在 .env 中配置 LLM_API_KEY 后即可获得真正的模型答案。")
        return "\n".join(lines)


_generator: AnswerGenerator | None = None


def get_answer_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator
