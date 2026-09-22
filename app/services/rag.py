"""RAG 问答编排：检索 → 父块装配 → 大模型生成 → 返回带引用的答案。"""

from __future__ import annotations

import time

from app.config import settings
from app.schemas import ChatResponse, Citation
from app.services.llm import get_answer_generator
from app.services.retriever import get_retrieval_service


class RagService:
    """把检索层与生成层串成一条完整的问答链路。"""

    def __init__(self) -> None:
        self.retriever = get_retrieval_service()
        self.generator = get_answer_generator()

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        final_top_k: int | None = None,
        use_sparse: bool = True,
        max_parent_chars: int | None = None,
    ) -> ChatResponse:
        t0 = time.perf_counter()

        parents, timings = self.retriever.retrieve(
            query=query, top_k=top_k, final_top_k=final_top_k, use_sparse=use_sparse
        )
        t_retrieve = time.perf_counter()

        contexts = [
            {
                "parent_id": p.parent_id,
                "title": p.title,
                "heading_path": p.heading_path,
                "content": p.content,
            }
            for p in parents
        ]
        answer, mode = self.generator.generate(
            query, contexts, max_parent_chars=max_parent_chars or settings.max_parent_chars
        )
        t_llm = time.perf_counter()

        timings["llm_ms"] = round((t_llm - t_retrieve) * 1000, 2)
        timings["total_ms"] = round((t_llm - t0) * 1000, 2)

        citations = [
            Citation(
                parent_id=p.parent_id,
                doc_id=p.doc_id,
                title=p.title,
                heading_path=p.heading_path,
            )
            for p in parents
        ]
        return ChatResponse(
            query=query,
            answer=answer,
            citations=citations,
            llm_mode=mode,
            parents=parents,
            timings=timings,
        )


_rag_service: RagService | None = None


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service
