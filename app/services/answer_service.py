from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.services.cache_service import CacheService


class AnswerService:
    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
    ):
        self.retrieval_service = RetrievalService(
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
        )
        self.llm_service = LLMService()
        self.cache_service = CacheService()

    def answer(self, question: str, top_k: int = 8):
        cache_key = f"{question}:{top_k}"
        cached = self.cache_service.get(cache_key)
        if cached is not None:
            return cached

        hits = self.retrieval_service.retrieve(question=question, top_k=top_k)

        contexts = []
        citations = []

        for hit in hits:
            payload = hit.payload
            contexts.append(payload["text"])
            citations.append(
                {
                    "document_id": payload["document_id"],
                    "filename": payload["filename"],
                    "chunk_id": payload["chunk_id"],
                    "text": payload["text"],
                    "score": hit.score,
                }
            )

        answer = self.llm_service.answer_with_context(
            question=question,
            contexts=contexts,
        )

        result = (answer, citations)
        self.cache_service.set(cache_key, result)
        return result