from sentence_transformers import CrossEncoder
from app.core.config import settings


class RerankService:
    def __init__(self):
        self.model = CrossEncoder(settings.reranker_model)

    def rerank_hits(self, question: str, hits: list):
        """
        对 Qdrant 返回的 hits 进行重排。
        返回值仍然是 hit 对象列表，只是顺序被重新排序。
        """
        if not hits:
            return []

        pairs = []
        for hit in hits:
            text = hit.payload["text"]
            pairs.append([question, text])

        scores = self.model.predict(pairs)

        scored_hits = list(zip(hits, scores))
        scored_hits.sort(key=lambda x: x[1], reverse=True)

        reranked_hits = [hit for hit, _ in scored_hits]
        return reranked_hits