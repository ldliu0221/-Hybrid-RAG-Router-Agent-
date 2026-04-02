from app.services.embed_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.rerank_service import RerankService
from app.services.bm25_service import BM25Service
from app.services.rewrite_service import RewriteService


class RetrievalService:
    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
    ):
        self.embed_service = EmbeddingService()
        self.vector_service = VectorService()
        self.rerank_service = RerankService()
        self.bm25_service = BM25Service()
        self.rewrite_service = RewriteService()

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.enable_rewrite = enable_rewrite
        self.enable_rerank = enable_rerank

    def retrieve(self, question: str, top_k: int = 8):
        rewritten_query = question
        if self.enable_rewrite:
            rewritten_query = self.rewrite_service.rewrite(question)
            print("Rewrite:", rewritten_query)

        dense_top_k = max(top_k * 2, 10)

        # 1. dense retrieval
        query_vector = self.embed_service.embed_query(rewritten_query)
        dense_hits = self.vector_service.search(
            query_vector=query_vector,
            top_k=dense_top_k
        )

        # 2. sparse retrieval
        all_points = self.vector_service.get_all_points()
        docs = []
        for point in all_points:
            payload = point.payload
            docs.append(
                {
                    "document_id": payload["document_id"],
                    "filename": payload["filename"],
                    "chunk_id": payload["chunk_id"],
                    "text": payload["text"],
                }
            )

        self.bm25_service.build_index(docs)
        sparse_hits = self.bm25_service.search(rewritten_query, top_k=dense_top_k)

        # 3. dense + sparse 融合
        merged = {}

        for hit in dense_hits:
            chunk_id = hit.payload["chunk_id"]
            merged[chunk_id] = {
                "payload": hit.payload,
                "dense_score": float(hit.score),
                "sparse_score": 0.0
            }

        for doc, score in sparse_hits:
            chunk_id = doc["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "payload": doc,
                    "dense_score": 0.0,
                    "sparse_score": float(score)
                }
            else:
                merged[chunk_id]["sparse_score"] = float(score)

        fused_hits = []
        for item in merged.values():
            fused_score = (
                self.dense_weight * item["dense_score"] +
                self.sparse_weight * item["sparse_score"]
            )
            fused_hits.append(
                {
                    "payload": item["payload"],
                    "fused_score": fused_score
                }
            )

        # 4. 去重 + 排序
        seen = set()
        unique_hits = []

        for item in fused_hits:
            cid = item["payload"]["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                unique_hits.append(item)

        unique_hits.sort(key=lambda x: x["fused_score"], reverse=True)

        # 5. 构造候选对象
        class SimpleHit:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        candidate_hits = [
            SimpleHit(item["payload"], item["fused_score"])
            for item in unique_hits[:dense_top_k]
        ]

        # 6. rerank
        if self.enable_rerank:
            reranked_hits = self.rerank_service.rerank_hits(
                question=question,
                hits=candidate_hits
            )
            return reranked_hits[:top_k]

        return candidate_hits[:top_k]