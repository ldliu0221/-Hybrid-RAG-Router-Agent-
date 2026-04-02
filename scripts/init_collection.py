from app.services.embed_service import EmbeddingService
from app.services.vector_service import VectorService


if __name__ == "__main__":
    embed_service = EmbeddingService()
    vector_service = VectorService()

    sample_vector = embed_service.embed_query("初始化集合")
    vector_service.ensure_collection(vector_size=len(sample_vector))
    print("Qdrant collection initialized.")