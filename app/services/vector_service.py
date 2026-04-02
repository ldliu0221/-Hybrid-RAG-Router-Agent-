import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
QDRANT_LOCAL_PATH = ROOT_DIR / "data" / "qdrant_storage"


class VectorService:
    def __init__(self):
        QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(QDRANT_LOCAL_PATH))
        self.collection_name = settings.qdrant_collection

    def has_collection(self) -> bool:
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        return self.collection_name in collection_names

    def ensure_collection(self, vector_size: int) -> None:
        if not self.has_collection():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
            )

    def upsert_chunks(
        self,
        chunks: list[str],
        vectors: list[list[float]],
        filename: str,
        document_id: str
    ) -> None:
        points = []

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_id": f"{document_id}_{idx}",
                        "text": chunk,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: list[float], top_k: int = 8):
        if not self.has_collection():
            raise ValueError(
                f"Collection {self.collection_name} not found. "
                f"请先通过 /ingest 导入文档，或确认本地库路径是否正确: {QDRANT_LOCAL_PATH}"
            )

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    def get_all_points(self):
        if not self.has_collection():
            raise ValueError(
                f"Collection {self.collection_name} not found. "
                f"请先通过 /ingest 导入文档，或确认本地库路径是否正确: {QDRANT_LOCAL_PATH}"
            )

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=1000
        )
        return records