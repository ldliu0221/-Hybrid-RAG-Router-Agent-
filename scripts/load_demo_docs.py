import os
import uuid

from app.services.parser_service import ParserService
from app.services.chunk_service import ChunkService
from app.services.embed_service import EmbeddingService
from app.services.vector_service import VectorService
from app.core.config import settings


def load_docs_from_folder(folder: str = "data/raw"):
    embed_service = EmbeddingService()
    vector_service = VectorService()

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue

        document_id = str(uuid.uuid4())
        text = ParserService.parse_file(path)
        chunks = ChunkService.chunk_text(
            text=text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )

        vectors = embed_service.embed_texts(chunks)
        vector_service.ensure_collection(vector_size=len(vectors[0]))
        vector_service.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
            filename=filename,
            document_id=document_id
        )
        print(f"Loaded: {filename}, chunks={len(chunks)}")


if __name__ == "__main__":
    load_docs_from_folder()