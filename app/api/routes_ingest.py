import os
import uuid
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestResponse
from app.services.parser_service import ParserService
from app.services.chunk_service import ChunkService
from app.services.embed_service import EmbeddingService
from app.services.vector_service import VectorService
from app.core.config import settings

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    try:
        embed_service = EmbeddingService()
        vector_service = VectorService()

        os.makedirs("data/raw", exist_ok=True)

        document_id = str(uuid.uuid4())
        save_path = f"data/raw/{document_id}_{file.filename}"

        with open(save_path, "wb") as f:
            f.write(await file.read())

        text = ParserService.parse_file(save_path)

        chunks = ChunkService.chunk_text(
            text=text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="切块结果为空")

        vectors = embed_service.embed_texts(chunks)
        vector_service.ensure_collection(vector_size=len(vectors[0]))
        vector_service.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
            filename=file.filename,
            document_id=document_id,
        )

        return IngestResponse(
            document_id=document_id,
            filename=file.filename,
            chunks=len(chunks),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))