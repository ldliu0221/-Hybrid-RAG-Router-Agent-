from app.services.chunk_service import ChunkService


def test_chunk_text():
    text = "a" * 1200
    chunks = ChunkService.chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)