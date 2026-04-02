from dataclasses import dataclass


@dataclass
class DocumentChunk:
    document_id: str
    filename: str
    chunk_id: str
    text: str