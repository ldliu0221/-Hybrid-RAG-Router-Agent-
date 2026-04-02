from typing import List


class ChunkService:
    @staticmethod
    def clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        text = ChunkService.clean_text(text)
        chunks = []

        start = 0
        n = len(text)

        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == n:
                break

            start = max(0, end - overlap)

        return chunks