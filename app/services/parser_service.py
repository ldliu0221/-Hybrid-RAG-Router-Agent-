from pathlib import Path
import fitz

from app.core.exceptions import UnsupportedFileTypeError, EmptyDocumentError


class ParserService:
    @staticmethod
    def parse_file(file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = ParserService._parse_pdf(path)
        elif suffix in [".md", ".txt"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")

        text = text.strip()
        if not text:
            raise EmptyDocumentError("文档内容为空，无法入库")

        return text

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n".join(texts)