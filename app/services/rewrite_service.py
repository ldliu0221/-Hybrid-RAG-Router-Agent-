from app.services.llm_service import LLMService


class RewriteService:
    def __init__(self):
        self.llm = LLMService()

    def rewrite(self, query: str) -> str:
        rewritten = self.llm.rewrite_query(query)
        return rewritten.strip()