import jieba
from rank_bm25 import BM25Okapi


class BM25Service:
    def __init__(self):
        self.corpus = []
        self.metadata = []
        self.bm25 = None

    def build_index(self, docs: list[dict]):
        self.metadata = docs
        self.corpus = [doc["text"] for doc in docs]

        tokenized_corpus = [list(jieba.cut(text)) for text in self.corpus]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5):
        if not self.bm25:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        scored_docs = list(zip(self.metadata, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]