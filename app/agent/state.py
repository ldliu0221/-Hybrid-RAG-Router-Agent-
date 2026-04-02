from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    question: str
    rewritten_query: str
    retrieved_docs: List[Dict[str, Any]]
    answer: str