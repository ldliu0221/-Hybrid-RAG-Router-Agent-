from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    use_agent: bool = False
    top_k: int = 8


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    text: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    route: str


class HealthResponse(BaseModel):
    status: str
    app_name: str


class EvalRequest(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)
    use_agent: bool = True
    top_k: int = 5


class EvalResponse(BaseModel):
    question: str
    ground_truth: str
    prediction: str
    score: float
    reason: str
    route: str