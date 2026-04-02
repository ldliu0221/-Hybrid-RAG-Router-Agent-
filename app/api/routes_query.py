import traceback
from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, Citation
from app.services.answer_service import AnswerService
from app.services.router_service import RouterService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        answer_service = AnswerService()
        router_service = RouterService()

        # use_agent=False：固定走 RAG
        if not req.use_agent:
            route = "rag"
        else:
            route = router_service.route(req.question)

        if route == "rag":
            answer, citations = answer_service.answer(
                question=req.question,
                top_k=req.top_k,
            )
        else:
            answer = answer_service.llm_service.simple_chat(req.question)
            citations = []

        return QueryResponse(
            answer=answer,
            citations=[Citation(**c) for c in citations],
            route=route,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))