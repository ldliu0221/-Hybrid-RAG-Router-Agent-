import traceback
from fastapi import APIRouter, HTTPException

from app.models.schemas import EvalRequest, EvalResponse
from app.services.eval_service import EvalService

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("", response_model=EvalResponse)
async def run_eval(req: EvalRequest):
    try:
        eval_service = EvalService()

        result = eval_service.run_single_eval(
            question=req.question,
            ground_truth=req.ground_truth,
            use_agent=req.use_agent,
            top_k=req.top_k,
        )

        return EvalResponse(**result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))