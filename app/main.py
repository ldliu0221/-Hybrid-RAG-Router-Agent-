from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.api.routes_eval import router as eval_router

setup_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="基于 RAG + 阿里云百炼 的企业知识问答系统"
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(eval_router)