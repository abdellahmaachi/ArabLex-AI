"""Point d'entrée FastAPI : uvicorn backend.app.main:app."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from backend.app.routers.classification import router as classification_router
from backend.app.routers.lexical import router as lexical_router
from backend.app.routers.ner import router as ner_router
from backend.app.routers.rag import close_rag_service
from backend.app.routers.rag import router as rag_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await run_in_threadpool(close_rag_service)


app = FastAPI(title="Arabic Legal Text Mining", lifespan=lifespan)
app.include_router(classification_router)
app.include_router(lexical_router)
app.include_router(ner_router)
app.include_router(rag_router)
