"""
Indian Legal RAG System
FastAPI entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, health
from app.services.legal_pipeline import LegalPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup, release at shutdown."""
    logger.info("Loading Legal RAG pipeline...")
    pipeline = LegalPipeline()
    pipeline.load()
    app.state.pipeline = pipeline
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down pipeline.")
    del app.state.pipeline


app = FastAPI(
    title="Indian Legal RAG System",
    description="Legal issue extraction, NER, FAISS+BM25 retrieval, and T5 summarization pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(health.router)
app.include_router(analyze.router)
