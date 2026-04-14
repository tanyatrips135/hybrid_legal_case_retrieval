"""
Analyze router
==============
POST /api/analyze  — runs the full pipeline
GET  /             — serves the HTML frontend
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.models.schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(
            content=f.read(),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, body: AnalyzeRequest) -> AnalyzeResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet.")

    try:
        t0 = time.perf_counter()
        result = pipeline.run(
            case_description=body.case_description,
            top_k=body.top_k,
        )
        logger.info(
            "analyze completed in %.0f ms", (time.perf_counter() - t0) * 1000
        )
        return result
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
