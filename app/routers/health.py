"""Health check router."""

from fastapi import APIRouter
from fastapi import Request

from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return HealthResponse(
            status="loading",
            models_loaded={},
        )
    return HealthResponse(
        status="ok",
        models_loaded=pipeline.models_loaded,
    )
