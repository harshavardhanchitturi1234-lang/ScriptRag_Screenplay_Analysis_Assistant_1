"""
API v1 Router Aggregator.
"""

from fastapi import APIRouter
from app.api.v1.endpoints.theory import router as theory_router
from app.api.v1.endpoints.screenplays import router as screenplays_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(theory_router)
api_v1_router.include_router(screenplays_router)

