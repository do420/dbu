from fastapi import APIRouter
from api.api_v1.endpoints import agents, mini_services, api_keys, rag_documents

api_router = APIRouter()
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(rag_documents.router, prefix="/agents", tags=["rag"])
api_router.include_router(mini_services.router, prefix="/mini-services", tags=["mini-services"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"])