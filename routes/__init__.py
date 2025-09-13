from .training import router as training_router
from .jobs import router as jobs_router  
from .artifacts import router as artifacts_router
from .health import router as health_router

__all__ = ["training_router", "jobs_router", "artifacts_router", "health_router"]
