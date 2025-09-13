from fastapi import APIRouter
import pandas as pd

router = APIRouter(title="SproutML Health API", version="1.0.0")

@router.get("/")
async def root():
    import os
    return {
        "message": "SproutML API is running",
        "status": "healthy", 
        "port": os.getenv("PORT", "8000"),
        "timestamp": pd.Timestamp.now().isoformat()
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    import os
    return {
        "message": "SproutML Training API is running", 
        "status": "healthy",
        "port": os.getenv("PORT", "8000"),
        "timestamp": pd.Timestamp.now().isoformat()
    }