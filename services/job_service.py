from typing import Dict, Any
import uuid
from datetime import datetime
job_store: Dict[str, Any] = {}

def create_job( file_name: str, target_column: str):
    """
    Creates a job in the job store
    """
    created_at = datetime.now().isoformat()
    
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": file_name,
            "target_column": target_column,
            "created_at": created_at,
            "updated_at": created_at
        }
    return job_id

def update_job_status(job_id: str, status: str, updated_at: str, **kwargs):
    """
    Updates the status of a job in the job store
    """
    try:
        job_store[job_id]["status"] = status
        job_store[job_id]["updated_at"] = updated_at
        job_store[job_id].update(kwargs)
    except Exception:
        return None

def get_job(job_id: str):
    """
    Gets a job from the job store
    """
    try:
        return job_store[job_id]
    except Exception:
        return None

def list_jobs():
    """
    Gets all jobs from the job store
    """
    try:
        return list(job_store.values())
    except Exception:
        return None