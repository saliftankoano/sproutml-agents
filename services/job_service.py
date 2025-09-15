from typing import Dict, Any
import uuid
import json
import os
from datetime import datetime

# Use persistent file storage for Railway deployment
JOB_STORE_FILE = "/tmp/job_store.json"

def _load_job_store() -> Dict[str, Any]:
    """Load job store from persistent file"""
    try:
        if os.path.exists(JOB_STORE_FILE):
            with open(JOB_STORE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading job store: {e}")
    return {}

def _save_job_store(job_store: Dict[str, Any]):
    """Save job store to persistent file"""
    try:
        with open(JOB_STORE_FILE, 'w') as f:
            json.dump(job_store, f, indent=2)
    except Exception as e:
        print(f"Error saving job store: {e}")

# Initialize job store from file
job_store: Dict[str, Any] = _load_job_store()

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
    _save_job_store(job_store)
    return job_id

def update_job_status(job_id: str, status: str, updated_at: str, **kwargs):
    """
    Updates the status of a job in the job store
    """
    try:
        job_store[job_id]["status"] = status
        job_store[job_id]["updated_at"] = updated_at
        job_store[job_id].update(kwargs)
        _save_job_store(job_store)
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

def recover_job_from_sandbox(job_id: str):
    """
    Attempt to recover job data from persistent sandbox
    """
    try:
        from services.daytona_service import get_persistent_sandbox
        
        sandbox = get_persistent_sandbox(job_id)
        if sandbox is None:
            print(f"No sandbox found for job {job_id}")
            return None
        
        ws = "/home/daytona/volume/workspace"
        
        # Check if training artifacts exist
        try:
            # List all files in workspace
            files_result = sandbox.process.exec("ls -1", cwd=ws, timeout=20)
            files = [f.strip() for f in files_result.result.split('\n') if f.strip()]
            
            # Check for trained model files
            model_files = [f for f in files if f.startswith('trained_') and f.endswith('.pkl')]
            
            # Check for preprocessed files
            preprocessed_files = [f for f in files if f.startswith('preprocessed_step') and f.endswith('.csv')]
            
            if model_files or preprocessed_files:
                print(f"Found artifacts for job {job_id}: {len(model_files)} models, {len(preprocessed_files)} preprocessed files")
                
                # Create a recovered job entry
                recovered_job = {
                    "job_id": job_id,
                    "status": "completed",
                    "filename": "recovered",
                    "target_column": "unknown",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "recovered": True,
                    "result": {
                        "trained_models": [{"model_name": f.replace('trained_', '').replace('.pkl', ''), "pickle_file": f} for f in model_files],
                        "model_files_ready": len(model_files) > 0,
                        "preprocessed_files": preprocessed_files
                    }
                }
                
                # Add to job store
                job_store[job_id] = recovered_job
                _save_job_store(job_store)
                
                return recovered_job
            else:
                print(f"No training artifacts found for job {job_id}")
                return None
                
        except Exception as e:
            print(f"Error checking sandbox for job {job_id}: {e}")
            return None
            
    except Exception as e:
        print(f"Error recovering job {job_id}: {e}")
        return None