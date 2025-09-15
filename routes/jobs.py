from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import base64
import mimetypes
import io
import os
import shlex
from services.daytona_service import get_persistent_sandbox
from services.job_service import get_job
from services.job_service import list_jobs

router = APIRouter()

@router.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {"jobs": list_jobs()}

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a training job"""
    try:
        job_data = get_job(job_id)
        if job_data is None:
            print(f"Job {job_id} not found in job store, attempting recovery...")
            # Try to recover from sandbox
            from services.job_service import recover_job_from_sandbox
            job_data = recover_job_from_sandbox(job_id)
            if job_data is None:
                print(f"Job {job_id} not found in job store and recovery failed")
                raise HTTPException(status_code=404, detail="Job not found")
            else:
                print(f"Successfully recovered job {job_id}")
        
        print(f"Returning job status for {job_id}: {job_data.get('status', 'unknown')}")
        return job_data
    except Exception as e:
        print(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {e}")

@router.get("/job/{job_id}/artifact/{filename:path}")
async def download_job_artifact(job_id: str, filename: str):
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    sandbox = get_persistent_sandbox(job_id)
    if sandbox is None:
        raise HTTPException(status_code=404, detail="Sandbox not found or already cleaned up")
    ws = get_job(job_id).get("daytona", {}).get("workspace", "/home/daytona/volume/workspace")
    remote_path = f"{ws}/{filename}"
    try:
        content_bytes = None
        if hasattr(sandbox, "fs") and hasattr(sandbox.fs, "download_file"):
            content_bytes = sandbox.fs.download_file(remote_path)
        else:
            encoded = sandbox.process.exec(
                f"sh -lc 'base64 -w 0 {shlex.quote(remote_path)}'", cwd="/", timeout=120
            ).result.strip()
            content_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found or unreadable: {e}")
    media_type, _ = mimetypes.guess_type(filename)
    media_type = media_type or "application/octet-stream"
    return StreamingResponse(io.BytesIO(content_bytes), media_type=media_type, headers={
        "Content-Disposition": f"attachment; filename={os.path.basename(filename)}"
    })

@router.get("/job/{job_id}/models")
async def list_trained_models(job_id: str):
    """List all trained model files for a job"""
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = get_job(job_id)
    trained_models = job_info.get("result", {}).get("trained_models", [])
    
    return {
        "job_id": job_id,
        "trained_models": trained_models,
        "model_files_ready": len(trained_models) > 0,
        "total_models": len(trained_models)
    }

@router.get("/job/{job_id}/model/{model_name}")
async def download_trained_model(job_id: str, model_name: str):
    """Download a specific trained model pickle file"""
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Construct the pickle filename
    pickle_filename = f"trained_{model_name.lower()}.pkl"
    
    # Use the existing download endpoint
    return await download_job_artifact(job_id, pickle_filename)

@router.post("/job/{job_id}/recover")
async def recover_job(job_id: str):
    """Manually recover a job from its sandbox"""
    try:
        from services.job_service import recover_job_from_sandbox
        recovered_job = recover_job_from_sandbox(job_id)
        
        if recovered_job:
            return {
                "success": True,
                "message": f"Job {job_id} recovered successfully",
                "job": recovered_job
            }
        else:
            raise HTTPException(status_code=404, detail="No artifacts found for recovery")
    except Exception as e:
        print(f"Error recovering job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Recovery failed: {e}")
