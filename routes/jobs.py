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

router = APIRouter(title="SproutML Jobs API", version="1.0.0")

@router.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {"jobs": list_jobs()}

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a training job"""
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return get_job(job_id)

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
