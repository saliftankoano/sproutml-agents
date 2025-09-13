from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from services.daytona_service import get_persistent_sandbox
from services.job_service import get_job

router = APIRouter()

@router.get("/job/{job_id}/artifacts")
async def list_job_artifacts(job_id: str):
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    sandbox = get_persistent_sandbox(job_id)
    if sandbox is None:
        raise HTTPException(status_code=404, detail="Sandbox not found or already cleaned up")
    ws = get_job(job_id).get("daytona", {}).get("workspace", "/home/daytona/volume/workspace")
    try:
        listing = sandbox.process.exec("pwd && ls -la", cwd=ws, timeout=30)
        files_plain = sandbox.process.exec("ls -1", cwd=ws, timeout=20).result
        latest = sandbox.process.exec(
            "sh -lc 'ls -1t preprocessed_step*.csv 2>/dev/null | head -1'", cwd=ws, timeout=20
        ).result.strip()
        files_list = [f for f in (files_plain or "").split("\n") if f.strip()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list artifacts: {e}")
    return JSONResponse({
        "workspace": ws,
        "listing": listing.result,
        "files": files_list,
        "latest_csv": latest or None,
        "daytona": get_job(job_id).get("daytona", {}),
    })
