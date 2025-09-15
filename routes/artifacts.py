from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from services.daytona_service import get_persistent_sandbox
from services.job_service import get_job

router = APIRouter()

@router.get("/job/{job_id}/artifacts")
async def list_job_artifacts(job_id: str):
    try:
        if get_job(job_id) is None:
            print(f"Job {job_id} not found for artifacts")
            raise HTTPException(status_code=404, detail="Job not found")
        
        sandbox = get_persistent_sandbox(job_id)
        if sandbox is None:
            print(f"Sandbox not found for job {job_id}")
            raise HTTPException(status_code=404, detail="Sandbox not found or already cleaned up")
        
        ws = get_job(job_id).get("daytona", {}).get("workspace", "/home/daytona/volume/workspace")
        print(f"Listing artifacts for job {job_id} in workspace {ws}")
        
        try:
            listing = sandbox.process.exec("pwd && ls -la", cwd=ws, timeout=30)
            files_plain = sandbox.process.exec("ls -1", cwd=ws, timeout=20).result
            latest = sandbox.process.exec(
                "sh -lc 'ls -1t preprocessed_step*.csv 2>/dev/null | head -1'", cwd=ws, timeout=20
            ).result.strip()
            files_list = [f for f in (files_plain or "").split("\n") if f.strip()]
            
            # Get trained model pickle files
            try:
                model_files = sandbox.process.exec(
                    "ls -1 trained_*.pkl 2>/dev/null || echo ''", cwd=ws, timeout=20
                ).result.strip()
                model_files_list = [f for f in model_files.split("\n") if f.strip()] if model_files else []
                print(f"Found {len(model_files_list)} model files: {model_files_list}")
            except Exception as e:
                print(f"Error listing model files: {e}")
                model_files_list = []
        except Exception as e:
            print(f"Error listing files: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list artifacts: {e}")
        
        # Get job information including trained models
        job_info = get_job(job_id)
        trained_models = job_info.get("result", {}).get("trained_models", []) if job_info else []
        print(f"Job has {len(trained_models)} trained models in result")
        
        return JSONResponse({
            "workspace": ws,
            "listing": listing.result,
            "files": files_list,
            "latest_csv": latest or None,
            "model_files": model_files_list,
            "trained_models": trained_models,
            "model_files_ready": len(model_files_list) > 0,
            "daytona": get_job(job_id).get("daytona", {}),
        })
    except Exception as e:
        print(f"Error in list_job_artifacts for {job_id}: {e}")
        raise
