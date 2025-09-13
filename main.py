from agents import Agent, Runner, function_tool
from agents.tool_context import ToolContext
from dotenv import load_dotenv
import asyncio
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import tempfile
import shutil
import json
from datetime import datetime
import base64
import mimetypes
import shlex
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from prompts import orchestrator_prompt, preprocessing_agent_prompt
from daytona import Sandbox
from types import SimpleNamespace
from config import DAYTONA_KEY, OPENAI_KEY, ALLOW_ORIGINS, ALLOW_METHODS, ALLOW_HEADERS
from services.agent_service import create_preprocessor_agent, create_orchestrator_agent
from services.job_service import create_job, update_job_status, get_job, list_jobs
from services.daytona_service import create_volume, get_volume, delete_volume, create_sandbox, get_sandbox, delete_sandbox, persistent_sandboxes, persistent_volumes, get_persistent_sandbox, get_persistent_volume
"""
The agent needed are:
    1- Orchestator
    2- Preprocessing
    3- Master trainer
    4- Evaluator
    5- Tuning
"""
load_dotenv()

app = FastAPI(title="SproutML Training API", version="1.0.0")

# Track a persistent sandbox/volume per job


# In-memory job store (in production, use Redis or database)
executor = ThreadPoolExecutor(max_workers=3)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOW_METHODS,
    allow_headers=ALLOW_HEADERS,
)

async def process_training_job(job_id: str, training_request: str, temp_file_path: str):
    """Create a persistent Daytona sandbox per job, upload dataset, run, then clean up."""
    try:
        # Update job status to processing
        update_job_status(job_id, "processing", datetime.now().isoformat())
        
        print(f"Starting training job {job_id}")
        
        # Create/get a persistent volume for this job and mount it to the sandbox
        volume = get_persistent_volume(job_id)
        if volume is None:
            volume = create_volume(job_id)
        sandbox = get_persistent_sandbox(job_id)
        if sandbox is None:
            sandbox = create_sandbox(job_id)
        # Record infrastructure details for later inspection/download
        update_job_status(job_id, "daytona", {
            "volume_id": getattr(volume, "id", None),
            "sandbox_id": getattr(sandbox, "id", None),
            "workspace": "/home/daytona/volume/workspace",
            "retain": True,
        })
        # Use the mounted volume as the working directory so files persist and are visible
        ws = "/home/daytona/volume/workspace"
        try:
            sandbox.process.exec(f"mkdir -p {ws}", cwd=".", timeout=30)
        except Exception:
            pass
        # Upload the dataset into the volume path and log
        with open(temp_file_path, "rb") as f:
            sandbox.fs.upload_file(f.read(), f"{ws}/{os.path.basename(temp_file_path)}")
        try:
            listing = sandbox.process.exec("pwd && ls -la", cwd=ws, timeout=30)
            print(f"[Daytona] After dataset upload, workspace listing for job {job_id}:\n{listing.result}")
        except Exception as e:
            print(f"[Daytona] Listing failed for job {job_id}: {e}")

        # Create preprocessor agent (stateless tool)
        persistent_preprocessor = create_preprocessor_agent()
        orchestrator = create_orchestrator_agent()
        
        # Create orchestrator with the persistent preprocessor
        job_orchestrator = orchestrator
        
        # Run the orchestrator agent with the training context (include dataset_path so tool can self-heal)
        tool_context = SimpleNamespace(
            job_id=job_id,
            dataset_filename=os.path.basename(temp_file_path),
            dataset_path=temp_file_path,
        )
        result = await Runner.run(job_orchestrator, training_request, context=tool_context, max_turns=50)

        # Handoff-based preprocessing loop: orchestrator explicitly hands off to preprocessing agent for each step
        def _extract_step(json_text: str) -> tuple[str | None, dict[str, Any] | None]:
            try:
                data = json.loads(json_text)
                if isinstance(data, dict) and "output_csv" in data:
                    return data.get("output_csv"), data
            except Exception:
                pass
            try:
                start = json_text.find("{")
                end = json_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(json_text[start : end + 1])
                    if isinstance(data, dict) and "output_csv" in data:
                        return data.get("output_csv"), data
            except Exception:
                return None, None
            return None, None

        current_csv = os.path.basename(temp_file_path)
        step_num = 1
        max_steps = 10
        step_results = []
        
        # Initial step result
        initial_output = str(getattr(result, "final_output", ""))
        step_results.append(f"Initial: {initial_output}")
        
        out_csv, _ = _extract_step(initial_output)
        
        while out_csv and out_csv != current_csv and step_num <= max_steps:
            current_csv = out_csv
            tool_context.dataset_filename = current_csv
            step_num += 1
            
            # Orchestrator explicitly hands off to preprocessing agent for this step
            handoff_request = (
                f"Hand off to preprocessing agent for step {step_num}. "
                f"Current dataset: '{current_csv}'. "
                f"The preprocessing agent should analyze the current dataset and execute the next logical preprocessing step. "
                f"If this is step 1, do initial analysis and planning. If this is a later step, analyze the current data state and continue preprocessing. "
                f"Return the structured JSON result with output_csv for the next iteration."
            )
            
            result = await Runner.run(job_orchestrator, handoff_request, context=tool_context, max_turns=50)
            step_output = str(getattr(result, "final_output", ""))
            step_results.append(f"Step {step_num}: {step_output}")
            
            out_csv, _ = _extract_step(step_output)
            
            if not out_csv:
                print(f"Step {step_num}: No output_csv found. Stopping preprocessing.")
                break

        # Update job with results after loop
        update_job_status(job_id, "completed", datetime.now().isoformat(), {
            "status": "completed",
            "result": {
                "orchestrator_output": result.final_output if hasattr(result, 'final_output') else str(result),
                "final_input_csv": current_csv,
                "step_results": step_results,
                "total_steps": step_num,
            },
            "completed_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        print(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        # Update job with error
        update_job_status(job_id, "failed", datetime.now().isoformat(), {
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.now().isoformat()
        })
        print(f"Training job {job_id} failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up {temp_file_path}: {cleanup_error}")
        
# Default preprocessor agent (for backwards compatibility)
preprocessor_agent = create_preprocessor_agent()
orchestrator = create_orchestrator_agent()

@app.post("/train")
async def train_model(
    file: UploadFile = File(..., description="CSV file containing the dataset"),
    target_column: str = Form(..., description="Name of the target column for training")
):
    """
    Train endpoint that receives a CSV file and target column to start the ML training process.
    Returns immediately with a job ID for status tracking.
    
    Args:
        file: CSV file upload
        target_column: Name of the column to use as target variable
        
    Returns:
        Job ID for tracking training progress
    """
    try:
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp(prefix="sproutml_")
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file to temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create job record
        job_id = create_job(file_name=file.filename, target_column=target_column)
        
        # Prepare training request and pass dataset context for tools
        training_request = f"""
        I have a machine learning training request:
        
        - CSV File Path: {temp_file_path}
        - Original Filename: {file.filename}
        - Target Column: {target_column}
        
        Please orchestrate the complete machine learning pipeline for this dataset.
        The CSV file is saved at the provided path. Use your specialized agents to:
        1. Load and validate the CSV file
        2. Handle data preprocessing 
        3. Train models
        4. Evaluate performance
        5. Tune hyperparameters
        """
        
        # Start async processing (fire and forget) using persistent sandbox + volume
        asyncio.create_task(process_training_job(job_id, training_request, temp_file_path))
        
        return {
            "status": "success",
            "message": "Training job queued successfully",
            "job_id": job_id,
            "filename": file.filename,
            "target_column": target_column
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    import os
    return {
        "message": "SproutML Training API is running", 
        "status": "healthy",
        "port": os.getenv("PORT", "8000"),
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Additional health check endpoint"""
    return {"status": "ok", "service": "sproutml-api"}

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a training job"""
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return get_job(job_id)

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {"jobs": list_jobs()}

@app.get("/job/{job_id}/artifacts")
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

@app.get("/job/{job_id}/artifact/{filename:path}")
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

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable, default to 8000
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)