from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, function_tool
from agents.tool_context import ToolContext
from agents.exceptions import InputGuardrailTripwireTriggered
from dotenv import load_dotenv
import asyncio
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import tempfile
import shutil
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from prompts import orchestrator_prompt, preprocessing_agent_prompt
from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams
"""
The agent needed are:
    1- Orchestator
    2- Preprocessing
    3- Master trainer
    4- Evaluator
    5- Tuning
"""
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="SproutML Training API", version="1.0.0")
DAYTONA_KEY = os.getenv("DAYTONA_API_KEY")
daytona = Daytona(DaytonaConfig(api_key=DAYTONA_KEY))

# In-memory job store (in production, use Redis or database)
job_store: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=3)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sproutml.com",
        "https://www.sproutml.com",
        "https://sproutml.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

 

async def process_training_job(job_id: str, training_request: str, temp_file_path: str):
    """Process training job asynchronously"""
    try:
        # Update job status to processing
        job_store[job_id]["status"] = "processing"
        job_store[job_id]["updated_at"] = datetime.now().isoformat()
        
        print(f"Starting training job {job_id}")
        
        # Create preprocessor agent (stateless tool)
        persistent_preprocessor = create_preprocessor_agent()
        
        # Create orchestrator with the persistent preprocessor
        job_orchestrator = Agent(
            name="Orchestrator Agent",
            instructions=orchestrator_prompt,
            handoffs=[persistent_preprocessor],
        )
        
        # Run the orchestrator agent with the training context
        result = await Runner.run(job_orchestrator, training_request)
        
        # Update job with results
        job_store[job_id].update({
            "status": "completed",
            "result": {
                "orchestrator_output": result.final_output if hasattr(result, 'final_output') else str(result)
            },
            "completed_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        
        print(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        # Update job with error
        job_store[job_id].update({
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
        
        # No container cleanup required

async def test_orchestrator():
    """Test function for the orchestrator - can be used for debugging"""
    try:
        result = await Runner.run(orchestrator, "who was the first president of the united states?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

@function_tool
def daytona_run_script(
    ctx: ToolContext,
    script_name: str,
    script: str,
    requirements: Optional[str] = None,
    dataset_destination: Optional[str] = None,
    timeout: int = 300
) -> str:
    """Upload and run a Python script in a Daytona sandbox.

    - script_name: filename to save (e.g., 'preprocess.py')
    - script: full Python source code
    - requirements: optional contents for requirements.txt (installed if provided)
    - dataset_destination: filename to upload the current dataset as (defaults to context filename)
    - timeout: seconds to allow the process to run

    Returns combined stdout/stderr and exit_code as text.
    """
    sandbox = daytona.create(CreateSandboxFromSnapshotParams(language="python"))
    try:
        # Ensure workspace exists implicitly by using paths under it
        ws = "workspace"
        # Upload script
        sandbox.fs.upload_file(script.encode("utf-8"), f"{ws}/{script_name}")

        # Upload dataset from server temp path if present
        ds_path = None
        ds_name = dataset_destination
        try:
            ds_path = getattr(ctx.context, "dataset_path", None)
        except Exception:
            ds_path = None
        if ds_name is None:
            try:
                ds_name = getattr(ctx.context, "dataset_filename", None)
            except Exception:
                ds_name = None
        if ds_path and ds_name:
            with open(ds_path, "rb") as f:
                sandbox.fs.upload_file(f.read(), f"{ws}/{ds_name}")

        # Upload requirements and install if provided
        if requirements and requirements.strip():
            sandbox.fs.upload_file(requirements.encode("utf-8"), f"{ws}/requirements.txt")
            sandbox.process.exec("pip install -r requirements.txt", cwd=ws, timeout=180)

        # Run script
        result = sandbox.process.exec(f"python {script_name}", cwd=ws, timeout=timeout)
        return f"exit_code={result.exit_code}\n{result.result}"
    finally:
        try:
            sandbox.delete()
        except Exception:
            pass

def create_preprocessor_agent():
    """Create a preprocessing agent with Daytona execution tool."""
    hint = (
        "You must emit two files per step: '\n"
        "- preprocess.py: Python code for ONLY the current step, reading from the dataset filename in your CWD (provided below) and writing outputs (e.g., preprocessed_stepN.csv).\n"
        "- requirements.txt: only the extra pip dependencies needed for this step (if any).\n\n"
        "Then call the 'daytona_run_script' tool with arguments: script_name='preprocess.py', script=<file contents>, requirements=<requirements.txt contents or ''>, dataset_destination=<dataset filename>.\n\n"
        "Dataset filename to use in code is provided in context; assume it's in your working directory. Do not use absolute paths.\n\n"
    )
    return Agent(
        name="Preprocessing Agent",
        handoff_description="Agent specializing in preprocessing datasets",
        instructions=hint + preprocessing_agent_prompt,
        tools=[daytona_run_script],
    )

# Default preprocessor agent (for backwards compatibility)
preprocessor_agent = create_preprocessor_agent()

master_trainer_agent = Agent(
    name="Master training Agent",
    instructions="You determine which agent to use based on the user's homework question",
)

evaluator_agent = Agent(
    name="Evaluator Agent",
    instructions="You determine which agent to use based on the user's homework question",

)

tuning_agent = Agent(
    name="Tuning Agent",
    instructions="You determine which agent to use based on the user's homework question",
)

orchestrator = Agent(
    name="Orchestrator Agent",
    instructions=orchestrator_prompt,
    handoffs=[preprocessor_agent],
)

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
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp(prefix="sproutml_")
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file to temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create job record
        job_store[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "target_column": target_column,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
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
        
        # Start async processing (fire and forget)
        # Attach dataset info to context so daytona tool can upload it
        tool_context = type("ToolCtx", (), {
            "dataset_path": temp_file_path,
            "dataset_filename": file.filename,
        })()
        asyncio.create_task(Runner.run(orchestrator, training_request, context=tool_context))
        
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
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_store[job_id]

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {"jobs": list(job_store.values())}

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable, default to 8000
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)