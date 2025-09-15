from agents import Agent, function_tool
from prompts import (
    preprocessing_agent_prompt, 
    orchestrator_prompt, 
    master_training_agent_prompt,
    get_sub_training_agent_prompt,
    evaluator_agent_prompt
)
from services.daytona_service import (
    persistent_sandboxes,
    persistent_volumes,
    get_persistent_sandbox,
    create_sandbox,
    create_ephemeral_sandbox,
    get_ephemeral_sandbox,
)
from agents.tool_context import ToolContext
from daytona import Sandbox
from typing import Optional
import json
import asyncio

def create_preprocessor_agent():
    """Create a preprocessing agent with Daytona execution tool."""
    return Agent(
        name="Preprocessing Agent",
        handoff_description="Agent specializing in preprocessing datasets",
        instructions=preprocessing_agent_prompt,
        tools=[daytona_agent_tool],
    )

def _get_raw_context(ctx: ToolContext | object):
    """Return the raw context object whether ctx is ToolContext or already a context object."""
    return getattr(ctx, "context", ctx)


def _daytona_run_script_impl(
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
    raw_ctx = _get_raw_context(ctx)
    job_id = getattr(raw_ctx, "job_id", None)
    sandbox: Sandbox | None = None
    # Optional ephemeral key and resource hints
    eph_key = getattr(raw_ctx, "sandbox_key", None)
    eph_cpu = getattr(raw_ctx, "sandbox_cpu", None)
    eph_mem = getattr(raw_ctx, "sandbox_memory", None)
    if eph_key:
        sandbox = get_ephemeral_sandbox(job_id, eph_key) or create_ephemeral_sandbox(
            job_id, eph_key, cpu=eph_cpu or 2, memory=eph_mem or 4
        )
    elif job_id and job_id in persistent_sandboxes:
        sandbox = persistent_sandboxes[job_id]
    else:
        # Create ad-hoc sandbox mounted to the job's volume if present
        if job_id and job_id in persistent_volumes:
            sandbox = get_persistent_sandbox(job_id)
            if sandbox is None:
                sandbox = create_sandbox(job_id)
            persistent_sandboxes[job_id] = sandbox
        else:
            sandbox = get_persistent_sandbox(job_id)
            if sandbox is None:
                sandbox = create_sandbox(job_id)
            persistent_sandboxes[job_id] = sandbox
    
    # Ensure sandbox is available
    if sandbox is None:
        return json.dumps({
            "exit_code": 1,
            "output": "Error: Failed to create or retrieve sandbox",
            "workspace": "",
            "files": [],
            "latest_csv": None,
        })
    try:
        # Ensure workspace exists and is usable
        # Workspace on the mounted volume
        ws = "/home/daytona/volume/workspace"
        try:
            sandbox.process.exec(f"mkdir -p {ws}", cwd=".", timeout=30)
        except Exception:
            pass

        # If using ephemeral sandbox, try to sync shared artifacts from the persistent sandbox
        if eph_key and job_id and job_id in persistent_sandboxes:
            try:
                src_sb: Sandbox = persistent_sandboxes[job_id]
                # Files we need for training
                needed = [
                    "X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "encoded_meta.json",
                ]
                for fname in needed:
                    try:
                        content = src_sb.fs.download_file(f"{ws}/{fname}")
                        sandbox.fs.upload_file(content, f"{ws}/{fname}")
                    except Exception:
                        continue
                # Also sync latest preprocessed CSVs for fallback
                try:
                    csv_list = src_sb.process.exec("sh -lc 'ls -1 preprocessed_step*.csv 2>/dev/null'", cwd=ws, timeout=15).result
                    for line in (csv_list or "").splitlines():
                        name = line.strip()
                        if not name:
                            continue
                        try:
                            content = src_sb.fs.download_file(f"{ws}/{name}")
                            sandbox.fs.upload_file(content, f"{ws}/{name}")
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
        # Ensure base ML dependencies are present (best-effort)
        try:
            sandbox.process.exec("pip install -q numpy pandas scikit-learn xgboost", cwd=ws, timeout=420)
        except Exception:
            pass

        # Upload script
        sandbox.fs.upload_file(script.encode("utf-8"), f"{ws}/{script_name}")

        # Upload dataset only if not already present and we have path on first call
        ds_name = dataset_destination or getattr(raw_ctx, "dataset_filename", None)
        ds_path = getattr(raw_ctx, "dataset_path", None)
        if ds_name:
            # Check if file exists already in persistent sandbox
            try:
                sandbox.process.exec(f"test -f {ds_name} || test -f {ws}/{ds_name} || false", cwd=ws, timeout=10)
                already = True
            except Exception:
                already = False
            if not already and ds_path:
                with open(ds_path, "rb") as f:
                    sandbox.fs.upload_file(f.read(), f"{ws}/{ds_name}")

        # Upload requirements and install if provided
        if requirements and requirements.strip():
            sandbox.fs.upload_file(requirements.encode("utf-8"), f"{ws}/requirements.txt")
            sandbox.process.exec("pip install -r requirements.txt", cwd=ws, timeout=180)

        # Diagnostics: ensure python available and list directory
        try:
            sandbox.process.exec("python --version || python3 --version", cwd=ws, timeout=30)
            sandbox.process.exec("pwd && ls -la", cwd=ws, timeout=30)
        except Exception:
            pass

        # Run script (fallback to python3 if needed)
        try:
            result = sandbox.process.exec(f"python {script_name}", cwd=ws, timeout=timeout)
        except Exception:
            result = sandbox.process.exec(f"python3 {script_name}", cwd=ws, timeout=timeout)

        # If using ephemeral sandbox, sync trained model files (and other artifacts) back to persistent sandbox
        if eph_key and job_id and job_id in persistent_sandboxes:
            try:
                src_sb: Sandbox = persistent_sandboxes[job_id]
                # Sync trained model files back to persistent sandbox (search recursively up to depth 3)
                try:
                    find_cmd = "sh -lc 'find . -maxdepth 3 -type f -name \"trained_*.pkl\" -printf \"%P\\n\" 2>/dev/null'"
                    model_files = sandbox.process.exec(find_cmd, cwd=ws, timeout=30).result.strip()
                    if model_files:
                        for relpath in model_files.split("\n"):
                            relpath = relpath.strip()
                            if not relpath:
                                continue
                            try:
                                content = sandbox.fs.download_file(f"{ws}/{relpath}")
                                # upload to same relative path to keep structure
                                src_sb.fs.upload_file(content, f"{ws}/{relpath}")
                                print(f"Synced trained model {relpath} back to persistent sandbox")
                            except Exception as e:
                                print(f"Failed to sync {relpath}: {e}")
                except Exception as e:
                    print(f"Failed to list model files for sync: {e}")

                # Best-effort: also sync evaluator outputs, logs, and plots (common extensions)
                try:
                    other_cmd = "sh -lc 'find . -maxdepth 2 -type f \\(" \
                               "-name \"*.txt\" -o -name \"*.json\" -o -name \"*.png\" -o -name \"*.csv\" \\) " \
                               "-printf \"%P\\n\" 2>/dev/null'"
                    other_files = sandbox.process.exec(other_cmd, cwd=ws, timeout=30).result.strip()
                    for relpath in (other_files or "").split("\n"):
                        relpath = relpath.strip()
                        if not relpath:
                            continue
                        try:
                            content = sandbox.fs.download_file(f"{ws}/{relpath}")
                            src_sb.fs.upload_file(content, f"{ws}/{relpath}")
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception as e:
                print(f"Failed to sync model files back to persistent sandbox: {e}")

        # Summarize workspace outputs for the agent to advance steps deterministically
        latest_csv = None
        try:
            lst = sandbox.process.exec("ls -1", cwd=ws, timeout=20).result
            latest = sandbox.process.exec("sh -lc 'ls -1t preprocessed_step*.csv 2>/dev/null | head -1'", cwd=ws, timeout=20).result.strip()
            latest_csv = latest if latest else None
        except Exception:
            lst = ""

        payload = {
            "exit_code": result.exit_code,
            "output": result.result,
            "workspace": ws,
            "files": lst.splitlines() if lst else [],
            "latest_csv": latest_csv,
        }
        import json as _json
        return _json.dumps(payload)
    finally:
        # Do not delete the persistent sandbox; retain for multi-step run & user inspection
        if not (job_id and job_id in persistent_sandboxes):
            # Only clean up ad-hoc sandboxes created for missing job mapping
            try:
                sandbox.delete()
            except Exception:
                pass

# Create the function tool for agents
daytona_agent_tool = function_tool(_daytona_run_script_impl)

# Create a regular function that can be called directly from other services
def daytona_direct(
    ctx: ToolContext,
    script_name: str,
    script: str,
    requirements: Optional[str] = None,
    dataset_destination: Optional[str] = None,
    timeout: int = 300
) -> str:
    """Direct callable version of daytona_run_script for use in services."""
    return _daytona_run_script_impl(ctx, script_name, script, requirements, dataset_destination, timeout)

@function_tool
def training_agent_tool(
    ctx: ToolContext,
    dataset_filename: str,
    target_column: str,
    max_iterations: int = 3
) -> str:
    """Run the complete training pipeline with Master Training Agent orchestration.
    
    - dataset_filename: Name of the preprocessed dataset file
    - target_column: Name of the target column for training
    - max_iterations: Maximum number of improvement iterations (default: 3)
    
    Returns comprehensive training results and model artifacts.
    """
    try:
        # Get job context
        raw_ctx = _get_raw_context(ctx)
        job_id = getattr(raw_ctx, "job_id", None)
        if not job_id:
            return json.dumps({
                "status": "error",
                "message": "No job_id found in context"
            })
        
        # Get sandbox for this job
        sandbox = persistent_sandboxes.get(job_id)
        if not sandbox:
            return json.dumps({
                "status": "error", 
                "message": "No sandbox found for job"
            })
        
        # Determine workspace path
        try:
            sandbox.process.exec("test -d /home/daytona/volume", cwd=".", timeout=10)
            workspace = "/home/daytona/volume/workspace"
        except Exception:
            workspace = "/home/daytona/workspace"
        
        # Construct full dataset path
        dataset_path = f"{workspace}/{dataset_filename}"

        # Prefer the latest preprocessed CSV if the provided file is missing or not a preprocessed file
        try:
            # Check if provided file exists
            exists_check = sandbox.process.exec(
                f"sh -lc 'test -f {dataset_filename} && echo EXIST || echo MISSING'",
                cwd=workspace,
                timeout=10,
            ).result.strip()
            needs_discovery = exists_check != "EXIST" or not dataset_filename.startswith("preprocessed_")
            if needs_discovery:
                # Prefer train split if available, otherwise any preprocessed_step*.csv
                latest_pre = sandbox.process.exec(
                    "sh -lc 'ls -1t preprocessed_step*_train.csv preprocessed_step*.csv 2>/dev/null | head -1'",
                    cwd=workspace,
                    timeout=20,
                ).result.strip()
                if latest_pre:
                    dataset_filename = latest_pre
                    dataset_path = f"{workspace}/{dataset_filename}"
        except Exception:
            pass
        
        # Create TrainingService and run pipeline (import locally to avoid circular dependency)
        from services.training_service import TrainingService
        training_service = TrainingService()
        
        # Run the training pipeline asynchronously
        import concurrent.futures
        
        def run_in_new_loop():
            """Run the training pipeline in a new event loop in a separate thread"""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    training_service.run_training_pipeline(
                        dataset_path=dataset_path,
                        target_column=target_column,
                        ctx=raw_ctx,
                        max_iterations=max_iterations
                    )
                )
            finally:
                new_loop.close()
        
        # Run in a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            result = future.result(timeout=1800)  # 30 minute timeout
        
        return json.dumps({
            "status": "success",
            "message": "Training pipeline completed successfully",
            "results": result,
            "selected_dataset": dataset_filename
        })
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Training pipeline failed: {str(e)}"
        })

def create_master_training_agent():
    """Create the Master Training Agent that orchestrates model selection and training."""
    return Agent(
        name="Master Training Agent",
        instructions=master_training_agent_prompt,
        tools=[daytona_agent_tool, training_agent_tool],
    )

def create_sub_training_agent(model_name: str, model_category: str, dataset_characteristics: dict, performance_expectations: str):
    """Create a specialized Sub Training Agent for a specific model."""
    
    prompt = get_sub_training_agent_prompt(model_name, model_category, dataset_characteristics, performance_expectations)
    
    return Agent(
        name=f"{model_name} Training Agent",
        instructions=prompt,
        tools=[daytona_agent_tool],
    )

def create_evaluator_agent():
    """Create the Evaluator Agent for performance analysis and tuning recommendations."""
    return Agent(
        name="Evaluator Agent",
        instructions=evaluator_agent_prompt,
        tools=[daytona_agent_tool],
    )


def create_orchestrator_agent():
    return Agent(
        name="Orchestrator Agent",
        instructions=orchestrator_prompt,
        handoffs=[create_preprocessor_agent(), create_master_training_agent()],
    )
