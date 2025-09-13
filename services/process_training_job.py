from services.daytona_service import get_persistent_volume, create_volume, get_persistent_sandbox, create_sandbox, wait_for_volume_ready
from services.job_service import update_job_status
from services.agent_service import create_preprocessor_agent, create_orchestrator_agent
from agents import Runner
from datetime import datetime
from typing import Any
import json
import os
from types import SimpleNamespace

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
            if volume is not None:
                # Wait for volume to be ready before creating sandbox
                volume = wait_for_volume_ready(job_id)
        
        if volume is None:
            raise Exception("Failed to create or get ready volume for job")
            
        sandbox = get_persistent_sandbox(job_id)
        if sandbox is None:
            sandbox = create_sandbox(job_id)
        # Record infrastructure details for later inspection/download
        update_job_status(job_id, "daytona", datetime.now().isoformat(),
                         volume_id=getattr(volume, "id", None),
                         sandbox_id=getattr(sandbox, "id", None),
                         workspace="/home/daytona/volume/workspace",
                         retain=True)
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
        update_job_status(job_id, "completed", datetime.now().isoformat(),
                         status="completed",
                         result={
                             "orchestrator_output": result.final_output if hasattr(result, 'final_output') else str(result),
                             "final_input_csv": current_csv,
                             "step_results": step_results,
                             "total_steps": step_num,
                         },
                         completed_at=datetime.now().isoformat(),
                         updated_at=datetime.now().isoformat())
        print(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        # Update job with error
        update_job_status(job_id, "failed", datetime.now().isoformat(), 
                         status="failed",
                         error=str(e),
                         updated_at=datetime.now().isoformat())
        print(f"Training job {job_id} failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up {temp_file_path}: {cleanup_error}")
     