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
        
        # Try to create sandbox (with or without volume)
        sandbox = get_persistent_sandbox(job_id)
        if sandbox is None:
            sandbox = create_sandbox(job_id)
            
        if sandbox is None:
            raise Exception("Failed to create sandbox for job")
        # Determine workspace path based on whether volume is available
        try:
            sandbox.process.exec("test -d /home/daytona/volume", cwd=".", timeout=10)
            ws = "/home/daytona/volume/workspace"
            volume_available = True
        except Exception:
            ws = "/home/daytona/workspace"
            volume_available = False
            
        # Record infrastructure details for later inspection/download
        update_job_status(job_id, "daytona", datetime.now().isoformat(),
                         volume_id=getattr(volume, "id", None) if volume else None,
                         sandbox_id=getattr(sandbox, "id", None),
                         workspace=ws,
                         volume_available=volume_available,
                         retain=True)
        
        # Create workspace directory
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

       
        orchestrator = create_orchestrator_agent()
        # Also create a dedicated preprocessing agent to run deterministic multi-step preprocessing
        preprocessor_agent = create_preprocessor_agent()
        
        # Create orchestrator with the persistent preprocessor
        job_orchestrator = orchestrator
        
        # Extract target column from training request
        target_column = None
        for line in training_request.split('\n'):
            if 'Target Column:' in line:
                target_column = line.split('Target Column:')[1].strip()
                break
        
        # Prepare shared tool context (include dataset_path so tool can self-heal)
        tool_context = SimpleNamespace(
            job_id=job_id,
            dataset_filename=os.path.basename(temp_file_path),
            dataset_path=temp_file_path,
            target_column=target_column,
        )
        # Kick off preprocessing directly with the preprocessing agent (Step 1)
        initial_handoff_request = (
            f"Preprocessing step 1. "
            f"Dataset: '{tool_context.dataset_filename}'. "
            f"Target Column: {target_column}. "
            f"Perform initial analysis (stats/plots) AND produce a concrete preprocessing plan, then execute the FIRST actual preprocessing operation if applicable. "
            f"Always emit preprocess.py + requirements.txt, call the daytona tool, and save a versioned CSV (e.g., preprocessed_step1.csv). "
            f"Return structured JSON with output_csv and preprocessing_complete when truly ready for training."
        )
        result = await Runner.run(preprocessor_agent, initial_handoff_request, context=tool_context, max_turns=50)

        # Handoff-based preprocessing loop: orchestrator explicitly hands off to preprocessing agent for each step
        def _extract_step_and_latest_csv(json_text: str) -> tuple[str | None, str | None, dict[str, Any] | None, bool]:
            """Extract output_csv from JSON and also look for latest_csv from tool execution"""
            output_csv = None
            latest_csv = None
            data = None
            preprocessing_complete = False
            
            try:
                data = json.loads(json_text)
                if isinstance(data, dict):
                    output_csv = data.get("output_csv")
                    preprocessing_complete = data.get("preprocessing_complete", False)
                    # Also look for latest_csv from tool execution results
                    if "latest_csv" in data:
                        latest_csv = data.get("latest_csv")
            except Exception:
                pass
            
            if not data:
                try:
                    start = json_text.find("{")
                    end = json_text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        data = json.loads(json_text[start : end + 1])
                        if isinstance(data, dict):
                            output_csv = data.get("output_csv")
                            preprocessing_complete = data.get("preprocessing_complete", False)
                            if "latest_csv" in data:
                                latest_csv = data.get("latest_csv")
                except Exception:
                    pass
            
            # If output_csv is empty but latest_csv exists, use latest_csv
            if not output_csv and latest_csv:
                output_csv = latest_csv
                
            return output_csv, latest_csv, data, preprocessing_complete

        current_csv = os.path.basename(temp_file_path)
        step_num = 1
        max_steps = 10
        step_results = []
        
        # Initial step result
        initial_output = str(getattr(result, "final_output", ""))
        update_job_status(job_id, "preprocessing", datetime.now().isoformat(), latest_output=initial_output)
        step_results.append(f"Initial: {initial_output}")
        
        out_csv, latest_csv, _, preprocessing_complete = _extract_step_and_latest_csv(initial_output)
        
        # Use latest_csv if available, otherwise use out_csv; if none, continue with current file
        next_csv = latest_csv if latest_csv else out_csv
        if not next_csv:
            # Force continuation so the agent can begin concrete preprocessing
            next_csv = current_csv
        
        while next_csv and step_num <= max_steps and not preprocessing_complete:
            current_csv = next_csv
            tool_context.dataset_filename = current_csv
            step_num += 1
            
            # Directly instruct the preprocessing agent to continue with the next concrete step
            handoff_request = (
                f"Continue preprocessing step {step_num}. "
                f"Current dataset: '{current_csv}'. "
                f"Target Column: {target_column}. "
                f"Analyze current data and execute the next logical preprocessing operation (e.g., drop ID columns, encode categoricals, impute, scale, split). "
                f"MUST emit preprocess.py + requirements.txt, call the daytona tool, and SAVE a NEW versioned CSV named preprocessed_step{step_num}.csv. "
                f"If no transformation is needed yet, simply COPY the input dataset to preprocessed_step{step_num}.csv so the pipeline can proceed. "
                f"CRITICAL: Use target column '{target_column}' throughout preprocessing and never scale the target. "
                f"Return structured JSON with output_csv and set preprocessing_complete=true ONLY when fully ready for training."
            )
            
            result = await Runner.run(preprocessor_agent, handoff_request, context=tool_context, max_turns=50)
            step_output = str(getattr(result, "final_output", ""))
            step_results.append(f"Step {step_num}: {step_output}")
            update_job_status(job_id, "preprocessing", datetime.now().isoformat(), latest_output=step_output)
            
            out_csv, latest_csv, step_data, preprocessing_complete = _extract_step_and_latest_csv(step_output)
            
            if preprocessing_complete:
                print(f"Step {step_num}: Preprocessing marked as complete by agent.")
                break
            
            # Use latest_csv if available, otherwise use out_csv; if none, continue with current file
            next_csv = latest_csv if latest_csv else out_csv
            
            if not next_csv:
                print(f"Step {step_num}: No output_csv or latest_csv found. Continuing with current dataset '{current_csv}'.")
                next_csv = current_csv
                
            # Update for next iteration
            out_csv = next_csv

        # After preprocessing is complete, hand off to Master Training Agent
        print(f"Preprocessing completed. Final dataset: {current_csv}")
        print("Handing off to Master Training Agent for model training...")
        
        # Extract training dataset filename (handle case where both train and test files are present)
        training_dataset = current_csv
        if ',' in current_csv:
            # If multiple files are present, extract the training file
            files = [f.strip() for f in current_csv.split(',')]
            training_file = next((f for f in files if 'train' in f.lower()), files[0])
            training_dataset = training_file
            print(f"Extracted training dataset: {training_dataset}")
        
        # Update status to training before starting
        update_job_status(job_id, "training", datetime.now().isoformat(), 
                         latest_output=f"Starting training with dataset: {training_dataset}")
        
        # Hand off to Master Training Agent
        training_handoff_request = (
            f"Hand off to Master Training Agent. "
            f"Preprocessed dataset: '{training_dataset}'. "
            f"Target Column: {target_column}. "
            f"Preprocessing complete with {step_num} steps. "
            f"Please begin model selection and training pipeline."
        )
        
        training_result = await Runner.run(job_orchestrator, training_handoff_request, context=tool_context, max_turns=50)
        training_output = str(getattr(training_result, "final_output", ""))
        
        # Parse training results to extract model information
        trained_models = []
        try:
            # Try to extract model information from training output
            import re
            import json
            json_match = re.search(r'\{.*"trained_models".*\}', training_output, re.DOTALL)
            if json_match:
                training_data = json.loads(json_match.group())
                trained_models = training_data.get('trained_models', [])
        except Exception as e:
            print(f"Error parsing training results for models: {e}")
        
        # Update job with results after training
        update_job_status(job_id, "completed", datetime.now().isoformat(),
                         result={
                             "preprocessing_output": training_result.final_output if hasattr(training_result, 'final_output') else str(training_result),
                             "training_output": training_output,
                             "final_input_csv": current_csv,
                             "training_dataset": training_dataset,
                             "preprocessing_steps": step_results,
                             "total_preprocessing_steps": step_num,
                             "trained_models": trained_models,
                             "model_files_ready": len(trained_models) > 0,
                         },
                         completed_at=datetime.now().isoformat())
        print(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        # Update job with error
        update_job_status(job_id, "failed", datetime.now().isoformat(), 
                         error=str(e))
        print(f"Training job {job_id} failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up {temp_file_path}: {cleanup_error}")
     