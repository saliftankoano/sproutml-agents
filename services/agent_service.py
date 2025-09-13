from agents import Agent, function_tool
from prompts import preprocessing_agent_prompt, orchestrator_prompt
from services.daytona_service import persistent_sandboxes, persistent_volumes, get_persistent_sandbox, create_sandbox
from agents.tool_context import ToolContext
from daytona import Sandbox
from fastapi import HTTPException
from typing import Optional

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
    job_id = getattr(ctx.context, "job_id", None)
    sandbox: Sandbox | None = None
    if job_id and job_id in persistent_sandboxes:
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
    if sandbox is None:
        raise HTTPException(status_code=404, detail="Sandbox not found or already cleaned up")
    try:
        # Ensure workspace exists and is usable
        # Workspace on the mounted volume
        ws = "/home/daytona/volume/workspace"
        try:
            sandbox.process.exec(f"mkdir -p {ws}", cwd=".", timeout=30)
        except Exception:
            pass
        # Upload script
        sandbox.fs.upload_file(script.encode("utf-8"), f"{ws}/{script_name}")

        # Upload dataset only if not already present and we have path on first call
        ds_name = dataset_destination or getattr(ctx.context, "dataset_filename", None)
        ds_path = getattr(ctx.context, "dataset_path", None)
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

def create_orchestrator_agent():
    return Agent(
        name="Orchestrator Agent",
        instructions=orchestrator_prompt,
        handoffs=[create_preprocessor_agent()],
    )
