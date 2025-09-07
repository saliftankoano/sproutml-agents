# Try to import real agents module, fall back to mock if not available
try:
    from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
    from agents.exceptions import InputGuardrailTripwireTriggered
except ImportError:
    print("Warning: 'agents' module not found, using mock implementation")
    from agents_mock import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, InputGuardrailTripwireTriggered
from dotenv import load_dotenv
import asyncio
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import tempfile
import shutil
from typing import Optional

"""
The agent needed are:
    1- Orchestator
    2- Preprocessing
    3- Master trainer
    4- Evaluator
    5- Tuning

Task 2: Use fastapi to make an api endpoint where the csv file can be transferred through 
to kickstart the orchestrator agent.
"""
load_dotenv()
openai_key= os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="SproutML Training API", version="1.0.0")

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

async def test_orchestrator():
    """Test function for the orchestrator - can be used for debugging"""
    try:
        result = await Runner.run(orchestrator, "who was the first president of the united states?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

# instructions, name, handoff_description completed ✅
preprocessor_agent = Agent(
    name="Preprocessing Agent",
    handoff_description="Agent specializing in preprocessing datasets",
    instructions="You're an expert in the data preprocessing for machine learning pipelines. You make graphs, stats, devise step by step plans, generate code you run by calling tools to help you with, save processed datasets and also provide an executive summary to allow other agents to continue the process.",
)

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
    instructions="You are an expert ML orchestrator, your only role is to own the end-to-end run of machine learning agentic architecture. Validate inputs, choose a plan, call downstream in the appropriate sequences to deliver the highest quality. Manage the entire pipeline to return trained models, results and an executive summary.",
    handoffs=[preprocessor_agent],
)

@app.post("/train")
async def train_model(
    file: UploadFile = File(..., description="CSV file containing the dataset"),
    target_column: str = Form(..., description="Name of the target column for training")
):
    """
    Train endpoint that receives a CSV file and target column to start the ML training process.
    
    Args:
        file: CSV file upload
        target_column: Name of the column to use as target variable
        
    Returns:
        Training results and model information
    """
    temp_dir = None
    try:
        # Save uploaded file temporarily
        
        # Create temporary directory (will be auto-cleaned on system reboot)
        temp_dir = tempfile.mkdtemp(prefix="sproutml_")
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file to temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare context with file path instead of raw content
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
        
        # Let the orchestrator and its agents handle everything
        result = await Runner.run(orchestrator, training_request)
        
        return {
            "status": "success",
            "message": "Training request sent to orchestrator",
            "filename": file.filename,
            "target_column": target_column,
            "temp_file_path": temp_file_path,  # For debugging - remove in production
            "orchestrator_output": result.final_output if hasattr(result, 'final_output') else str(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up {temp_dir}: {cleanup_error}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)