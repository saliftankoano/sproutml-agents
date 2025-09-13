from fastapi import HTTPException, APIRouter, File, UploadFile, Form
import asyncio
import tempfile
import os
import shutil
from services.job_service import create_job
from services.process_training_job import process_training_job

router = APIRouter(title="SproutML Training API", version="1.0.0")

@router.post("/train")
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
