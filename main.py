
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from config import ALLOW_ORIGINS, ALLOW_METHODS, ALLOW_HEADERS
from services.agent_service import create_preprocessor_agent, create_orchestrator_agent
from routes import training, jobs, artifacts, health

load_dotenv()

app = FastAPI(title="SproutML API", version="1.0.0")
app.include_router(training.router)
app.include_router(jobs.router)
app.include_router(artifacts.router)
app.include_router(health.router)

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
   
# Default preprocessor agent (for backwards compatibility)
preprocessor_agent = create_preprocessor_agent()
orchestrator = create_orchestrator_agent()

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable, default to 8000
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)