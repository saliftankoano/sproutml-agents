import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
DAYTONA_KEY = os.getenv("DAYTONA_API_KEY")

# Middleware configurations
ALLOW_ORIGINS = ["https://sproutml.com", "https://www.sproutml.com", "https://sproutml.vercel.app"]
ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
ALLOW_HEADERS = ["*"]