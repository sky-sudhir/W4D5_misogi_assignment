import uvicorn
from config.logging_config import setup_logging
from api.routes import app

# Setup logging
# setup_logging()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    ) 