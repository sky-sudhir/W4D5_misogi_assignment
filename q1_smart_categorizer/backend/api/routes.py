from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from models.model_factory import ModelFactory
from utils.data_loader import load_dataset, prepare_train_test_split, get_dataset_stats
from utils.visualization import create_umap_visualization
from utils.json_encoder import safe_json_response
from config.logging_config import log_request, log_error, log_model_training, log_prediction
from config.settings import API_CONFIG

logger = logging.getLogger("api")

# Request/Response models
class TrainingRequest(BaseModel):
    model: str

class PredictionRequest(BaseModel):
    model: str
    text: str

class TrainingResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    success: bool
    model: str
    predicted_category: str
    confidence: float
    message: Optional[str] = None

class UMAPResponse(BaseModel):
    success: bool
    model: str
    data: List[Dict[str, Any]]
    stats: Dict[str, Any]

# Global variables to store models and data
trained_models: Dict[str, Any] = {}
dataset_cache = None
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"]
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    try:
        # Extract request info
        method = request.method
        url = str(request.url)
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        log_request(url, method, user_ip=client_ip)
        
        # Process request
        response = await call_next(request)
        
        return response
        
    except Exception as e:
        log_error(e, "Request middleware")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Smart Article Categorizer API", "version": API_CONFIG["version"]}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    try:
        models = ModelFactory.get_available_models()
        return {"models": models}
    except Exception as e:
        log_error(e, "Get available models")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/stats")
async def get_dataset_statistics():
    """Get dataset statistics"""
    try:
        stats = get_dataset_stats()
        return safe_json_response({"success": True, "stats": stats})
    except Exception as e:
        log_error(e, "Get dataset stats")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainingRequest):
    """Train a specific model"""
    try:
        model_name = request.model.lower()
        
        # Validate model name
        if not ModelFactory.is_valid_model(model_name):
            available_models = ModelFactory.get_available_models()
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model: {model_name}. Available models: {available_models}"
            )
        
        log_model_training(model_name, "started")
        
        # Load dataset if not cached
        global dataset_cache
        if dataset_cache is None:
            dataset_cache = load_dataset()
        
        # Prepare train/test split
        train_df, test_df = prepare_train_test_split(dataset_cache)
        
        # Create model instance
        model = ModelFactory.create_model(model_name)
        
        # Train model in executor to avoid blocking
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            executor, 
            model.train_classifier, 
            train_df, 
            test_df
        )
        
        # Store trained model
        trained_models[model_name] = model
        
        log_model_training(model_name, "completed", metrics)
        
        return TrainingResponse(
            success=True,
            message=f"Model {model_name} trained successfully",
            metrics=safe_json_response(metrics)
        )
        
    except Exception as e:
        log_error(e, f"Training model {request.model}")
        log_model_training(request.model, "failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_category(request: PredictionRequest):
    """Predict category for given text"""
    try:
        model_name = request.model.lower()
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Validate model name
        if not ModelFactory.is_valid_model(model_name):
            available_models = ModelFactory.get_available_models()
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model: {model_name}. Available models: {available_models}"
            )
        
        # Get or create model instance
        if model_name not in trained_models:
            model = ModelFactory.create_model(model_name)
            # Try to load saved model
            if not model.load_saved_model():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model {model_name} is not trained. Please train it first."
                )
            trained_models[model_name] = model
        else:
            model = trained_models[model_name]
        
        # Make prediction
        loop = asyncio.get_event_loop()
        predicted_category, confidence = await loop.run_in_executor(
            executor,
            model.predict,
            text
        )
        
        # Log prediction
        log_prediction(model_name, len(text.split()), predicted_category, confidence)
        
        return PredictionResponse(
            success=True,
            model=model_name,
            predicted_category=predicted_category,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"Prediction with model {request.model}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/umap/{model_name}")
async def get_umap_visualization(model_name: str):
    """Get UMAP visualization data for a model"""
    try:
        model_name = model_name.lower()
        
        # Validate model name
        if not ModelFactory.is_valid_model(model_name):
            available_models = ModelFactory.get_available_models()
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model: {model_name}. Available models: {available_models}"
            )
        
        # Get or create model instance
        if model_name not in trained_models:
            model = ModelFactory.create_model(model_name)
            # Try to load saved model
            if not model.load_saved_model():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model {model_name} is not trained. Please train it first."
                )
            trained_models[model_name] = model
        else:
            model = trained_models[model_name]
        
        # Get embeddings for UMAP
        embeddings, labels = model.get_embeddings_for_umap()
        
        # Create UMAP visualization
        loop = asyncio.get_event_loop()
        umap_data = await loop.run_in_executor(
            executor,
            create_umap_visualization,
            embeddings,
            labels
        )
        
        return UMAPResponse(
            success=True,
            model=model_name,
            data=safe_json_response(umap_data["data"]),
            stats=safe_json_response(umap_data["stats"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"UMAP visualization for model {model_name}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_models_status():
    """Get training status of all models"""
    try:
        status = {}
        for model_name in ModelFactory.get_available_models():
            if model_name in trained_models:
                status[model_name] = "trained"
            else:
                # Check if saved model exists
                model = ModelFactory.create_model(model_name)
                if model.load_saved_model():
                    status[model_name] = "saved"
                    trained_models[model_name] = model
                else:
                    status[model_name] = "not_trained"
        
        return {"success": True, "models_status": status}
        
    except Exception as e:
        log_error(e, "Get models status")
        raise HTTPException(status_code=500, detail=str(e)) 