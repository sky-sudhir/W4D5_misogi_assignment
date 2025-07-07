import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

def setup_logging():
    """Configure logging for the application"""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers
    api_logger = logging.getLogger("api")
    model_logger = logging.getLogger("model")
    data_logger = logging.getLogger("data")
    
    return {
        "api": api_logger,
        "model": model_logger,
        "data": data_logger
    }

def log_request(endpoint: str, method: str, params: dict = None, user_ip: str = None):
    """Log API requests"""
    logger = logging.getLogger("api")
    message = f"Request: {method} {endpoint}"
    if params:
        message += f" - Params: {params}"
    if user_ip:
        message += f" - IP: {user_ip}"
    logger.info(message)

def log_error(error: Exception, context: str = None):
    """Log errors with context"""
    logger = logging.getLogger("api")
    message = f"Error: {str(error)}"
    if context:
        message += f" - Context: {context}"
    logger.error(message, exc_info=True)

def log_model_training(model_name: str, status: str, metrics: dict = None):
    """Log model training events"""
    logger = logging.getLogger("model")
    message = f"Model Training - {model_name}: {status}"
    if metrics:
        message += f" - Metrics: {metrics}"
    logger.info(message)

def log_prediction(model_name: str, article_length: int, predicted_category: str, confidence: float):
    """Log prediction events"""
    logger = logging.getLogger("model")
    message = f"Prediction - {model_name}: Category={predicted_category}, Confidence={confidence:.3f}, Article_length={article_length}"
    logger.info(message) 