import json
import numpy as np
from typing import Any

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return super().default(obj)

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def safe_json_response(data: Any) -> Any:
    """Ensure data is safe for JSON serialization"""
    return convert_numpy_types(data) 