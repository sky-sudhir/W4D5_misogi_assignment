import streamlit as st
import requests
import time
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8000"

def check_training_status(model_name: str) -> str:
    """Check if a model is currently being trained"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models_status = data.get("models_status", {})
            return models_status.get(model_name, "not_trained")
    except:
        pass
    return "unknown"

def wait_for_training_completion(model_name: str, max_wait_time: int = 1800) -> bool:
    """Wait for training to complete with periodic status checks"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status = check_training_status(model_name)
        
        if status in ["trained", "saved"]:
            return True
        elif status == "not_trained":
            # Training might have failed
            return False
        
        # Wait 30 seconds before next check
        time.sleep(30)
    
    return False  # Timeout

def get_model_recommendations() -> Dict[str, str]:
    """Get recommendations for which model to try first"""
    return {
        "glove": "🟢 **Recommended for beginners** - Fast training, good baseline performance",
        "sbert": "🟡 **Good balance** - Moderate training time, excellent performance", 
        "bert": "🟠 **Advanced** - Longer training time, very good performance",
        "gemini": "🔴 **Requires API key** - Fast but needs Google Gemini API access"
    }

def format_training_time_estimate(model_name: str) -> str:
    """Get estimated training time for each model"""
    estimates = {
        "glove": "5-10 minutes",
        "sbert": "10-15 minutes", 
        "bert": "15-30 minutes",
        "gemini": "5-15 minutes (API dependent)"
    }
    return estimates.get(model_name, "10-20 minutes")

def show_training_tips():
    """Display helpful training tips"""
    st.markdown("### 💡 Training Tips")
    
    with st.expander("Click for training recommendations"):
        st.markdown("""
        **For first-time users:**
        1. Start with **GloVe** - fastest training and good results
        2. Make sure you have a stable internet connection (for model downloads)
        3. Training time varies by system performance
        
        **If training takes too long:**
        - Try GloVe first (fastest)
        - Close other applications to free up memory
        - Check backend logs for progress updates
        
        **Model characteristics:**
        - **GloVe**: Uses pre-trained word vectors, fastest training
        - **Sentence-BERT**: Best balance of speed and accuracy
        - **BERT**: Highest accuracy but slower training
        - **Gemini**: Requires API key but very fast once set up
        """)

def display_system_requirements():
    """Display system requirements and recommendations"""
    st.markdown("### 🖥️ System Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Minimum:**")
        st.markdown("- 4GB RAM")
        st.markdown("- 2GB free disk space")
        st.markdown("- Stable internet connection")
    
    with col2:
        st.markdown("**Recommended:**")
        st.markdown("- 8GB+ RAM")
        st.markdown("- GPU (for faster BERT training)")
        st.markdown("- SSD storage") 