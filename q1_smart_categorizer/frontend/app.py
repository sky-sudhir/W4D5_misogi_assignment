import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import json
import time

# Page configuration
st.set_page_config(
    page_title="Smart Article Categorizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: dict = None, timeout: int = 1800) -> Dict[str, Any]:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Request timed out after {timeout} seconds. The operation may still be running in the background.")
        return {"success": False, "error": "timeout"}
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {"success": False, "error": str(e)}
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse API response: {str(e)}")
        return {"success": False, "error": str(e)}

def display_metrics(metrics: Dict[str, Any]):
    """Display model metrics in a formatted way"""
    if not metrics:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

def plot_umap_visualization(umap_data: List[Dict[str, Any]], model_name: str):
    """Create UMAP visualization plot"""
    if not umap_data:
        st.warning("No UMAP data available")
        return
    
    df = pd.DataFrame(umap_data)
    
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='label',
        title=f'UMAP Visualization - {model_name.upper()}',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Smart Article Categorizer</h1>', unsafe_allow_html=True)
    st.markdown("### Classify news articles using multiple embedding approaches")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Model Training", "Article Prediction", "Model Comparison", "UMAP Visualization"]
    )
    
    # Get available models
    models_response = make_api_request("/models")
    available_models = models_response.get("models", []) 
    
    # Get models status
    status_response = make_api_request("/models/status")
    models_status = status_response.get("models_status", {}) 
    
    if page == "Home":
        display_home_page()
    elif page == "Model Training":
        display_training_page(available_models, models_status)
    elif page == "Article Prediction":
        display_prediction_page(available_models, models_status)
    elif page == "Model Comparison":
        display_comparison_page(available_models, models_status)
    elif page == "UMAP Visualization":
        display_umap_page(available_models, models_status)

def display_home_page():
    """Display home page with project information"""
    st.markdown("## Welcome to Smart Article Categorizer!")
    
    st.markdown("""
    This application uses four different embedding approaches to classify news articles into categories:
    
    ### 🔧 Available Models:
    - **GloVe**: Pre-trained word vectors with averaging
    - **BERT**: Using [CLS] token from BERT-base-uncased
    - **Sentence-BERT**: Direct sentence embeddings
    - **Gemini**: Google's text-embedding-004 model
    
    ### 📊 Categories:
    - Tech
    - Finance (Business)
    - Sport
    - Politics
    - Entertainment
    
    ### 🚀 Features:
    - Train models individually
    - Compare model performance
    - Predict article categories
    - Visualize embeddings with UMAP
    """)
    
    # Display dataset statistics
    st.markdown("### 📈 Dataset Statistics")
    stats_response = make_api_request("/dataset/stats")
    if stats_response.get("success"):
        stats = stats_response["stats"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", stats.get("total_samples", 0))
        with col2:
            st.metric("Categories", len(stats.get("categories", [])))
        with col3:
            st.metric("Avg. Words per Article", f"{stats.get('avg_text_length', 0):.1f}")
        
        # Category distribution
        if "category_distribution" in stats:
            st.markdown("#### Category Distribution")
            dist_df = pd.DataFrame(
                list(stats["category_distribution"].items()),
                columns=["Category", "Count"]
            )
            fig = px.bar(dist_df, x="Category", y="Count", title="Articles per Category")
            st.plotly_chart(fig, use_container_width=True)

def display_training_page(available_models: List[str], models_status: Dict[str, str]):
    """Display model training page"""
    st.markdown("## 🏋️ Model Training")
    
    st.markdown("Train individual models and view their performance metrics.")
    
    # Import utilities
    try:
        from utils import get_model_recommendations, format_training_time_estimate, show_training_tips
        
        # Show training tips
        show_training_tips()
        
        # Model recommendations
        st.markdown("### 🎯 Model Recommendations")
        recommendations = get_model_recommendations()
        
        for model in available_models:
            if model in recommendations:
                st.markdown(f"**{model.upper()}**: {recommendations[model]}")
        
    except ImportError:
        pass
    
    # Model selection
    selected_model = st.selectbox(
        "Select a model to train:",
        available_models,
        format_func=lambda x: f"{x.upper()} - {models_status.get(x, 'not_trained').replace('_', ' ').title()}"
    )
    
    if selected_model:
        current_status = models_status.get(selected_model, "not_trained")
        
        # Display current status and estimated training time
        if current_status == "trained":
            st.success(f"✅ {selected_model.upper()} is currently trained and ready")
        elif current_status == "saved":
            st.info(f"💾 {selected_model.upper()} has a saved model")
        else:
            st.warning(f"⚠️ {selected_model.upper()} is not trained")
            
            # Show estimated training time
            try:
                from utils import format_training_time_estimate
                estimated_time = format_training_time_estimate(selected_model)
                st.info(f"⏱️ Estimated training time: {estimated_time}")
            except ImportError:
                pass
        
        # Training button
        if st.button(f"Train {selected_model.upper()} Model", type="primary"):
            # Show training progress
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            with progress_placeholder.container():
                st.info(f"🚀 Starting training for {selected_model.upper()} model...")
                st.info("⏱️ This may take 10-30 minutes depending on your system and the model.")
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Show initial progress
                progress_bar.progress(10)
                status_text.text("Initializing training...")
                
                # Make training request with longer timeout
                try:
                    url = f"{API_BASE_URL}/train"
                    
                    # Start training in background
                    status_text.text("Sending training request...")
                    progress_bar.progress(20)
                    
                    response = requests.post(
                        url, 
                        json={"model": selected_model}, 
                        timeout=1800  # 30 minutes timeout
                    )
                    response.raise_for_status()
                    response = response.json()
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                except requests.exceptions.Timeout:
                    progress_placeholder.empty()
                    st.error("⏱️ Training request timed out after 30 minutes.")
                    st.info("💡 **What this means:**")
                    st.info("- The training process might still be running on the server")
                    st.info("- Check the backend terminal for progress logs")
                    st.info("- You can refresh the page and check model status")
                    st.info("- Consider training a smaller model first (e.g., GloVe)")
                    return
                except requests.exceptions.RequestException as e:
                    progress_placeholder.empty()
                    st.error(f"❌ Training request failed: {str(e)}")
                    st.info("💡 **Troubleshooting:**")
                    st.info("- Make sure the backend server is running")
                    st.info("- Check your internet connection (for downloading models)")
                    st.info("- Look at the backend logs for detailed error information")
                    return
                
                if response.get("success"):
                    st.success(f"✅ {selected_model.upper()} model trained successfully!")
                    
                    # Display metrics
                    metrics = response.get("metrics", {})
                    if metrics:
                        st.markdown("### 📊 Training Results")
                        display_metrics(metrics)
                        
                        # Detailed classification report
                        if "classification_report" in metrics:
                            st.markdown("#### Detailed Classification Report")
                            report = metrics["classification_report"]
                            
                            # Convert to DataFrame for better display
                            report_data = []
                            for category, stats in report.items():
                                if isinstance(stats, dict) and "precision" in stats:
                                    report_data.append({
                                        "Category": category,
                                        "Precision": f"{stats['precision']:.4f}",
                                        "Recall": f"{stats['recall']:.4f}",
                                        "F1-Score": f"{stats['f1-score']:.4f}",
                                        "Support": int(stats['support'])
                                    })
                            
                            if report_data:
                                st.dataframe(pd.DataFrame(report_data), use_container_width=True)
                else:
                    st.error(f"❌ Training failed: {response.get('error', 'Unknown error')}")

def display_prediction_page(available_models: List[str], models_status: Dict[str, str]):
    """Display article prediction page"""
    st.markdown("## 🔮 Article Prediction")
    
    st.markdown("Enter an article text to get category predictions from all trained models.")
    
    # Text input
    article_text = st.text_area(
        "Enter article text:",
        height=200,
        placeholder="Paste your article text here..."
    )
    
    if article_text.strip():
        if st.button("Predict Category", type="primary"):
            st.markdown("### 🎯 Prediction Results")
            
            # Get predictions from all trained models
            predictions = {}
            
            for model in available_models:
                if models_status.get(model) in ["trained", "saved"]:
                    with st.spinner(f"Getting prediction from {model.upper()}..."):
                        response = make_api_request(
                            "/predict",
                            method="POST",
                            data={"model": model, "text": article_text}
                        )
                        
                        if response.get("success"):
                            predictions[model] = {
                                "category": response["predicted_category"],
                                "confidence": response["confidence"]
                            }
                        else:
                            st.error(f"❌ Prediction failed for {model}: {response.get('error', 'Unknown error')}")
            
            # Display predictions
            if predictions:
                cols = st.columns(len(predictions))
                
                for i, (model, pred) in enumerate(predictions.items()):
                    with cols[i]:
                        st.markdown(f"#### {model.upper()}")
                        st.markdown(f"**Category:** {pred['category'].title()}")
                        st.markdown(f"**Confidence:** {pred['confidence']:.3f}")
                        
                        # Progress bar for confidence
                        st.progress(pred['confidence'])
                
                # Summary table
                st.markdown("#### Summary")
                summary_data = []
                for model, pred in predictions.items():
                    summary_data.append({
                        "Model": model.upper(),
                        "Predicted Category": pred['category'].title(),
                        "Confidence": f"{pred['confidence']:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            else:
                st.warning("No trained models available for prediction. Please train models first.")

def display_comparison_page(available_models: List[str], models_status: Dict[str, str]):
    """Display model comparison page"""
    st.markdown("## 📊 Model Comparison")
    
    st.markdown("Compare performance metrics across different models.")
    
    # Get metrics for all trained models
    # Note: This would require storing metrics in the API or database
    # For now, we'll show a placeholder
    
    st.info("📝 Model comparison will show metrics once models are trained.")
    
    # Show current model status
    st.markdown("### Current Model Status")
    status_data = []
    for model in available_models:
        status = models_status.get(model, "not_trained")
        status_data.append({
            "Model": model.upper(),
            "Status": status.replace("_", " ").title(),
            "Ready for Prediction": "✅" if status in ["trained", "saved"] else "❌"
        })
    
    st.dataframe(pd.DataFrame(status_data), use_container_width=True)

def display_umap_page(available_models: List[str], models_status: Dict[str, str]):
    """Display UMAP visualization page"""
    st.markdown("## 🗺️ UMAP Visualization")
    
    st.markdown("Visualize embedding clusters in 2D space using UMAP dimensionality reduction.")
    
    # Model selection for UMAP
    trained_models = [m for m in available_models if models_status.get(m) in ["trained", "saved"]]
    
    if not trained_models:
        st.warning("No trained models available for UMAP visualization. Please train models first.")
        return
    
    selected_model = st.selectbox(
        "Select a model for UMAP visualization:",
        trained_models,
        format_func=lambda x: x.upper()
    )
    
    if selected_model and st.button("Generate UMAP Visualization", type="primary"):
        with st.spinner(f"Generating UMAP visualization for {selected_model.upper()}..."):
            response = make_api_request(f"/umap/{selected_model}")
            
            if response.get("success"):
                st.success(f"✅ UMAP visualization generated for {selected_model.upper()}")
                
                # Display statistics
                stats = response.get("stats", {})
                if stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", stats.get("n_samples", 0))
                    with col2:
                        st.metric("Categories", stats.get("n_categories", 0))
                    with col3:
                        st.metric("Embedding Dimension", stats.get("embedding_dim", 0))
                
                # Plot UMAP
                umap_data = response.get("data", [])
                if umap_data:
                    plot_umap_visualization(umap_data, selected_model)
                else:
                    st.warning("No UMAP data available")
            else:
                st.error(f"❌ UMAP visualization failed: {response.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 