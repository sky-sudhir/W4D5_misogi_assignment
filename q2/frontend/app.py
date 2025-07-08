import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import json
import os
from dotenv import load_dotenv

# from utils import format_results, calculate_display_metrics, create_comparison_chart

load_dotenv()

# Configuration
BACKEND_URL = f"http://localhost:{os.getenv('BACKEND_PORT', 8000)}"

# Page configuration
st.set_page_config(
    page_title="Indian Legal Document Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-header {
        font-size: 1.5rem;
        color: #2c5282;
        border-bottom: 2px solid #2c5282;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2c5282;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .similarity-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c5282;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Indian Legal Document Search System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Health Check
        if st.button("Check API Health"):
            try:
                response = requests.get(f"{BACKEND_URL}/health/")
                if response.status_code == 200:
                    health_data = response.json()
                    st.success("✅ API is healthy")
                    st.json(health_data)
                else:
                    st.error("❌ API is not responding")
            except Exception as e:
                st.error(f"❌ Error connecting to API: {str(e)}")
        
        # Collection Info
        if st.button("View Collections Info"):
            try:
                response = requests.get(f"{BACKEND_URL}/collections/info/")
                if response.status_code == 200:
                    collections_data = response.json()
                    st.success("📊 Collections Information")
                    st.json(collections_data)
                else:
                    st.error("❌ Failed to fetch collections info")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Search Configuration
        st.header("🔍 Search Settings")
        top_k = st.slider("Results per method", min_value=1, max_value=10, value=5)
        
        similarity_methods = st.multiselect(
            "Similarity Methods",
            ["cosine", "euclidean", "mmr", "hybrid"],
            default=["cosine", "euclidean", "mmr", "hybrid"]
        )
        
        # Document Type Filter
        document_type_filter = st.selectbox(
            "Document Type Filter",
            ["All", "act", "judgment", "regulation", "circular"],
            index=0
        )
        
        # Law Name Filter
        law_name_filter = st.text_input("Law Name Filter (optional)")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📄 Upload Documents", "📊 Analytics"])
    
    with tab1:
        search_interface(top_k, similarity_methods, document_type_filter, law_name_filter)
    
    with tab2:
        upload_interface()
    
    with tab3:
        analytics_interface()

def search_interface(top_k, similarity_methods, document_type_filter, law_name_filter):
    """Search interface with 4-column result view"""
    st.header("🔍 Search Legal Documents")
    
    # Search input
    query = st.text_input(
        "Enter your legal query:",
        placeholder="e.g., Section 80C deduction limit for income tax",
        help="Enter your legal question or search for specific provisions"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    
    with col2:
        if st.button("💡 Get AI Summary"):
            if 'search_results' in st.session_state and st.session_state.search_results:
                get_ai_summary(query, st.session_state.search_results)
            else:
                st.warning("Please perform a search first")
    
    with col3:
        if st.button("🧠 Ask AI"):
            if 'search_results' in st.session_state and st.session_state.search_results:
                ai_qa_interface(query, st.session_state.search_results)
            else:
                st.warning("Please perform a search first")
    
    if search_button and query:
        # Prepare search parameters
        params = {
            "query": query,
            "top_k": top_k,
            "similarity_methods": similarity_methods
        }
        
        # Add filters if specified
        where_clause = {}
        if document_type_filter != "All":
            where_clause["document_type"] = document_type_filter
        if law_name_filter:
            where_clause["law_name"] = law_name_filter
        
        # Perform search
        with st.spinner("Searching legal documents..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/search/",
                    params=params
                )
                
                if response.status_code == 200:
                    results = response.json()
                    st.session_state.search_results = results
                    display_search_results(results, query)
                else:
                    st.error(f"Search failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error during search: {str(e)}")

def display_search_results(results: Dict[str, Any], query: str):
    """Display search results in 4-column layout"""
    st.success(f"Found results for: '{query}'")
    
    # Results overview
    total_results = sum(len(method_results) for method_results in results['results'].values())
    st.info(f"Total results: {total_results} across {len(results['results'])} methods")
    
    # Create columns for each method
    methods = list(results['results'].keys())
    
    if len(methods) == 1:
        cols = [st.container()]
    elif len(methods) == 2:
        cols = st.columns(2)
    elif len(methods) == 3:
        cols = st.columns(3)
    else:
        cols = st.columns(4)
    
    # Display results for each method
    for i, method in enumerate(methods):
        with cols[i % len(cols)]:
            method_results = results['results'][method]
            
            # Method header
            st.markdown(f'<div class="method-header">{method.upper()} Similarity</div>', unsafe_allow_html=True)
            
            if not method_results:
                st.warning(f"No results found for {method}")
                continue
            
            # Method metrics
            avg_score = sum(r['similarity_score'] for r in method_results) / len(method_results)
            st.markdown(f'<div class="metric-card">Avg Score: <span class="similarity-score">{avg_score:.3f}</span></div>', unsafe_allow_html=True)
            
            # Display each result
            for j, result in enumerate(method_results, 1):
                with st.expander(f"Result {j} (Score: {result['similarity_score']:.3f})"):
                    # Document content
                    st.markdown("**Document Content:**")
                    st.text_area(
                        f"Content {method}_{j}",
                        value=result['document'][:500] + "..." if len(result['document']) > 500 else result['document'],
                        height=150,
                        key=f"content_{method}_{j}",
                        label_visibility="collapsed"
                    )
                    
                    # Metadata
                    metadata = result.get('metadata', {})
                    st.markdown("**Metadata:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Law:** {metadata.get('law_name', 'N/A')}")
                        st.write(f"**Type:** {metadata.get('document_type', 'N/A')}")
                    with col2:
                        st.write(f"**Sections:** {', '.join(metadata.get('sections', []))}")
                        st.write(f"**Keywords:** {', '.join(metadata.get('keywords', []))}")
                    
                    # Additional scores for hybrid method
                    if method == 'hybrid' and 'cosine_score' in result:
                        st.markdown("**Score Breakdown:**")
                        st.write(f"Cosine Score: {result['cosine_score']:.3f}")
                        st.write(f"Entity Score: {result['entity_score']:.3f}")
    
    # Comparison chart
    if len(methods) > 1:
        st.subheader("📊 Method Comparison")
        create_method_comparison_chart(results['results'])

def create_method_comparison_chart(results: Dict[str, List[Dict[str, Any]]]):
    """Create comparison chart for different methods"""
    method_scores = {}
    
    for method, method_results in results.items():
        if method_results:
            scores = [r['similarity_score'] for r in method_results]
            method_scores[method] = {
                'avg_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'result_count': len(scores)
            }
    
    if method_scores:
        # Create comparison DataFrame
        df = pd.DataFrame(method_scores).T
        
        # Average scores bar chart
        fig = px.bar(
            x=df.index,
            y=df['avg_score'],
            title="Average Similarity Scores by Method",
            labels={'x': 'Method', 'y': 'Average Score'},
            color=df['avg_score'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.dataframe(df.round(3), use_container_width=True)

def get_ai_summary(query: str, search_results: Dict[str, Any]):
    """Get AI summary of search results"""
    st.subheader("🤖 AI Summary")
    
    # Collect all documents for summarization
    all_documents = []
    for method_results in search_results['results'].values():
        all_documents.extend(method_results)
    
    # Remove duplicates based on document content
    unique_documents = []
    seen_content = set()
    for doc in all_documents:
        content_hash = hash(doc['document'])
        if content_hash not in seen_content:
            unique_documents.append(doc)
            seen_content.add(content_hash)
    
    if unique_documents:
        with st.spinner("Generating AI summary..."):
            try:
                summary_data = {
                    "query": query,
                    "documents": unique_documents[:10]  # Limit to top 10 unique documents
                }
                
                response = requests.post(
                    f"{BACKEND_URL}/summarize/",
                    json=summary_data
                )
                
                if response.status_code == 200:
                    summary_result = response.json()
                    
                    st.markdown("**Summary:**")
                    st.write(summary_result['summary'])
                    
                    if summary_result.get('key_provisions'):
                        st.markdown("**Key Provisions:**")
                        for provision in summary_result['key_provisions']:
                            st.write(f"• {provision}")
                    
                    if summary_result.get('laws_covered'):
                        st.markdown("**Laws Covered:**")
                        for law in summary_result['laws_covered']:
                            st.write(f"• {law}")
                
                else:
                    st.error(f"Failed to generate summary: {response.text}")
                    
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
    else:
        st.warning("No documents available for summarization")

def ai_qa_interface(query: str, search_results: Dict[str, Any]):
    """AI Q&A interface"""
    st.subheader("🧠 Ask AI about Legal Documents")
    
    question = st.text_input(
        "Ask a specific question about the search results:",
        placeholder="e.g., What are the conditions for claiming Section 80C deduction?"
    )
    
    if st.button("Get Answer") and question:
        # Collect context documents
        context_documents = []
        for method_results in search_results['results'].values():
            context_documents.extend(method_results)
        
        # Remove duplicates
        unique_documents = []
        seen_content = set()
        for doc in context_documents:
            content_hash = hash(doc['document'])
            if content_hash not in seen_content:
                unique_documents.append(doc)
                seen_content.add(content_hash)
        
        if unique_documents:
            with st.spinner("Getting AI answer..."):
                try:
                    # Note: This would require implementing the answer_legal_question endpoint
                    # For now, we'll use the summarize endpoint with a modified query
                    qa_data = {
                        "query": f"Question: {question}\nContext: {query}",
                        "documents": unique_documents[:5]
                    }
                    
                    response = requests.post(
                        f"{BACKEND_URL}/summarize/",
                        json=qa_data
                    )
                    
                    if response.status_code == 200:
                        answer_result = response.json()
                        
                        st.markdown("**Answer:**")
                        st.write(answer_result['summary'])
                        
                        if answer_result.get('key_provisions'):
                            st.markdown("**Relevant Provisions:**")
                            for provision in answer_result['key_provisions']:
                                st.write(f"• {provision}")
                    
                    else:
                        st.error(f"Failed to get answer: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
        else:
            st.warning("No context documents available")

def upload_interface():
    """Document upload interface"""
    st.header("📄 Upload Legal Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a legal document",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files containing legal documents"
        )
    
    with col2:
        law_name = st.text_input(
            "Law Name",
            placeholder="e.g., Income Tax Act",
            help="Enter the name of the law or act"
        )
        
        document_type = st.selectbox(
            "Document Type",
            ["act", "judgment", "regulation", "circular", "notification"],
            help="Select the type of legal document"
        )
    
    if st.button("📤 Upload and Process", type="primary"):
        if uploaded_file and law_name:
            with st.spinner("Processing document..."):
                try:
                    files = {"file": uploaded_file}
                    data = {
                        "law_name": law_name,
                        "document_type": document_type
                    }
                    
                    response = requests.post(
                        f"{BACKEND_URL}/upload-document/",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Document uploaded successfully!")
                        
                        st.info(f"Created {result['chunks_created']} chunks from {result['filename']}")
                        
                        # Display processing results
                        with st.expander("Processing Details"):
                            st.json(result)
                    
                    else:
                        st.error(f"Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")
        else:
            st.warning("Please provide both a file and law name")

def analytics_interface():
    """Analytics and metrics interface"""
    st.header("📊 Analytics Dashboard")
    
    # Placeholder for analytics
    st.info("Analytics features will be available after performing searches and uploads")
    
    if 'search_results' in st.session_state:
        st.subheader("Search Results Analysis")
        
        results = st.session_state.search_results['results']
        
        # Method performance comparison
        method_performance = {}
        for method, method_results in results.items():
            if method_results:
                scores = [r['similarity_score'] for r in method_results]
                method_performance[method] = {
                    'avg_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'std_score': pd.Series(scores).std()
                }
        
        if method_performance:
            df = pd.DataFrame(method_performance).T
            
            st.subheader("Method Performance Metrics")
            st.dataframe(df.round(3))
            
            # Visualization
            fig = go.Figure()
            
            for method in df.index:
                fig.add_trace(go.Scatter(
                    x=[method],
                    y=[df.loc[method, 'avg_score']],
                    error_y=dict(
                        type='data',
                        array=[df.loc[method, 'std_score']],
                        visible=True
                    ),
                    mode='markers+lines',
                    name=method,
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title="Method Performance with Standard Deviation",
                xaxis_title="Method",
                yaxis_title="Average Score",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 