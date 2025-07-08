import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import numpy as np

def format_results(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Format search results into a pandas DataFrame"""
    formatted_data = []
    
    for method, method_results in results.items():
        for i, result in enumerate(method_results):
            formatted_data.append({
                'method': method,
                'rank': i + 1,
                'similarity_score': result['similarity_score'],
                'document': result['document'][:200] + "..." if len(result['document']) > 200 else result['document'],
                'law_name': result.get('metadata', {}).get('law_name', 'N/A'),
                'document_type': result.get('metadata', {}).get('document_type', 'N/A'),
                'sections': ', '.join(result.get('metadata', {}).get('sections', [])),
                'keywords': ', '.join(result.get('metadata', {}).get('keywords', []))
            })
    
    return pd.DataFrame(formatted_data)

def calculate_display_metrics(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for display"""
    metrics = {}
    
    for method, method_results in results.items():
        if method_results:
            scores = [r['similarity_score'] for r in method_results]
            
            # Calculate diversity score
            diversity_score = 0.0
            if len(method_results) > 1:
                documents = [r['document'] for r in method_results]
                diversity_scores = []
                
                for i in range(len(documents)):
                    for j in range(i + 1, len(documents)):
                        # Simple diversity calculation based on document length difference
                        len_diff = abs(len(documents[i]) - len(documents[j]))
                        max_len = max(len(documents[i]), len(documents[j]))
                        diversity_scores.append(len_diff / max_len if max_len > 0 else 0)
                
                diversity_score = np.mean(diversity_scores) if diversity_scores else 0.0
            
            metrics[method] = {
                'avg_similarity': np.mean(scores),
                'max_similarity': np.max(scores),
                'min_similarity': np.min(scores),
                'std_similarity': np.std(scores),
                'diversity_score': diversity_score,
                'result_count': len(method_results)
            }
        else:
            metrics[method] = {
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'std_similarity': 0.0,
                'diversity_score': 0.0,
                'result_count': 0
            }
    
    return metrics

def create_comparison_chart(results: Dict[str, List[Dict[str, Any]]]) -> go.Figure:
    """Create a comparison chart for different methods"""
    methods = list(results.keys())
    avg_scores = []
    max_scores = []
    min_scores = []
    
    for method in methods:
        method_results = results[method]
        if method_results:
            scores = [r['similarity_score'] for r in method_results]
            avg_scores.append(np.mean(scores))
            max_scores.append(np.max(scores))
            min_scores.append(np.min(scores))
        else:
            avg_scores.append(0)
            max_scores.append(0)
            min_scores.append(0)
    
    fig = go.Figure()
    
    # Add bars for average scores
    fig.add_trace(go.Bar(
        name='Average Score',
        x=methods,
        y=avg_scores,
        marker_color='lightblue'
    ))
    
    # Add bars for max scores
    fig.add_trace(go.Bar(
        name='Max Score',
        x=methods,
        y=max_scores,
        marker_color='darkblue'
    ))
    
    # Add bars for min scores
    fig.add_trace(go.Bar(
        name='Min Score',
        x=methods,
        y=min_scores,
        marker_color='lightgray'
    ))
    
    fig.update_layout(
        title='Similarity Scores Comparison Across Methods',
        xaxis_title='Method',
        yaxis_title='Similarity Score',
        barmode='group',
        height=400
    )
    
    return fig

def create_diversity_chart(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create a diversity comparison chart"""
    methods = list(metrics.keys())
    diversity_scores = [metrics[method]['diversity_score'] for method in methods]
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=diversity_scores,
            marker_color='green',
            text=[f'{score:.3f}' for score in diversity_scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Result Diversity by Method',
        xaxis_title='Method',
        yaxis_title='Diversity Score',
        height=400
    )
    
    return fig

def create_score_distribution_chart(results: Dict[str, List[Dict[str, Any]]]) -> go.Figure:
    """Create score distribution chart"""
    fig = go.Figure()
    
    for method, method_results in results.items():
        if method_results:
            scores = [r['similarity_score'] for r in method_results]
            
            fig.add_trace(go.Box(
                y=scores,
                name=method,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
    
    fig.update_layout(
        title='Score Distribution by Method',
        yaxis_title='Similarity Score',
        height=400
    )
    
    return fig

def format_document_preview(document: str, max_length: int = 300) -> str:
    """Format document content for preview"""
    if len(document) <= max_length:
        return document
    
    # Try to break at a sentence boundary
    truncated = document[:max_length]
    last_period = truncated.rfind('.')
    last_space = truncated.rfind(' ')
    
    if last_period > max_length - 50:  # If period is near the end
        return document[:last_period + 1] + "..."
    elif last_space > max_length - 20:  # If space is near the end
        return document[:last_space] + "..."
    else:
        return document[:max_length] + "..."

def extract_key_terms(documents: List[Dict[str, Any]]) -> List[str]:
    """Extract key terms from documents"""
    all_keywords = []
    all_sections = []
    
    for doc in documents:
        metadata = doc.get('metadata', {})
        keywords = metadata.get('keywords', [])
        sections = metadata.get('sections', [])
        
        all_keywords.extend(keywords)
        all_sections.extend(sections)
    
    # Count frequency and return top terms
    from collections import Counter
    
    keyword_counts = Counter(all_keywords)
    section_counts = Counter(all_sections)
    
    top_keywords = [term for term, count in keyword_counts.most_common(10)]
    top_sections = [section for section, count in section_counts.most_common(10)]
    
    return top_keywords + top_sections

def calculate_method_overlap(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """Calculate overlap between different methods"""
    methods = list(results.keys())
    overlaps = {}
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:  # Avoid duplicate comparisons
                docs1 = set(r['document'] for r in results[method1])
                docs2 = set(r['document'] for r in results[method2])
                
                if docs1 or docs2:
                    overlap = len(docs1.intersection(docs2)) / len(docs1.union(docs2))
                    overlaps[f"{method1}_vs_{method2}"] = overlap
                else:
                    overlaps[f"{method1}_vs_{method2}"] = 0.0
    
    return overlaps

def create_overlap_heatmap(results: Dict[str, List[Dict[str, Any]]]) -> go.Figure:
    """Create overlap heatmap between methods"""
    methods = list(results.keys())
    n_methods = len(methods)
    
    # Create overlap matrix
    overlap_matrix = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                overlap_matrix[i][j] = 1.0  # Perfect overlap with itself
            else:
                docs1 = set(r['document'] for r in results[method1])
                docs2 = set(r['document'] for r in results[method2])
                
                if docs1 or docs2:
                    overlap = len(docs1.intersection(docs2)) / len(docs1.union(docs2))
                    overlap_matrix[i][j] = overlap
                else:
                    overlap_matrix[i][j] = 0.0
    
    fig = go.Figure(data=go.Heatmap(
        z=overlap_matrix,
        x=methods,
        y=methods,
        colorscale='Blues',
        text=np.round(overlap_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title='Method Overlap Heatmap',
        xaxis_title='Method',
        yaxis_title='Method',
        height=400
    )
    
    return fig

def generate_search_summary(results: Dict[str, List[Dict[str, Any]]], query: str) -> Dict[str, Any]:
    """Generate a summary of search results"""
    total_results = sum(len(method_results) for method_results in results.values())
    
    # Get all unique documents
    all_documents = []
    for method_results in results.values():
        all_documents.extend(method_results)
    
    unique_documents = list({doc['document']: doc for doc in all_documents}.values())
    
    # Extract metadata statistics
    laws_covered = set()
    document_types = set()
    all_sections = []
    
    for doc in unique_documents:
        metadata = doc.get('metadata', {})
        if metadata.get('law_name'):
            laws_covered.add(metadata['law_name'])
        if metadata.get('document_type'):
            document_types.add(metadata['document_type'])
        if metadata.get('sections'):
            all_sections.extend(metadata['sections'])
    
    return {
        'query': query,
        'total_results': total_results,
        'unique_documents': len(unique_documents),
        'methods_used': len(results),
        'laws_covered': list(laws_covered),
        'document_types': list(document_types),
        'sections_found': list(set(all_sections)),
        'avg_similarity_across_methods': np.mean([
            np.mean([r['similarity_score'] for r in method_results])
            for method_results in results.values()
            if method_results
        ]) if any(results.values()) else 0.0
    } 