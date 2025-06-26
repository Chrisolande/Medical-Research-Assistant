"""Streamlined Document Assistant."""

import logging
import random
import time
from datetime import datetime
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
)

# Simple CSS for better appearance
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .metric-container { 
        background: white; 
        padding: 1rem; 
        border-radius: 5px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class AppState:
    """Application state management."""
    pipeline_initialized: bool = False
    query_history: list = None
    uploaded_files: list = None
    stats: dict = None
    knowledge_graph: nx.Graph = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.query_history is None:
            self.query_history = []
        if self.uploaded_files is None:
            self.uploaded_files = []  
        if self.stats is None:
            self.stats = {"documents": 0, "nodes": 0, "edges": 0, "queries": 0}


def initialize_session_state():
    """Initialize session state."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()


def extract_text_from_pdf(file):
    """Extract text from PDF file."""
    try:
        return f"Extracted text from {file.name}. Demo content."
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None


def process_documents(files):
    """Process uploaded documents."""
    documents = []
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        st.text(f"Processing {file.name}...")
        time.sleep(0.2)  # Simulate processing
        
        text = extract_text_from_pdf(file)
        if text:
            documents.append({"content": text, "filename": file.name})
        
        progress_bar.progress((i + 1) / len(files))
    
    progress_bar.empty()
    return documents


def create_knowledge_graph():
    """Create a sample knowledge graph."""
    G = nx.Graph()
    
    concepts = ["AI", "Machine Learning", "Data Science", "Neural Networks", 
                "Deep Learning", "NLP", "Computer Vision", "Algorithms"]
    
    for concept in concepts:
        G.add_node(concept)
    
    edges = [
        ("AI", "Machine Learning"), ("Machine Learning", "Deep Learning"),
        ("Deep Learning", "Neural Networks"), ("AI", "NLP"), 
        ("AI", "Computer Vision"), ("Machine Learning", "Data Science"),
        ("Deep Learning", "NLP"), ("Deep Learning", "Computer Vision")
    ]
    
    G.add_edges_from(edges)
    return G


def visualize_graph(graph, highlighted_nodes=None):
    """Create graph visualization."""
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Node positions
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]
    node_text = list(graph.nodes())
    
    # Edge positions
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color='#888'), hoverinfo='none'
    ))
    
    # Add nodes
    node_colors = ['red' if node in (highlighted_nodes or []) else '#4CAF50' 
                   for node in graph.nodes()]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="middle center",
        marker=dict(size=20, color=node_colors),
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Knowledge Graph",
        showlegend=False, hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    
    return fig


def process_query(query):
    """Process user query and return results."""
    time.sleep(0.5)  # Simulate processing
    
    # Simple answer generation
    if "what" in query.lower():
        answer = "Based on document analysis, key concepts and relationships were identified."
    elif "how" in query.lower():
        answer = "The process involves text extraction, analysis, and graph construction."
    else:
        answer = "Analysis complete. Relevant information found in knowledge graph."
    
    # Random traversal path
    traversal_path = random.sample(["AI", "Machine Learning", "Data Science"], 2)
    return answer, traversal_path


def main():
    """Main application."""
    initialize_session_state()
    state = st.session_state.app_state
    
    st.title("📚 Document Assistant")
    st.markdown("Upload documents and ask questions")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", state.stats["documents"])
    with col2:
        st.metric("Graph Nodes", state.stats["nodes"])
    with col3:
        st.metric("Graph Edges", state.stats["edges"])
    with col4:
        st.metric("Queries", state.stats["queries"])
    
    # Main tabs
    tab1, tab2 = st.tabs(["📁 Upload", "🔍 Query"])
    
    with tab1:
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files", type=["pdf"], accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        documents = process_documents(uploaded_files)
                        
                        # Update state
                        state.pipeline_initialized = True
                        state.uploaded_files = uploaded_files
                        state.knowledge_graph = create_knowledge_graph()
                        state.stats.update({
                            "documents": len(uploaded_files),
                            "nodes": len(state.knowledge_graph.nodes()),
                            "edges": len(state.knowledge_graph.edges())
                        })
                        
                        st.success(f"Processed {len(documents)} documents!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        logger.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Query Documents")
        
        if not state.pipeline_initialized:
            st.warning("Please upload documents first.")
            return
        
        query = st.text_area("Enter your question:", height=100)
        
        if st.button("Run Query", type="primary", disabled=not query.strip()):
            with st.spinner("Processing query..."):
                try:
                    answer, traversal_path = process_query(query)
                    
                    # Update history
                    if query not in state.query_history:
                        state.query_history.append(query)
                    state.stats["queries"] += 1
                    
                    # Display results
                    st.subheader("Answer")
                    st.success(answer)
                    
                    # Show graph
                    if state.knowledge_graph:
                        st.subheader("Knowledge Graph")
                        fig = visualize_graph(state.knowledge_graph, traversal_path)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Traversal details
                    with st.expander("Traversal Path"):
                        for i, node in enumerate(traversal_path):
                            st.write(f"{i+1}. {node}")
                
                except Exception as e:
                    st.error(f"Query error: {e}")
                    logger.error(f"Query error: {e}")
        
        # Query history in sidebar
        with st.sidebar:
            st.subheader("Recent Queries")
            for query in state.query_history[-5:]:
                st.write(f"• {query[:30]}...")


if __name__ == "__main__":
    main()