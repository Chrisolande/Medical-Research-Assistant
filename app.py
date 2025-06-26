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