# config.py
"""Config module."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

_ = load_dotenv()
# Dataclass Definitions


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization parameters."""

    figure_size: tuple[int, int] = (16, 12)
    node_size: int = 3000
    edge_width: int = 2
    traversal_edge_width: int = 3
    font_size: int = 8
    curve_radius: float = 0.3
    layout_iterations: int = 50
    layout_k: float = 1
    max_label_length: int = 40
    content_preview_length: int = 200


@dataclass
class NodeStyle:
    """Style configuration for different node types in graph visualization."""

    regular: str = "lightblue"
    start: str = "lightgreen"
    end: str = "lightcoral"
    visited: str = "gold"


@dataclass
class EdgeStyle:
    """Style configuration for different edge types in graph visualization."""

    regular_color: str = "blue"
    traversal_color: str = "red"
    traversal_style: str = "--"


# Model Names

EMBEDDING_MODEL_NAME = "abhinand/MedEmbed-small-v0.1"
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v1-turbo-en"
FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
LLM_MODEL_NAME = "deepseek-chat"


# Paths & Directories

# General vector store persistence directory
PERSIST_DIRECTORY = "faiss_index"
# Cache directory for FlashRank models
FLASHRANK_CACHE_DIR = os.path.expanduser("~/.cache/flashrank")


# API Keys/Bases

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# Reranker Settings

RERANKER_TOP_N = 4


# VectorStore Settings

# Batch size for processing documents into the vector store
BATCH_SIZE = 500
# Maximum concurrent operations for vector store updates/processing
MAX_CONCURRENT = 10
# Maximum tokens for LLM interactions
LLM_MAX_TOKENS = 4000


# Semantic Cache Configuration

DUMMY_DOC_CONTENT = "Langchain Document Initializer"
DEFAULT_DATABASE_PATH = ".langchain.db"
DEFAULT_FAISS_INDEX_PATH = "../semantic_cache_index"


DEFAULT_MAX_DISTANCE_THRESHOLD = 0.4
SEMANTIC_CACHE_TOP_K = 5

DEFAULT_MAX_CACHE_SIZE = 1000
DEFAULT_MEMORY_CACHE_SIZE = 100
DEFAULT_BATCH_SIZE = 10
ENABLE_QUANTIZATION = False


# PMCBatchProcessor Settings

PMC_BATCH_SIZE = 96
PMC_MAX_CONCURRENT_BATCHES = 3
PMC_RETRY_ATTEMPTS = 2
PMC_RETRY_DELAY = 1.0
PMC_INTER_BATCH_DELAY = 0.1  # Delay between batches
MIN_ABSTRACT_CONTENT_LENGTH = 50  # Minimum content length for a valid document abstract


# Query Engine Settings
MIN_NODES_TO_TRAVERSE = 5  # Default minimum nodes to traverse for QueryEngine
MAX_NODES_TO_TRAVERSE = 25
LLM_MAX_CONTEXT_LENGTH = 4000  # Max context length for the LLM
ANSWER_CHECK_INTERVAL = 3
GRAPH_EDGE_SIMILARITY_THRESHOLD = 0.8
USE_LLM_CHAIN_EXTRACTOR = False


# NER Concept Filtering

BIOMEDICAL_ENTITY_GROUPS = frozenset(
    {
        "DISEASE",
        "CHEMICAL",
        "SPECIES",
        "DNA",
        "CELL_TYPE",
        "CELL_LINE",
        "RNA",
        "PROTEIN",
    }
)
MIN_CONCEPT_LENGTH = 3


# PubMed Settings

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
PUBMED_DEFAULT_MAX_RESULTS = 50
PUBMED_MAX_RESULTS_LIMIT = 200
