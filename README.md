# Medical Graph RAG: A Comprehensive RAG Pipeline for Knowledge Discovery
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/chrisolande/Medical-Graph-RAG/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Medical Graph RAG is a Python-based project that implements a Retrieval Augmented Generation (RAG) pipeline. It's designed to process documents, build a knowledge graph, and utilize this graph along with vector search and reranking to answer queries based on the ingested information.

## Features

- **Comprehensive Document Processing:** Ingests documents from various sources with focus on medical literature (PubMed Central). Supports Raw PMC JSON and Pre-chunked JSON formats with batch processing for large datasets.
- **Knowledge Graph Construction:** Builds dynamic knowledge graphs representing entities, concepts, and relationships.
- **Advanced Retrieval System:** FAISS vector store integration with reranking (Jina AI, FlashRank) for improved relevance.
- **Retrieval Augmented Generation:** Combines vector search and knowledge graph traversal with LLMs for comprehensive answers.
- **Interactive Streamlit Application:** User-friendly interface with dynamic API key input, pipeline control, real-time graph traversal visualization, conversation history, and quick query suggestions.
- **Semantic Caching:** LLM response caching with configurable similarity thresholds using Langchain's SQLite and FAISS-backed cache.
- **Modular Design:** Highly configurable via environment variables and extensible framework for knowledge discovery.

## Installation

1. **Clone and setup:**
    ```bash
    git clone https://github.com/chrisolande/Medical-Graph-RAG.git
    cd Medical-Graph-RAG
    python3 -m venv venv  # Python 3.11+ required
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2. **Install dependencies (choose one):**
    
    **Poetry (Recommended):**
    ```bash
    pip install poetry
    poetry install --with dev,test  # Or just: poetry install
    ```
    
    **Using uv:**
    ```bash
    pip install uv
    uv pip sync pyproject.toml --all-extras
    ```
    
    **Traditional pip:**
    ```bash
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    pip install -r requirements.txt
    ```

3. **Environment setup:**
    Create `.env` file with your OpenRouter API key:
    ```env
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    ```

## Usage

**Quick Start:**
1. Run the Streamlit application: `streamlit run app.py`
2. Enter API key in sidebar if not in environment
3. Click "Initialize Pipeline" 
4. Load documents (default: `pmc_chunks.json` or upload custom JSON)
5. Query the knowledge graph

**Application Workflow:**

**Sidebar Controls (`:dna: Medical RAG Config`):**
- **API Configuration:** Enter `OPENROUTER_API_KEY` if not in environment
- **Pipeline Control:** Initialize/Reset pipeline
- **Settings:** Toggle semantic caching and adjust similarity threshold
- **Load Documents:** Process default data or upload custom JSON files

**Supported JSON Formats:**
- **Pre-chunked:** `{"documents": [{"content": "...", "metadata": {...}}, ...]}`
- **Raw PMC:** `[{"abstract": "...", ...}, ...]` (auto-chunked)

**Main Interface:**
- **Query Input:** Ask questions with real-time graph traversal visualization
- **Results:** LLM answers, traversal paths, content snippets, and graph statistics
- **Conversation History:** Session-based query/response tracking

## Configuration

**Environment Variables (`.env` file):**
```env
OPENROUTER_API_KEY="required"
LLM_MODEL_NAME="nousresearch/nous-hermes-2-mixtral-8x7b-dpo"  # optional
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"  # optional
```

**Key Backend Settings (`src/medical_graph_rag/core/config.py`):**
- **Models:** `LLM_MODEL_NAME`, `EMBEDDING_MODEL_NAME`, `RERANKER_MODEL_NAME`
- **Paths:** `PERSIST_DIRECTORY`, cache directories for knowledge graph and semantic cache
- **Processing:** `BATCH_SIZE`, `RERANKER_TOP_N`, traversal limits
- **Cache:** Similarity thresholds, max cache size

**Configuration Priority:**
1. Streamlit UI settings (session-specific)
2. Environment variables 
3. `config.py` defaults

## Project Structure

```
src/medical_graph_rag/
├── core/           # Main pipeline, config, utilities
├── data_processing/# Document ingestion, chunking, batch processing
├── knowledge_graph/# Graph building, querying, visualization
└── nlp/           # Vector store, RAG engine, semantic caching

data/
├── input/         # Raw input documents
└── output/        # Processed data (pmc_chunks.json)

app.py             # Streamlit web application
streaming.py       # Real-time graph traversal display
tests/             # Unit and integration tests
```

## Contributing

1. Fork repository and create feature branch
2. Install dev dependencies: `poetry install --with dev,test`
3. Make changes following project style (black, isort)
4. Write tests and update documentation
5. Set up pre-commit hooks: `poetry run pre-commit install`
6. Submit pull request with clear description

## License

MIT License - see `LICENSE` file for details.