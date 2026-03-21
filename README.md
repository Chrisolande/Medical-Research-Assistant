# Medical RAG Knowledge Graph Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/Chrisolande/Medical-Research-Assistant/actions/workflows/main.yml/badge.svg)](https://github.com/Chrisolande/Medical-Research-Assistant/actions/workflows/main.yml)
[![Code Quality](https://github.com/Chrisolande/Medical-Research-Assistant/actions/workflows/code-quality.yml/badge.svg)](https://github.com/Chrisolande/Medical-Research-Assistant/actions/workflows/code-quality.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A RAG pipeline that combines a biomedical knowledge graph with vector search to answer medical research questions. Documents are embedded, linked by semantic similarity and shared biomedical entities, and traversed with a priority-based graph algorithm that expands context until the LLM confirms a sufficient answer.

## Demo

[![Demo video](https://img.youtube.com/vi/euVsXqd9A5c/maxresdefault.jpg)](https://youtu.be/euVsXqd9A5c)

---

## How it works

When you submit a query:

1. **Retrieval** - FAISS finds the most relevant document chunks using `abhinand/MedEmbed-small-v0.1` embeddings. A cross-encoder reranker (`jinaai/jina-reranker-v1-turbo-en`) re-scores the top 20 results and returns the top 4.

2. **Graph traversal** - Starting from the nodes closest to the retrieved documents, a priority queue explores the knowledge graph. Neighbours are penalised if their biomedical concepts are already covered by visited nodes, steering traversal toward new information.

3. **Sufficiency check** - Every 3 nodes (after a minimum of 8), DeepSeek evaluates whether the accumulated context is sufficient to answer the query. Traversal stops early if it is, or continues up to 25 nodes.

4. **Answer generation** - If early stopping triggers, the synthesised answer is returned directly. Otherwise, the full accumulated context is passed to DeepSeek for a final answer.

The knowledge graph is built once per dataset. Nodes are document chunks; edges connect chunks that exceed 0.8 cosine similarity and share at least one biomedical named entity. Concept extraction uses `d4data/biomedical-ner-all`, filtered to eight entity groups (DISEASE, CHEMICAL, SPECIES, DNA, CELL_TYPE, CELL_LINE, RNA, PROTEIN). MeSH terms from PubMed articles are merged in when available, giving higher-quality concept coverage than NER alone.

---

## Features

- **Live PubMed search** - search by topic, fetch abstracts and MeSH terms via the NCBI Entrez API, and ingest directly into the pipeline without any file uploads
- **Pre-built FAISS index** - the repo ships with a pre-built index so you can query immediately after initialization
- **Semantic response cache** - a three-tier cache (memory, SQLite, FAISS semantic search) avoids redundant LLM calls for similar queries
- **Real-time graph traversal display** - the sidebar shows each node as it is visited during traversal
- **Traversal-scoped statistics** - graph statistics report only the subgraph actually visited, not the full corpus graph
- **Configurable retrieval pipeline** - reranking, LLM chain extraction, edge threshold, traversal depth, and cache distance are all tunable via config or the UI

---

## Project structure

```
Medical-Research-Assistant/
├── app.py                        # Streamlit application
├── streaming.py                  # Real-time traversal display
├── medical_graph_rag/
│   ├── config.py                 # All constants and tunable parameters
│   ├── pipeline.py               # Main orchestrator (Main class)
│   ├── graph.py                  # Knowledge graph construction and traversal
│   ├── graph_viz.py              # Matplotlib graph visualization
│   ├── rag_chain.py              # QueryEngine: traversal algorithm and LLM calls
│   ├── vectorstore.py            # FAISS index, reranking, retrieval pipeline
│   ├── cache.py                  # Three-tier semantic response cache
│   ├── batch_processor.py        # Async batch document processing
│   ├── document_processor.py     # Text splitting and metadata extraction
│   └── pubmed_downloader.py      # NCBI Entrez API client
├── data/
│   └── output/processed_pmc_data/
│       └── pmc_chunks.json       # Default pre-chunked dataset
├── faiss_index/                  # Pre-built FAISS index
└── tests/                        # pytest test suite
```

---

## Installation

**Requirements:** Python 3.11+, a DeepSeek API key

```bash
git clone https://github.com/Chrisolande/Medical-Research-Assistant.git
cd Medical-Research-Assistant
```

**Using uv (recommended):**

```bash
pip install uv
uv sync --all-extras
```

**Using pip:**

```bash
pip install -e ".[dev,test]"
```

**Environment setup:**

```bash
cp .env.example .env
# Edit .env and add your keys
```

`.env.example`:

```
DEEPSEEK_API_KEY=your_deepseek_key_here
NCBI_EMAIL=your@email.com
```

`NCBI_EMAIL` is required only if you want to use the live PubMed search feature. Any valid email address works; it is sent to NCBI in request headers as required by their API terms of service.

---

## Usage

```bash
streamlit run app.py
```

### Option A - Query immediately with the pre-built index

1. Click **Initialize Pipeline** in the sidebar
2. Click **Load pmc_chunks.json** to load the default dataset
3. Type a question and press Enter

### Option B - Search PubMed and build a custom knowledge base

1. Click **Initialize Pipeline**
2. Enter your NCBI email in the **Search PubMed** section
3. Type a search topic (e.g. `COVID-19 long term effects children`)
4. Set the number of articles and click **Fetch and Ingest**
5. Wait for graph construction to complete, then query

### Option C - Upload your own data

Upload a JSON file in the **Upload Custom JSON** section. Two formats are supported:

**Pre-chunked format:**

```json
{
  "documents": [
    {"content": "Abstract text...", "metadata": {"pmid": "12345", "title": "..."}}
  ]
}
```

**Raw PMC format:**

```json
[
  {"abstract": "Abstract text...", "pmid": "12345", "title": "..."}
]
```

---

## Configuration

All tunable parameters live in `medical_graph_rag/config.py`. Key values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL_NAME` | `deepseek-chat` | DeepSeek model |
| `EMBEDDING_MODEL_NAME` | `abhinand/MedEmbed-small-v0.1` | Biomedical embedding model |
| `GRAPH_EDGE_SIMILARITY_THRESHOLD` | `0.8` | Minimum cosine similarity to create a graph edge |
| `MIN_NODES_TO_TRAVERSE` | `8` | Nodes visited before sufficiency checks begin |
| `MAX_NODES_TO_TRAVERSE` | `25` | Hard traversal limit |
| `ANSWER_CHECK_INTERVAL` | `3` | How often the LLM checks for a sufficient answer |
| `RERANKER_TOP_N` | `4` | Documents returned after reranking |
| `USE_LLM_CHAIN_EXTRACTOR` | `False` | Enable LLM-based sentence extraction in retrieval |
| `DEFAULT_MAX_DISTANCE_THRESHOLD` | `0.4` | L2 distance ceiling for semantic cache hits |
| `PUBMED_DEFAULT_MAX_RESULTS` | `50` | Default article count for PubMed searches |

The Streamlit sidebar also exposes cache on/off and the semantic distance threshold at runtime without needing to edit files.

---

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=medical_graph_rag --cov-report=html

# Lint and format
ruff check . --fix && ruff format .

# Security scan
bandit -r medical_graph_rag/ app.py -ll --skip B101,B403,B601

# Install pre-commit hooks
pre-commit install
```

---

## Models used

| Model | Purpose | Source |
|-------|---------|--------|
| `deepseek-chat` | Answer generation and sufficiency checking | DeepSeek API |
| `abhinand/MedEmbed-small-v0.1` | Document and query embeddings | HuggingFace |
| `d4data/biomedical-ner-all` | Biomedical named entity extraction | HuggingFace |
| `jinaai/jina-reranker-v1-turbo-en` | Cross-encoder reranking | HuggingFace |
| `ms-marco-MiniLM-L-12-v2` | FlashRank reranking | HuggingFace |

---

## Contributing

1. Fork the repository and create a feature branch
2. Install development dependencies: `uv sync --all-extras`
3. Make your changes following the project style (ruff)
4. Write tests and update documentation as needed
5. Set up pre-commit hooks: `pre-commit install`
6. Submit a pull request with a clear description of your changes

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
