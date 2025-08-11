# Medical Research Assistant: A Comprehensive RAG Pipeline for Knowledge Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Chrisolande/Medical-Research-Assistant/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![wakatime](https://wakatime.com/badge/user/c31184d5-02d4-4117-8619-29f018aa2840/project/feb670b2-23b5-4fe9-8d32-ed4f8e615b99.svg)](https://wakatime.com/badge/user/c31184d5-02d4-4117-8619-29f018aa2840/project/feb670b2-23b5-4fe9-8d32-ed4f8e615b99)

Medical research assistant is a Python-based project that implements a Retrieval Augmented Generation (RAG) pipeline. It's designed to process documents, build a knowledge graph, and utilize this graph along with vector search and reranking to answer queries based on the ingested information.

## Features

- **Comprehensive Document Processing:** Ingests documents from various sources with focus on medical literature (PubMed Central). Supports Raw PMC JSON and Pre-chunked JSON formats with batch processing for large datasets.
- **Knowledge Graph Construction:** Builds dynamic knowledge graphs representing entities, concepts, and relationships.
- **Advanced Retrieval System:** FAISS vector store integration with reranking (Jina AI, FlashRank) for improved relevance.
- **Pre-built FAISS Index:** Includes a pre-built FAISS index (`faiss_index/`) for immediate use, based on the default dataset (`data/output/processed_pmc_data/pmc_chunks.json`), saving setup time.
- **Retrieval Augmented Generation:** Combines vector search and knowledge graph traversal with LLMs for comprehensive answers.
- **Interactive Streamlit Application:** User-friendly interface with dynamic API key input, pipeline control, real-time graph traversal visualization, conversation history, and quick query suggestions.
- **Semantic Caching:** LLM response caching with configurable similarity thresholds using Langchain's SQLite and FAISS-backed cache.
- **Modular Design:** Highly configurable via environment variables and extensible framework for knowledge discovery.

## Demo

[![Video Title](https://img.youtube.com/vi/euVsXqd9A5c/maxresdefault.jpg)](https://youtu.be/euVsXqd9A5c)

## Installation

1. **Clone and setup:**

    ```bash
    git clone https://github.com/Chrisolande/Medical-Research-Assistant.git
    cd medical_graph_rag
    python3 -m venv venv  # Python 3.11+ required
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2. **Install dependencies:**

    **Using uv (Recommended):**
    ```bash
    pip install uv
    uv sync --all-extras
    ```

    **Using pip:**
    ```bash
    pip install -e ".[dev,test]"
    ```

3. **Environment setup:**
    Create a `.env` file in the root directory. See `.env.example` for required variables.

## Usage

**Quick Start:**

1. Run the Streamlit application: `streamlit run app.py`
2. Enter your API key in the sidebar (if not set in `.env`)
3. Click "Initialize Pipeline" - the pre-built FAISS index allows immediate querying
4. Start asking questions or load custom documents

**Application Interface:**

- **Sidebar (`:dna: Medical RAG Config`):** API configuration, pipeline control, settings, and document loading
- **Main Interface:** Query input with real-time graph visualization, LLM answers, and conversation history

## Configuration

Configuration is managed through environment variables (`.env` file) and defaults in `src/medical_graph_rag/core/config.py`.

**Priority order:**
1. Streamlit UI settings (session-specific)
2. Environment variables
3. Config defaults

Key settings include model names (`LLM_MODEL_NAME`, `EMBEDDING_MODEL_NAME`), paths, and processing parameters.

## Project Structure

```text
.
├── .env.example
├── .github/
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── data/
│   ├── input/
│   └── output/
├── faiss_index/
├── pyproject.toml
├── pytest.ini
├── src/
│   └── medical_graph_rag/
│       ├── core/
│       ├── data_processing/
│       ├── knowledge_graph/
│       └── nlp/
├── streaming.py
├── tests/
└── uv.lock
```

## Contributing

1. Fork the repository and create a feature branch
2. Install development dependencies: `uv sync --all-extras`
3. Make your changes, following the project's style (black, isort)
4. Write tests and update documentation as needed
5. Set up pre-commit hooks: `pre-commit install`
6. Submit a pull request with a clear description of your changes

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
