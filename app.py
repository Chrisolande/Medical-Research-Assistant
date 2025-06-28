import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from medical_graph_rag.core.main import Main

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration."""

    cache_dir: str = "my_cache"
    faiss_index_dir: str = "/home/olande/Desktop/FinalRAG/faiss_index"


# TODO: Implement a class to handle the document uploads either by importing the document processing class or building a new one


class StreamlitApp:
    def __init__(self):
        self.config = AppConfig()
        self.main_engine: Optional | Main = None

    def _check_cache_exists(self):
        faiss_exists = Path(self.config.faiss_index_dir).exists()
        cache_exists = Path(self.config.cache_dir).exists()
        return cache_exists, faiss_exists

    async def _initialize_engine(self):
        try:
            if self.main_engine is None:
                self.main_engine = Main(cache_dir=self.config.cache_dir)

            # Try loading the cache first
            cache_loaded = await self.main_engine.load_from_cache()
            return cache_loaded
        except Exception as e:
            logger.error(f"Engine initialization failed: {str(e)}")
            st.error(f"Failed to initialize RAG engine: {str(e)}")
            return False

    async def _process_and_build(self, documents: list[dict[str, Any]]) -> bool:
        try:
            with st.spinner("Building knowledge graph and vector store ..."):
                await self.main_engine.process_documents(documents)
            st.success("Documents processed successfully!")
            return True
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            st.error(f"Failed to process documents: {str(e)}")
            return False

    async def execute_query(self, query):
        try:
            with st.spinner("Processing query..."):
                result = await self.main_engine.query(query)
            return result
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            st.error(f"Query failed: {str(e)}")
            return None

    def _render_sidebar(self):
        st.sidebar.title("System status")

        faiss_exists, cache_exists = self._check_cache_exists()

        # Cache status
        st.sidebar.subheader("Cache Status")
        st.sidebar.write(f"FAISS Index: {'✅' if faiss_exists else '❌'}")
        st.sidebar.write(f"Knowledge Graph: {'✅' if cache_exists else '❌'}")

        # Environment check
        st.sidebar.subheader("Environment")
        api_key_set = bool(os.getenv("OPENROUTER_API_KEY"))
        st.sidebar.write(f"API Key: {'✅' if api_key_set else '❌'}")

    def _render_query_interface(self):
        st.subheader("Query Interface")
        st.divider()

        query = st.text_input(
            "Enter your query:",
            placeholder="Ask a question about your documents...",
            help="Enter a natural language query to search through your documents",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            query_button = st.button("Query", type="primary", use_container_width=True)

        if query_button and query.strip():
            return query.strip()
        elif query_button:
            st.warning("Please enter a query")

        return None

    def _render_upload_interface(self):
        st.subheader("Document Upload")
        st.divider()

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF files to build a new knowledge base",
        )

        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                mb_size = file.size / (1024 * 1024)
                st.write(f"- {file.name} ({mb_size:.2f} MB)")

            if st.button("Process Documents", type="primary"):
                return uploaded_files

            return None

    def _render_results(self, response, traversal_path, filtered_content):
        st.subheader("Response")
        st.divider()

        if traversal_path:
            with st.expander("Traversal Path", expanded=False):
                st.json(traversal_path)

        if filtered_content:
            with st.expander("Filtered Content", expanded=False):
                st.json(filtered_content)

    async def run(self):
        st.set_page_config(
            page_title="Medical RAG System",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🏥 Medical RAG System")
        st.markdown("Retrieval-Augmented Generation for Medical Documents")

        # Render sidebar
        self._render_sidebar()

        # Initialize the engine
        if "engine_initialized" not in st.session_state:
            with st.spinner("Initializing RAG engine..."):
                cache_loaded = await self._initialize_engine()
                st.session_state.engine_initialized = True
                st.session_state.cache_available = cache_loaded
                if cache_loaded:
                    st.success("Loaded existing knowledge base from cache!")
                else:
                    st.info(
                        "No cache found. Please upload documents to build knowledge base."
                    )

        # Main interface tabs
        if st.session_state.get("cache_available", False):
            tab1, tab2 = st.tabs(["Query System", "Upload New Documents"])
            with tab1:
                query = self._render_query_interface()
                if query:
                    result = await self._execute_query(query)
                    if result:
                        response, traversal_path, filtered_content = result
                        self._render_results(response, traversal_path, filtered_content)

            with tab2:
                uploaded_files = self._render_upload_interface()
                if uploaded_files:
                    documents = await self.doc_handler.process_uploaded_files(
                        uploaded_files
                    )
                    if documents:
                        success = await self._process_and_build(documents)
                        if success:
                            st.session_state.cache_available = True
                            st.rerun()
        # self._render_query_interface()
        # self._render_upload_interface()
        # self._render_results(response, traversal_path, filtered_content)


async def main():
    """Application entry point."""
    app = StreamlitApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
