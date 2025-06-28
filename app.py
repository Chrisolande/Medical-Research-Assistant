import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
