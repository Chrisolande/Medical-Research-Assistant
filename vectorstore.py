from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_community.storage import LocalFileStore

from typing import List, Optional
from knowledge_graph import KnowledgeGraph
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os

EMBEDDING_MODEL = "embed-english-light-v3.0"

class VectorStore:
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        embedding_model: Optional[str] = None,
        batch_size: int = 96,
        max_workers = os.cpu_count() - 1,
        cache_type = "file",
        cache_dir = "embedding_cache"
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_type = cache_type
        self.cache_dir = cache_dir
        
        # Cache the embeddings instance
        self._embeddings = None

        # Initialize the vector index
        self.vector_index = None

    @property
    def embeddings(self):
        """Lazy load embeddings with caching to avoid re-computing embeddings"""
        if self._embeddings is None:
            # Create base embeddings
            base_embeddings = CohereEmbeddings(
                model = self.embedding_model,
                cohere_api_key = os.getenv("COHERE_API_KEY1")
            )
            if self.cache_type == "file":
                # Set up caching
                fs = LocalFileStore(self.cache_dir)
                self._embeddings = CacheBackedEmbeddings.from_bytes_store(
                    base_embeddings,
                    fs,
                    namespace=f"cohere_{self.embedding_model}"
                )
            else:
                # No caching
                self._embeddings = base_embeddings

        return self._embeddings

    


    def create_vector_index(
        self,
        documents: List[Document],
        node_label: str = "Document Embeddings",
        text_node_property: list = ["text"],
        embedding_node_property: str = "embedding"
    ):
        """Create vector index from documents"""

        self.vector_index = Neo4jVector.from_documents(
            documents,
            self.embeddings,
            url = self.knowledge_graph.url,
            username = self.knowledge_graph.username,
            password = self.knowledge_graph.password,
            index_name = "vector",
            node_label = node_label,
            text_node_property = text_node_property,
            embedding_node_property = embedding_node_property
        )

    def create_hybrid_index(
        self,
        node_label: str = "Document",
        text_node_properties: List[str] = ["text"],
        embedding_node_property:str = "embedding"
    ):
        """Create hybrid index"""
        self.vector_index = Neo4jVector.from_existing_graph(
            self.embeddings,
            url = self.knowledge_graph.url,
            username = self.knowledge_graph.username,
            password = self.knowledge_graph.password,
            search_type = "hybrid", # Combines both semantic and keyword matching
            node_label = node_label,
            text_node_properties = text_node_properties,
            embedding_node_property = embedding_node_property
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector index"""
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call create_vector_index or create_hybrid_index first.")
        return self.vector_index.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores on the vector index"""
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call create_vector_index or create_hybrid_index first")
        return self.vector_index.similarity_search_with_score(query, k=k)