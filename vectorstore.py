"""Based on this documentation from langchain https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.neo4j_vector.Neo4jVector.html
It turns out neo4j has async method"""

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_community.storage import LocalFileStore
from langchain_community.embeddings import CacheBackedEmbeddings

from typing import List, Optional
from knowledge_graph import KnowledgeGraph
import asyncio
from asyncio import Semaphore
import time
import os

EMBEDDING_MODEL = "embed-english-light-v3.0"

class AdaptiveBatcher:
    def __init__(self, initial_batch_size:int = 96, min_batch_size: int = 16, max_batch_size:int = 96):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.latency_history = []
        self.target_latency = 5.0  # Target 5 seconds per batch
        self.adjustment_factor = 0.2  # 20pc adjustments
        self.max_history_length = 5
        
    def adjust_batch_size(self, latency: float):
        """Adjust based on observed latency"""
        self.latency_history.append(latency)

        # Keep only the last 5 history
        if len(self.latency_history) > self.max_history_length:
            self.latency_history.pop(0)

        avg_latency = sum(self.latency_history) / len(self.latency_history)

        if avg_latency > self.target_latency * 1.2: # Too slow, reduce batch size
            new_size = int(self.current_batch_size * (1 - self.adjustment_factor))
            self.current_batch_size = max(new_size, self.min_batch_size)
            print(f"Batch too slow ({avg_latency:.2f}s). Reducing batch size to {self.current_batch_size}")
        
        elif avg_latency < self.target_latency * 0.8:  # Too fast, increase batch size
            new_size = int(self.current_batch_size * (1 + self.adjustment_factor))
            self.current_batch_size = min(new_size, self.max_batch_size)
            print(f"Batch too fast ({avg_latency:.2f}s). Increasing batch size to {self.current_batch_size}")
            
        return self.current_batch_size
class VectorStore:
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        embedding_model: Optional[str] = None,
        initial_batch_size: int = 96,
        max_concurrent: int = 15,
        cache_type = "file",
        cache_dir = "embedding_cache",
        adaptive_batching_enabled:bool = True
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.initial_batch_size = initial_batch_size
        self.cache_type = cache_type
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        self.adaptive_batching_enabled = adaptive_batching_enabled

        #Initialize adaptive batching if enabled
        self.batcher = AdaptiveBatcher(initial_batch_size=self.batch_size) if adaptive_batching else None # Use fixed batching as the fallback
        
        # Cache the embeddings instance
        self._embeddings = None

        # Initialize the vector index
        self.vector_index = None

        self.semaphore = Semaphore(self.max_concurrent)  # For concurrency

    @property # ensures that the CohereEmbeddings instance is only created when it's first accessed, rather than during object initialization. "
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

    def _get_current_batch_size(self):
        """Get the current batch size either adaptive or fixed batching"""
        return self.batcher.current_batch_size if self.adaptive_batching_enabled else self.initial_batch_size

    async def create_vector_index(
        self,
        documents: List[Document],
        node_label: str = "Document Embeddings",
        text_node_property: list = ["text"],
        embedding_node_property: str = "embedding"
    ):
        """Create vector index from documents with batching"""
        if not documents:
            print("No documents to process")
            return
        
        
        current_batch_size = self._get_current_batch_size()

        if len(documents) > current_batch_size:
            async with semaphore:

                return await self._create_vector_index_batched(
                    documents, node_label, text_node_property, embedding_node_property
                )
        
        self.vector_index = await Neo4jVector.afrom_documents(
            documents,
            self.embeddings,
            url=self.knowledge_graph.url,
            username=self.knowledge_graph.username,
            password=self.knowledge_graph.password,
            index_name="vector",
            node_label=node_label,
            text_node_property=text_node_property,
            embedding_node_property=embedding_node_property
        )

    async def _create_vector_index_batched(self,
            documents:List[Document],
            node_label: str,
            text_node_property:list,
            embedding_node_property:str
        ):
            """Process documents in batches for faster processing"""
            semaphore = self.semaphore

            current_batch_size = self._get_current_batch_size()

            # Create the first batch and initialize the vector with it
            first_batch = documents[:current_batch_size]
            start_time = time.time()

            self.vector_index = await Neo4jVector.afrom_documents(
                    first_batch,
                    self.embeddings,
                    url = self.knowledge_graph.url,
                    username = self.knowledge_graph.username,
                    password = self.knowledge_graph.password,
                    index_name = "vector",
                    node_label = node_label,
                    text_node_property = text_node_property,
                    embedding_node_property = embedding_node_property
                )

            # Track latency and adjust if need be
            if self.adaptive_batching:
                latency = time.time() - start_time
                current_batch_size = self.batcher.adjust_batch_size(latency)
                print(f"Initial batch ({len(first_batch)} docs) processed in {latency:.2f}s. New batch size: {self.batcher.current_batch_size}")

            remaining_documents = documents[current_batch_size:]
            if not remaining_documents:
                print("All documents processed in the initial batch.")
                return

            # Process remaining documents with adaptive batching
            async def process_batch_adaptive(batch: List[Document]):
                async with semaphore:
                    start_time = time.time()
                    await self.vector_index.aadd_documents(batch)
                    
                    # Adjust batch size based on latency
                    if self.adaptive_batching:
                        latency = time.time() - start_time
                        self.batcher.adjust_batch_size(latency)

            # Create batches dynamically
            tasks = []
            i = 0
            while i < len(remaining_documents):
                current_batch_size = self._get_current_batch_size()
                batch = remaining_documents[i:i + current_batch_size]
                tasks.append(process_batch_adaptive(batch))
                i += current_batch_size

            # Run all tasks
            await asyncio.gather(*tasks, return_exceptions=True)

            
            remaining_batches = []

            # Create the batches list for concurrent processing
            for i in range(current_batch_size, len(documents), current_batch_size):
                batch = documents[i:i + current_batch_size]
                remaining_batches.append(batch)

            # Return early if no remaining batches
            if not remaining_batches:
                return

            # Process asynchronously
            async def process_batch(batch: List[Document]):
                async with semaphore:
                    await self.vector_index.aadd_documents(batch)

            # Create tasks, run everything asynchronously
            tasks = [process_batch(batch) for batch in remaining_batches]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def create_hybrid_index(
        self,
        node_label: str = "Document Embedding",
        text_node_properties: List[str] = ["text"],
        embedding_node_property:str = "embedding"
    ):
        """Create hybrid index"""
        self.vector_index = await Neo4jVector.afrom_existing_graph(
                self.embeddings,
                url = self.knowledge_graph.url,
                username = self.knowledge_graph.username,
                password = self.knowledge_graph.password,
                search_type = "hybrid", # Combines both semantic and keyword matching
                node_label = node_label,
                text_node_properties = text_node_properties,
                embedding_node_property = embedding_node_property
            )
   
    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector index"""
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call create_vector_index or create_hybrid_index first.")
        return await self.vector_index.asimilarity_search(query, k=k)

    async def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores on the vector index"""
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call create_vector_index or create_hybrid_index first")
        return await self.vector_index.asimilarity_search_with_score(query, k=k)

    async def query(self, queries: List[str], k = 4):
        """Perform similarity search"""
        if self.vector_index is None:
            raise ValueError("Vector Index is not initialized")

        tasks = [self.similarity_search(query, k) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def __aenter__(self):
        """Add a simple context manager"""
        return self

    async def __aexit__(self):
        """Clean up resources"""
        if hasattr(self.vector_index, 'close'):
            self.vector_index.close()