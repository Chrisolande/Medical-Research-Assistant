from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_community.storage import LocalFileStore

from typing import List, Optional
from knowledge_graph import KnowledgeGraph
from concurrent.futures import ThreadPoolExecutor
import asyncio
from asyncio import Semaphore
import os

EMBEDDING_MODEL = "embed-english-light-v3.0"

class VectorStore:
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        embedding_model: Optional[str] = None,
        batch_size: int = 96,
        max_workers = os.cpu_count() - 1,
        max_concurrent: int = 15,
        cache_type = "file",
        cache_dir = "embedding_cache"
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_type = cache_type
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        
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

    async def create_vector_index(
        self,
        documents: List[Document],
        node_label: str = "Document Embeddings",
        text_node_property: list = ["text"],
        embedding_node_property: str = "embedding"
    ):
        """Create vector index from documents with batching"""
        semaphore = Semaphore(self.max_concurrent)

        if len(documents) > self.batch_size:
            async with semaphore:

                return await self._create_vector_index_batched(
                    documents, node_label, text_node_property, embedding_node_property
                )
        # Wrap the synchronous vector index call in an executor to make it async
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.vector_index = await loop.run_in_executor(
                executor,

                lambda: Neo4jVector.from_documents(
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
            )

    async def _create_vector_index_batched(self,
            documents:List[Document],
            node_label: str,
            text_node_property:list,
            embedding_node_property:str
        ):
            """Process documents in batches for faster processing"""
            semaphore = Semaphore(self.max_concurrent)
            # Create the first batch and initialize the vector with it
            first_batch = documents[:self.batch_size]

            # Loop the sync vector index call to make it async as before
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers = self.max_workers) as executor:
                self.vector_index = await loop.run_in_executor(
                    executor,
                    lambda: Neo4jVector.from_documents(
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
                )

            remaining_batches = []

            # Create the batches list for concurrent processing
            for i in range(self.batch_size, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                remaining_batches.append(batch)

            # Return early if no remaining batches
            if not remaining_batches:
                return

            # Process asynchronously
            async def process_batch(batch: List[Document]):
                async with semaphore:
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        await loop.run_in_executor(
                            executor,
                             lambda: self.vector_index.add_documents(batch)
                        )

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
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers = self.max_workers) as executor:
            self.vector_index = await loop.run_in_executor(
                executor,
                lambda: Neo4jVector.from_existing_graph(
                self.embeddings,
                url = self.knowledge_graph.url,
                username = self.knowledge_graph.username,
                password = self.knowledge_graph.password,
                search_type = "hybrid", # Combines both semantic and keyword matching
                node_label = node_label,
                text_node_properties = text_node_properties,
                embedding_node_property = embedding_node_property
            )
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

    def query(self, queries: List[str], k = 4):
        """Perform similarity search"""
        if self.vector_index is None:
            raise ValueError("Vector Index is not initialized")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.similarity_search, query, k) for query in queries]
            return [future.result() for future in futures]

    def __enter__(self):
        """Add a simple context manager"""
        return self

    def __exit__(self):
        """Clean up resources"""
        if hasattr(self.vector_index, 'close'):
            self.vector_index.close()