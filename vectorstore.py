"""Based on this documentation from langchain https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.neo4j_vector.Neo4jVector.html
It turns out neo4j has async method"""

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from typing import List, Optional
from knowledge_graph import KnowledgeGraph
import asyncio
from asyncio import Semaphore
import time
import os
from tqdm.asyncio import tqdm
from math import ceil

EMBEDDING_MODEL = "embed-english-light-v3.0"

# TODO: Implement the reranking using cohere
class AdaptiveBatcher:
    def __init__(self, initial_batch_size:int = 96, min_batch_size: int = 16, max_batch_size:int = 96):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.latency_history = []
        self.target_latency = 30.0  # Target 5 seconds per batch
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
            old_size = self.current_batch_size
            new_size = int(self.current_batch_size * (1 - self.adjustment_factor))
            self.current_batch_size = max(new_size, self.min_batch_size)
            if self.current_batch_size != old_size:
                print(f"Batch too slow ({avg_latency:.2f}s). Reducing batch size to {self.current_batch_size}")
        
        elif avg_latency < self.target_latency * 0.8:  # Too fast, increase batch size
            old_size = self.current_batch_size
            new_size = int(self.current_batch_size * (1 + self.adjustment_factor))
            self.current_batch_size = min(new_size, self.max_batch_size)
            if self.current_batch_size != old_size:
                print(f"Batch too fast ({avg_latency:.2f}s). Increasing batch size to {self.current_batch_size}")
            
        return self.current_batch_size 

class VectorStore:
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        embedding_model: Optional[str] = None,
        initial_batch_size: int = 96,
        max_concurrent: int = 10,
        cache_type = "file",
        cache_dir = "embedding_cache",
        adaptive_batching_enabled:bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.initial_batch_size = initial_batch_size
        self.cache_type = cache_type
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        self.adaptive_batching_enabled = adaptive_batching_enabled
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        #Initialize adaptive batching if enabled
        self.batcher = AdaptiveBatcher(initial_batch_size=self.initial_batch_size) if adaptive_batching_enabled else None # Use fixed batching as the fallback

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
        return self.batcher.current_batch_size if self.adaptive_batching_enabled and self.batcher else self.initial_batch_size

    async def _retry_operation(self, operation, *args, **kwargs):
        """Retry database operations with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                if "SessionExpired" in str(e) or "TimeoutError" in str(e) or "ConnectionError" in str(e):
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Connection failed (attempt {attempt + 1}/{self.max_retries}). Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                raise e
        raise Exception(f"Operation failed after {self.max_retries} attempts")

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

        # Filter out documents with empty or None text content
        valid_documents = [
            doc for doc in documents 
            if doc.page_content and doc.page_content.strip() and len(doc.page_content.strip()) > 0
        ]
        if len(valid_documents) != len(documents):
            print(f"Filtered out {len(documents) - len(valid_documents)} documents with empty text content")
        
        if not valid_documents:
            print("No valid documents to process")
            return

        current_batch_size = self._get_current_batch_size()

        if len(valid_documents) >= current_batch_size:
            return await self._create_vector_index_batched(
                valid_documents, node_label, text_node_property, embedding_node_property
            )

        print(f"Processing all {len(valid_documents)} documents in a single batch...")
        self.vector_index = await Neo4jVector.afrom_documents(
            valid_documents,
            self.embeddings,
            url=self.knowledge_graph.url,
            username=self.knowledge_graph.username,
            password=self.knowledge_graph.password,
            index_name="vector",
            node_label=node_label,
            text_node_property=text_node_property,
            embedding_node_property=embedding_node_property
        )
        print("Single batch processing complete.")

    async def cleanup_empty_nodes(self):
        """Remove nodes with empty or null text properties"""
        try:
            cleanup_query = """
            MATCH (n:`Document Embeddings`) 
            WHERE n.text IS NULL OR n.text = '' OR trim(n.text) = ''
            DETACH DELETE n
            """
            result = self.knowledge_graph.query(cleanup_query)
            print(f"Cleaned up nodes with empty text properties")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    async def _create_vector_index_batched(self,
            documents:List[Document],
            node_label: str,
            text_node_property:list,
            embedding_node_property:str
        ):
            """Process documents in batches for faster processing with tqdm progress bar"""

            current_batch_size = self._get_current_batch_size()

            valid_documents = [
                doc for doc in documents 
                if doc.page_content and doc.page_content.strip() and len(doc.page_content.strip()) > 0
            ]
            if len(valid_documents) != len(documents):
                print(f"Filtered out {len(documents) - len(valid_documents)} documents with empty text content")
            
            if not valid_documents:
                print("No valid documents to process in batches.")
                return

            index_exists = False
            try:
                # Try to connect to existing index
                print("Checking for existing vector index...")
                self.vector_index = Neo4jVector.from_existing_index(
                    self.embeddings,
                    url=self.knowledge_graph.url,
                    username=self.knowledge_graph.username,
                    password=self.knowledge_graph.password,
                    index_name="vector",
                    text_node_property=text_node_property[0]  # Use first property
                )
                print("Connected to existing vector index.")
                index_exists = True
            except Exception as e:
                print(f"Vector index does not exist or connection failed: {e}")
                index_exists = False

            if index_exists:
                existing_docs = await self._get_existing_document_ids()
                valid_new_documents = [doc for doc in valid_documents if self._get_doc_id(doc) not in existing_docs]
                print(f"Found {len(existing_docs)} existing documents. Processing {len(valid_new_documents)} new documents.")
                
                if len(valid_new_documents) == 0:
                    print("No new documents to add to the existing index. Loading complete.")
                    return
                
                remaining_documents = valid_new_documents
            else:
                # Create new index if none exists
                first_batch = valid_documents[:current_batch_size]

                if not first_batch:
                    print("No documents to process in the initial batch for new index creation.")
                    return

                print(f"Creating new vector index with the first {len(first_batch)} documents...")

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
                if self.adaptive_batching_enabled:
                    latency = time.time() - start_time
                    current_batch_size = self.batcher.adjust_batch_size(latency)
                    print(f"Initial batch ({len(first_batch)} docs) processed in {latency:.2f}s. Next batch size: {self._get_current_batch_size()}")
                else:
                    print(f"Initial batch ({len(first_batch)} docs) processed in {time.time() - start_time:.2f}s.")

                # Set remaining documents after processing first batch
                remaining_documents = valid_documents[current_batch_size:]

            if not remaining_documents:
                print("All documents processed in the initial batch.")
                return

            print(f"Processing remaining {len(remaining_documents)} documents in batches...")

            # Process remaining documents with batching (adaptive or fixed)
            if self.adaptive_batching_enabled:
                # Adaptive batching with dynamic batch sizes
                async def process_batch_adaptive(batch: List[Document]):
                    async with self.semaphore:
                        start_time = time.time()
                        
                        async def add_batch():
                            await self.vector_index.aadd_documents(batch)
        
                        # Add retry logic for adaptive batch processing
                        await self._retry_operation(add_batch)

                        latency = time.time() - start_time
                        self.batcher.adjust_batch_size(latency)
                        return len(batch)

                # Create batches dynamically
                tasks = []
                i = 0
                while i < len(remaining_documents):
                    # Always get the latest batch size from the batcher
                    current_batch_size = self._get_current_batch_size()
                    batch = remaining_documents[i:i + current_batch_size]
                    if not batch:
                        break
                    tasks.append(process_batch_adaptive(batch))
                    i += current_batch_size
                
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), unit="batch", desc="Adding documents to vector index"):
                    await f

            else:
                # Fix batching with tqdm
                async def process_batch_fixed(batch: List[Document]):
                    async with self.semaphore:
                        async def add_batch():
                            await self.vector_index.aadd_documents(batch)
                        
                        # Add retry logic for batch processing
                        await self._retry_operation(add_batch)
                    return len(batch)

                # Create all batches upfront since batch size is fixed
                tasks = []
                for i in range(0, len(remaining_documents), current_batch_size):
                    batch = remaining_documents[i:i + current_batch_size]
                    tasks.append(process_batch_fixed(batch))
                
                for f in tqdm(asyncio.as_completed(tasks), total=ceil(len(remaining_documents) / current_batch_size), unit="batch", desc="Adding documents to vector index"):
                    await f

            print("All remaining documents processed.")

    async def delete_existing_embeddings(self):
        """ Delete all existing Document Embeddings nodes and vector index"""
        try:
            print("Deleting existing Document Embeddings nodes...")
            # Delete all nodes with the label and their relationships
            delete_query = "MATCH (n:`Document Embeddings`) DETACH DELETE n"
            self.knowledge_graph.query(delete_query)

            # Drop the vector index if it exists
            try:
                drop_index_query = "DROP INDEX vector IF EXISTS"
                self.knowledge_graph.query(drop_index_query)
                print("Existing vector index and nodes deleted successfully.")
            except:
                print("No existing vector index to delete.")

        except Exception as e:
            print(f"Error deleting existing embeddings: {e}")

    async def _get_existing_document_ids(self) -> set:
        """Get IDS of documents already in the vector index"""
        try:

            query = f"MATCH (n:`Document Embeddings`) RETURN n.pmid as pmid, n.seq_num as seq_num"
            result = self.knowledge_graph.query(query)
            return {f"{record['pmid']}_{record['seq_num']}" for record in result if record['pmid'] and record['seq_num']}
        except:
            return set()

    def _get_doc_id(self, doc: Document) -> str:
        """Extract unique ID from document using pmid + seq_num"""
        pmid = doc.metadata.get('pmid', '')
        seq_num = doc.metadata.get('seq_num', '')
        return f"{pmid}_{seq_num}"

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

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources"""
        if hasattr(self.vector_index, 'close'):
            self.vector_index.close()