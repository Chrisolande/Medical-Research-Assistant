from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from typing import List, Optional
from knowledge_graph import KnowledgeGraph
import asyncio
from asyncio import Semaphore
import time
from tqdm.asyncio import tqdm
from collections import defaultdict, Counter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class AdaptiveBatcher:
    def __init__(self, initial_batch_size: int = 96, min_batch_size: int = 16, max_batch_size: int = 96):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.latency_history = []
        self.target_latency = 30.0
        self.adjustment_factor = 0.2
        
    def adjust_batch_size(self, latency: float):
        self.latency_history.append(latency)
        if len(self.latency_history) > 5:
            self.latency_history.pop(0)

        avg_latency = sum(self.latency_history) / len(self.latency_history)

        if avg_latency > self.target_latency * 1.2:
            new_size = max(int(self.current_batch_size * 0.8), self.min_batch_size)
            if new_size != self.current_batch_size:
                print(f"Reducing batch size to {new_size}")
                self.current_batch_size = new_size
        elif avg_latency < self.target_latency * 0.8:
            new_size = min(int(self.current_batch_size * 1.2), self.max_batch_size)
            if new_size != self.current_batch_size:
                print(f"Increasing batch size to {new_size}")
                self.current_batch_size = new_size
            
        return self.current_batch_size

class VectorStore:
    def __init__(self, knowledge_graph: KnowledgeGraph, embedding_model: Optional[str] = None, 
                 initial_batch_size: int = 96, max_concurrent: int = 10, cache_dir: str = "embedding_cache"):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.batcher = AdaptiveBatcher(initial_batch_size)
        self.semaphore = Semaphore(max_concurrent)
        self.cache_dir = cache_dir
        self._embeddings = None
        self.vector_index = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            base_embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            fs = LocalFileStore(self.cache_dir)
            self._embeddings = CacheBackedEmbeddings.from_bytes_store(
                base_embeddings, fs, namespace=f"hf_{self.embedding_model.replace('/', '_')}")
        return self._embeddings

    def _get_doc_id(self, doc: Document) -> str:
        return f"{doc.metadata.get('pmid', '')}_{doc.metadata.get('seq_num', '')}"

    def _filter_valid_documents(self, documents: List[Document]) -> List[Document]:
        valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if len(valid_docs) != len(documents):
            print(f"Filtered out {len(documents) - len(valid_docs)} empty documents")
        return valid_docs

    def fix_seq_numbering(self, documents: List[Document]) -> List[Document]:
        pmid_groups = defaultdict(list)
        for i, doc in enumerate(documents):
            pmid = doc.metadata.get('pmid', '')
            pmid_groups[pmid].append((i, doc))
        
        for pmid, doc_list in pmid_groups.items():
            doc_list.sort(key=lambda x: x[0])
            for new_seq, (_, doc) in enumerate(doc_list):
                doc.metadata['seq_num'] = new_seq
        
        return documents

    async def _get_existing_document_ids(self) -> set:
        try:
            query = """
            MATCH (n:`Document Embeddings`) 
            WHERE n.text IS NOT NULL AND trim(n.text) <> ''
            RETURN n.pmid as pmid, n.seq_num as seq_num
            """
            result = self.knowledge_graph.query(query)
            return {f"{r['pmid']}_{r['seq_num']}" for r in result if r['pmid'] and r['seq_num']}
        except Exception as e:
            print(f"Error getting existing IDs: {e}")
            return set()

    async def _clean_corrupted_nodes(self) -> set:
        try:
            # Get corrupted IDs first
            query = """
            MATCH (n:`Document Embeddings`) 
            WHERE n.text IS NULL OR trim(n.text) = ''
            RETURN n.pmid as pmid, n.seq_num as seq_num
            """
            result = self.knowledge_graph.query(query)
            corrupted_ids = {f"{r['pmid']}_{r['seq_num']}" for r in result if r['pmid'] and r['seq_num']}
            
            if corrupted_ids:
                delete_query = """
                MATCH (n:`Document Embeddings`) 
                WHERE n.text IS NULL OR trim(n.text) = ''
                DETACH DELETE n
                """
                self.knowledge_graph.query(delete_query)
                print(f"Cleaned {len(corrupted_ids)} corrupted nodes")
            
            return corrupted_ids
        except Exception as e:
            print(f"Error cleaning nodes: {e}")
            return set()

    async def _process_batch(self, batch: List[Document]) -> int:
        async with self.semaphore:
            start_time = time.time()
            await self.vector_index.aadd_documents(batch)
            
            latency = time.time() - start_time
            self.batcher.adjust_batch_size(latency)
            return len(batch)

    async def create_vector_index(self, documents: List[Document], node_label: str = "Document Embeddings"):
        if not documents:
            print("No documents to process")
            return

        documents = self.fix_seq_numbering(documents)
        valid_documents = self._filter_valid_documents(documents)
        
        if not valid_documents:
            return

        # Try to connect to existing index
        try:
            self.vector_index = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self.knowledge_graph.url,
                username=self.knowledge_graph.username,
                password=self.knowledge_graph.password,
                index_name="vector",
                text_node_property="text"
            )
            print("Connected to existing vector index")
            
            # Clean corrupted nodes and get existing IDs
            corrupted_ids = await self._clean_corrupted_nodes()
            existing_ids = await self._get_existing_document_ids()
            
            # Filter for new documents
            valid_documents = [doc for doc in valid_documents 
                             if self._get_doc_id(doc) not in existing_ids or self._get_doc_id(doc) in corrupted_ids]
            
            print(f"Processing {len(valid_documents)} new/corrupted documents")
            
        except Exception:
            # Create new index
            first_batch = valid_documents[:self.batcher.current_batch_size]
            
            self.vector_index = await Neo4jVector.afrom_documents(
                first_batch,
                self.embeddings,
                url=self.knowledge_graph.url,
                username=self.knowledge_graph.username,
                password=self.knowledge_graph.password,
                index_name="vector",
                node_label=node_label,
                text_node_property=["text"],
                embedding_node_property="embedding"
            )
            print(f"Created new index with {len(first_batch)} documents")
            valid_documents = valid_documents[self.batcher.current_batch_size:]

        if not valid_documents:
            print("All documents processed")
            return

        # Process remaining documents in batches
        tasks = []
        i = 0
        while i < len(valid_documents):
            batch_size = self.batcher.current_batch_size
            batch = valid_documents[i:i + batch_size]
            if batch:
                tasks.append(self._process_batch(batch))
            i += batch_size

        async for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches"):
            await task

        print("Vector index creation complete")

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.vector_index:
            raise ValueError("Vector index not initialized")
        return await self.vector_index.asimilarity_search(query, k=k)

    async def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        if not self.vector_index:
            raise ValueError("Vector index not initialized")
        return await self.vector_index.asimilarity_search_with_score(query, k=k)

    async def query(self, queries: List[str], k: int = 4):
        if not self.vector_index:
            raise ValueError("Vector index not initialized")
        tasks = [self.similarity_search(query, k) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def delete_existing_embeddings(self):
        try:
            self.knowledge_graph.query("MATCH (n:`Document Embeddings`) DETACH DELETE n")
            self.knowledge_graph.query("DROP INDEX vector IF EXISTS")
            print("Existing embeddings deleted")
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
