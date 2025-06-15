import logging
import os
import hashlib
import asyncio
from asyncio import Semaphore
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm

from knowledge_graph import KnowledgeGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorStore:
    knowledge_graph: KnowledgeGraph 
    batch_size: int = 200
    persist_directory: str = "faiss_index"
    max_concurrent: int = 10
    use_reranker: bool = True
    reranker_model: str = "jinaai/jina-reranker-v1-turbo-en"
    reranker_top_n: Optional[int] = 4
    vector_index: Optional[FAISS] = None
    added_doc_hashes: set = field(default_factory=set) 
    compression_retriever: Optional[ContextualCompressionRetriever] = None
    
    embeddings: HuggingFaceEmbeddings = field(default_factory=lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    
    semaphore: Semaphore = None

    def __post_init__(self):
        self.semaphore = Semaphore(self.max_concurrent)
        self._load_local_index()
        self._setup_reranker()

    def _setup_reranker(self):
        if not self.use_reranker or self.vector_index is None:
            return []

        try:
            model = HuggingFaceCrossEncoder(model_name=self.reranker_model)
            compressor = CrossEncoderReranker(model=model, top_n=self.reranker_top_n)
            base_retriever = self.vector_index.as_retriever(search_kwargs={"k": 20})
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
                )

            logger.info(f"Reranker initialized with model: {self.reranker_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.use_reranker = False

    def _get_document_hash(self, doc: Document) -> str:
        if not doc.page_content:
            logger.debug(f"Attempted to hash a document with empty page_content: {doc}")
            return ""
        return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    def _filter_valid_docs(self, documents: List[Document]) -> List[Document]:
        valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        num_filtered = len(documents) - len(valid_docs)
        if num_filtered > 0:
            logger.info(f"Filtered out {num_filtered} documents due to empty or whitespace-only content.")
        return valid_docs

    def _load_local_index(self):
        if not os.path.exists(self.persist_directory):
            logger.info(f"No existing FAISS index found at {self.persist_directory}. A new one will be created upon first addition.")
            return

        logger.info(f"Attempting to load index from {self.persist_directory}...")
        try:
            self.vector_index = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
            total_docs_in_index = len(self.vector_index.docstore._dict)

            try:
                progress_iterator = tqdm(self.vector_index.docstore._dict.keys(), desc="Reconstructing document hashes", total=total_docs_in_index)
            except ImportError:
                logger.warning("tqdm not installed. Progress bar for hash reconstruction will not be shown.")
                progress_iterator = self.vector_index.docstore._dict.keys()

            for doc_id in progress_iterator:
                doc = self.vector_index.docstore.search(doc_id)
                if doc and isinstance(doc, Document):
                    self.added_doc_hashes.add(self._get_document_hash(doc))
            
            logger.info(f"Successfully loaded FAISS index from {self.persist_directory} with {total_docs_in_index} documents.")
            
            # Setup reranker after successful index loading
            self._setup_reranker()
            
        except Exception as e:
            logger.error(f"Failed to load vector index from {self.persist_directory}. Error: {e}", exc_info=True)
            self.vector_index = None
            self.added_doc_hashes.clear()

    def _save_local_index(self):
        if self.vector_index is None:
            logger.info("No FAISS index initialized or loaded; skipping save operation.")
            return

        logger.info(f"Saving FAISS index to {self.persist_directory}...")
        try:
            self.vector_index.save_local(self.persist_directory)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
            logger.error(f"Error saving FAISS index to {self.persist_directory}: {e}", exc_info=True)

    async def _add_batch_and_persist(self, batch: List[Document]):
        if not batch:
            logger.warning("Attempted to add an empty batch.")
            return 0

        async with self.semaphore:
            try:
                if self.vector_index is None:
                    logger.info(f"Creating new FAISS index from a batch of {len(batch)} documents.")
                    self.vector_index = FAISS.from_documents(batch, self.embeddings)
                else:
                    logger.info(f"Adding {len(batch)} documents to existing FAISS index.")
                    self.vector_index.add_documents(batch)
                
                self._save_local_index()

                # Track the documents in batches
                for doc in batch:
                    doc_hash = self._get_document_hash(doc)
                    if doc_hash: # Only add if hash is not empty
                        self.added_doc_hashes.add(doc_hash)
                
                return len(batch)
            except Exception as e:
                logger.error(f"Error processing batch of {len(batch)} documents: {e}", exc_info=True)
                return 0

    async def _create_vector_index(self, documents: List[Document]):
        """
        Creates or updates the vector index with new, valid documents.
        This method handles filtering, de-duplication, batching, and concurrent processing.
        """
        if not documents:
            logger.info("No documents provided to create/update vector index.")
            return
        
        logger.info(f"Starting vector index creation/update with {len(documents)} initial documents.")
        
        valid_documents = self._filter_valid_docs(documents)
        
        if not valid_documents:
            logger.info("No valid documents after filtering; skipping vector index update.")
            return

        new_documents = []
        try:
            progress_iterator = tqdm(valid_documents, desc="Filtering new documents...")
        except ImportError:
            logger.warning("tqdm not installed. Progress bar for new document filtering will not be shown.")
            progress_iterator = valid_documents

        for doc in progress_iterator:
            doc_hash = self._get_document_hash(doc)
            if doc_hash and doc_hash not in self.added_doc_hashes:
                new_documents.append(doc)

        if not new_documents:
            logger.info("No new documents to add to the vector index.")
            return 
        
        logger.info(f"Found {len(new_documents)} new documents to process.")
        
        # Prepare batches
        batches = [new_documents[i: i + self.batch_size] for i in range(0, len(new_documents), self.batch_size)]

        if not batches: 
            logger.warning("No batches formed from the new documents, this should not happen if new_documents is not empty.")
            return 
        
        logger.info(f"Processing {len(new_documents)} new documents in {len(batches)} batches.")
        
        tasks = [self._add_batch_and_persist(batch) for batch in batches]
        processed_counts = await asyncio.gather(*tasks) # Gather results of each batch

        total_processed = sum(processed_counts)
        logger.info(f"Vector store creation/update successful. Total new documents added: {total_processed}.")

    async def similarity_search(self, query: str, k: int = 4):
        if self.vector_index is None:
            logger.warning("Vector index is not initialized. Cannot perform similarity search.")
            return [] 

        try:
            if self.use_reranker and self.compression_retriever:
                # Update the top_n dynamically if not set
                if self.compression_retriever.base_compressor.top_n is None:
                    self.compression_retriever.base_compressor.top_n = k

                results = self.compression_retriever.invoke(query)
                logger.debug(f"Performed reranked similarity search for query: '{query}'")
                return results

            else:
                results = self.vector_index.similarity_search(query, k=k)
                logger.debug(f"Performed similarity search for query: '{query}'")
                return results
        except Exception as e:
            logger.error(f"Error during similarity search for query '{query}': {e}", exc_info=True)
            return []

    async def similarity_search_with_score(self, query: str, k: int = 4):
        if self.vector_index is None:
            logger.warning("Vector index is not initialized. Cannot perform similarity search with score.")
            return []
        
        try:
            results = self.vector_index.similarity_search_with_score(query, k=k)
            logger.debug(f"Performed similarity search with score for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search with score for query '{query}': {e}", exc_info=True)
            return []

    async def batch_query(self, queries: List[str], k: int = 4):
        if self.vector_index is None:
            logger.warning("Vector index is not initialized. Cannot perform batch query.")
            return [[] for _ in queries]

        tasks = [self.similarity_search(query, k=k) for query in queries]
        results = await asyncio.gather(*tasks)
        logger.debug(f"Performed batch query for {len(queries)} queries.")
        return results

    async def delete_index(self):
        """Deletes the local FAISS index directory and resets the vector store state."""
        if os.path.exists(self.persist_directory):
            try:
                import shutil
                shutil.rmtree(self.persist_directory) 
                logger.info(f"Successfully deleted the FAISS index directory: {self.persist_directory}")
                self.vector_index = None
                self.added_doc_hashes.clear()
            except OSError as e: 
                logger.error(f"Error deleting FAISS index directory {self.persist_directory}: {e}", exc_info=True)
            except Exception as e: 
                logger.error(f"An unexpected error occurred during index deletion: {e}", exc_info=True)
        else:
            logger.info("No vector index directory to delete!")