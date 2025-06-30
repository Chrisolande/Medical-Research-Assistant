"""Vectorstore module."""

import asyncio
import hashlib
import logging
import os
import shutil
from asyncio import Semaphore
from dataclasses import dataclass, field

from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    CrossEncoderReranker,
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from medical_graph_rag.core.config import (
    BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    FLASHRANK_CACHE_DIR,
    FLASHRANK_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    MAX_CONCURRENT,
    OPENROUTER_API_BASE,
    OPENROUTER_API_KEY,
    PERSIST_DIRECTORY,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
)
from medical_graph_rag.core.utils import pretty_print_docs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ranker = Ranker(model_name=FLASHRANK_MODEL_NAME, cache_dir=FLASHRANK_CACHE_DIR)


@dataclass
class VectorStore:
    """VectorStore class."""

    # Configuration
    batch_size: int = BATCH_SIZE
    persist_directory: str = PERSIST_DIRECTORY
    max_concurrent: int = MAX_CONCURRENT
    model_name: str = LLM_MODEL_NAME

    # Reranker settings
    use_reranker: bool = True
    reranker_model: str = RERANKER_MODEL_NAME
    reranker_top_n: int | None = RERANKER_TOP_N

    # Internal state
    vector_index: FAISS | None = None
    compression_retriever: ContextualCompressionRetriever | None = None
    added_doc_hashes: set = field(default_factory=set)
    semaphore: Semaphore = None

    embeddings: HuggingFaceEmbeddings = field(
        default_factory=lambda: HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    )

    def __post_init__(self):
        """Initialize post_init."""
        # ensure_semantic_cache()  # Ensure semantic cache is initialized
        self.semaphore = Semaphore(self.max_concurrent)
        self.llm = ChatOpenAI(
            model=self.model_name,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_API_BASE,
            temperature=0,
        )
        self._load_local_index()

    # ============ INDEX PERSISTENCE ============
    def _load_local_index(self):
        """Load Local Index method."""
        if not os.path.exists(self.persist_directory):
            logger.info(
                f"No existing FAISS index found at {self.persist_directory}. A new one will be created upon first addition."
            )
            return

        logger.info(f"Attempting to load index from {self.persist_directory}...")
        try:
            self.vector_index = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self._reconstruct_hashes()
            logger.info(
                f"Successfully loaded FAISS index with {len(self.vector_index.index_to_docstore_id)} documents."
            )
            self._setup_reranker()
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}", exc_info=True)
            self.vector_index = None
            self.added_doc_hashes.clear()

    def _save_local_index(self):
        """Save Local Index method."""
        if not self.vector_index:
            logger.info(
                "No FAISS index initialized or loaded; skipping save operation."
            )
            return

        logger.info(f"Saving FAISS index to {self.persist_directory}...")
        try:
            self.vector_index.save_local(self.persist_directory)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}", exc_info=True)

    async def delete_index(self) -> None:
        if not os.path.exists(self.persist_directory):
            logger.info("No vector index directory to delete!")
            return

        try:
            shutil.rmtree(self.persist_directory)
            logger.info("Deleted the FAISS index directory.")
        except Exception as e:
            logger.error(f"Error deleting FAISS index directory: {e}", exc_info=True)

        self.vector_index = None
        self.added_doc_hashes.clear()

    # ============ DOCUMENT HASHING ============
    def _get_document_hash(self, doc: Document) -> str:
        """Get Document Hash method."""
        return (
            hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()  # nosec
            if doc.page_content
            else ""
        )

    def _update_doc_hashes(self, batch: list[Document]):
        """Update Doc Hashes method."""
        for doc in batch:
            self.added_doc_hashes.add(self._get_document_hash(doc))

    def _reconstruct_hashes(self):
        """Reconstruct Hashes method."""
        if self.vector_index and self.vector_index.docstore:
            for doc_id in self.vector_index.docstore._dict.keys():
                doc = self.vector_index.docstore._dict.get(doc_id)
                if isinstance(doc, Document):
                    self.added_doc_hashes.add(self._get_document_hash(doc))

    def _is_new_document(self, doc: Document) -> bool:
        """Is New Document method."""
        return (
            doc_hash := self._get_document_hash(doc)
        ) and doc_hash not in self.added_doc_hashes

    # ============ DOCUMENT PROCESSING ============
    def _filter_valid_docs(self, documents: list[Document]) -> list[Document]:
        """Filter Valid Docs method."""
        valid_docs = [
            doc for doc in documents if doc.page_content and doc.page_content.strip()
        ]
        if num_filtered := len(documents) - len(valid_docs):
            logger.info(
                f"Filtered out {num_filtered} documents due to empty or whitespace-only content."
            )
        return valid_docs

    def _create_batches(self, documents: list[Document]) -> list[list[Document]]:
        """Create Batches method."""
        return [
            documents[i : i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]

    async def _add_batch_and_persist(self, batch: list[Document]):
        """Add to batch and save."""
        if not batch:
            logger.warning("Attempted to add an empty batch.")
            return 0

        async with self.semaphore:
            try:
                if not self.vector_index:
                    self.vector_index = FAISS.from_documents(batch, self.embeddings)
                    self._setup_reranker()
                else:
                    self.vector_index.add_documents(batch)

                self._save_local_index()
                self._update_doc_hashes(batch)
                return len(batch)
            except Exception as e:
                logger.error(
                    f"Error processing batch of {len(batch)} documents: {e}",
                    exc_info=True,
                )
                return 0

    async def _create_vector_index(self, documents: list[Document]):
        """Build the vector index."""
        if not documents:
            logger.info("No documents provided to create/update vector index.")
            return

        valid_documents = self._filter_valid_docs(documents)
        if not valid_documents:
            logger.info(
                "No valid documents after filtering; skipping vector index update."
            )
            return

        new_documents = [doc for doc in valid_documents if self._is_new_document(doc)]
        if not new_documents:
            logger.info("No new documents to add to the vector index.")
            return

        logger.info(
            f"Processing {len(new_documents)} new documents in {len(new_documents) // self.batch_size + 1} batches."
        )
        await asyncio.gather(
            *(
                self._add_batch_and_persist(batch)
                for batch in self._create_batches(new_documents)
            )
        )

    # ============ RERANKER SETUP ============
    def _setup_reranker(self):
        """Setup Reranker method."""
        if not self.use_reranker or not self.vector_index:
            return
        # TODO: Check the effets of the compression if its an overkill or not
        # TODO: Check if multiquery retrieval will be needed
        try:
            model = HuggingFaceCrossEncoder(model_name=self.reranker_model)
            compressor = CrossEncoderReranker(model=model, top_n=self.reranker_top_n)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vector_index.as_retriever(search_kwargs={"k": 20}),
            )
            logger.info(f"Reranker initialized with model: {self.reranker_model}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.use_reranker = False

    # ============ SEARCH OPERATIONS ============
    async def _perform_reranked_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform reranked search."""
        if self.use_reranker and self.compression_retriever:
            self.compression_retriever.base_compressor.top_n = k
            return await self.compression_retriever.ainvoke(query)
        return []

    async def similarity_search(self, query: str, k: int = 4):
        if not self.vector_index:
            logger.warning(
                "Vector index is not initialized. Cannot perform similarity search."
            )
            return []

        try:
            return (
                await self._perform_reranked_search(query, k)
                if self.use_reranker
                else await self.vector_index.asimilarity_search(query, k=k)
            )
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return []

    async def similarity_search_with_score(self, query: str, k: int = 4):
        """Perform similarity search with score."""
        if not self.vector_index:
            logger.warning(
                "Vector index is not initialized. Cannot perform similarity search with score."
            )
            return []

        try:
            return self.vector_index.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(
                f"Error during similarity search with score: {e}", exc_info=True
            )
            return []

    async def batch_query(self, queries: list[str], k: int = 4):
        """Process multiple queries at once."""
        if not self.vector_index:
            logger.warning(
                "Vector index is not initialized. Cannot perform batch query."
            )
            return [[] for _ in queries]

        return await asyncio.gather(
            *(self.similarity_search(query, k=k) for query in queries)
        )

    # ============ RETRIEVAL ============
    def retrieve_relevant_documents(self, query: str, filter_threshold: float = 0.6):
        """Retrieve relevant documents using compression pipeline."""
        if not self.vector_index:
            return []

        retriever = self.vector_index.as_retriever(search_kwargs={"k": 8})
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[
                EmbeddingsFilter(
                    embeddings=self.embeddings, similarity_threshold=filter_threshold
                ),
                EmbeddingsRedundantFilter(
                    embeddings=self.embeddings, similarity_threshold=0.95
                ),
                FlashrankRerank(client=ranker),
                LLMChainExtractor.from_llm(self.llm),
            ]
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )

        try:
            results = compression_retriever.invoke(query)
            if results:
                pretty_print_docs(results)
            else:
                print("No relevant documents found for the query.")

            return results or []
        except ValueError as e:
            if "token_type_ids" in str(e) or "missing from input feed" in str(e):
                return []
            raise
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
