"""Prompt Caching module."""

import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from langchain.schema import Generation
from langchain_community.cache import SQLiteCache
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from medical_graph_rag.core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATABASE_PATH,
    DEFAULT_FAISS_INDEX_PATH,
    DEFAULT_MAX_CACHE_SIZE,
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DUMMY_DOC_CONTENT,
    EMBEDDING_MODEL_NAME,
    ENABLE_QUANTIZATION,
)
from medical_graph_rag.core.utils import log_error, log_info, run_in_executor

logger = logging.getLogger(__name__)


@dataclass
class SemanticCache(SQLiteCache):
    """Optimized SemanticCache with minimal overhead."""

    database_path: str = DEFAULT_DATABASE_PATH
    faiss_index_path: str = DEFAULT_FAISS_INDEX_PATH
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_cache_size: int = DEFAULT_MAX_CACHE_SIZE
    memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_quantization: bool = ENABLE_QUANTIZATION

    # Internal fields
    embedding_cache: OrderedDict = field(default_factory=OrderedDict, init=False)
    memory_cache: dict[str, Any] = field(default_factory=dict, init=False)
    metrics: dict[str, float] = field(
        default_factory=lambda: {
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_hits": 0,
            "embedding_time": 0,
            "search_time": 0,
            "memory_hits": 0,
        },
        init=False,
    )
    executor: ThreadPoolExecutor = field(default=None, init=False)
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    embeddings: HuggingFaceEmbeddings = field(default=None, init=False)
    vector_store: FAISS | None = field(default=None, init=False)
    _lazy_loaded: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize cache components."""
        super().__init__(self.database_path)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Vector Store Management
    def _lazy_load_vector_store(self):
        """Load FAISS only when needed."""
        if not self._lazy_loaded:
            with self.lock:
                if not self._lazy_loaded:
                    self._init_semantic_store()
                    self._lazy_loaded = True

    def _init_semantic_store(self):
        """Initialize semantic vector store."""
        if os.path.exists(self.faiss_index_path) and os.path.isdir(
            self.faiss_index_path
        ):
            try:
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                log_info(f"Loaded FAISS index from {self.faiss_index_path}")
            except Exception as e:
                log_error(f"Error loading FAISS index: {e}", exc_info=True)
                self._create_new_faiss_index()
        else:
            self._create_new_faiss_index()

    def _create_new_faiss_index(self):
        """Create new FAISS index with dummy content."""
        try:
            self.vector_store = FAISS.from_texts(
                [DUMMY_DOC_CONTENT],
                self.embeddings,
                metadatas=[{"type": "initializer", "is_dummy": True}],
            )

            if self.enable_quantization:
                self._apply_quantization()

            log_info("Created new FAISS index")
        except Exception as e:
            log_error(f"Failed to initialize FAISS: {e}", exc_info=True)
            self.vector_store = None

    def _apply_quantization(self):
        """Apply quantization to FAISS index if conditions are met."""
        if self.vector_store and hasattr(self.vector_store, "index"):
            try:
                import faiss

                ntotal = self.vector_store.index.ntotal
                if ntotal > 100:
                    quantizer = faiss.IndexFlatL2(self.vector_store.index.d)
                    index_ivf = faiss.IndexIVFPQ(
                        quantizer,
                        self.vector_store.index.d,
                        min(100, ntotal // 10),
                        8,
                        8,
                    )
                    vectors = self.vector_store.index.reconstruct_n(0, ntotal)
                    index_ivf.train(vectors)
                    index_ivf.add(vectors)
                    self.vector_store.index = index_ivf
                    log_info("Applied quantization to FAISS index")
            except ImportError:
                log_info("faiss-cpu not available for quantization")

    # Embedding Management
    def _get_embedding_with_cache(self, text: str) -> list[float]:
        """Get embedding with LRU cache."""
        if text in self.embedding_cache:
            self.embedding_cache.move_to_end(text)
            return self.embedding_cache[text]

        start_time = time.time()
        embedding = self.embeddings.embed_query(text)
        self.metrics["embedding_time"] += time.time() - start_time

        # LRU eviction
        if len(self.embedding_cache) >= self.memory_cache_size:
            self.embedding_cache.popitem(last=False)
        self.embedding_cache[text] = embedding

        return embedding

    # Cache Operations
    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        """Lookup with memory, SQLite, and semantic search."""
        cache_key = f"{prompt}:{llm_string}"

        # Memory cache check
        if cache_key in self.memory_cache:
            self.metrics["memory_hits"] += 1
            log_info("Memory cache hit")
            return self.memory_cache[cache_key]

        # SQLite cache check
        result = super().lookup(prompt, llm_string)
        if result:
            self.metrics["cache_hits"] += 1
            log_info("SQLite cache hit")
            self._add_to_memory_cache(cache_key, result)
            return result

        # Semantic search
        return self._semantic_lookup(prompt, llm_string, cache_key)

    def _semantic_lookup(
        self, prompt: str, llm_string: str, cache_key: str
    ) -> list[Generation] | None:
        """Perform semantic similarity search."""
        self._lazy_load_vector_store()

        if not self.vector_store or self._is_dummy_only():
            self.metrics["cache_misses"] += 1
            return None

        try:
            start_time = time.time()
            query_embedding = self._get_embedding_with_cache(prompt)
            docs_with_score = self.vector_store.similarity_search_with_score_by_vector(
                query_embedding, k=3
            )
            self.metrics["search_time"] += time.time() - start_time

            for doc, score in docs_with_score:
                if self._is_dummy_doc(doc) or score < self.similarity_threshold:
                    continue

                cached_llm_string = doc.metadata.get("llm_string_key")
                if cached_llm_string == llm_string:
                    result = super().lookup(doc.page_content, cached_llm_string)
                    if result:
                        self.metrics["semantic_hits"] += 1
                        log_info(f"Semantic match found with score {score:.4f}")
                        self._add_to_memory_cache(cache_key, result)
                        return result

        except Exception as e:
            log_error(f"Error during semantic lookup: {e}", exc_info=True)

        self.metrics["cache_misses"] += 1
        return None

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]):
        """Update all cache layers."""
        super().update(prompt, llm_string, return_val)

        cache_key = f"{prompt}:{llm_string}"
        self._add_to_memory_cache(cache_key, return_val)
        self._update_semantic_store(prompt, llm_string)

    async def update_async(
        self, prompt: str, llm_string: str, return_val: list[Generation]
    ):
        """Async version of update for better performance."""
        await run_in_executor(
            self.executor, self.update, prompt, llm_string, return_val
        )

    def _update_semantic_store(self, prompt: str, llm_string: str):
        """Update semantic vector store."""
        self._lazy_load_vector_store()
        if not self.vector_store:
            return

        try:
            self._remove_dummy_doc()

            if len(self.vector_store.docstore._dict) >= self.max_cache_size:
                self._evict_oldest_entries()

            metadata = {
                "llm_string_key": llm_string,
                "type": "cache_entry",
                "timestamp": time.time(),
            }

            self.vector_store.add_texts([prompt], metadatas=[metadata])
            self._save_vector_store()

        except Exception as e:
            log_error(f"Error updating semantic store: {e}", exc_info=True)

    # Utility Methods
    def _is_dummy_only(self) -> bool:
        """Check if vector store contains only dummy documents."""
        if len(self.vector_store.index_to_docstore_id) <= 1:
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = self.vector_store.docstore.search(doc_id)
                return self._is_dummy_doc(doc) if doc else False
        return False

    def _is_dummy_doc(self, doc) -> bool:
        """Check if document is a dummy document."""
        return doc.page_content == DUMMY_DOC_CONTENT and doc.metadata.get(
            "is_dummy", False
        )

    def _add_to_memory_cache(self, key: str, value: list[Generation]):
        """Add to memory cache with size limit."""
        if len(self.memory_cache) >= self.memory_cache_size:
            first_key = next(iter(self.memory_cache))
            del self.memory_cache[first_key]
        self.memory_cache[key] = value

    def _remove_dummy_doc(self):
        """Remove dummy documents from vector store."""
        dummy_ids = [
            doc_id
            for doc_id, doc in self.vector_store.docstore._dict.items()
            if doc and self._is_dummy_doc(doc)
        ]
        if dummy_ids:
            self.vector_store.delete(dummy_ids)
            log_info(f"Removed {len(dummy_ids)} dummy documents")

    def _evict_oldest_entries(self):
        """Evict oldest entries when cache is full."""
        docs_with_timestamps = [
            (doc_id, doc.metadata.get("timestamp", 0))
            for doc_id, doc in self.vector_store.docstore._dict.items()
            if not doc.metadata.get("is_dummy", False)
        ]

        if len(docs_with_timestamps) > self.max_cache_size * 0.8:
            docs_with_timestamps.sort(key=lambda x: x[1])
            evict_count = int(len(docs_with_timestamps) * 0.2)
            evict_ids = [doc_id for doc_id, _ in docs_with_timestamps[:evict_count]]
            self.vector_store.delete(evict_ids)
            log_info(f"Evicted {len(evict_ids)} oldest entries")

    def _save_vector_store(self):
        """Save vector store to disk."""
        os.makedirs(self.faiss_index_path, exist_ok=True)
        self.vector_store.save_local(self.faiss_index_path)

    # Metrics and Management
    def get_metrics(self) -> dict[str, float]:
        """Get comprehensive cache metrics."""
        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        total_hits = (
            self.metrics["cache_hits"]
            + self.metrics["semantic_hits"]
            + self.metrics["memory_hits"]
        )

        return {
            **self.metrics,
            "hit_rate": total_hits / max(total_requests, 1),
            "total_requests": total_requests,
            "avg_embedding_time": (
                self.metrics["embedding_time"] / max(self.metrics["cache_misses"], 1)
            ),
            "avg_search_time": (
                self.metrics["search_time"] / max(self.metrics["cache_misses"], 1)
            ),
        }

    def clear_cache(self):
        """Clear all caches and reset state."""
        # Clear memory caches
        self.memory_cache.clear()
        self.embedding_cache.clear()

        # Delete the SQLite database file FIRST
        if os.path.exists(self.database_path):
            try:
                os.remove(self.database_path)
                log_info(f"Deleted SQLite database file: {self.database_path}")
            except Exception as e:
                log_error(f"Error deleting SQLite database: {e}")

        # Then try to clear SQLite cache
        try:
            super().clear()
        except Exception:
            # Ignore errors since we already deleted the file
            pass

        # Clear FAISS index
        if os.path.exists(self.faiss_index_path):
            shutil.rmtree(self.faiss_index_path)
            log_info(f"Deleted FAISS index: {self.faiss_index_path}")

        # Reset state
        self.vector_store = None
        self._lazy_loaded = False
        self.metrics = {k: 0 for k in self.metrics}
        log_info("All caches cleared completely")

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=False)


#
