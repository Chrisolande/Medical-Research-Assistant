"""Prompt Caching module."""

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

from langchain.schema import Generation
from langchain_community.cache import SQLiteCache
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
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
from utils import log_error, log_info, run_in_executor


@dataclass
class SemanticCache(SQLiteCache):
    """SemanticCache class."""

    database_path: str = DEFAULT_DATABASE_PATH
    faiss_index_path: str = DEFAULT_FAISS_INDEX_PATH
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_cache_size: int = DEFAULT_MAX_CACHE_SIZE
    memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_quantization: bool = ENABLE_QUANTIZATION

    def __post_init__(self):
        """Initialize post_init."""
        super().__init__(self.database_path)
        self.embedding_cache = {}
        self.memory_cache = {}
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_hits": 0,
            "embedding_time": 0,
            "search_time": 0,
            "memory_hits": 0,
        }

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self._lazy_loaded = False

    # @property
    def _lazy_load_vector_store(self):
        """Only load FAISS when needed."""
        if not self._lazy_loaded:
            with self.lock:
                if not self._lazy_loaded:
                    self._init_semantic_store()
                    self._lazy_loaded = True

    def _init_semantic_store(self):
        """Init Semantic Store method."""
        if os.path.exists(self.faiss_index_path) and os.path.isdir(
            self.faiss_index_path
        ):
            try:
                self.vector_store = self._load_faiss_index()
                log_info(f"Loaded FAISS index from {self.faiss_index_path}")
            except Exception as e:
                log_error(f"Error loading the faiss index: {str(e)}", exc_info=True)
                self._create_new_faiss_index()
        else:
            log_info(
                f"FAISS index path {self.faiss_index_path} not found or not a directory. Creating a new index."
            )
            self._create_new_faiss_index()

    async def _init_semantic_store_async(self):
        if os.path.exists(self.faiss_index_path) and os.path.isdir(
            self.faiss_index_path
        ):
            try:
                self.vector_store = await run_in_executor(
                    self.executor, self._load_faiss_index
                )  # No parentheses for func
                log_info(f"Loaded existing FAISS index from {self.faiss_index_path}")

            except Exception as e:
                log_error(f"Error loading the FAISS index: {str(e)}", exc_info=True)
                await self._create_new_faiss_index_async()  # Call async version
        else:
            await self._create_new_faiss_index_async()

    def _load_faiss_index(self):
        """Load Faiss Index method."""
        return FAISS.load_local(
            self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True
        )

    def _create_new_faiss_index(self):
        """Create New Faiss Index method."""
        try:
            self.vector_store = self._create_faiss_from_texts(
                [DUMMY_DOC_CONTENT], [{"type": "initializer", "is_dummy": True}]
            )
            log_info("Created new FAISS index")

        except Exception as e:
            log_error(
                f"Error: Failed to initialize FAISS vector store: {e}", exc_info=True
            )
            self.vector_store = None

    async def _create_new_faiss_index_async(self):
        try:
            self.vector_store = await run_in_executor(
                self.executor,
                self._create_faiss_from_texts,
                [DUMMY_DOC_CONTENT],
                [{"type": "initializer", "is_dummy": True}],
            )
            log_info("Created new FAISS index successfully")

        except Exception as e:
            log_error(
                f"Error: Failed to initialize FAISS vector store: {str(e)}",
                exc_info=True,
            )
            self.vector_store = None

    def _create_faiss_from_texts(self, texts: List[str], metadatas: List[Dict]):
        """Create Faiss From Texts method."""

        faiss_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

        if self.enable_quantization and len(texts) > 100:
            try:
                import faiss

                quantizer = faiss.IndexFlatL2(faiss_store.index.d)
                index_ivf = faiss.IndexIVFPQ(
                    quantizer, faiss_store.index.d, min(100, len(texts) // 10), 8, 8
                )
                index_ivf.train(
                    faiss_store.index.reconstruct_n(0, faiss_store.index.ntotal)
                )
                index_ivf.add(
                    faiss_store.index.reconstruct_n(0, faiss_store.index.ntotal)
                )
                faiss_store.index = index_ivf
                log_info("Applied quantization to FAISS index")
            except ImportError:
                log_info("faiss-cpu not available for quantization")

        return faiss_store

    def _get_cached_embedding(self, text: List[str]):
        """Get Cached Embedding method."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        return None

    def _cache_embedding(self, text: List[str], embedding: List[float]):
        """Cache Embedding method."""
        # Perform LRU eviction
        if len(self.embedding_cache) >= self.memory_cache_size:
            first_item = next(iter(self.embedding_cache))
            self.embedding_cache.pop(first_item)

        self.embedding_cache[text] = embedding

    def _get_embedding_with_cache(self, text: str):
        """Get Embedding With Cache method."""
        cached = self._get_cached_embedding(text)
        if cached:
            return cached

        start_time = time.time()
        embedding = self.embeddings.embed_query(text)
        self.metrics["embedding_time"] += time.time() - start_time

        self._cache_embedding(text, embedding)
        return embedding

    def lookup(self, prompt: str, llm_string: str):
        """Lookup method."""
        cache_key = f"{prompt}:{llm_string}"

        if cache_key in self.memory_cache:
            self.metrics["memory_hits"] += 1
            log_info("Memory cache hit")

            return self.memory_cache[cache_key]

        result = super().lookup(prompt, llm_string)
        if result:
            self.metrics["cache_hits"] += 1
            log_info("Exact match found in SQLite cache")

            self._add_to_memory_cache(cache_key, result)
            return result

        self._lazy_load_vector_store()

        if self.vector_store is None:
            log_info(
                "Semantic vector store is not initialized. Skipping semantic search"
            )
            self.metrics["cache_misses"] += 1
            return None

        try:
            if self._is_dummy_only():
                log_info(
                    "The vector store contains only the initializer document. Skipping semantic search"
                )
                self.metrics["cache_misses"] += 1
                return None

            # Use cached embeddings to improve speed
            start_time = time.time()
            query_embedding = self._get_embedding_with_cache(prompt)

            # Avoid re-embedding using vector search -smarter similarity search
            docs_with_score = self.vector_store.similarity_search_with_score_by_vector(
                query_embedding, k=3
            )

            self.metrics["search_time"] += time.time() - start_time

            if docs_with_score:
                for doc, score in docs_with_score:
                    if doc.page_content == DUMMY_DOC_CONTENT and doc.metadata.get(
                        "is_dummy"
                    ):
                        continue

                    if score <= self.similarity_threshold:  # Less is better
                        cached_llm_string = doc.metadata.get("llm_string_key")
                        original_cached_prompt = doc.page_content

                        if cached_llm_string and original_cached_prompt:
                            log_info(f"Semantic match found with score {score:.4f}.")
                            result = super().lookup(
                                original_cached_prompt, cached_llm_string
                            )
                            if result:
                                self.metrics["semantic_hits"] += 1
                                # Add to memory cache
                                self._add_to_memory_cache(cache_key, result)
                                return result
                    else:
                        log_info(
                            f"Best match score {
    score:.4f} above threshold ({
    self.similarity_threshold})."
                        )
                        break

        except Exception as e:
            log_error(f"Error during semantic lookup: {e}", exc_info=True)

        self.metrics["cache_misses"] += 1
        return None

    def _is_dummy_only(self):
        """Is Dummy Only method."""
        if len(self.vector_store.index_to_docstore_id) <= 1:
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = self.vector_store.docstore.get(doc_id)
                if (
                    doc
                    and doc.page_content == DUMMY_DOC_CONTENT
                    and doc.metadata.get("is_dummy")
                ):
                    return True
            return False  # If count is 1 but not dummy, it's not dummy only
        return False

    def _add_to_memory_cache(self, key: str, value: List[Generation]):
        """Add To Memory Cache method."""
        if len(self.memory_cache) >= self.memory_cache_size:
            first_item = next(iter(self.memory_cache))
            self.memory_cache.pop(first_item)

        self.memory_cache[key] = value

    def update(self, prompt: str, llm_string: str, return_val: List[Generation]):
        """Update method."""
        super().update(prompt, llm_string, return_val)

        # Add to memory cache
        cache_key = f"{prompt}:{llm_string}"
        self._add_to_memory_cache(cache_key, return_val)

        self._lazy_load_vector_store()

        if self.vector_store is None:
            log_info(
                "Semantic vector store not initialized, attempting to re-initialize."
            )
            self._init_semantic_store()
            if self.vector_store is None:
                log_info("Failed to re-initialize semantic vector store.")
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
            self._add_to_vector_store(prompt, metadata)
            self._save_vector_store()

        except Exception as e:
            log_error(f"Error during semantic index update: {e}", exc_info=True)

    async def update_async(
        self, prompt: str, llm_string: str, return_val: List[Generation]
    ):
        super().update(prompt, llm_string, return_val)

        # Add to memory cache
        cache_key = f"{prompt}:{llm_string}"
        self._add_to_memory_cache(cache_key, return_val)

        self._lazy_load_vector_store()

        if self.vector_store is None:
            log_info(
                "Semantic vector store not initialized, attempting to re-initialize."
            )
            await self._init_semantic_store_async()
            if self.vector_store is None:
                log_info("Failed to re-initialize semantic vector store.")
                return

        try:
            await self._remove_dummy_doc_async()
            if len(self.vector_store.docstore._dict) >= self.max_cache_size:
                await self._evict_oldest_entries_async()

            metadata = {
                "llm_string_key": llm_string,
                "type": "cache_entry",
                "timestamp": time.time(),
            }

            await run_in_executor(
                self.executor, self._add_to_vector_store, prompt, metadata
            )

            await self._save_vector_store_async()

        except Exception as e:
            log_error(f"Error during semantic index update: {e}", exc_info=True)

    def _remove_dummy_doc(self):
        """Remove Dummy Doc method."""
        dummy_ids = [
            doc_id
            for doc_id in self.vector_store.docstore._dict.keys()
            if self.vector_store.docstore._dict.get(doc_id)
            and self.vector_store.docstore._dict.get(doc_id).page_content
            == DUMMY_DOC_CONTENT
        ]
        if dummy_ids:
            self.vector_store.delete(dummy_ids)
            log_info(f"Removed {len(dummy_ids)} dummy documents.")

    async def _remove_dummy_doc_async(self):
        await run_in_executor(self.executor, self._remove_dummy_doc)

    def _evict_oldest_entries(self):
        """Evict Oldest Entries method."""
        # Use a set for faster lookups
        docs_with_timestamps = sorted(
            [
                (doc_id, doc.metadata.get("timestamp", 0))
                for doc_id, doc in self.vector_store.docstore._dict.items()
                if not doc.metadata.get("is_dummy")
            ],
            key=lambda x: x[1],
        )

        if (
            len(docs_with_timestamps) > self.max_cache_size * 0.8
        ):  # Evict 20% when near limit
            to_evict = docs_with_timestamps[: int(len(docs_with_timestamps) * 0.2)]
            evict_ids = [doc_id for doc_id, _ in to_evict]
            self.vector_store.delete(evict_ids)
            log_info(f"Evicted {len(evict_ids)} oldest entries from FAISS index.")

    async def _evict_oldest_entries_async(self):
        await run_in_executor(self.executor, self._evict_oldest_entries)

    def _add_to_vector_store(self, prompt: str, metadata: Dict):
        """Add To Vector Store method."""
        self.vector_store.add_texts([prompt], metadatas=[metadata])

    def _save_vector_store(self):
        """Save Vector Store method."""
        os.makedirs(self.faiss_index_path, exist_ok=True)
        self.vector_store.save_local(self.faiss_index_path)
        log_info(f"Updated FAISS index and saved to {self.faiss_index_path}")

    async def _save_vector_store_async(self):
        await run_in_executor(self.executor, self._save_vector_store)

    def get_metrics(self):
        """Get Metrics method."""
        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = (
            self.metrics["cache_hits"]
            + self.metrics["semantic_hits"]
            + self.metrics["memory_hits"]
        ) / max(total_requests, 1)

        return {
            **self.metrics,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "avg_embedding_time": self.metrics["embedding_time"]
            / max(self.metrics["cache_misses"], 1),
            "avg_search_time": self.metrics["search_time"]
            / max(self.metrics["cache_misses"], 1),
        }

    def clear_cache(self):
        """Clear Cache method."""
        super().clear()
        self.memory_cache.clear()
        self.embedding_cache.clear()
        if os.path.exists(self.faiss_index_path):
            shutil.rmtree(self.faiss_index_path)
        self.vector_store = None
        self._lazy_loaded = False
        self.metrics = {k: 0 for k in self.metrics}

        log_info("All caches cleared.")
