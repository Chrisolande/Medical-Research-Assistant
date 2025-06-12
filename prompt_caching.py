"""main logic comes from here https://api.python.langchain.com/en/latest/cache/langchain_community.cache.SQLiteCache.html"""


import os
import shutil
from typing import Optional, List, Dict

from langchain_community.cache import SQLiteCache
from langchain_community.vectorstores import FAISS
from langchain.schema import Generation
from langchain_cohere import CohereEmbeddings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
DUMMY_DOC_CONTENT = "Langchain Document Initializer"

@dataclass
class SemanticCache(SQLiteCache):
    database_path: str = ".langchain.db",
    faiss_index_path: str = "./faiss_index",
    similarity_threshold: float = 0.5,
    max_cache_size: int = 1000, 
    memory_cache_size: int = 100,
    batch_size: int = 10,
    enable_quantization: bool = False

    def __post_init__(self):
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path, exist_ok = True)
        super().__init__(self.database_path)
        self.embedding_cache = dict()
        self.memory_cache = dict()
        self.metrics = {
            "cache_hits": 0, "cache_misses": 0, "semantic_hits": 0,
            'embedding_time': 0, 'search_time': 0, 'memory_hits': 0
        }

        self.executor = ThreadPoolExecutor(max_workers = 4)
        self.lock = threading.RLock() # ReEntrant Lock, a single thread can access it multiple times
        cohere_api_key = os.getenv("COHERE_API_KEY1")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY1 environment variable not set. Please set it to use CohereEmbeddings.")

        self.embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")
        self.vector_store: Optional[FAISS] = None
        self._lazy_loaded = False

    #@property
    def _lazy_load_vector_store(self):
        """Only load FAISS when needed"""
        if not self._lazy_loaded:
            with self.lock():
                if not self._lazy_loaded:
                    self._init_semantic_store()
                    self._lazy_loaded = True
        
    def _init_semantic_store(self):
        # Check if the path exists
        if os.path.exists(self.faiss_index_path) and os.path.isdir(self.faiss_index_path):
            try:
                self.vector_store = self._load_faiss_index
                print(f"Loaded FAISS indec from {self.faiss_index_path}")
            except Exception as e:
                print(f"Error loading the faiss index: {str(e)}")
                self._create_new_faiss_index()
        else:
            print(f"FAISS index path {self.faiss_index_path} not found or not a directory. Creating a new index.")
            self._create_new_faiss_index()

    async def _init_semantic_store_async(self):
        if os.path.exists(self.faiss_index_path) and os.path.isdir(self.faiss_index_path):
            try:
                #Wrap the blocking function in an event loop
                self.vector_store = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_faiss_index
                )
                print(f"Loaded existing FAISS index from {self.faiss_index_path}")
            
            except Exception as e:
                print(f"Error loading the FAISS index: {str(e)}")
                self._create_new_faiss_index_async()
        else:
            self._create_new_faiss_index_async()

    def _load_faiss_index(self):
        return FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_serialization = True)
    
    def _create_new_faiss_index(self):
        try:
            self.vector_store = self._create_index_from_texts([DUMMY_DOC_CONTENT], [{"type": "initializer", "is_dummy": True}])
            print("Created new FAISS index")
        
        except Exception as e:
            print(f"Error: Failed to initialize FAISS vector store: {e}")
            self.vector_store = None

    async def _create_new_faiss_index_async(self):
        try:
            self.vector_store = await asyncio.get_event_loop().run_in_executor(self.executor, self._create_index_from_texts([DUMMY_DOC_CONTENT], [{"type": "initializer", "is_dummy": True}]))
            print("Created new FAISS index successfully")
        
        except Exception as e:
            print(f"Error: Failed to initialize FAISS vector store: {str(e)}")
            self.vector_store = None

    def _create_faiss_from_texts(self, texts:List[str], metadatas: List[Dict]):
        
        faiss_store = FAISS.from_texts(texts, self.embedding, metadata = metadatas)

        if self.enable_quantization and len(texts) > 100:
            try:
                import faiss
                quantizer = faiss.IndexFlatL2(faiss_store.index.d)
                index_ivf = faiss.IndexIVFPQ(quantizer, faiss_store.index.d, min(100, len(texts)//10), 8, 8)
                index_ivf.train(faiss_store.index.reconstruct_n(0, faiss_store.index.ntotal))
                index_ivf.add(faiss_store.index.reconstruct_n(0, faiss_store.index.ntotal))
                faiss_store.index = index_ivf
                print("Applied quantization to FAISS index")
            except ImportError:
                print("faiss-cpu not available for quantization")
        
        return faiss_store
       
    def _get_cached_embedding(self, text:List[str]):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        return None
    
    def _cache_embedding(self, text: List[str], embedding: List[float]):
        # Perform LRU eviction
        if len(self.embedding_cache) >= self.memory_cache_size:
            first_item = next(iter(self.embedding_cache))
            self.embedding_cache.pop(first_item)

        self.embedding_cache[text] = embedding
    
    def _get_embedding_with_cache(self, text: List[str]):
        cached = self._get_cached_embedding(text)
        if cached:
            return cached
        
        start_time = time.time()
        embedding = self.embedding.embed_query(text)
        self.metrics['embedding_time'] += time.time() - start_time

        self._cache_embedding(text, embedding)
        return embedding

    def lookup(self, prompt: str, llm_string: str):
        cache_key = f"{prompt}:{llm_string}"

        if cache_key in self.memory_cache:
            self.metrics["memory_hits"] += 1
            print("Memory cache hit")

            return self.memory_cache[cache_key]

        result = super().lookup(prompt, llm_string)
        if result:
            self.metrics["cache_hits"] += 1
            print("Exact match found in SQLite cache")

            self._add_to_memory_cache(cache_key, result)
            return result

        self._lazy_load_vector_store()

        if self.vector_store is None:
            print("Semantic vector store is not initialized. Skipping semantic search")
            self.metrics["cache_misses"] += 1
            return None

        try:
            if self._is_dummy_only():
                print("The vector store contains only the initializer document. Skipping semantic search")
                self.metrics["cache_misses"] += 1
                return None
            
            # Use cached embeddings to improve speed
            start_time = time.time()
            query_embedding = self._get_embedding_with_cache(prompt)

            # Avoid re-embedding using vector search -smarter similarity search
            docs_with_score = self.vector_store.similarity_search_with_score_by_vector(
                query_embedding, k=3
            )

            self.metrics['search_time'] += time.time() - start_time

            if docs_with_score:
                for doc, score in docs_with_score:
                    if doc.page_content == DUMMY_DOC_CONTENT and doc.metadata.get("is_dummy"):
                        continue

                    if score <= self.similarity_threshold: # Less is better
                        cached_llm_string = doc.metadata.get('llm_string_key')
                        original_cached_prompt = doc.page_content

                        if cached_llm_string and original_cached_prompt:
                            print(f"Semantic match found with score {score:.4f}.")
                            result = super().lookup(original_cached_prompt, cached_llm_string)
                            if result:
                                self.metrics['semantic_hits'] += 1
                                # CHANGE: Add to memory cache - caching strategy
                                self._add_to_memory_cache(cache_key, result)
                                return result
                    else:
                        print(f"Best match score {score:.4f} above threshold ({self.similarity_threshold}).")
                        break


        except Exception as e:
            print(f"Error during semantic lookup: {e}")

        self.metrics['cache_misses'] += 1
        return None
    
    def _is_dummy_only(self):
        if len(self.vector_store.docstore._dict) <= 1:
            return True

        real_docs = 0
        for doc in self.vector_store.docstore._dict.values():
            if doc.page_content != DUMMY_DOC_CONTENT:
                real_docs += 1
                if real_docs >= 1:
                    return False
        
        return True

    def _add_memory_to_cache(self, key: str, value: List[Generation]):
        if len(self.memory_cache) >= self.memory_cache_size:
            first_item = next(iter(self.memory_cache))
            self.memory_cache.pop(first_item)
        
        self.memory_cache[key] = value

    def update(self, prompt: str, llm_string: str, return_val: List[Generation]):
        super().update(prompt, llm_string, return_val)

        # Add to memory cache
        cache_key = f"{prompt}:{llm_string}"
        self._add_to_memory_cache(cache_key, llm_string)

        self._lazy_load_vector_store()

        if self.vector_store is None:
            print("Semantic vector store not initialized, attempting to re-initialize.")
            self._init_semantic_store()
            if self.vector_store is None:
                print("Failed to re-initialize semantic vector store.")
                return

        try:
            self._remove_dummy_doc()
            if len(self.vector_store.docstore._dict) >= self.max_cache_size:
                self._evict_oldest_entries()

            metadata = {"llm_string_key": llm_string, "type": "cache_entry", "timestamp": time.time()}
            self._add_to_vector_store(prompt, metadata)
            self._save_vector_store()
            
        except Exception as e:
            print(f"Error during semantic index update: {e}")

    async def update_async(self, prompt: str, llm_string: str, return_val: List[Generation]):
        super().update(prompt, llm_string, return_val)

        # Add to memory cache
        cache_key = f"{prompt}:{llm_string}"
        self._add_to_memory_cache(cache_key, llm_string)

        self._lazy_load_vector_store()

        if self.vector_store is None:
            print("Semantic vector store not initialized, attempting to re-initialize.")
            self._init_semantic_store()
            if self.vector_store is None:
                print("Failed to re-initialize semantic vector store.")
                return

        try:
            self._remove_dummy_doc()
            if len(self.vector_store.docstore._dict) >= self.max_cache_size:
                self._evict_oldest_entries()

            metadata = {"llm_string_key": llm_string, "type": "cache_entry", "timestamp": time.time()}
            await asyncio.get_event_loop().run_in_executor(self.executor, 
                                                            self._add_to_vector_store(prompt, metadata))
            
            await self._save_vector_store_async()
            
        except Exception as e:
            print(f"Error during semantic index update: {e}")


    
    def update(self, prompt: str, llm_string: str, return_val: List[Generation]):
        super().update(prompt, llm_string, return_val)

        # Add to memory cache
        cache_key = f"{prompt}:{llm_string}"
        self._add_to_memory_cache(cache_key, llm_string)

        self._lazy_load_vector_store()

        if self.vector_store is None:
            print("Semantic vector store not initialized, attempting to re-initialize.")
            self._init_semantic_store()
            if self.vector_store is None:
                print("Failed to re-initialize semantic vector store.")
                return

        try:
            self._remove_dummy_doc()
            if len(self.vector_store.docstore._dict) >= self.max_cache_size:
                self._evict_oldest_entries()

            metadata = {"llm_string_key": llm_string, "type": "cache_entry", "timestamp": time.time()}
            self._add_to_vector_store(prompt, metadata)
            self._save_vector_store()
            
        except Exception as e:
            print(f"Error during semantic index update: {e}")

    def _remove_dummy_doc(self):
        dummy_ids = [doc_id for doc_id, doc in self.vector_store.docstore._dict.items() if doc.page_content == DUMMY_DOC_CONTENT and doc.metadata.get("is_dummy")]
        if dummy_ids:
            self.vector_store.delete(dummy_ids)
            print(f"Removed {len(dummy_ids)} dummy documents.")

    async def _remove_dummy_doc_async(self):
        await asyncio.get_event_loop().run_in_executor(
            self.executor, self._remove_dummy_doc
        )

    def _evict_oldest_entries(self):

        docs_with_timestamps = [(doc_id, doc.metadata.get("timestamp", 0)) 
                               for doc_id, doc in self.vector_store.docstore._dict.items()
                               if not doc.metadata.get("is_dummy")]

        if len(docs_with_timestamps) > self.max_cache_size * 0.8:  # Evict 20% when near limit
            docs_with_timestamps.sort(key=lambda x: x[1])
            to_evict = docs_with_timestamps[:int(len(docs_with_timestamps) * 0.2)]
            evict_ids = [doc_id for doc_id, _ in to_evict]
            self.vector_store.delete(evict_ids)
            print(f"Evicted {len(evict_ids)} oldest entries from FAISS index.")

    async def _evict_oldest_entries_async(self):
        await asyncio.get_event_loop().run_in_executor(
            sel.executor, self._evict_oldest_entries
        )

    def _add_to_vector_store(self, prompt: str, metadata: Dict):
        self.vector_store.add_texts([prompt], metadatas=[metadata])

    def _save_to_vector_store(self):
        os.makedirs(self.faiss_index_path, exist_ok = True)
        self.vector_store.save_local(self.faiss_index_path)
        print(f"Updated FAISS index and saved to {self.faiss_index_path}")

    def get_metrics(self):
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        hit_rate = (self.metrics['cache_hits'] + self.metrics['semantic_hits'] + self.metrics['memory_hits']) / max(total_requests, 1)
        
        return {
            **self.metrics,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'avg_embedding_time': self.metrics['embedding_time'] / max(self.metrics['cache_misses'], 1),
            'avg_search_time': self.metrics['search_time'] / max(self.metrics['cache_misses'], 1)
        }

    def clear_cache(self):
        super().clear()
        self.memory_cache.clear()
        self.embedding_cache.clear()
        if os.path.exists(self.faiss_index_path):
            shutil.rmtree(self.faiss_index_path)
        self.vector_store = None
        self._lazy_loaded = False
        self.metrics = {k: 0 for k in self.metrics}
        print("All caches cleared.")