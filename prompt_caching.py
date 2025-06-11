"""main logic comes from here https://api.python.langchain.com/en/latest/cache/langchain_community.cache.SQLiteCache.html"""


import os
from typing import Optional, List

from langchain_community.cache import SQLiteCache
from langchain_community.vectorstores import FAISS
from langchain.schema import Generation
from langchain_cohere import CohereEmbeddings

DUMMY_DOC_CONTENT = "Langchain Document Initializer"

class SemanticSQLiteCache(SQLiteCache): # Inherit from sqlitecache
    def __init__(self, database_path: str = ".langchain.db", faiss_index_path: str = "./faiss_index", similarity_threshold: float = 0.5):
        super().__init__(database_path)
        self.similarity_threshold = similarity_threshold
        self.faiss_index_path = faiss_index_path

        cohere_api_key = os.getenv("COHERE_API_KEY1")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY1 environment variable not set. Please set it to use CohereEmbeddings.")

        self.embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")
        self.vector_store: Optional[FAISS] = None
        self._init_semantic_store()

    def _init_semantic_store(self):
        if os.path.exists(self.faiss_index_path) and os.path.isdir(self.faiss_index_path):
            try:
                self.vector_store = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"Loaded existing FAISS index from {self.faiss_index_path}")

            except Exception as e:
                print(f"Warning: Could not load FAISS index from {self.faiss_index_path}: {e}")
                print("Creating new vector index")
                self._create_new_faiss_index()
        else:
            print(f"FAISS index path {self.faiss_index_path} not found or not a directory. Creating a new index.")
            self._create_new_faiss_index()

    def _create_new_faiss_index(self):
        try:
            self.vector_store = FAISS.from_texts(
                [DUMMY_DOC_CONTENT], self.embeddings, metadatas=[{"type": "initializer", "is_dummy": True}]
            )

            print("Created a new FAISS index.")
        except Exception as e:
            print(f"Error: Failed to initialize FAISS vector store: {e}")
            self.vector_store = None

    def _create_new_faiss_index(self):
        try:
            self.vector_store = FAISS.from_texts(
                [DUMMY_DOC_CONTENT], self.embeddings, metadatas=[{"type": "initializer", "is_dummy": True}]
            )
            print("Created a new FAISS index.")
        except Exception as e:
            print(f"ERROR: Failed to initialize the new vector store {str(e)}")
            self.vector_store = None

    def lookup(self, prompt: str, llm_string: str)-> Optional[List[Generation]]: # From langchain official documentation
        result = super().lookup(prompt, llm_string)

        if result:
            print("Exact match found in SQLite cache.")
            return result

        if self.vector_store is None:
            print("Semantic vector store not initialized. Skipping semantic search.")
            return None

        try:
            # If the index is NOT dummy-only (i.e., contains actual cached data), proceed with semantic search.
            is_dummy_only = True
            if len(self.vector_store.docstore._dict) > 0:
                for doc_id, doc in self.vector_store.docstore._dict.items():
                    if doc.page_content != DUMMY_DOC_CONTENT:
                        is_dummy_only = False
                        break

            if is_dummy_only:
                print("FAISS index contains only initializer doc. Skipping semantic search.")
                return None

            docs_with_score = self.vector_store.similarity_search_with_score(
                prompt, k=1
            )
            
            if docs_with_score:
                doc, score = docs_with_score[0]
                if doc.page_content == DUMMY_DOC_CONTENT and doc.metadata.get("is_dummy"):
                    print("Semantic search returned dummy document. Skipping.")
                    return None

            if score <= self.similarity_threshold:
                cached_llm_string = doc.metadata.get("llm_string_key")
                original_cached_prompt = doc.page_content

                if cached_llm_string and original_cached_prompt:
                        print(f"Semantic match found with score {score:.4f}. Retrieving from SQLite cache.")
                        return super().lookup(original_cached_prompt, cached_llm_string)

                else:
                    print(f"Semantic match found with score {score:.4f}, but it's above the threshold ({self.similarity_threshold}). Not using cache.")

        except Exception as e:
            print(f"Error during semantic lookup: {e}")
        
        return None

    def update(self, prompt: str, llm_string: str, return_val: List[Generation]):
        super().update(prompt, llm_string, return_val)

        if self.vector_store is None:
            print("Semantic vector store not initialized during update, attempting to re-initialize.")
            self._init_semantic_store()
            if self.vector_store is None:
                print("Failed to re-initialize semantic vector store. Skipping semantic update.")
                return
        
        try:
            dummy_doc_id = None
            for doc_id, doc in self.vector_store.docstore._dict.items():
                if doc.page_content == DUMMY_DOC_CONTENT and doc.metadata.get("is_dummy"):
                    dummy_doc_id = doc_id
                    break

            if dummy_doc_id:
                self.vector_store.delete([dummy_doc_id])
                print("Removed dummy initializer document from FAISS index.")

            metadata = {"llm_string_key": llm_string, "type": "cache_entry"}
            self.vector_store.add_texts([prompt], metadatas=[metadata])

            os.makedirs(self.faiss_index_path, exist_ok=True)
            self.vector_store.save_local(self.faiss_index_path)
            print(f"Updated FAISS index and saved to {self.faiss_index_path}")
            
        except Exception as e:
            print(f"Error during semantic index update or save: {e}")
        

            


    
