import os
import hashlib
import json
from typing import Optional, List, Any
import shutil 

from langchain.cache import SQLiteCache
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS as FAISSStore
# from langchain_community.embeddings import HuggingFaceEmbeddings 
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
    self.vector_store: Optional[FAISSStore] = None
    self._init_semantic_store()

    
