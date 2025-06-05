import asyncio
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Generator
from pathlib import Path
import logging
from dataclasses import dataclass, field
from langchain_core.documents import Document
from threading import Semaphore, Lock
from queue import Queue
import math

@dataclass
class PMCBatchProcessor:
    document_processor: "DocumentProcessor"
    batch_size: int = 96
    max_concurrent_batches: int = 3
    max_api_calls_per_minute: int = 2000
    retry_attempts: int = 3
    retry_delay: float = 1.0 # Initial delay between retries
    inter_batch_delay: float = 0.5 # Delay between batches

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_semaphore = Semaphore(self.max_api_calls_per_minute)
        self.rate_limit_lock = Lock()
        self.api_call_times = Queue()

    def _rate_limit_wait(self):
        """Implement rate limiting for api calls"""
        with self.rate_limit_lock:
            current_time = time.time()

            # Remove calls older than 1 minute
            temp_queue = Queue()
            while not self.api_call_times.empty():
                call_time = self.api_call_times.get()
                if current_time - call_time <= 60:
                    temp_queue.put(call_time)
            
            # Replace the queue with filtered call times
            self.api_call_times = temp_queue

            # Check if we need to wait
            if self.api_call_times.qsize() >= self.max_api_calls_per_minute:
                # Get the oldest call time
                oldest_call = min(self.api_call_times.queue)
                wait_time = 60 - (current_time - oldest_call) + 1
                if wait_time > 0:
                    self.logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
            
            # Record this API call
            self.api_call_times.put(current_time)

    def load_pmc_data(self, file_path:str, max_docs: Optional[int] = None):
        try:
            with open(file_path, "r", encoding = "utf-8") as f:
                data = json.load(f)
            fields = [
                    "pmid", "title", "authors", "journal", "volume", "issues", 
                    "year", "month", "day", "pub_date", "doi", "pmc_id", 
                    "mesh_terms", "publication_types", "doi_url", "pubmed_url", "abstract"
                    ]
            # Handle different JSON structures
            if isinstance(data, list):
                pmc_docs = data
            elif isinstance(data, dict):
                # Try common keys that might contain the array
                for key in fields:
                    if key in data and isinstance(data[key], list):
                        pmc_docs = data[key]
                        break
            else:
                raise ValueError(f"Unexpected JSON structure: {type(data)}")

            # Limit the number of documents
            if max_docs and max_docs > 0:
                pmc_docs = pmc_docs[:max_docs]
                self.logger.info(f"Limited to first {max_docs} documents")
        
            self.logger.info(f"Loaded {len(pmc_docs)} PMC documents from {file_path}")

            # Make sure that the documents(all of them have abstracts)   
            valid_docs = []
            for i, doc in enumerate(pmc_docs):
                if 'abstract' in doc and doc['abstract'] and doc['abstract'].strip():
                    valid_docs.append(doc)
                else:
                    self.logger.warning(f"Document {i} missing or empty abstract")
            
            self.logger.info(f"Found {len(valid_docs)} documents with valid abstracts")
            return valid_docs
        
        except Exception as e:
            self.logger.error(f"Error loading PMC data: {str(e)}")
            raise ValueError(f"Failed to load PMC data: {str(e)}")

