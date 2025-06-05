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
from typing import Optional
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

    def create_document_batches(
        self, 
        pmc_docs: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Split the documents to batches"""
        if batch_size is None:
            batch_size = self.batch_size

        total_batches = math.ceil(len(pmc_docs) / batch_size)
        self.logger.info(f"Creating {total_batches} batches of size {batch_size}")

        for i in range(0, len(pmc_docs), batch_size):
            batch = pmc_docs[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            self.logger.debug(f"Created batch {batch_num}/{total_batches} with {len(batch)} documents")
            yield batch

    def _process_batch_with_retry(
        self, 
        batch: List[Dict[str, Any]], 
        batch_num: int,
        total_batches: int
    ) -> Dict[str, Any]:

        """Process the batches with exponential backoff"""
        for attempt in range(self.retry_attempts):
            try:
                # Apply rate limiting
                self._rate_limit_wait()
                self.logger.info(f"Processing batch {batch_num}/{total_batches} "
                               f"({len(batch)} documents, attempt {attempt + 1})")

                # Create temporary JSON for the batch
                temp_file = f"temp_batch_{batch_num}_{time.time()}.json"
                try:
                    with open(temp_file, "w", encoding = "utf-8") as f:
                        json.dump(batch, f, ensure_ascii=False, indent=2)

                    # Process the batch using the existing document processor
                    documents = self.document_processor.load_and_process_documents(
                        file_path=temp_file,
                        content_key="abstract",
                        jq_schema=".[]",
                        max_docs=None,  # Process all documents in the batch
                        min_chunk_size=50
                    )

                    # Get statistics
                    stats = self.document_processor.get_stats(documents)
                    
                    # Clean up temporary file
                    Path(temp_file).unlink(missing_ok=True)

                    return {
                        "batch_num": batch_num,
                        "success": True,
                        "documents": documents,
                        "stats": stats,
                        "original_count": len(batch),
                        "chunk_count": len(documents),
                        "error": None,
                        "attempt": attempt + 1
                    }
                    
                except Exception as e:
                    # Clean up temporary file on error
                    Path(temp_file).unlink(missing_ok=True)
                    raise e

            except Exception as e:
                self.logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {str(e)}")

                if attempt - self.retry_attempts - 1:
                    # Perform exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying batch {batch_num} in {wait_time} seconds...")
                    time.sleep(wait_time)

                else:
                    self.logger.error(f"All attempts failed for batch {batch_num}")
                    return {
                        "batch_num": batch_num,
                        "success": False,
                        "documents": [],
                        "stats": {},
                        "original_count": len(batch),
                        "chunk_count": 0,
                        "error": str(e),
                        "attempt": attempt + 1
                    }
    
    def process_pmc_file(
        self,
        file_path: str,
        max_docs: Optional[int] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ):

        start_time = time.time()
        self.logger.info(f"Loading PMC data from {file_path}")
        pmc_docs = self.load_pmc_data(file_path, max_docs)
        if not pmc_docs:
            self.logger.warning("No valid PMC documents found")
            return {
                "successful_batches": [],
                "failed_batches": [],
                "all_documents": [],
                "combined_stats": {},
                "processing_summary": {
                    "total_documents": 0,
                    "total_batches": 0,
                    "successful_batches": 0,
                    "failed_batches": 0,
                    "total_chunks": 0,
                    "success_rate": 0.0,
                    "processing_time": 0.0
                }
            }
        
        # Create batches
        batches = list(self.create_document_batches(pmc_docs, batch_size))
        total_batches = len(batches)
        
        self.logger.info(f"Processing {len(pmc_docs)} documents in {total_batches} batches")
        self.logger.info(f"Max concurrent batches: {self.max_concurrent_batches}")
        
        results = {
            "successful_batches": [],
            "failed_batches": [],
            "all_documents": [],
            "combined_stats": {},
            "processing_summary": {}
        }
        
        completed_batches = 0

        # Process batches with controlled concurrency
        with ThreadPoolExecutor(max_workers=self.max_concurrent_batches) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(
                    self._process_batch_with_retry,
                    batch,
                    i + 1,
                    total_batches
                ): (batch, i + 1) for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                result = future.result()
                completed_batches += 1
                
                if result["success"]:
                    results["successful_batches"].append(result)
                    results["all_documents"].extend(result["documents"])
                    self.logger.info(f" Batch {result['batch_num']} completed: "
                                   f"{result['original_count']} docs → {result['chunk_count']} chunks")
                else:
                    results["failed_batches"].append(result)
                    self.logger.error(f" Batch {result['batch_num']} failed: {result['error']}")
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed_batches, total_batches, result)
                
                if completed_batches < total_batches:
                    time.sleep(self.inter_batch_delay)
        
        # Generate final statistics
        processing_time = time.time() - start_time
        results["combined_stats"] = self._generate_combined_stats(results["all_documents"])
        results["processing_summary"] = {
            "total_documents": len(pmc_docs),
            "total_batches": total_batches,
            "successful_batches": len(results["successful_batches"]),
            "failed_batches": len(results["failed_batches"]),
            "total_chunks": len(results["all_documents"]),
            "success_rate": len(results["successful_batches"]) / total_batches * 100,
            "processing_time": processing_time,
            "avg_time_per_batch": processing_time / total_batches,
            "docs_per_second": len(pmc_docs) / processing_time if processing_time > 0 else 0
        }
        
        self.logger.info(f"PMC processing complete in {processing_time:.2f} seconds")
        self.logger.info(f"Success rate: {results['processing_summary']['success_rate']:.1f}%")
        self.logger.info(f"Total chunks created: {results['processing_summary']['total_chunks']}")
        
        return results

    def _generate_combined_stats(self, all_documents: List[Document]) -> Dict[str, Any]:
        """Generate combined statistics for all processed documents"""
        if not all_documents:
            return {}
        
        chunk_sizes = [len(doc.page_content) for doc in all_documents]
        total_chars = sum(chunk_sizes)
        
        # Count unique PMIDs
        unique_pmids = set()
        for doc in all_documents:
            pmid = doc.metadata.get("pmid")
            if pmid:
                unique_pmids.add(str(pmid))
        
        return {
            "total_chunks": len(all_documents),
            "total_characters": total_chars,
            "average_chunk_size": round(total_chars / len(all_documents), 2),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "unique_pmids": len(unique_pmids),
            "chunk_size_distribution": {
                "q25": sorted(chunk_sizes)[len(chunk_sizes)//4],
                "q50": sorted(chunk_sizes)[len(chunk_sizes)//2],
                "q75": sorted(chunk_sizes)[3*len(chunk_sizes)//4]
            }
        }

    def save_results(
        self, 
        results: Dict[str, Any], 
        output_dir: str,
        save_batch_details: bool = False
    ) -> None:
        """Save results to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results file
        main_output = {
            "processing_info": {
                "source_type": "pmc_abstracts",
                "batch_processing": True,
                "chunking_method": "semantic_chunking",
                "embeddings_model": self.document_processor.embeddings_model,
                "batch_size": self.batch_size,
                "total_batches": results["processing_summary"]["total_batches"]
            },
            "summary": results["processing_summary"],
            "combined_stats": results["combined_stats"],
            "documents": []
        }

        # Add all documents
        for i, doc in enumerate(results["all_documents"]):
            main_output["documents"].append({
                "chunk_id": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "content_length": len(doc.page_content)
            })
        
        # Save main results
        main_path = output_path / "pmc_semantic_chunks.json"
        with open(main_path, 'w', encoding='utf-8') as f:
            json.dump(main_output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(results['all_documents'])} chunks to {main_path}")

        # Save processing log
        log_data = {
            "processing_summary": results["processing_summary"],
            "successful_batches": [
                {
                    "batch_num": b["batch_num"],
                    "original_count": b["original_count"],
                    "chunk_count": b["chunk_count"],
                    "attempts": b["attempt"]
                }
                for b in results["successful_batches"]
            ],
            "failed_batches": [
                {
                    "batch_num": b["batch_num"],
                    "original_count": b["original_count"],
                    "error": b["error"],
                    "attempts": b["attempt"]
                }
                for b in results["failed_batches"]
            ]
        }
        
        log_path = output_path / "processing_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved processing log to {log_path}")
        
        # Save detailed batch information if requested
        if save_batch_details:
            batch_dir = output_path / "batch_details"
            batch_dir.mkdir(exist_ok=True)
            
            for batch_result in results["successful_batches"]:
                batch_file = batch_dir / f"batch_{batch_result['batch_num']:03d}.json"
                batch_data = {
                    "batch_info": {
                        "batch_num": batch_result["batch_num"],
                        "original_count": batch_result["original_count"],
                        "chunk_count": batch_result["chunk_count"]
                    },
                    "stats": batch_result["stats"],
                    "documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "content_length": len(doc.page_content)
                        }
                        for doc in batch_result["documents"]
                    ]
                }
                
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, indent=2, ensure_ascii=False)
                    

