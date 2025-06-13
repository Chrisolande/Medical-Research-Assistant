import asyncio
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Generator
from pathlib import Path
import logging
from dataclasses import dataclass, field
from langchain_core.documents import Document
from threading import Semaphore
import math
from typing import Optional
@dataclass
class PMCBatchProcessor:
    document_processor: "DocumentProcessor"
    batch_size: int = 96
    max_concurrent_batches: int = 3
    retry_attempts: int = 2
    retry_delay: float = 1.0
    inter_batch_delay: float = 0.1 # Delay between batches

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.processing_semaphore = Semaphore(self.max_concurrent_batches)

    
    def load_pmc_data(self, file_path:str, max_docs: Optional[int] = None):
        try:
            with open(file_path, "r", encoding = "utf-8") as f:
                data = json.load(f)

            pmc_docs = data if isinstance(data, list) else list(data.values())[0]

            # Limit the number of documents
            if max_docs and max_docs > 0:
                pmc_docs = pmc_docs[:max_docs]
                self.logger.info(f"Limited to first {max_docs} documents")
        
            self.logger.info(f"Loaded {len(pmc_docs)} PMC documents from {file_path}")

            # Make sure that the documents(all of them have abstracts)  
            valid_docs = [doc for doc in pmc_docs if doc.get('abstract', '').strip()]
                        
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
        batch_size = batch_size or self.batch_size

        total_batches = math.ceil(len(pmc_docs) / batch_size)
        self.logger.info(f"Creating {total_batches} batches of size {batch_size}")

        for i in range(0, len(pmc_docs), batch_size):
            yield pmc_docs[i:i + batch_size]
            

    def _process_batch_documents(self, batch):
        documents = []
        for doc in batch:
            content = doc.get("abstract", ' ').strip()
            if content and len(content) >= 50:
                langchain_doc = Document(
                    page_content = content, 
                    metadata = {k:v for k, v in doc.items() if k != 'abstract' and v is not None}
                )
                documents.append(langchain_doc)
        
        return documents
    
    async def _process_batch_async(self, batch, batch_num, total_batches):
        for attempt in range(self.retry_attempts):
            try:
                self.logger.info(f"Processing batch {batch_num}/{total_batches} "
                               f"({len(batch)} documents, attempt {attempt + 1})")
                
                documents = await asyncio.get_event_loop().run_in_executor(None, self._process_batch_documents, batch)
                return {
                    "batch_num": batch_num,
                    "success": True,
                    "documents": documents,
                    "original_count": len(batch),
                    "chunk_count": len(documents),
                    "error": None,
                    "attempt": attempt + 1
                }   
            except Exception as e:
                self.logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return {
                        "batch_num": batch_num,
                        "success": False,
                        "documents": [],
                        "original_count": len(batch),
                        "chunk_count": 0,
                        "error": str(e),
                        "attempt": attempt + 1
                    }   
    
    async def process_pmc_file_async(
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
            return self._empty_result()
        
        # Create batches
        batches = list(self.create_document_batches(pmc_docs, batch_size))
        total_batches = len(batches)
        
        self.logger.info(f"Processing {len(pmc_docs)} documents in {total_batches} batches")
        self.logger.info(f"Max concurrent batches: {self.max_concurrent_batches}")
        
        results = {
            "successful_batches": [],
            "failed_batches": [],
            "all_documents": [],
        }

        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def process_with_semaphore(batch, batch_num):
            async with semaphore:
                result = await self._process_batch_async(batch, batch_num, total_batches)
                if self.inter_batch_delay > 0:
                    await asyncio.sleep(self.inter_batch_delay)
                    return result
            
        tasks = [process_with_semaphore(batch, i + 1) for i, batch in enumerate(batches)] 
        
        completed_batches = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
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
                    
        # Generate final statistics
        processing_time = time.time() - start_time
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
        
        self.logger.info(f"Processing complete in {processing_time:.2f} seconds")
        self.logger.info(f"Success rate: {results['processing_summary']['success_rate']:.1f}%")
        
        return results
    
    async def process_pmc_file(self, *args, **kwargs):
        """Sync wrapper for async method"""
        return await self.process_pmc_file_async(*args, **kwargs)

    def _empty_result(self):
        return {
            "successful_batches": [],
            "failed_batches": [],
            "all_documents": [],
            "processing_summary": {
                "total_documents": 0,
                "total_batches": 0,
                "successful_batches": 0,
                "failed_batches": 0,
                "total_chunks": 0,
                "success_rate": 0.0,
                "processing_time": 0.0,
                "avg_time_per_batch": 0.0,
                "docs_per_second": 0.0
            }
        }

    def save_results(self, results: Dict[str, Any], output_dir: str, 
                    save_batch_details: bool = False) -> None:
        """Save results to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results file
        main_output = {
            "processing_info": {
                "source_type": "pmc_abstracts",
                "batch_processing": True,
                "batch_size": self.batch_size,
                "total_batches": results["processing_summary"]["total_batches"]
            },
            "summary": results["processing_summary"],
            "documents": [
                {
                    "chunk_id": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                }
                for i, doc in enumerate(results["all_documents"])
            ]
        }
        
        # Save main results
        main_path = output_path / "pmc_semantic_chunks.json"
        with open(main_path, 'w', encoding='utf-8') as f:
            json.dump(main_output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(results['all_documents'])} chunks to {main_path}")
        
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
                    

