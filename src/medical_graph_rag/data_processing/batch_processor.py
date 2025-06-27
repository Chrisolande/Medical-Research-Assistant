"""Batchprocessor module."""

import asyncio
import logging
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Semaphore
from typing import Any

from langchain_core.documents import Document

from medical_graph_rag.core.config import (
    MIN_ABSTRACT_CONTENT_LENGTH,
    PMC_BATCH_SIZE,
    PMC_INTER_BATCH_DELAY,
    PMC_MAX_CONCURRENT_BATCHES,
    PMC_RETRY_ATTEMPTS,
    PMC_RETRY_DELAY,
)
from medical_graph_rag.core.utils import (
    create_batches,
    load_json_data,
    save_processing_results,
)


@dataclass
class PMCBatchProcessor:
    """PMCBatchProcessor class."""

    document_processor: Any
    batch_size: int = PMC_BATCH_SIZE
    max_concurrent_batches: int = PMC_MAX_CONCURRENT_BATCHES
    retry_attempts: int = PMC_RETRY_ATTEMPTS
    retry_delay: float = PMC_RETRY_DELAY
    inter_batch_delay: float = PMC_INTER_BATCH_DELAY

    def __post_init__(self):
        """Initialize post_init."""
        self.logger = logging.getLogger(__name__)
        self.processing_semaphore = Semaphore(self.max_concurrent_batches)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_batches)

    def load_pmc_data(
        self, file_path: str, max_docs: int | None = None
    ) -> list[dict[str, Any]]:
        pmc_docs = load_json_data(file_path, max_docs)

        # Make sure that the documents(all of them have abstracts)
        valid_docs = [
            doc
            for doc in pmc_docs
            if doc.get("abstract", "").strip()
            and len(doc.get("abstract", "").strip()) >= MIN_ABSTRACT_CONTENT_LENGTH
        ]

        if len(pmc_docs) != len(valid_docs):
            self.logger.info(
                f"Filtered {len(pmc_docs) - len(valid_docs)} documents lacking valid abstracts or too short."
            )
        self.logger.info(f"Found {len(valid_docs)} documents with valid abstracts.")
        return valid_docs

    def create_document_batches(
        self, pmc_docs: list[dict[str, Any]], batch_size: int | None = None
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Split the documents to batches."""
        effective_batch_size = batch_size or self.batch_size
        yield from create_batches(pmc_docs, effective_batch_size)

    def _process_batch_documents(self, batch: list[dict[str, Any]]) -> list[Document]:
        documents = []
        for doc in batch:
            content = doc.get("abstract", " ").strip()
            if content and len(content) >= MIN_ABSTRACT_CONTENT_LENGTH:
                langchain_doc = Document(
                    page_content=content,
                    metadata={
                        k: v
                        for k, v in doc.items()
                        if k != "abstract" and v is not None
                    },
                )
                documents.append(langchain_doc)

        processed_documents = self.document_processor.process_documents(documents)
        return processed_documents

    async def _process_batch_async(
        self, batch: list[dict[str, Any]], batch_num: int
    ) -> dict[str, Any]:
        for attempt in range(self.retry_attempts):
            try:
                self.logger.info(
                    f"Processing batch {batch_num} "
                    f"({len(batch)} documents, attempt {attempt + 1})"
                )

                documents = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._process_batch_documents, batch
                )
                return {
                    "batch_num": batch_num,
                    "success": True,
                    "documents": documents,
                    "original_count": len(batch),
                    "chunk_count": len(documents),
                    "error": None,
                    "attempt": attempt + 1,
                }
            except Exception as e:
                self.logger.warning(
                    f"Batch {batch_num} attempt {attempt + 1} failed: {str(e)}",
                    exc_info=True,
                )
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                else:
                    return {
                        "batch_num": batch_num,
                        "success": False,
                        "documents": [],
                        "original_count": len(batch),
                        "chunk_count": 0,
                        "error": str(e),
                        "attempt": attempt + 1,
                    }

    async def process_pmc_file_async(
        self,
        file_path: str,
        max_docs: int | None = None,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Loading PMC data from {file_path}")
        pmc_docs = self.load_pmc_data(file_path, max_docs)
        if not pmc_docs:
            self.logger.warning("No valid PMC documents found")
            return self._empty_result()

        # Create batches
        batches = list(self.create_document_batches(pmc_docs, batch_size))
        total_batches = len(batches)

        self.logger.info(
            f"Processing {len(pmc_docs)} documents in {total_batches} batches"
        )
        self.logger.info(f"Max concurrent batches: {self.max_concurrent_batches}")

        results = {"successful_batches": [], "failed_batches": [], "all_documents": []}

        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def process_with_semaphore(batch, batch_num):
            async with semaphore:
                result = await self._process_batch_async(batch, batch_num)
                if self.inter_batch_delay > 0:
                    await asyncio.sleep(self.inter_batch_delay)
                return result

        tasks = [
            process_with_semaphore(batch, i + 1) for i, batch in enumerate(batches)
        ]

        completed_batches = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed_batches += 1

            if result["success"]:
                results["successful_batches"].append(result)
                results["all_documents"].extend(result["documents"])
                self.logger.info(
                    f" Batch {result['batch_num']} completed: "
                    f"{result['original_count']} docs → {result['chunk_count']} chunks"
                )
            else:
                results["failed_batches"].append(result)
                self.logger.error(
                    f" Batch {result['batch_num']} failed: {result['error']}"
                )

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
            "success_rate": (
                len(results["successful_batches"]) / total_batches * 100
                if total_batches > 0
                else 0.0
            ),
            "processing_time": processing_time,
            "avg_time_per_batch": (
                processing_time / total_batches if total_batches > 0 else 0.0
            ),
            "docs_per_second": (
                len(pmc_docs) / processing_time if processing_time > 0 else 0
            ),
        }

        self.logger.info(f"Processing complete in {processing_time:.2f} seconds")
        self.logger.info(
            f"Success rate: {results['processing_summary']['success_rate']:.1f}%"
        )
        self.logger.info(
            f"Total chunks: {results['processing_summary']['total_chunks']}"
        )
        self.logger.info(
            f"Total documents: {results['processing_summary']['total_documents']}"
        )
        self.logger.info(
            f"Total batches: {results['processing_summary']['total_batches']}"
        )
        self.logger.info(
            f"Failed batches: {results['processing_summary']['failed_batches']}"
        )

        self.executor.shutdown(wait=True)

        return results

    async def process_pmc_file(self, *args, **kwargs):
        """Sync wrapper for async method."""
        return await self.process_pmc_file_async(*args, **kwargs)

    def _empty_result(self) -> dict[str, Any]:
        """Empty Result method."""
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
                "docs_per_second": 0.0,
            },
        }

    def save_results(
        self, results: dict[str, Any], output_dir: str, save_batch_details: bool = False
    ) -> None:
        """Save results to file."""
        save_processing_results(
            results=results,
            output_dir=output_dir,
            base_filename="pmc_chunks",
            batch_size=self.batch_size,
            source_type="pmc_abstracts",
            save_batch_details=save_batch_details,
        )
