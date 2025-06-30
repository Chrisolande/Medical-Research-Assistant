"""Utils module."""

import hashlib
import json
import logging
import math
import pickle  # nosec - pickle usage reviewed for security
import re
import textwrap
from asyncio import get_event_loop
from collections.abc import Generator
from pathlib import Path
from typing import Any

from langchain.globals import set_llm_cache
from langchain_core.documents import Document

from medical_graph_rag.core.common_helpers import log_error, log_info
from medical_graph_rag.core.config import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_FAISS_INDEX_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    ENABLE_QUANTIZATION,
)

# ---------------------------------------------------------------------------- #
#                               Logging Configuration                          #
# ---------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#                             Semantic Cache Setup                             #
# ---------------------------------------------------------------------------- #

# This block sets up the global LLM cache. It will run when utils.py is imported.
_semantic_cache_instance = None


def ensure_semantic_cache():
    """Ensure semantic cache is initialized globally."""
    global _semantic_cache_instance
    if _semantic_cache_instance is None:
        from medical_graph_rag.nlp.prompt_caching import SemanticCache

        _semantic_cache_instance = SemanticCache(
            database_path=DEFAULT_DATABASE_PATH,
            faiss_index_path=DEFAULT_FAISS_INDEX_PATH,
            similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
            enable_quantization=ENABLE_QUANTIZATION,
        )
        set_llm_cache(_semantic_cache_instance)
        logger.info("Semantic cache initialized and set globally.")
    return _semantic_cache_instance


# ---------------------------------------------------------------------------- #
#                             Document Printing Utilities                      #
# ---------------------------------------------------------------------------- #


def pretty_print_docs(docs, wrap_width: int = 80, queries: list[str] | None = None):
    """Prints a list of Document objects or batch results with their page_content
    cleaned and wrapped, separated by a visual delimiter.

    Args:
        docs: A list of Document objects OR a list of lists of Document objects (batch results).
        wrap_width (int): The maximum width for wrapping the text.
        queries (List[str]): Optional list of query strings for batch results.
    """
    # Handle batch results (list of lists)
    if docs and isinstance(docs[0], list):
        for query_idx, query_docs in enumerate(docs):
            print(f"\n{'='*wrap_width}")
            query_text = (
                f": {queries[query_idx]}"
                if queries and query_idx < len(queries)
                else ""
            )
            print(f"QUERY {query_idx + 1} RESULTS{query_text}")
            print(f"{'='*wrap_width}")
            _print_single_query_docs(query_docs, wrap_width)
        return

    # single query handling
    _print_single_query_docs(docs, wrap_width)


def _print_single_query_docs(docs: list[Document], wrap_width: int):
    """Helper function for printing a single list of documents."""
    if not docs:
        print("No documents to display.")
        return

    formatted_docs = []
    for i, d in enumerate(docs):
        content = d.page_content if d.page_content is not None else ""
        wrapped_content = textwrap.fill(content.strip(), width=wrap_width)

        metadata_str = ""
        if hasattr(d, "metadata") and d.metadata:
            metadata_str = f"\nMetadata: {d.metadata}"

        formatted_docs.append(f"Document {i+1}:{metadata_str}\n\n{wrapped_content}")

    separator = f"\n{'-' * (wrap_width if wrap_width > 10 else 10)}\n"
    print(separator.join(formatted_docs))


def print_filtered_content(
    traversal_path: list[int],
    filtered_content: dict[int, str],
    content_preview_length: int = 200,
) -> None:
    """Print the filtered content of visited nodes in traversal order.

    Args:
    traversal_path (List[int]): The list of nodes to print the filtered content for.
    filtered_content (Dict[int, str]): A mapping of node IDs to the filtered content.
    content_preview_length (int, optional): The length of the content preview. Defaults to 200.
    """
    logger.info(f"Printing filtered content for {len(traversal_path)} nodes")

    if not traversal_path:
        logger.warning("Empty traversal path provided.")
        return

    print("\n" + "=" * 80)
    print("FILTERED CONTENT OF VISITED NODES (IN TRAVERSAL ORDER)")
    print("=" * 80)

    for i, node in enumerate(traversal_path):
        content = filtered_content.get(node, "No filtered content available")
        preview = content[:content_preview_length]
        if len(content) > content_preview_length:
            preview += "..."

        print(f"\n Step {i + 1} - Node {node}")
        print("-" * 50)
        print(f"Content Preview: {preview}")
        print("-" * 50)

    print(f"\n Completed traversal of {len(traversal_path)} nodes")
    logger.info("Content printing completed.")


# ---------------------------------------------------------------------------- #
#                               Cache Management Utilities                     #
# ---------------------------------------------------------------------------- #


class CacheManager:
    """Handles caching operations for embeddings and concepts using pickle."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize init."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        log_info(f"CacheManager initialized. Cache directory: {self.cache_dir}")

    def load_cache(self) -> dict[str, Any]:
        """Loads cache from disk."""
        cache_file = self.cache_dir / "cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)  # nosec
                    log_info(f"Cache loaded from {cache_file}.")
                    return data
            except Exception as e:
                log_error(f"Failed to load cache from {cache_file}: {e}", exc_info=True)
                return {}
        log_info(f"No cache file found at {cache_file}. Starting with empty cache.")
        return {}

    def save_cache(self, data: dict[str, Any]) -> None:
        """Saves cache to disk."""
        cache_file = self.cache_dir / "cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            log_info(f"Cache saved to {cache_file}.")
        except Exception as e:
            log_error(f"Failed to save cache to {cache_file}: {e}", exc_info=True)


# ---------------------------------------------------------------------------- #
#                            JSON & Text Processing Utilities                  #
# ---------------------------------------------------------------------------- #


def extract_and_parse_json(text: str) -> dict[str, list[str]] | None:
    """Extract and parse JSON with multiple fallback strategies from a given text."""

    json_patterns = [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Handle nested braces
        r"\{.*?\}(?=\s*$|\s*\n)",  # JSON followed by end/newline
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                cleaned = re.sub(r"^[^{]*(\{)", r"\1", cleaned)
                cleaned = re.sub(r"(\})[^}]*$", r"\1", cleaned)
                cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
                cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
                cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)

                cleaned = "".join(
                    char for char in cleaned if ord(char) >= 32 or char in "\n\t"
                )

                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                continue

    # Line-by-line extraction
    concept_dict = {}
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        patterns = [
            r'"?(\d+)"?\s*:\s*\[(.*?)\]',  # e.g., "0": ["concept1", "concept2"] or 0: ["concept1"]
            r'"(\d+)":\s*"([^"]+)"',  # e.g., "0": "concept1"
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                key = match.group(1)
                if len(match.groups()) == 2 and pattern == r'"(\d+)":\s*"([^"]+)"':
                    if pattern == r'"(\d+)":\s*"([^"]+)"':
                        concepts = [match.group(2)]
                    else:
                        values = match.group(2)
                        # Try to find quoted concepts first
                        concepts = re.findall(r'"([^"]+)"', values)
                        if not concepts:
                            # Fallback to comma-separated unquoted concepts
                            concepts = [
                                c.strip() for c in values.split(",") if c.strip()
                            ]

                    if concepts:
                        concept_dict[key] = concepts
                break

    # Extract any quoted strings as potential concepts (last resort)
    if not concept_dict:
        all_quotes = re.findall(r'"([^"]+)"', text)
        if all_quotes and len(all_quotes) >= 2:
            # Group quotes into potential concept lists (heuristic)
            for i in range(0, len(all_quotes), 3):
                if i // 3 < 10:  # Limit to ~10 documents for this heuristic
                    concepts = all_quotes[i : i + 3]
                    concept_dict[str(i // 3)] = concepts

    return concept_dict if concept_dict else None


def create_text_hash(text: str) -> str:
    """Creates an MD5 hash for text, useful for caching."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # nosec


# ---------------------------------------------------------------------------- #
#                               Graph Utilities                                #
# ---------------------------------------------------------------------------- #


def calculate_edge_weight(
    similarity_score: float,
    shared_concepts: list[str],
    node1_concepts: list[str],
    node2_concepts: list[str],
    similarity_weight: float = 0.7,
) -> float:
    """Calculates an edge weight based on a similarity score and shared concepts between
    two nodes."""
    max_shared = min(len(node1_concepts), len(node2_concepts))
    concept_score = len(shared_concepts) / max_shared if max_shared > 0 else 0
    return (
        similarity_weight * similarity_score + (1 - similarity_weight) * concept_score
    )


# ---------------------------------------------------------------------------- #
#                             Batch Processing Utilities                       #
# ---------------------------------------------------------------------------- #


def load_json_data(
    file_path: str, max_items: int | None = None
) -> list[dict[str, Any]]:
    """Loads JSON data from a file, with an option to limit the number of items."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if (
            isinstance(data, dict)
            and len(data) == 1
            and isinstance(list(data.values())[0], list)
        ):
            items = list(data.values())[0]
        elif isinstance(data, list):
            items = data
        else:
            logger.warning(
                f"Unexpected JSON structure in {file_path}. Expected a list or a dict with one list value."
            )
            return []

        if max_items and max_items > 0:
            items = items[:max_items]
            logger.info(f"Limited to first {max_items} items.")

        logger.info(f"Loaded {len(items)} items from {file_path}.")
        return items
    except Exception as e:
        logger.error(
            f"Error loading JSON data from {file_path}: {str(e)}", exc_info=True
        )
        raise ValueError(f"Failed to load data from {file_path}: {str(e)}") from e


def create_batches(
    items: list[Any], batch_size: int
) -> Generator[list[Any], None, None]:
    """Generates batches from a list of items."""
    if not items:
        return  # Yield nothing for empty input

    total_batches = math.ceil(len(items) / batch_size)
    logger.info(f"Creating {total_batches} batches of size {batch_size}.")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def save_processing_results(
    results: dict[str, Any],
    output_dir: str,
    base_filename: str,
    batch_size: int,
    source_type: str,
    save_batch_details: bool = False,
) -> None:
    """Saves processing results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    main_output = {
        "processing_info": {
            "source_type": source_type,
            "batch_processing": True,
            "batch_size": batch_size,
            "total_batches": results["processing_summary"]["total_batches"],
        },
        "summary": results["processing_summary"],
        "documents": [
            {
                "chunk_id": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "content_length": len(doc.page_content),
            }
            for i, doc in enumerate(results["all_documents"])
        ],
    }

    main_path = output_path / f"{base_filename}.json"
    try:
        with open(main_path, "w", encoding="utf-8") as f:
            json.dump(main_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results['all_documents'])} chunks to {main_path}")
    except Exception as e:
        logger.error(
            f"Error saving main results to {main_path}: {str(e)}", exc_info=True
        )

    if save_batch_details:
        batch_dir = output_path / "batch_details"
        batch_dir.mkdir(exist_ok=True)

        for batch_result in results["successful_batches"]:
            batch_file = batch_dir / f"batch_{batch_result['batch_num']:03d}.json"
            batch_data = {
                "batch_info": {
                    "batch_num": batch_result["batch_num"],
                    "original_count": batch_result["original_count"],
                    "chunk_count": batch_result["chunk_count"],
                },
                "documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "content_length": len(doc.page_content),
                    }
                    for doc in batch_result["documents"]
                ],
            }
            try:
                with open(batch_file, "w", encoding="utf-8") as f:
                    json.dump(batch_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(
                    f"Error saving batch {batch_result['batch_num']} to {batch_file}: {str(e)}"
                )


# ---------------------------------------------------------------------------- #
#                              Asynchronous operations                         #
# ---------------------------------------------------------------------------- #


async def run_in_executor(func, *args, **kwargs):
    return await get_event_loop().run_in_executor(None, func, *args, **kwargs)
