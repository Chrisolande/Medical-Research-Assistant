"""Document Processor module."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentProcessor:
    """DocumentProcessor class."""

    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    metadata_fields: list[str] = field(
        default_factory=lambda: [
            "pmid",
            "title",
            "authors",
            "journal",
            "volume",
            "issues",
            "year",
            "month",
            "day",
            "pub_date",
            "doi",
            "pmc_id",
            "mesh_terms",
            "publication_types",
            "doi_url",
            "pubmed_url",
        ]
    )

    def __post_init__(self):
        """Initialize post_init."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=75
        )
        logger.info(f"Initialized DocumentProcessor with {self.embeddings_model}")

    def metadata_func(
        self, record: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Metadata Func."""
        metadata.update(
            {field: str(record.get(field, "")) for field in self.metadata_fields}
        )
        metadata["embeddings_model"] = self.embeddings_model
        return metadata

    def _validate_and_clean(self, file_path: str, text: str) -> tuple[Path, str]:
        """Validate And Clean method."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if not text:
            return path, ""
        return path, re.sub(
            r"\s+", " ", re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text.strip())
        )

    def load_and_process_documents(
        self,
        file_path: str,
        content_key: str = "abstract",
        jq_schema: str = ".[]",
        max_docs: int | None = None,
        min_chunk_size: int = 50,
    ) -> list[Document]:
        """Load and process documents."""
        try:
            validated_path, _ = self._validate_and_clean(file_path, "")
            loader = JSONLoader(
                str(validated_path), jq_schema, content_key, self.metadata_func
            )
            documents = loader.load()[:max_docs] if max_docs else loader.load()
            documents = [doc for doc in documents if doc.page_content]

            if not documents:
                return []

            # Filter and preprocess in one pass
            valid_docs = []
            for doc in documents:
                _, clean_content = self._validate_and_clean("", doc.page_content)
                if len(clean_content.strip()) >= min_chunk_size:
                    doc.page_content = clean_content
                    valid_docs.append(doc)

            if not valid_docs:
                return []

            # This method now just loads and validates, it doesn't split here.
            # Splitting happens in process_documents
            logger.info(
                f"Loaded and pre-processed {len(valid_docs)} documents from {file_path}"
            )
            return valid_docs

        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise ValueError(f"Failed to process documents: {e}") from e

    def process_documents(
        self, documents: list[Document], min_chunk_size: int = 50
    ) -> list[Document]:
        """Processes a list of LangChain Document objects by splitting them into chunks
        and filtering based on minimum chunk size."""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        final_chunks = [
            doc for doc in chunks if len(doc.page_content.strip()) >= min_chunk_size
        ]

        logger.info(f"Processed {len(documents)} docs into {len(final_chunks)} chunks.")
        return final_chunks

    def get_stats(self, documents: list[Document]) -> dict[str, Any]:
        """Get Stats method."""
        if not documents:
            return {"total_chunks": 0, "total_characters": 0, "average_chunk_size": 0}

        chunk_sizes = [len(doc.page_content) for doc in documents]
        return {
            "total_chunks": len(documents),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": round(sum(chunk_sizes) / len(documents), 2),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "embeddings_model": self.embeddings_model,
        }

    def save_processed_documents(
        self, documents: list[Document], output_path: str
    ) -> None:
        """Save proessed documents to a json format."""
        data = {
            "processing_info": {
                "embeddings_model": self.embeddings_model,
                "total_chunks": len(documents),
            },
            "documents": [
                {"chunk_id": i, "content": doc.page_content, "metadata": doc.metadata}
                for i, doc in enumerate(documents)
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} documents to {output_path}")
