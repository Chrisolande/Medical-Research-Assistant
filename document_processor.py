from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import os
import re
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessor:
    embeddings_model: str = "embed_english_v3.0"
    breakpoint_threshold_type: str = "semantic"
    breakpoint_threshold_amount: Optional[int] = None
    number_of_chunks: Optional[int] = None
    metadata_fields: List[str] = field(default_factory=lambda: [
        "pmid", "title", "authors", "journal", "volume", "issues", 
        "year", "month", "day", "pub_date", "doi", "pmc_id", 
        "mesh_terms", "publication_types", "doi_url", "pubmed_url"
    ])
    cohere_api_key: Optional[str] = None

    def __post_init__(self):
        self.api_key = cohere_api_key or self.cohere_api_key
        if not self.api_key:
            raise ValueError("Cohere API KEY not found or is invalid. Please enter a valid API KEY")
        
        # Initialize Embeddings
        try:
            self.embeddings = CohereEmbeddings(model = self.embeddings_model,
                            api_key = self.api_key)
            # Initialize semantic chunker with appropriate parameters
            chunker_kwargs = {
                "embeddings": self.embeddings,
                "breakpoint_threshold_type": self.breakpoint_threshold_type
            }

            # Add parameters based on chunking type
            if self.breakpoint_threshold_type == "percentile":
                if self.breakpoint_threshold_amount:
                    chunker_kwargs["breakpoint_threshold_amount"] = self.breakpoint_threshold_amount
                if self.number_of_chunks:
                    chunker_kwargs["number_of_chunks"] = self.number_of_chunks
            
            elif self.breakpoint_threshold_type in ["interquartile", "standard_deviation"]:
                if self.breakpoint_threshold_amount:
                    chunker_kwargs["breakpoint_threshold_amount"] = self.breakpoint_threshold_amount
            elif self.breakpoint_threshold_type == "gradient":
                pass # No further parameters for gradient type

            else:
                print("Please enter a valid threshold type")

            self.text_splitter = SemanticChunker(**chunker_kwargs)
            logger.info(f"Initialized DocumentProcessor with semantic chunking")
            logger.info(f"Embeddings model: {self.embeddings_model}")
            logger.info(f"Breakpoint threshold type: {self.breakpoint_threshold_type}")

        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {str(e)}")
            raise ValueError(f"Failed to initialize semantic chunker: {str(e)}") from e

    def metadata_func(self, record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        for field in self.metadata_fields:
            if field in record:
                value = record[field]
                if isinstance(value, (list, dict)):
                    # Convert complex types to strings for metadata
                    metadata[field] = str(value) if value else ""
                else:
                    metadata[field] = str(value) if value is not None else ""

        # Add document source information
        metadata["source_type"] = "research_paper"
        metadata["chunk_method"] = "semantic_chunking"
        metadata["embeddings_model"] = self.embeddings_model
        metadata["threshold_type"] = self.breakpoint_threshold_type
        
        return metadata

    def preprocess_text(self, text:str) -> str:
        """Preprocess the data for better semantic chunking"""
        if not text:
            return ""
        
        text = text.strip()

        # Remove excessive whitespace
        
        text = re.sub(r'\s+', " ", text)

        # Ensure sentences end properly for better semantic boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better semantic chunking.

        """
        if not text:
            return ""
        
        # Basic text cleaning for research papers
        text = text.strip()
        
        # Remove excessive whitespace
  
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure sentences end properly for better semantic boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

    def load_and_process_documents(
        self, 
        file_path: str, 
        content_key: str = "abstract",
        jq_schema: str = ".[]",
        max_docs: Optional[int] = None,
        min_chunk_size: int = 50  # Minimum chunk size to avoid very small chunks
    ) -> List[Document]:
        """Load and process documents from the JSON using semantic chunking"""

        # Validate if the file path is ok

        # Validate file path
        validated_path = self.validate_file_path(file_path)
        logger.info(f"Processing file: {validated_path}")
        
        # Create loader with custom metadata function
        loader = JSONLoader(
            file_path=str(validated_path),
            jq_schema=jq_schema,
            content_key=content_key,
            metadata_func=self.metadata_func
        )

        # Load documents
        logger.info("Loading documents")
        documents = loader.load()

        if not documents:
            logger.warning("No documents were loaded from the file")
            return []
        
        # Process the first n documents
        if max_docs and max_docs > 0:
            documents = documents[:max_docs]
            logger.info(f"Limited to the first {max_docs} documents")
        
        logger.info(f"Loaded {len(documents)} documents")