from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import os
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

    embeddings_model: str = "embed-english-v3.0"  # More stable model
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: Optional[float] = None
    number_of_chunks: Optional[int] = None
    metadata_fields: List[str] = field(default_factory=lambda: [
        "pmid", "title", "authors", "journal", "volume", "issues", 
        "year", "month", "day", "pub_date", "doi", "pmc_id", 
        "mesh_terms", "publication_types", "doi_url", "pubmed_url"
    ])
    cohere_api_key: Optional[str] = None
    
    def __post_init__(self):
        
        api_key = self.cohere_api_key or os.getenv("COHERE_API_KEY1")
        
        if not api_key:
            raise ValueError("Cohere API key not found. Please set COHERE_API_KEY environment variable or provide it directly.")
        
        try:
            # Initialize embeddings
            self.embeddings = CohereEmbeddings(
                model=self.embeddings_model,
                cohere_api_key=api_key
            )
            
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
            elif self.breakpoint_threshold_type in ["standard_deviation", "interquartile"]:
                if self.breakpoint_threshold_amount:
                    chunker_kwargs["breakpoint_threshold_amount"] = self.breakpoint_threshold_amount
            elif self.breakpoint_threshold_type == "gradient":
                # No further parameters for gradient type
                pass
            
            self.text_splitter = SemanticChunker(**chunker_kwargs)
            
            logger.info(f"Initialized DocumentProcessor with semantic chunking")
            logger.info(f"Embeddings model: {self.embeddings_model}")
            logger.info(f"Breakpoint threshold type: {self.breakpoint_threshold_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {str(e)}")
            raise ValueError(f"Failed to initialize semantic chunker: {str(e)}") from e
    
    def metadata_func(self, record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
      
        # Only extract fields that are specified and exist in the record
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

    def validate_file_path(self, file_path: str) -> Path:
        """Validate the file path's existence"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
            
        if path.suffix.lower() != '.json':
            logger.warning(f"File extension is not .json: {path.suffix}")
            
        return path

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better semantic chunking.
        
        """
        if not text:
            return ""
        
        # Basic text cleaning for research papers
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure sentences end properly for better semantic boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text

    def load_and_process_documents(
        self, 
        file_path: str, 
        content_key: str = "abstract",
        jq_schema: str = ".[]",
        max_docs: Optional[int] = None,
        min_chunk_size: int = 50  # Minimum chunk size to avoid very small chunks
    ) -> List[Document]:

        try:
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
            logger.info("Loading documents...")
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents were loaded from the file")
                return []
            
            # Limit documents if specified
            if max_docs and max_docs > 0:
                documents = documents[:max_docs]
                logger.info(f"Limited to first {max_docs} documents")
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Preprocess and filter documents
            valid_documents = []
            for doc in documents:
                preprocessed_content = self.preprocess_text(doc.page_content)
                if len(preprocessed_content.strip()) >= min_chunk_size:
                    # Update document with preprocessed content
                    doc.page_content = preprocessed_content
                    valid_documents.append(doc)
            
            if len(valid_documents) != len(documents):
                logger.warning(f"Filtered out {len(documents) - len(valid_documents)} "
                             f"documents with insufficient content (< {min_chunk_size} chars)")
            
            if not valid_documents:
                logger.warning("No valid documents found after filtering")
                return []
            
            # Perform semantic chunking
            logger.info("Performing semantic chunking...")
            try:
                chunked_documents = self.text_splitter.split_documents(valid_documents)
            except Exception as e:
                logger.error(f"Error during semantic chunking: {str(e)}")
                # Fallback: return original documents if chunking fails
                logger.warning("Falling back to original documents without chunking")
                chunked_documents = valid_documents
            
            # Filter out very small chunks
            final_chunks = [doc for doc in chunked_documents 
                          if len(doc.page_content.strip()) >= min_chunk_size]
            
            if len(final_chunks) != len(chunked_documents):
                logger.info(f"Filtered out {len(chunked_documents) - len(final_chunks)} "
                          f"chunks smaller than {min_chunk_size} characters")
            
            logger.info(f"Created {len(final_chunks)} semantic chunks from "
                       f"{len(valid_documents)} documents")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise ValueError(f"Failed to process documents: {str(e)}") from e

    def get_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the processed documents.

        """
        if not documents:
            return {
                "total_chunks": 0, 
                "total_characters": 0, 
                "average_chunk_size": 0,
                "unique_documents": 0,
                "metadata_fields": []
            }
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        total_chars = sum(chunk_sizes)
        unique_sources = len(set(doc.metadata.get("pmid", "") for doc in documents if doc.metadata.get("pmid")))
        
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": round(total_chars / len(documents), 2),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "unique_documents": unique_sources,
            "metadata_fields": list(documents[0].metadata.keys()) if documents else [],
            "chunking_method": "semantic",
            "embeddings_model": self.embeddings_model,
            "threshold_type": self.breakpoint_threshold_type
        }

    def save_processed_documents(self, documents: List[Document], output_path: str) -> None:
        """
        Save processed documents to a JSON file.

        """
        import json
        
        output_data = {
            "processing_info": {
                "chunk_method": "semantic_chunking",
                "embeddings_model": self.embeddings_model,
                "threshold_type": self.breakpoint_threshold_type,
                "total_chunks": len(documents)
            },
            "documents": []
        }
        
        for i, doc in enumerate(documents):
            output_data["documents"].append({
                "chunk_id": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "content_length": len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} processed documents to {output_path}")

    def analyze_chunk_boundaries(self, documents: List[Document], sample_size: int = 5) -> None:
        """
        Analyze and log information about chunk boundaries for debugging.
        
        """
        if not documents:
            logger.info("No documents to analyze")
            return
        
        logger.info(f"\n=== Semantic Chunking Analysis ===")
        logger.info(f"Total chunks: {len(documents)}")
        
        # Sample a few chunks for detailed analysis
        sample_docs = documents[:min(sample_size, len(documents))]
        
        for i, doc in enumerate(sample_docs):
            logger.info(f"\nChunk {i + 1}:")
            logger.info(f"  Length: {len(doc.page_content)} characters")
            logger.info(f"  PMID: {doc.metadata.get('pmid', 'N/A')}")
            logger.info(f"  First 100 chars: {doc.page_content[:100]}...")
            logger.info(f"  Last 100 chars: ...{doc.page_content[-100:]}")


