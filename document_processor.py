from dataclasses import dataclass
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
import os
from typing import dict
@dataclass
class DocumentProcessor:
    chunk_size: int = 2000
    chunk_overlap: int = 200

    def __post_init__(self):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len
        )
    
    def metadata_func(self, record: dict, metadata: dict) -> dict:
        """Extract all the fields except the abstract as the metadata from the JSON record"""
        metadata.update({
            "pmid": record.get("pmid", ""),
            "title": record.get("title", ""),
            "authors": record.get("authors", ""),
            "journal": record.get("journal", ""),
            "volume": record.get("volume", ""),
            "issues": record.get("issues", ""),
            "year": record.get("year", ""),
            "month": record.get("month", ""),
            "day": record.get("day", ""),
            "pub_date": record.get("pub_date", ""),
            "doi": record.get("doi", ""),
            "pmc_id": record.get("pmc_id", ""),
            "mesh_terms": record.get("mesh_terms", ""),
            "publication_types": record.get("publication_types", ""),
            "doi_url": record.get("doi_url", ""),
            "pubmed_url": record.get("pubmed_url", ""),
        })

    def load_documents(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError("The file path you specified is not valid")

        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".[]",  # Assumes JSON is an array of records
            content_key="abstract",  # The abstract field is the main content
            metadata_func=self.metadata_func
        )

        # Load documents
        documents = loader.load()
        # Split documents into chunks using the text splitter
        chunked_documents = self.text_splitter.split_documents(documents)
        
        return chunked_documents




    
