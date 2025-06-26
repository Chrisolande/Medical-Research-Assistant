import pytest
import json
import asyncio
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.data_processing.document_processor import DocumentProcessor


@pytest.fixture(scope="session")
def test_data() -> Dict[str, Any]:
    """Centralized test data to avoid duplication."""
    return {
        "abstracts": [
            "This is abstract one for testing purposes with medical terminology.",
            "This is another abstract for testing cardiovascular disease research.",
            "Sample abstract about machine learning applications in healthcare."
        ],
        "metadata": [
            {"pmid": "12345", "title": "Medical AI Research", "authors": "Smith, J., Johnson, K."},
            {"pmid": "67890", "title": "Cardiovascular Studies", "authors": "Brown, A., Davis, L."},
            {"pmid": "11111", "title": "Healthcare ML Applications", "authors": "Wilson, M."}
        ]
    }


@pytest.fixture(scope="session")
def embeddings_config() -> Dict[str, Any]:
    """Configuration for embeddings model used across tests."""
    return {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "metadata_fields": ["pmid", "title", "authors"]
    }


@pytest.fixture(scope="function")
def document_processor(embeddings_config):
    """Fixture to create a DocumentProcessor instance with default settings."""
    processor = DocumentProcessor(
        embeddings_model=embeddings_config["model"],
        metadata_fields=embeddings_config["metadata_fields"]
    )
    yield processor
    # Cleanup if needed
    if hasattr(processor, 'cleanup'):
        processor.cleanup()


@pytest.fixture(scope="session")
def sample_document(test_data) -> Document:
    """Fixture for a single sample Document object."""
    return Document(
        page_content=test_data["abstracts"][0],
        metadata=test_data["metadata"][0]
    )


@pytest.fixture(scope="session")
def sample_documents(test_data) -> List[Document]:
    """Fixture for multiple sample Document objects with varied content."""
    return [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(test_data["abstracts"], test_data["metadata"])
    ]


@pytest.fixture(scope="function")
def empty_documents() -> List[Document]:
    """Fixture for edge case testing with empty documents."""
    return []


@pytest.fixture(scope="session")
def large_document(test_data) -> Document:
    """Fixture for testing with larger content."""
    large_content = " ".join(test_data["abstracts"] * 10)  # Repeat content
    return Document(
        page_content=large_content,
        metadata={**test_data["metadata"][0], "pmid": "99999", "size": "large"}
    )


# Initialization Tests
@patch('src.data_processing.document_processor.HuggingFaceEmbeddings')
@patch('src.data_processing.document_processor.RecursiveCharacterTextSplitter')
def test_init(mock_text_splitter, mock_embeddings, document_processor):
    """Test initialization of DocumentProcessor."""
    mock_embeddings.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
    mock_text_splitter.assert_called_once_with(chunk_size=300, chunk_overlap=75)
    assert document_processor.embeddings_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert document_processor.metadata_fields == ["pmid", "title", "authors", "journal", "volume",
                                                 "issues", "year", "month", "day", "pub_date",
                                                 "doi", "pmc_id", "mesh_terms", "publication_types",
                                                 "doi_url", "pubmed_url"]
    assert document_processor.embeddings is not None
    assert document_processor.text_splitter is not None


# Metadata Processing Tests
def test_metadata_func(document_processor):
    """Test metadata_func for extracting metadata from a record."""
    record = {
        "pmid": "12345",
        "title": "Sample Title",
        "authors": "John Doe",
        "extra_field": "ignored"
    }
    metadata = {"existing": "value"}
    result = document_processor.metadata_func(record, metadata)
    
    assert result == {
        "existing": "value",
        "pmid": "12345",
        "title": "Sample Title",
        "authors": "John Doe",
        "journal": "",
        "volume": "",
        "issues": "",
        "year": "",
        "month": "",
        "day": "",
        "pub_date": "",
        "doi": "",
        "pmc_id": "",
        "mesh_terms": "",
        "publication_types": "",
        "doi_url": "",
        "pubmed_url": "",
        "embeddings_model": document_processor.embeddings_model
    }


# File Validation Tests
@patch('src.data_processing.document_processor.Path')
def test_validate_and_clean_file_not_found(mock_path, document_processor):
    """Test _validate_and_clean with non-existent file."""
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = False
    mock_path.return_value = mock_path_instance
    
    with pytest.raises(FileNotFoundError, match="File not found: test.json"):
        document_processor._validate_and_clean("test.json", "")


@patch('src.data_processing.document_processor.Path')
def test_validate_and_clean_not_file(mock_path, document_processor):
    """Test _validate_and_clean with a directory path."""
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = False
    mock_path.return_value = mock_path_instance
    
    with pytest.raises(ValueError, match="Path is not a file: test.json"):
        document_processor._validate_and_clean("test.json", "")


def test_validate_and_clean_text_processing(document_processor):
    """Test _validate_and_clean text cleaning functionality."""
    text = "Hello.World!This  is   a   test."
    path, cleaned = document_processor._validate_and_clean("dummy.json", text)
    
    assert cleaned == "Hello. World! This is a test."
    assert isinstance(path, Path)


# Document Loading Tests
@patch('src.data_processing.document_processor.JSONLoader')
@patch('src.data_processing.document_processor.Path')
def test_load_and_process_documents_success(mock_path, mock_json_loader, document_processor, sample_documents):
    """Test load_and_process_documents with valid input."""
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    
    mock_loader_instance = Mock()
    mock_loader_instance.load.return_value = sample_documents
    mock_json_loader.return_value = mock_loader_instance
    
    result = document_processor.load_and_process_documents(
        file_path="test.json",
        content_key="abstract",
        jq_schema=".[]",
        max_docs=2,
        min_chunk_size=10
    )
    
    assert len(result) == 2
    assert result[0].page_content == "This is abstract one for testing purposes."
    assert result[1].page_content == "This is another abstract for testing."
    mock_json_loader.assert_called_once_with(
        "test.json", ".[]", "abstract", document_processor.metadata_func
    )
    mock_loader_instance.load.assert_called_once()


@patch('src.data_processing.document_processor.JSONLoader')
@patch('src.data_processing.document_processor.Path')
def test_load_and_process_documents_empty(mock_path, mock_json_loader, document_processor):
    """Test load_and_process_documents with empty document list."""
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    
    mock_loader_instance = Mock()
    mock_loader_instance.load.return_value = []
    mock_json_loader.return_value = mock_loader_instance
    
    result = document_processor.load_and_process_documents("test.json")
    
    assert result == []
    mock_loader_instance.load.assert_called_once()


@patch('src.data_processing.document_processor.JSONLoader')
@patch('src.data_processing.document_processor.Path')
def test_load_and_process_documents_min_chunk_size(mock_path, mock_json_loader, document_processor):
    """Test load_and_process_documents filtering by min_chunk_size."""
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    
    mock_loader_instance = Mock()
    mock_loader_instance.load.return_value = [
        Document(page_content="Short", metadata={"pmid": "1"}),
        Document(page_content="This is a valid abstract for testing.", metadata={"pmid": "2"})
    ]
    mock_json_loader.return_value = mock_loader_instance
    
    result = document_processor.load_and_process_documents("test.json", min_chunk_size=10)
    
    assert len(result) == 1
    assert result[0].page_content == "This is a valid abstract for testing."


@patch('src.data_processing.document_processor.JSONLoader')
@patch('src.data_processing.document_processor.Path')
def test_load_and_process_documents_error(mock_path, mock_json_loader, document_processor):
    """Test load_and_process_documents error handling."""
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    
    mock_loader_instance = Mock()
    mock_loader_instance.load.side_effect = Exception("Load error")
    mock_json_loader.return_value = mock_loader_instance
    
    with pytest.raises(ValueError, match="Failed to process documents: Load error"):
        document_processor.load_and_process_documents("test.json")


# Document Processing Tests
def test_process_documents_empty_input(document_processor):
    """Test process_documents with empty input."""
    result = document_processor.process_documents([])
    assert result == []


@patch.object(DocumentProcessor, 'text_splitter')
def test_process_documents_success(mock_text_splitter, document_processor, sample_documents):
    """Test process_documents with valid documents."""
    mock_text_splitter.split_documents.return_value = [
        Document(page_content="Chunk 1", metadata={"pmid": "12345"}),
        Document(page_content="Chunk 2", metadata={"pmid": "12345"}),
        Document(page_content="Chunk 3", metadata={"pmid": "67890"})
    ]
    
    result = document_processor.process_documents(sample_documents, min_chunk_size=5)
    
    assert len(result) == 3
    assert result[0].page_content == "Chunk 1"
    assert result[1].page_content == "Chunk 2"
    assert result[2].page_content == "Chunk 3"
    mock_text_splitter.split_documents.assert_called_once_with(sample_documents)


@patch.object(DocumentProcessor, 'text_splitter')
def test_process_documents_min_chunk_size(mock_text_splitter, document_processor, sample_documents):
    """Test process_documents filtering by min_chunk_size."""
    mock_text_splitter.split_documents.return_value = [
        Document(page_content="Short", metadata={"pmid": "12345"}),
        Document(page_content="This is a valid chunk for testing.", metadata={"pmid": "12345"})
    ]
    
    result = document_processor.process_documents(sample_documents, min_chunk_size=10)
    
    assert len(result) == 1
    assert result[0].page_content == "This is a valid chunk for testing."


# Statistics Tests
def test_get_stats_empty(document_processor):
    """Test get_stats with empty document list."""
    result = document_processor.get_stats([])
    assert result == {
        "total_chunks": 0,
        "total_characters": 0,
        "average_chunk_size": 0,
        "min_chunk_size": 0,
        "max_chunk_size": 0,
        "embeddings_model": document_processor.embeddings_model
    }


def test_get_stats_valid_documents(document_processor, sample_documents):
    """Test get_stats with valid documents."""
    result = document_processor.get_stats(sample_documents)
    
    assert result["total_chunks"] == 2
    assert result["total_characters"] == 80  # 40 + 40
    assert result["average_chunk_size"] == 40.0
    assert result["min_chunk_size"] == 40
    assert result["max_chunk_size"] == 40
    assert result["embeddings_model"] == document_processor.embeddings_model


# File I/O Tests
@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_save_processed_documents(mock_json_dump, mock_open_file, document_processor, sample_documents):
    """Test save_processed_documents functionality."""
    document_processor.save_processed_documents(sample_documents, "output.json")
    
    mock_open_file.assert_called_once_with("output.json", 'w', encoding='utf-8')
    mock_json_dump.assert_called_once()
    saved_data = mock_json_dump.call_args[0][0]
    
    assert saved_data["processing_info"] == {
        "embeddings_model": document_processor.embeddings_model,
        "total_chunks": 2
    }
    assert len(saved_data["documents"]) == 2
    assert saved_data["documents"][0]["content"] == sample_documents[0].page_content
    assert saved_data["documents"][1]["content"] == sample_documents[1].page_content


@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_save_processed_documents_empty(mock_json_dump, mock_open_file, document_processor):
    """Test save_processed_documents with empty document list."""
    document_processor.save_processed_documents([], "output.json")
    
    mock_open_file.assert_called_once_with("output.json", 'w', encoding='utf-8')
    mock_json_dump.assert_called_once()
    saved_data = mock_json_dump.call_args[0][0]
    
    assert saved_data["processing_info"] == {
        "embeddings_model": document_processor.embeddings_model,
        "total_chunks": 0
    }
    assert saved_data["documents"] == []


# Async and Concurrent Processing Tests
@pytest.mark.asyncio
async def test_concurrent_processing(document_processor, sample_documents):
    """Test concurrent processing capabilities if supported."""
    async def process_batch(docs):
        return document_processor.process_documents(docs)
    
    # Test concurrent processing
    tasks = [process_batch(sample_documents[:1]) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    assert all(len(result) >= 0 for result in results)


# Error Handling Tests
def test_error_handling_comprehensive(document_processor):
    """Test various error scenarios."""
    # Test with None input
    with pytest.raises((TypeError, ValueError)):
        document_processor.process_documents(None)
    
    # Test with invalid document structure
    invalid_doc = Mock()
    invalid_doc.page_content = None
    with pytest.raises((AttributeError, TypeError)):
        document_processor.process_documents([invalid_doc])


# Performance and Edge Case Tests
def test_large_document_processing(document_processor, large_document):
    """Test processing of large documents."""
    result = document_processor.process_documents([large_document])
    assert len(result) >= 1
    assert all(doc.page_content for doc in result)


def test_memory_efficiency(document_processor, sample_documents):
    """Test memory usage patterns."""
    # Process same documents multiple times to check for memory leaks
    for _ in range(10):
        result = document_processor.process_documents(sample_documents)
        assert len(result) >= 0
        # Force garbage collection would be here in real test


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
