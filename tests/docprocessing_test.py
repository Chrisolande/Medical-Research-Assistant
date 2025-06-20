from unittest.mock import MagicMock, patch

import pytest

from src.data_processing.document_processor import DocumentProcessor


@pytest.fixture(scope="session")
def test_data():
    """Centralized test data."""
    return {
        "abstracts": [
            "This is abstract one for testing purposes with medical terminology.",
            "This is another abstract for testing cardiovascular disease research.",
            "Sample abstract about machine learning applications in healthcare.",
        ],
        "metadata": [
            {
                "pmid": "12345",
                "title": "Medical AI Research",
                "authors": "Smith, J., Johnson, K.",
            },
            {
                "pmid": "67890",
                "title": "Cardiovascular Studies",
                "authors": "Brown, A., Davis, L.",
            },
            {
                "pmid": "11111",
                "title": "Healthcare ML Applications",
                "authors": "Wilson, M.",
            },
        ],
        "metadata_fields": ["pmid", "title", "authors"],
        "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


@pytest.fixture(scope="session")
def document_processor():
    """Fixture for DocumentProcessor with default settings."""
    return DocumentProcessor(
        embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        metadata_fields=["pmid", "title", "authors"],
    )


@patch("src.data_processing.document_processor.HuggingFaceEmbeddings")
@patch("src.data_processing.document_processor.RecursiveCharacterTextSplitter")
def test_init(mock_text_splitter, mock_embeddings, test_data):
    """Test DocumentProcessor initialization."""
    processor = DocumentProcessor(
        embeddings_model=test_data["embeddings_model"],
        metadata_fields=test_data["metadata_fields"],
    )
    assert (
        processor.embeddings_model == test_data["embeddings_model"]
    ), "Embeddings model should match configuration"
    assert (
        processor.metadata_fields == test_data["metadata_fields"]
    ), "Metadata fields should match configuration"
    (
        mock_embeddings.assert_called_once_with(
            model_name=test_data["embeddings_model"]
        ),
        "HuggingFaceEmbeddings should be called once with correct model",
    )
    (
        mock_text_splitter.assert_called_once_with(chunk_size=300, chunk_overlap=75),
        "RecursiveCharacterTextSplitter should be called once",
    )

    assert processor.text_splitter == mock_text_splitter.return_value
    assert processor.embeddings == mock_embeddings.return_value


def test_metadata_func(test_data):
    """Test DocumentProcessor metadata_func method."""
    processor = DocumentProcessor(
        embeddings_model=test_data["embeddings_model"],
        metadata_fields=test_data["metadata_fields"],
    )
    record = {"pmid": "123", "title": "Test", "authors": "Author1", "extra": "ignored"}
    metadata = {"existing": "data"}
    expected = {
        "existing": "data",
        "pmid": "123",
        "title": "Test",
        "authors": "Author1",
        "embeddings_model": test_data["embeddings_model"],
    }
    result = processor.metadata_func(record, metadata)
    assert (
        result == expected
    ), "Metadata should include specified fields and existing data"


@patch("src.data_processing.document_processor.Path")
def test_validate_and_clean_valid_file(mock_path, document_processor):
    """Test _validate_and_clean with valid file and text."""
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    text = "Hello.World!This is a test."
    path, cleaned = document_processor._validate_and_clean("test.json", text)
    assert path == mock_path_instance
    assert cleaned == "Hello. World! This is a test."


@patch("src.data_processing.document_processor.Path")
def test_validate_and_clean_invalid_file(mock_path, document_processor):
    """Test _validate_and_clean with invalid/non-existent file."""
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_path.return_value = mock_path_instance
    with pytest.raises(FileNotFoundError, match="File not found: test.json"):
        document_processor._validate_and_clean("test.json", "")
