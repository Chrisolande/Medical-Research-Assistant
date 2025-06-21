from unittest.mock import MagicMock, mock_open, patch

import pytest
from langchain_core.documents import Document

from src.data_processing.document_processor import DocumentProcessor


@pytest.fixture(scope="session")
def document_processor(test_data):
    """Fixture for DocumentProcessor with default settings."""
    return DocumentProcessor(
        embeddings_model=test_data["embeddings_model"],
        metadata_fields=test_data["metadata_fields"],
    )


@patch("src.data_processing.document_processor.HuggingFaceEmbeddings")
@patch("src.data_processing.document_processor.RecursiveCharacterTextSplitter")
def test_init(mock_text_splitter, mock_embeddings, test_data):
    """Test DocumentProcessor initialization."""
    processor = DocumentProcessor(
        embeddings_model=test_data["embeddings_model"],
        metadata_fields=test_data["metadata_fields"],
    )
    assert processor.embeddings_model == test_data["embeddings_model"]
    print("Embeddings model should match configuration")
    assert processor.metadata_fields == test_data["metadata_fields"]
    print("Metadata fields should match configuration")

    mock_embeddings.assert_called_once_with(model_name=test_data["embeddings_model"])
    mock_text_splitter.assert_called_once_with(chunk_size=300, chunk_overlap=75)

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


@patch("src.data_processing.document_processor.Path")
def test_validate_and_clean_not_file(mock_path, document_processor):
    """Test _validate_and_clean with non-file path."""
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = False
    mock_path.return_value = mock_path_instance
    with pytest.raises(ValueError, match="Path is not a file: test.json"):
        document_processor._validate_and_clean("test.json", "")


@patch("src.data_processing.document_processor.JSONLoader")
@patch("src.data_processing.document_processor.Path")
def test_load_and_process_documents_valid(
    mock_path, mock_loader, document_processor, test_data
):
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.isfile.return_value = True
    mock_path.return_value = mock_path_instance
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.load.return_value = [
        Document(
            page_content=test_data["abstracts"][0], metadata=test_data["metadata"][0]
        ),
        Document(
            page_content=test_data["abstracts"][1], metadata=test_data["metadata"][1]
        ),
        Document(
            page_content=test_data["abstracts"][2], metadata=test_data["metadata"][2]
        ),
        Document(
            page_content=test_data["abstracts"][3], metadata=test_data["metadata"][3]
        ),
    ]

    docs = document_processor.load_and_process_documents(
        file_path="test.json",
        content_key="abstract",
        jq_schema=".[]",
        max_docs=2,
        min_chunk_size=10,
    )
    mock_loader.assert_called_once_with(
        str(mock_path_instance), ".[]", "abstract", document_processor.metadata_func
    )
    assert len(docs) == 2
    assert docs[0].page_content == test_data["abstracts"][0]
    assert docs[0].metadata == test_data["metadata"][0]


@patch("src.data_processing.document_processor.JSONLoader")
@patch("src.data_processing.document_processor.Path")
def test_load_and_process_documents_empty(mock_path, mock_loader, document_processor):
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.isfile.return_value = True
    mock_path.return_value = mock_path_instance
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.loader.return_value = []
    docs = document_processor.load_and_process_documents("test.json")
    assert docs == []


@patch("src.data_processing.document_processor.JSONLoader")
@patch("src.data_processing.document_processor.Path")
def test_load_and_process_documents_error(mock_path, mock_loader, document_processor):
    """Test load_and_process_documents with loader error."""
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    mock_loader.side_effect = Exception("Loader error")
    with pytest.raises(ValueError, match="Failed to process documents: Loader error"):
        document_processor.load_and_process_documents("test.json")


def test_process_documents(document_processor, test_data):
    document_processor.text_splitter = MagicMock()
    document_processor.text_splitter.split_documents.return_value = [
        Document(
            page_content=test_data["abstracts"][0], metadata=test_data["metadata"][0]
        ),
        Document(
            page_content=test_data["abstracts"][1], metadata=test_data["metadata"][1]
        ),
        Document(
            page_content=test_data["abstracts"][2], metadata=test_data["metadata"][2]
        ),
    ]

    docs = [Document(page_content="Valid Content", metadata={"pmid": 123})]

    result = document_processor.process_documents(docs, min_chunk_size=5)
    document_processor.text_splitter.split_documents.assert_called_once_with(docs)
    assert len(result) == 3
    assert result[0].page_content == test_data["abstracts"][0]
    assert result[1].page_content == test_data["abstracts"][1]


def test_process_documents_empty(document_processor):
    """Test process_documents with empty input."""
    result = document_processor.process_documents([])
    assert result == []


def test_get_stats(document_processor, test_data):
    docs = [
        Document(
            page_content=test_data["abstracts"][0], metadata=test_data["metadata"][0]
        ),
        Document(
            page_content=test_data["abstracts"][1], metadata=test_data["metadata"][1]
        ),
    ]

    stats = document_processor.get_stats(docs)
    assert stats["total_chunks"] == 2
    assert stats["total_characters"] == 136  # 67 + 69
    assert stats["average_chunk_size"] == 68  # 136 / 2
    assert stats["min_chunk_size"] == 67
    assert stats["max_chunk_size"] == 69
    assert stats["embeddings_model"] == test_data["embeddings_model"]


def test_get_stats_empty(document_processor):
    """Test get_stats with empty documents."""
    stats = document_processor.get_stats([])
    assert stats == {"total_chunks": 0, "total_characters": 0, "average_chunk_size": 0}


@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_save_processed_documents(
    mock_json_dump, mock_file, document_processor, test_data
):
    """Test save_processed_documents writes correct JSON."""
    docs = [
        Document(
            page_content=test_data["abstracts"][0], metadata=test_data["metadata"][0]
        ),
        Document(
            page_content=test_data["abstracts"][3], metadata=test_data["metadata"][3]
        ),
    ]
    document_processor.save_processed_documents(docs, "output.json")
    mock_file.assert_called_once_with("output.json", "w", encoding="utf-8")
    mock_json_dump.assert_called_once()
    dumped_data = mock_json_dump.call_args[0][0]
    assert dumped_data["processing_info"] == {
        "embeddings_model": test_data["embeddings_model"],
        "total_chunks": 2,
    }
    assert len(dumped_data["documents"]) == 2
    assert dumped_data["documents"][0] == {
        "chunk_id": 0,
        "content": test_data["abstracts"][0],
        "metadata": test_data["metadata"][0],
    }
