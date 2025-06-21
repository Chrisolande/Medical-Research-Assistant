import json
import tempfile
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.data_processing.batch_processor import (
    MIN_ABSTRACT_CONTENT_LENGTH,
    PMCBatchProcessor,
)


@pytest.fixture
def sample_pmc_data(test_data):
    return [
        {
            "pmid": test_data["metadata"][i]["pmid"],
            "title": test_data["metadata"][i]["title"],
            "abstract": test_data["abstracts"][i],
            "authors": test_data["metadata"][i]["authors"],
        }
        for i in range(len(test_data["abstracts"]))
    ]


@pytest.fixture
def temp_json_file(sample_pmc_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_pmc_data, f)
        return f.name


@pytest.fixture
def batch_processor(document_processor, test_data):
    """Create PMCBatchProcessor instance."""
    return PMCBatchProcessor(
        document_processor=document_processor,
        batch_size=test_data["config"]["batch_size"],
        max_concurrent_batches=test_data["config"]["max_concurrent_batches"],
        retry_attempts=test_data["config"]["retry_attempts"],
        retry_delay=test_data["config"]["retry_delay"],
        inter_batch_delay=0.0,
    )


def test_init(document_processor, test_data):
    """Test initialization."""
    processor = PMCBatchProcessor(
        document_processor=document_processor,
        batch_size=5,
        max_concurrent_batches=3,
    )

    assert processor.document_processor == document_processor
    assert processor.batch_size == 5
    assert processor.max_concurrent_batches == 3
    assert processor.logger is not None
    assert processor.processing_semaphore is not None
    assert processor.executor is not None


@patch("src.data_processing.batch_processor.load_json_data")
@patch("src.data_processing.batch_processor.logging.Logger.info")
def test_load_pmc_data_valid_docs(
    mock_logger, mock_load, batch_processor, temp_json_file, sample_pmc_data
):
    mock_load.return_value = sample_pmc_data
    result = batch_processor.load_pmc_data(temp_json_file)

    expected_count = sum(
        1
        for doc in sample_pmc_data
        if doc.get("abstract", "").strip() and len(doc.get("abstract", "")) >= 50
    )

    # assertions
    assert isinstance(result, list)
    print(f"Expected a list, got {type(result)}")
    assert len(result) == expected_count
    assert all(
        doc.get("abstract", "").strip()
        and len(doc.get("abstract", "").strip()) >= MIN_ABSTRACT_CONTENT_LENGTH
        for doc in result
    )

    # Verify logging
    assert mock_logger.call_count >= 1, "Expected logging calls"
    mock_logger.assert_any_call(f"Found {len(result)} documents with valid abstracts.")
    if len(sample_pmc_data) != len(result):
        mock_logger.assert_any_call(
            f"Filtered {len(sample_pmc_data) - len(result)} documents lacking valid abstracts or too short."
        )


@patch("src.data_processing.batch_processor.load_json_data")
def test_load_pmc_data_max_docs_limit(
    mock_load, batch_processor, temp_json_file, sample_pmc_data
):
    """Test loading PMC data with max_docs limit."""
    mock_load.side_effect = lambda file_path, max_docs: (
        sample_pmc_data[:max_docs] if max_docs is not None else sample_pmc_data
    )

    result = batch_processor.load_pmc_data(temp_json_file, max_docs=1)

    expected_count = sum(
        1
        for doc in sample_pmc_data[:1]
        if doc.get("abstract", "").strip()
        and len(doc.get("abstract", "").strip()) >= MIN_ABSTRACT_CONTENT_LENGTH
    )

    # Assertions
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert (
        len(result) == expected_count
    ), f"Expected {expected_count} valid documents, got {len(result)}"
    assert len(result) <= 1, f"Expected at most 1 document, got {len(result)}"
    if result:
        assert (
            result[0]["pmid"] == "12345"
        ), f"Expected PMID 12345, got {result[0]['pmid']}"
        assert result[0].get("abstract", "").strip(), "Document has no valid abstract"
        assert (
            len(result[0].get("abstract", "").strip()) >= MIN_ABSTRACT_CONTENT_LENGTH
        ), f"Abstract too short: {len(result[0].get('abstract', '').strip())}"


def test_create_document_batches(batch_processor, sample_pmc_data):
    """Test document batch creation."""
    docs = sample_pmc_data

    batches = list(batch_processor.create_document_batches(docs, batch_size=2))

    # assertion
    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2


def test_create_document_batches_default_size(batch_processor, sample_pmc_data):
    """Test batch creation with default size."""
    docs = sample_pmc_data

    batches = list(batch_processor.create_document_batches(docs))

    assert len(batches) == 1


def test_process_batch_documents(batch_processor, sample_pmc_data):
    """Test processing batch of documents."""
    batch = sample_pmc_data[:2]

    result = batch_processor._process_batch_documents(batch)

    # Check that documents were processed
    assert len(result) >= 1  # At least some documents should be processed
    for doc in result:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert "pmid" in doc.metadata


@pytest.mark.asyncio
async def test_process_batch_async_success(batch_processor, sample_pmc_data):
    """Test successful async batch processing."""
    batch = sample_pmc_data[:1]

    result = await batch_processor._process_batch_async(batch, 1)

    assert result["success"] is True
    assert result["batch_num"] == 1
    assert result["original_count"] == 1
    assert result["error"] is None
    assert result["attempt"] == 1
