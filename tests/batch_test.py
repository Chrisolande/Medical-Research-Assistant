import json
import tempfile
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from medical_graph_rag.data_processing.batch_processor import (
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
        document_processor=document_processor, batch_size=5, max_concurrent_batches=3
    )

    assert processor.document_processor == document_processor
    assert processor.batch_size == 5
    assert processor.max_concurrent_batches == 3
    assert processor.logger is not None
    assert processor.executor is not None


@patch("medical_graph_rag.data_processing.batch_processor.load_json_data")
@patch("medical_graph_rag.data_processing.batch_processor.logging.Logger.info")
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

    # perform assertions
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


@patch("medical_graph_rag.data_processing.batch_processor.load_json_data")
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

    # perform assertions
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

    # perform assertions
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

    # perform assertions
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

    # perform assertions
    assert result["success"] is True
    assert result["batch_num"] == 1
    assert result["original_count"] == 1
    assert result["error"] is None
    assert result["attempt"] == 1


@pytest.mark.asyncio
async def test_process_batch_async_failure_with_retry(batch_processor, sample_pmc_data):
    """Test async batch processing with failure and retry."""
    batch = sample_pmc_data[:1]

    with patch.object(batch_processor, "_process_batch_documents") as mock_process:
        mock_process.side_effect = Exception("Processing failed")

        result = await batch_processor._process_batch_async(batch, 1)

        assert result["success"] is False
        assert result["batch_num"] == 1
        assert result["error"] == "Processing failed"
        assert result["attempt"] == batch_processor.retry_attempts
        assert mock_process.call_count == batch_processor.retry_attempts


@pytest.mark.asyncio
async def test_process_pmc_file_async_complete_flow(
    batch_processor, temp_json_file, sample_pmc_data
):
    """Test complete async processing flow."""
    with patch(
        "medical_graph_rag.data_processing.batch_processor.load_json_data"
    ) as mock_load:
        mock_load.return_value = sample_pmc_data

        result = await batch_processor.process_pmc_file_async(
            temp_json_file, max_docs=2
        )

        assert "successful_batches" in result
        assert "failed_batches" in result
        assert "all_documents" in result
        assert "processing_summary" in result

        summary = result["processing_summary"]
        assert summary["total_documents"] == 3
        assert summary["success_rate"] >= 0
        assert summary["processing_time"] > 0


@pytest.mark.asyncio
async def test_process_pmc_file_async_no_valid_docs(batch_processor, temp_json_file):
    """Test processing with no valid documents."""
    with patch(
        "medical_graph_rag.data_processing.batch_processor.load_json_data"
    ) as mock_load:
        mock_load.return_value = []

        result = await batch_processor.process_pmc_file_async(temp_json_file)

        assert result["processing_summary"]["total_documents"] == 0
        assert result["processing_summary"]["total_batches"] == 0
        assert len(result["all_documents"]) == 0


@pytest.mark.asyncio
async def test_process_pmc_file_async_with_progress_callback(
    batch_processor, temp_json_file, sample_pmc_data
):
    """Test processing with progress callback."""
    callback_calls = []

    def progress_callback(completed, total, result):
        callback_calls.append((completed, total, result))

    with patch(
        "medical_graph_rag.data_processing.batch_processor.load_json_data"
    ) as mock_load:
        mock_load.return_value = sample_pmc_data[:1]

        await batch_processor.process_pmc_file_async(
            temp_json_file, progress_callback=progress_callback
        )

        assert len(callback_calls) > 0
        completed, total, result = callback_calls[-1]
        assert completed <= total
        assert "batch_num" in result


def test_empty_result(batch_processor):
    """Test empty result structure."""
    result = batch_processor._empty_result()

    expected_keys = [
        "successful_batches",
        "failed_batches",
        "all_documents",
        "processing_summary",
    ]
    for key in expected_keys:
        assert key in result

    summary = result["processing_summary"]
    assert summary["total_documents"] == 0
    assert summary["success_rate"] == 0.0


def test_save_results(batch_processor, tmp_path):
    """Test saving results to file."""
    results = {
        "successful_batches": [],
        "failed_batches": [],
        "all_documents": [],
        "processing_summary": {"total_documents": 0},
    }

    with patch(
        "medical_graph_rag.data_processing.batch_processor.save_processing_results"
    ) as mock_save:
        batch_processor.save_results(results, str(tmp_path))

        mock_save.assert_called_once_with(
            results=results,
            output_dir=str(tmp_path),
            base_filename="pmc_chunks",
            batch_size=batch_processor.batch_size,
            source_type="pmc_abstracts",
            save_batch_details=False,
        )
