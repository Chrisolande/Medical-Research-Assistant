import pytest
import asyncio
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.core.batchprocessor import PMCBatchProcessor

@pytest.fixture
def batch_processor(document_processor, test_data):
    """Fixture for PMCBatchProcessor."""
    return PMCBatchProcessor(
        document_processor=document_processor,
        batch_size=test_data["config"]["batch_size"],
        max_concurrent_batches=test_data["config"]["max_concurrent_batches"],
        retry_attempts=test_data["config"]["retry_attempts"],
        retry_delay=test_data["config"]["retry_delay"],
        inter_batch_delay=test_data["config"]["inter_batch_delay"]
    )

@patch("src.core.batchprocessor.logging.getLogger")
def test_init(mock_get_logger, document_processor, test_data):
    """Test PMCBatchProcessor initialization."""
    processor = PMCBatchProcessor(
        document_processor=document_processor,
        batch_size=test_data["config"]["batch_size"],
        max_concurrent_batches=test_data["config"]["max_concurrent_batches"],
        retry_attempts=test_data["config"]["retry_attempts"],
        retry_delay=test_data["config"]["retry_delay"],
        inter_batch_delay=test_data["config"]["inter_batch_delay"]
    )
    assert processor.document_processor == document_processor
    assert processor.batch_size == test_data["config"]["batch_size"]
    assert processor.max_concurrent_batches == test_data["config"]["max_concurrent_batches"]
    assert processor.retry_attempts == test_data["config"]["retry_attempts"]
    assert processor.retry_delay == test_data["config"]["retry_delay"]
    assert processor.inter_batch_delay == test_data["config"]["inter_batch_delay"]
    assert processor.logger == mock_get_logger.return_value
    assert processor.processing_semaphore._value == test_data["config"]["max_concurrent_batches"]
    assert processor.executor._max_workers == test_data["config"]["max_concurrent_batches"]

@patch("src.core.batchprocessor.load_json_data")
def test_load_pmc_data_valid(mock_load_json, batch_processor, test_data):
    """Test load_pmc_data with valid and invalid documents."""
    mock_load_json.return_value = test_data["docs"]
    result = batch_processor.load_pmc_data("test.json", max_docs=4)
    assert len(result) == 3, "Should filter out invalid abstracts"
    assert result[0]["pmid"] == "12345"
    assert result[1]["pmid"] == "67890"
    assert result[2]["pmid"] == "33333"
    mock_load_json.assert_called_once_with("test.json", max_docs=4)
    batch_processor.logger.info.assert_called()

@patch("src.core.batchprocessor.load_json_data")
def test_load_pmc_data_empty(mock_load_json, batch_processor):
    """Test load_pmc_data with no valid documents."""
    mock_load_json.return_value = [{"abstract": ""}, {"abstract": "Short"}]
    result = batch_processor.load_pmc_data("test.json")
    assert result == [], "Should return empty list for no valid documents"
    batch_processor.logger.warning.assert_called()

def test_create_document_batches(batch_processor, test_data):
    """Test create_document_batches splits documents correctly."""
    docs = test_data["docs"][:3]
    batches = list(batch_processor.create_document_batches(docs, batch_size=2))
    assert len(batches) == 2, "Should create 2 batches"
    assert len(batches[0]) == 2, "First batch should have 2 documents"
    assert len(batches[1]) == 1, "Second batch should have 1 document"
    assert batches[0][0]["pmid"] == "12345"
    assert batches[0][1]["pmid"] == "67890"
    assert batches[1][0]["pmid"] == "11111"

def test_process_batch_documents(batch_processor, test_data):
    """Test _process_batch_documents processes valid documents."""
    batch = test_data["docs"][:2]
    result = batch_processor._process_batch_documents(batch)
    assert len(result) == 2, "Should process 2 valid documents"
    assert isinstance(result[0], Document)
    assert result[0].page_content == "Valid abstract one"
    assert result[0].metadata["pmid"] == "12345"
    assert batch_processor.document_processor.process_documents.called

@patch("src.core.batchprocessor.asyncio.get_event_loop")
async def test_process_batch_async_success(mock_get_loop, batch_processor, test_data):
    """Test _process_batch_async with successful processing."""
    batch = test_data["docs"][:2]
    mock_get_loop.return_value.run_in_executor.return_value = [
        Document(page_content="Valid abstract one", metadata={"pmid": "12345"}),
        Document(page_content="Valid abstract two", metadata={"pmid": "67890"})
    ]
    result = await batch_processor._process_batch_async(batch, batch_num=1)
    assert result["success"] is True
    assert result["batch_num"] == 1
    assert result["original_count"] == 2
    assert result["chunk_count"] == 2
    assert len(result["documents"]) == 2
    assert result["error"] is None
    batch_processor.logger.info.assert_called()

@patch("src.core.batchprocessor.asyncio.get_event_loop")
async def test_process_batch_async_failure(mock_get_loop, batch_processor, test_data):
    """Test _process_batch_async with failure after retries."""
    batch = test_data["docs"][:2]
    mock_get_loop.return_value.run_in_executor.side_effect = Exception("Processing error")
    result = await batch_processor._process_batch_async(batch, batch_num=1)
    assert result["success"] is False
    assert result["batch_num"] == 1
    assert result["original_count"] == 2
    assert result["chunk_count"] == 0
    assert result["documents"] == []
    assert result["error"] == "Processing error"
    assert result["attempt"] == batch_processor.retry_attempts
    batch_processor.logger.warning.assert_called()

@patch("src.core.batchprocessor.asyncio")
@patch("src.core.batchprocessor.load_json_data")
async def test_process_pmc_file_async_valid(mock_load_json, mock_asyncio, batch_processor, test_data):
    """Test process_pmc_file_async with valid documents."""
    mock_load_json.return_value = test_data["docs"][:3]
    batch_processor._process_batch_async = MagicMock(side_effect=[
        {
            "batch_num": 1,
            "success": True,
            "documents": [
                Document(page_content="Valid abstract one", metadata={"pmid": "12345"}),
                Document(page_content="Valid abstract two", metadata={"pmid": "67890"})
            ],
            "original_count": 2,
            "chunk_count": 2,
            "error": None,
            "attempt": 1
        },
        {
            "batch_num": 2,
            "success": True,
            "documents": [Document(page_content="Valid abstract three", metadata={"pmid": "33333"})],
            "original_count": 1,
            "chunk_count": 1,
            "error": None,
            "attempt": 1
        }
    ])
    results = await batch_processor.process_pmc_file_async("test.json", max_docs=3, batch_size=2)
    assert len(results["successful_batches"]) == 2
    assert len(results["failed_batches"]) == 0
    assert len(results["all_documents"]) == 3
    assert results["processing_summary"]["total_documents"] == 3
    assert results["processing_summary"]["total_batches"] == 2
    assert results["processing_summary"]["success_rate"] == 100.0
    mock_load_json.assert_called_once_with("test.json", max_docs=3)
    batch_processor.logger.info.assert_called()

@patch("src.core.batchprocessor.load_json_data")
async def test_process_pmc_file_async_empty(mock_load_json, batch_processor):
    """Test process_pmc_file_async with no valid documents."""
    mock_load_json.return_value = []
    results = await batch_processor.process_pmc_file_async("test.json")
    assert results["successful_batches"] == []
    assert results["failed_batches"] == []
    assert results["all_documents"] == []
    assert results["processing_summary"]["total_documents"] == 0
    assert results["processing_summary"]["total_batches"] == 0
    batch_processor.logger.warning.assert_called()

@patch("src.core.batchprocessor.save_processing_results")
def test_save_results(mock_save_results, batch_processor, test_data):
    """Test save_results calls save_processing_results correctly."""
    results = {
        "successful_batches": [{"batch_num": 1, "documents": [Document(page_content="Test", metadata={})]}],
        "failed_batches": [],
        "all_documents": [Document(page_content="Test", metadata={})],
        "processing_summary": {"total_documents": 1, "total_batches": 1}
    }
    batch_processor.save_results(results, output_dir="output", save_batch_details=True)
    mock_save_results.assert_called_once_with(
        results=results,
        output_dir="output",
        base_filename="pmc_chunks",
        batch_size=batch_processor.batch_size,
        source_type="pmc_abstracts",
        save_batch_details=True
    )

@patch('src.data_processing.batch_processor.load_json_data')
@patch('src.data_processing.batch_processor.logging.Logger.info')
def test_load_pmc_data_valid_docs(mock_logger, mock_load, batch_processor, temp_json_file, sample_pmc_data):
    """Test loading PMC data with valid documents."""
    # Set up mock
    mock_load.return_value = sample_pmc_data
    
    # Call the method
    result = batch_processor.load_pmc_data(temp_json_file)
    
    # Expected number of valid documents (abstracts with length >= MIN_ABSTRACT_CONTENT_LENGTH)
    expected_count = sum(
        1 for doc in sample_pmc_data
        if doc.get("abstract", "").strip() and len(doc.get("abstract", "").strip()) >= MIN_ABSTRACT_CONTENT_LENGTH
    )
    
    # Assertions
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == expected_count, f"Expected {expected_count} valid documents, got {len(result)}"
    assert all(
        doc.get("abstract", "").strip() and len(doc.get("abstract", "").strip()) >= MIN_ABSTRACT_CONTENT_LENGTH
        for doc in result
    ), "Some documents have invalid or too-short abstracts"
    
    # Verify specific document content
    expected_pmids = ["12345", "67890", "11111"]  # PMIDs of valid documents
    assert all(doc["pmid"] in expected_pmids for doc in result), "Unexpected PMIDs in result"
    assert len(result) == len(expected_pmids), f"Expected {len(expected_pmids)} PMIDs, got {len(result)}"
    
    # Verify logging
    assert mock_logger.call_count >= 1, "Expected logging calls"
    mock_logger.assert_any_call(f"Found {len(result)} documents with valid abstracts.")
    if len(sample_pmc_data) != len(result):
        mock_logger.assert_any_call(
            f"Filtered {len(sample_pmc_data) - len(result)} documents lacking valid abstracts or too short."
        )
