import json
import tempfile
from unittest.mock import patch

import pytest

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
