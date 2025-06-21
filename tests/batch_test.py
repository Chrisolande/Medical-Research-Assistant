import json
import tempfile

import pytest

from src.data_processing.batch_processor import PMCBatchProcessor


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
