import pytest

from medical_graph_rag.core.config import (
    MIN_ABSTRACT_CONTENT_LENGTH,
    PMC_BATCH_SIZE,
    PMC_INTER_BATCH_DELAY,
    PMC_MAX_CONCURRENT_BATCHES,
    PMC_RETRY_ATTEMPTS,
    PMC_RETRY_DELAY,
)
from medical_graph_rag.data_processing.document_processor import DocumentProcessor


@pytest.fixture(scope="session")
def document_processor(test_data):
    """Fixture for DocumentProcessor with default settings."""
    return DocumentProcessor(
        embeddings_model=test_data["embeddings_model"],
        metadata_fields=test_data["metadata_fields"],
    )


@pytest.fixture(scope="session")
def test_data():
    """Centralized test data."""
    return {
        "abstracts": [
            "This is abstract one for testing purposes with medical terminology.",
            "This is another abstract for testing cardiovascular disease research.",
            "Sample abstract about machine learning applications in healthcare.",
            "Applications of statistics in machine learning",
            "",
            "   ",
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
            {
                "pmid": "1234",
                "title": "Intersecting Machine Learning with Statistics.",
                "authors": "Cobain, K.",
            },
            {
                "pmid": "6571",
                "title": "Why pure mathematics is useful",
                "authors": "Cobain, K.",
            },
            {"pmid": "2039", "title": "proof pi = 3 = e", "authors": "Cobain, K."},
        ],
        "metadata_fields": ["pmid", "title", "authors"],
        "embeddings_model": "abhinand/MedEmbed-small-v0.1",
        "config": {
            "batch_size": PMC_BATCH_SIZE,
            "max_concurrent_batches": PMC_MAX_CONCURRENT_BATCHES,
            "retry_attempts": PMC_RETRY_ATTEMPTS,
            "retry_delay": PMC_RETRY_DELAY,
            "inter_batch_delay": PMC_INTER_BATCH_DELAY,
            "min_abstract_length": MIN_ABSTRACT_CONTENT_LENGTH,
        },
    }
