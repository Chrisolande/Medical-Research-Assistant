import pytest


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
            {
                "pmid": "2039",
                "title": "proof pi = 3 = e",
                "authors": "Cobain, K.",
            },
        ],
        "metadata_fields": ["pmid", "title", "authors"],
        "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
    }
