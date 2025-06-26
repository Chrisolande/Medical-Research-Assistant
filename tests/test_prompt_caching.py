import unittest
import os
import shutil
from unittest.mock import patch, MagicMock

# Set a dummy API key for testing purposes
os.environ['COHERE_API_KEY1'] = 'test-key'

from prompt_caching import SemanticCache

class TestSemanticCache(unittest.TestCase):

    def setUp(self):
        """Set up a test environment before each test."""
        self.db_path = "./test.db"
        self.faiss_path = "./test_faiss_index"
        # Ensure the FAISS path exists for the test
        os.makedirs(self.faiss_path, exist_ok=True)

        # Mock CohereEmbeddings to avoid actual API calls
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_query.return_value = [0.1] * 768

        with patch('prompt_caching.CohereEmbeddings', return_value=self.mock_embeddings):
            self.cache = SemanticCache(
                database_path=self.db_path,
                faiss_index_path=self.faiss_path
            )

        # Populate caches for testing clear_cache
        self.cache.memory_cache['test_key'] = 'test_value'
        self.cache.embedding_cache['test_prompt'] = [0.1] * 768
        # Add to SQLite cache
        self.cache.update('prompt', 'generation')
        self.cache.metrics["cache_hits"] = 1

    def tearDown(self):
        """Clean up the environment after each test."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.faiss_path):
            shutil.rmtree(self.faiss_path)

    def test_clear_cache(self):
        """Test the clear_cache method."""
        # Call the method to be tested
        self.cache.clear_cache()

        # Assert that the caches are cleared
        self.assertEqual(len(self.cache.memory_cache), 0)
        self.assertEqual(len(self.cache.embedding_cache), 0)
        self.assertFalse(os.path.exists(self.faiss_path))
        self.assertIsNone(self.cache.vector_store)
        self.assertFalse(self.cache._lazy_loaded)

        # Assert that metrics are reset
        self.assertEqual(self.cache.metrics['cache_hits'], 0)

        # Assert that the SQLite database is cleared
        self.assertIsNone(self.cache.lookup('prompt', 'llm_string'))

if __name__ == '__main__':
    unittest.main()
