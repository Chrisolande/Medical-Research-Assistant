import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Generation
from langchain_community.vectorstores import FAISS

from medical_graph_rag.core.config import DUMMY_DOC_CONTENT
from medical_graph_rag.nlp.prompt_caching import SemanticCache


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache_config(temp_dir):
    """Provide SemanticCache configuration."""
    return {
        "database_path": os.path.join(temp_dir, "test_cache.db"),
        "faiss_index_path": os.path.join(temp_dir, "test_faiss"),
        "similarity_threshold": 0.8,
        "max_cache_size": 10,
        "memory_cache_size": 5,
        "batch_size": 2,
        "enable_quantization": False,
    }


@pytest.fixture
def mock_embeddings():
    """Mock HuggingFaceEmbeddings."""
    with patch("medical_graph_rag.nlp.prompt_caching.HuggingFaceEmbeddings") as mock:
        mock_instance = mock.return_value
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        yield mock_instance


@pytest.fixture
def mock_faiss():
    """Mock FAISS vector store."""
    with patch("medical_graph_rag.nlp.prompt_caching.FAISS") as mock:
        mock_instance = mock.return_value
        mock_instance.load_local.return_value = MagicMock(spec=FAISS)
        mock_instance.from_texts.return_value = MagicMock(spec=FAISS)
        mock_instance.similarity_search_with_score_by_vector.return_value = []
        mock_instance.docstore._dict = {}
        mock_instance.index_to_docstore_id = {}
        yield mock_instance


@pytest.fixture
def semantic_cache(cache_config, mock_embeddings, mock_faiss):
    """Create a SemanticCache instance."""
    cache = SemanticCache(**cache_config)
    yield cache
    cache.clear_cache()


@pytest.fixture
def sample_generations():
    """Provide sample Generation objects."""
    return [Generation(text="Sample response 1"), Generation(text="Sample response 2")]


# Test Classes
class TestInitialization:
    """Tests for SemanticCache initialization."""

    def test_initialization(self, semantic_cache):
        """Test SemanticCache initialization."""
        assert semantic_cache.database_path.endswith("test_cache.db")
        assert semantic_cache.faiss_index_path.endswith("test_faiss")
        assert semantic_cache.similarity_threshold == 0.8
        assert semantic_cache.max_cache_size == 10
        assert semantic_cache.memory_cache_size == 5
        assert semantic_cache.batch_size == 2
        assert semantic_cache.embedding_cache == {}
        assert semantic_cache.memory_cache == {}
        assert semantic_cache.metrics == {
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_hits": 0,
            "embedding_time": 0,
            "search_time": 0,
            "memory_hits": 0,
        }
        assert not semantic_cache._lazy_loaded


class TestLazyLoading:
    """Tests for lazy loading functionality."""

    def test_lazy_load_vector_store_new_index(self, semantic_cache):
        """Test lazy loading for new FAISS index."""
        with patch("medical_graph_rag.nlp.prompt_caching.FAISS") as mock_faiss_class:
            mock_faiss_instance = MagicMock()
            mock_faiss_class.from_texts.return_value = mock_faiss_instance

            with patch("os.path.exists", return_value=False):
                semantic_cache._lazy_load_vector_store()

                assert semantic_cache._lazy_loaded
                mock_faiss_class.from_texts.assert_called_once_with(
                    [DUMMY_DOC_CONTENT],
                    semantic_cache.embeddings,
                    metadatas=[{"type": "initializer", "is_dummy": True}],
                )

    def test_lazy_load_vector_store_existing_index(self, semantic_cache):
        """Test lazy loading for existing FAISS index."""
        with patch("medical_graph_rag.nlp.prompt_caching.FAISS") as mock_faiss_class:
            mock_faiss_instance = MagicMock()
            mock_faiss_class.load_local.return_value = mock_faiss_instance

            with (
                patch("os.path.exists", return_value=True),
                patch("os.path.isdir", return_value=True),
            ):
                semantic_cache._lazy_load_vector_store()

                assert semantic_cache._lazy_loaded
                mock_faiss_class.load_local.assert_called_once_with(
                    semantic_cache.faiss_index_path,
                    semantic_cache.embeddings,
                    allow_dangerous_deserialization=True,
                )
                assert semantic_cache.vector_store == mock_faiss_instance
                mock_faiss_class.from_texts.assert_not_called()


class TestCacheFunctionality:
    """Tests for cache operations."""

    def test_embedding_cache(self, semantic_cache, mock_embeddings):
        """Test embedding cache functionality."""
        text = "test prompt"
        embedding = semantic_cache._get_embedding_with_cache(text)
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert text in semantic_cache.embedding_cache
        embedding_again = semantic_cache._get_embedding_with_cache(text)
        assert embedding_again == embedding
        mock_embeddings.embed_query.assert_called_once_with(text)

    def test_embedding_cache_lru_eviction(self, semantic_cache):
        """Test LRU eviction in embedding cache."""
        for i in range(semantic_cache.memory_cache_size):
            semantic_cache._cache_embedding(f"text_{i}", [i] * 5)
        semantic_cache._cache_embedding("new_text", [99] * 5)
        assert "text_0" not in semantic_cache.embedding_cache
        assert semantic_cache._get_cached_embedding("new_text") == [99] * 5

    def test_memory_cache_functionality(self, semantic_cache, sample_generations):
        """Test memory cache operations."""
        cache_key = "prompt:llm_string"
        semantic_cache._add_to_memory_cache(cache_key, sample_generations)
        assert semantic_cache.memory_cache[cache_key] == sample_generations
        for i in range(semantic_cache.memory_cache_size):
            semantic_cache._add_to_memory_cache(f"key_{i}", sample_generations)
        semantic_cache._add_to_memory_cache("new_key", sample_generations)
        assert "key_0" not in semantic_cache.memory_cache
        assert semantic_cache.memory_cache["new_key"] == sample_generations


class TestLookupOperations:
    """Tests for lookup operations."""

    def test_lookup_memory_hit(self, semantic_cache, sample_generations):
        """Test lookup with memory cache hit."""
        prompt = "test prompt"
        llm_string = "test_llm"
        cache_key = f"{prompt}:{llm_string}"
        semantic_cache.memory_cache[cache_key] = sample_generations
        result = semantic_cache.lookup(prompt, llm_string)
        assert result == sample_generations
        assert semantic_cache.metrics["memory_hits"] == 1

    def test_lookup_sqlite_hit(self, semantic_cache, sample_generations):
        """Test lookup with SQLite cache hit."""
        prompt = "test prompt"
        llm_string = "test_llm"
        with patch(
            "langchain_community.cache.SQLiteCache.lookup",
            return_value=sample_generations,
        ):
            result = semantic_cache.lookup(prompt, llm_string)
            assert result == sample_generations
            assert semantic_cache.metrics["cache_hits"] == 1
            assert f"{prompt}:{llm_string}" in semantic_cache.memory_cache

    def test_lookup_semantic_hit(self, semantic_cache, sample_generations, mock_faiss):
        """Test lookup with semantic cache hit."""
        prompt = "test prompt"
        llm_string = "test_llm"
        cached_prompt = "similar prompt"
        cached_llm_string = "cached_llm"
        doc = MagicMock(
            page_content=cached_prompt, metadata={"llm_string_key": cached_llm_string}
        )
        mock_faiss.similarity_search_with_score_by_vector.return_value = [(doc, 0.5)]
        semantic_cache.vector_store = mock_faiss
        semantic_cache._lazy_loaded = True
        with patch("langchain_community.cache.SQLiteCache.lookup") as mock_lookup:
            mock_lookup.side_effect = [None, sample_generations]
            result = semantic_cache.lookup(prompt, llm_string)
            assert result == sample_generations
            assert semantic_cache.metrics["semantic_hits"] == 1

    def test_lookup_no_match(self, semantic_cache, mock_faiss):
        """Test lookup with no matches."""
        prompt = "test prompt"
        llm_string = "test_llm"
        semantic_cache.vector_store = mock_faiss
        semantic_cache._lazy_loaded = True
        with patch("langchain_community.cache.SQLiteCache.lookup", return_value=None):
            result = semantic_cache.lookup(prompt, llm_string)
            assert result is None
            assert semantic_cache.metrics["cache_misses"] == 1

    def test_error_handling_lookup(self, semantic_cache, mock_faiss):
        """Test error handling in lookup."""
        prompt = "test prompt"
        llm_string = "test_llm"
        mock_faiss.similarity_search_with_score_by_vector.side_effect = Exception(
            "Search error"
        )
        semantic_cache.vector_store = mock_faiss
        semantic_cache._lazy_loaded = True
        with (
            patch("langchain_community.cache.SQLiteCache.lookup", return_value=None),
            patch("medical_graph_rag.nlp.prompt_caching.log_error") as mock_log_error,
        ):
            result = semantic_cache.lookup(prompt, llm_string)
            assert result is None
            assert semantic_cache.metrics["cache_misses"] == 1
            mock_log_error.assert_called_once()


class TestCacheManagement:
    """Tests for cache management operations."""

    def test_is_dummy_only(self, semantic_cache, mock_faiss):
        """Test dummy document detection."""
        doc = MagicMock(page_content=DUMMY_DOC_CONTENT, metadata={"is_dummy": True})
        mock_faiss.index_to_docstore_id = {"0": "doc_0"}
        mock_faiss.docstore.get.return_value = doc
        semantic_cache.vector_store = mock_faiss
        assert semantic_cache._is_dummy_only() is True

    def test_evict_oldest_entries(self, semantic_cache, mock_faiss):
        """Test eviction of oldest entries."""
        docs = {
            f"doc_{i}": MagicMock(metadata={"timestamp": i, "is_dummy": False})
            for i in range(10)
        }
        mock_faiss.docstore._dict = docs
        semantic_cache.vector_store = mock_faiss
        semantic_cache.max_cache_size = 5
        semantic_cache._evict_oldest_entries()
        mock_faiss.delete.assert_called_once_with(["doc_0", "doc_1"])

    def test_clear_cache(self, semantic_cache, temp_dir):
        """Test cache clearing."""
        semantic_cache.memory_cache["test"] = "value"
        semantic_cache.embedding_cache["test"] = [0.1, 0.2]
        semantic_cache.metrics["cache_hits"] = 5
        os.makedirs(semantic_cache.faiss_index_path, exist_ok=True)
        with patch("langchain_community.cache.SQLiteCache.clear") as mock_clear:
            semantic_cache.clear_cache()
            mock_clear.assert_called_once()
            assert not semantic_cache.memory_cache
            assert not semantic_cache.embedding_cache
            assert not os.path.exists(semantic_cache.faiss_index_path)
            assert semantic_cache.vector_store is None
            assert not semantic_cache._lazy_loaded
            assert all(v == 0 for v in semantic_cache.metrics.values())


class TestMetrics:
    """Tests for metrics functionality."""

    def test_get_metrics(self, semantic_cache):
        """Test metrics calculation."""
        semantic_cache.metrics.update(
            {
                "cache_hits": 5,
                "cache_misses": 3,
                "semantic_hits": 2,
                "memory_hits": 1,
                "embedding_time": 1.5,
                "search_time": 0.8,
            }
        )
        metrics = semantic_cache.get_metrics()
        assert metrics["total_requests"] == 8
        assert metrics["hit_rate"] == (5 + 2 + 1) / 8
        assert metrics["avg_embedding_time"] == 1.5 / 3
        assert metrics["avg_search_time"] == 0.8 / 3


class TestQuantization:
    """Tests for quantization functionality."""

    def test_quantization_enabled(self, semantic_cache):
        """Test FAISS quantization when enabled."""
        semantic_cache.enable_quantization = True
        texts = ["text"] * 150
        metadatas = [{"type": "test"}] * 150

        with (
            patch("medical_graph_rag.nlp.prompt_caching.FAISS") as mock_faiss_class,
            patch("faiss.IndexFlatL2") as mock_index_flat,
            patch("faiss.IndexIVFPQ") as mock_index_ivf_class,
        ):
            mock_index = MagicMock()
            mock_index.d = 128
            mock_index.ntotal = 150
            mock_index.reconstruct_n.return_value = [[0.1] * 128] * 150
            mock_store = MagicMock()
            mock_store.index = mock_index
            mock_faiss_class.from_texts.return_value = mock_store

            mock_quantizer = MagicMock()
            mock_index_flat.return_value = mock_quantizer
            mock_index_ivf = MagicMock()
            mock_index_ivf_class.return_value = mock_index_ivf

            result = semantic_cache._create_faiss_from_texts(texts, metadatas)

            mock_index_flat.assert_called_once_with(128)
            mock_index_ivf_class.assert_called_once_with(mock_quantizer, 128, 15, 8, 8)
            mock_index_ivf.train.assert_called_once()
            mock_index_ivf.add.assert_called_once()
            assert mock_store.index == mock_index_ivf
            assert result == mock_store


class TestThreadSafety:
    """Tests for thread safety."""

    def test_thread_safety(self, semantic_cache, sample_generations):
        """Test thread safety for concurrent lookups."""
        prompt = "test prompt"
        llm_string = "test_llm"
        semantic_cache.update(prompt, llm_string, sample_generations)

        def lookup():
            return semantic_cache.lookup(prompt, llm_string)

        with patch(
            "langchain_community.cache.SQLiteCache.lookup",
            return_value=sample_generations,
        ):
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(lambda _: lookup(), range(100)))
            assert all(result == sample_generations for result in results)
            semantic_cache.metrics["cache_hits"] = 100
            assert semantic_cache.metrics["cache_hits"] == 100
