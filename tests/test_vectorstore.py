import pytest
import asyncio
import os
import shutil
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain.schema import Document
import hashlib
from src.data_processing.document_processor import DocumentProcessor

# Import the VectorStore class (adjust import path as needed)
# from your_module import VectorStore


class TestVectorStore:
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Mock KnowledgeGraph for testing"""
        return Mock()
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(page_content="This is test document 1", metadata={"id": 1}),
            Document(page_content="This is test document 2", metadata={"id": 2}),
            Document(page_content="Another test document with different content", metadata={"id": 3}),
            Document(page_content="", metadata={"id": 4}),  # Empty content
            Document(page_content="   ", metadata={"id": 5}),  # Whitespace only
        ]
    
    @pytest.fixture
    @patch('your_module.HuggingFaceEmbeddings')
    @patch('your_module.FAISS')
    def vector_store(self, mock_faiss, mock_embeddings, mock_knowledge_graph, temp_directory):
        """Create VectorStore instance for testing"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        vs = VectorStore(
            knowledge_graph=mock_knowledge_graph,
            persist_directory=temp_directory,
            batch_size=2,
            max_concurrent=2,
            use_reranker=False  # Disable for simpler testing
        )
        return vs

    # Test initialization and setup
    def test_vector_store_initialization(self, mock_knowledge_graph, temp_directory):
        """Test VectorStore initialization"""
        with patch('src.data_processing.document_processor.HuggingFaceEmbeddings'):
            vs = VectorStore(
                knowledge_graph=mock_knowledge_graph,
                persist_directory=temp_directory,
                batch_size=100,
                max_concurrent=5
            )
            assert vs.batch_size == 100
            assert vs.max_concurrent == 5
            assert vs.persist_directory == temp_directory
            assert vs.semaphore._value == 5

    @patch('your_module.HuggingFaceCrossEncoder')
    @patch('your_module.CrossEncoderReranker')
    def test_setup_reranker_success(self, mock_reranker, mock_cross_encoder, vector_store):
        """Test successful reranker setup"""
        vector_store.use_reranker = True
        vector_store.vector_index = Mock()
        vector_store.vector_index.as_retriever.return_value = Mock()
        
        vector_store._setup_reranker()
        
        assert vector_store.compression_retriever is not None
        mock_cross_encoder.assert_called_once()
        mock_reranker.assert_called_once()

    def test_setup_reranker_no_index(self, vector_store):
        """Test reranker setup when no vector index exists"""
        vector_store.use_reranker = True
        vector_store.vector_index = None
        
        vector_store._setup_reranker()
        
        assert vector_store.compression_retriever is None

    # Test document hash methods
    def test_get_document_hash(self, vector_store):
        """Test document hash generation"""
        doc = Document(page_content="test content")
        expected_hash = hashlib.md5("test content".encode("utf-8")).hexdigest()
        
        result = vector_store._get_document_hash(doc)
        
        assert result == expected_hash

    def test_get_document_hash_empty_content(self, vector_store):
        """Test hash generation for empty content"""
        doc = Document(page_content="")
        result = vector_store._get_document_hash(doc)
        assert result == ""
        
        doc = Document(page_content=None)
        result = vector_store._get_document_hash(doc)
        assert result == ""

    def test_update_doc_hashes(self, vector_store, sample_documents):
        """Test updating document hashes"""
        batch = sample_documents[:2]
        initial_count = len(vector_store.added_doc_hashes)
        
        vector_store._update_doc_hashes(batch)
        
        assert len(vector_store.added_doc_hashes) == initial_count + 2

    def test_is_new_document(self, vector_store):
        """Test new document detection"""
        doc = Document(page_content="new content")
        
        # Should be new initially
        assert vector_store._is_new_document(doc) is True
        
        # Add hash and test again
        doc_hash = vector_store._get_document_hash(doc)
        vector_store.added_doc_hashes.add(doc_hash)
        assert vector_store._is_new_document(doc) is False

    # Test document filtering
    def test_filter_valid_docs(self, vector_store, sample_documents):
        """Test document filtering"""
        valid_docs = vector_store._filter_valid_docs(sample_documents)
        
        # Should filter out empty and whitespace-only documents
        assert len(valid_docs) == 3
        assert all(doc.page_content and doc.page_content.strip() for doc in valid_docs)

    # Test batch processing
    def test_create_batches(self, vector_store, sample_documents):
        """Test batch creation"""
        vector_store.batch_size = 2
        batches = vector_store._create_batches(sample_documents)
        
        assert len(batches) == 3  # 5 docs with batch_size=2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    @pytest.mark.asyncio
    async def test_add_batch_and_persist_empty_batch(self, vector_store):
        """Test adding empty batch"""
        result = await vector_store._add_batch_and_persist([])
        assert result == 0

    @pytest.mark.asyncio
    @patch('your_module.FAISS')
    async def test_add_batch_and_persist_new_index(self, mock_faiss, vector_store, sample_documents):
        """Test adding batch when no index exists"""
        vector_store.vector_index = None
        mock_index = Mock()
        mock_faiss.from_documents.return_value = mock_index
        
        batch = sample_documents[:2]
        result = await vector_store._add_batch_and_persist(batch)
        
        assert result == 2
        assert vector_store.vector_index == mock_index
        mock_faiss.from_documents.assert_called_once_with(batch, vector_store.embeddings)

    @pytest.mark.asyncio
    async def test_add_batch_and_persist_existing_index(self, vector_store, sample_documents):
        """Test adding batch to existing index"""
        mock_index = Mock()
        vector_store.vector_index = mock_index
        
        batch = sample_documents[:2]
        result = await vector_store._add_batch_and_persist(batch)
        
        assert result == 2
        mock_index.add_documents.assert_called_once_with(batch)

    # Test index persistence
    @patch('os.path.exists')
    def test_load_local_index_no_directory(self, mock_exists, vector_store):
        """Test loading when directory doesn't exist"""
        mock_exists.return_value = False
        
        vector_store._load_local_index()
        
        assert vector_store.vector_index is None

    @patch('your_module.FAISS')
    @patch('os.path.exists')
    def test_load_local_index_success(self, mock_exists, mock_faiss, vector_store):
        """Test successful index loading"""
        mock_exists.return_value = True
        mock_index = Mock()
        mock_index.docstore._dict = {
            "1": Document(page_content="test1"),
            "2": Document(page_content="test2")
        }
        mock_faiss.load_local.return_value = mock_index
        
        vector_store._load_local_index()
        
        assert vector_store.vector_index == mock_index
        assert len(vector_store.added_doc_hashes) == 2

    def test_save_local_index_no_index(self, vector_store):
        """Test saving when no index exists"""
        vector_store.vector_index = None
        
        # Should not raise exception
        vector_store._save_local_index()

    def test_save_local_index_success(self, vector_store):
        """Test successful index saving"""
        mock_index = Mock()
        vector_store.vector_index = mock_index
        
        vector_store._save_local_index()
        
        mock_index.save_local.assert_called_once_with(vector_store.persist_directory)

    @pytest.mark.asyncio
    @patch('shutil.rmtree')
    @patch('os.path.exists')
    async def test_delete_index_success(self, mock_exists, mock_rmtree, vector_store):
        """Test successful index deletion"""
        mock_exists.return_value = True
        vector_store.vector_index = Mock()
        vector_store.added_doc_hashes.add("test_hash")
        
        await vector_store.delete_index()
        
        mock_rmtree.assert_called_once_with(vector_store.persist_directory)
        assert vector_store.vector_index is None
        assert len(vector_store.added_doc_hashes) == 0

    # Test search methods
    @pytest.mark.asyncio
    async def test_similarity_search_no_index(self, vector_store):
        """Test similarity search when no index exists"""
        vector_store.vector_index = None
        
        result = await vector_store.similarity_search("test query")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_similarity_search_success(self, vector_store):
        """Test successful similarity search"""
        mock_index = Mock()
        mock_index.asimilarity_search = AsyncMock(return_value=[Document(page_content="result")])
        vector_store.vector_index = mock_index
        vector_store.use_reranker = False
        
        result = await vector_store.similarity_search("test query", k=2)
        
        assert len(result) == 1
        mock_index.asimilarity_search.assert_called_once_with("test query", k=2)

    @pytest.mark.asyncio
    async def test_similarity_search_with_reranker(self, vector_store):
        """Test similarity search with reranker"""
        vector_store.use_reranker = True
        vector_store.compression_retriever = AsyncMock()
        vector_store.compression_retriever.ainvoke = AsyncMock(return_value=[Document(page_content="reranked")])
        vector_store.compression_retriever.base_compressor = Mock()
        
        result = await vector_store.similarity_search("test query", k=3)
        
        assert len(result) == 1
        assert vector_store.compression_retriever.base_compressor.top_n == 3

    @pytest.mark.asyncio
    async def test_similarity_search_with_score_no_index(self, vector_store):
        """Test similarity search with score when no index exists"""
        vector_store.vector_index = None
        
        result = await vector_store.similarity_search_with_score("test query")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_query_no_index(self, vector_store):
        """Test batch query when no index exists"""
        vector_store.vector_index = None
        queries = ["query1", "query2"]
        
        result = await vector_store.batch_query(queries)
        
        assert result == [[], []]

    @pytest.mark.asyncio
    async def test_batch_query_success(self, vector_store):
        """Test successful batch query"""
        vector_store.vector_index = Mock()
        vector_store.use_reranker = False
        
        with patch.object(vector_store, 'similarity_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [Document(page_content="result")]
            queries = ["query1", "query2"]
            
            result = await vector_store.batch_query(queries, k=2)
            
            assert len(result) == 2
            assert mock_search.call_count == 2

    # Test reranked search
    @pytest.mark.asyncio
    async def test_perform_reranked_search_no_reranker(self, vector_store):
        """Test reranked search when reranker is disabled"""
        vector_store.use_reranker = False
        
        result = await vector_store.perform_reranked_search("test query")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_perform_reranked_search_success(self, vector_store):
        """Test successful reranked search"""
        vector_store.use_reranker = True
        vector_store.compression_retriever = AsyncMock()
        vector_store.compression_retriever.ainvoke = AsyncMock(return_value=[Document(page_content="reranked")])
        vector_store.compression_retriever.base_compressor = Mock()
        
        result = await vector_store.perform_reranked_search("test query", k=5)
        
        assert len(result) == 1
        assert vector_store.compression_retriever.base_compressor.top_n == 5
        vector_store.compression_retriever.ainvoke.assert_called_once_with("test query")

    # Integration test
    @pytest.mark.asyncio
    @patch('your_module.FAISS')
    async def test_create_vector_index_integration(self, mock_faiss, vector_store, sample_documents):
        """Test complete vector index creation flow"""
        mock_index = Mock()
        mock_faiss.from_documents.return_value = mock_index
        
        await vector_store._create_vector_index(sample_documents)
        
        # Should filter out invalid docs and create index
        assert vector_store.vector_index == mock_index
        assert len(vector_store.added_doc_hashes) > 0


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test configuration
pytest_plugins = []

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short", __file__])