import os
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import DeterministicFakeEmbedding
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from medical_graph_rag.nlp.vectorstore import VectorStore


@pytest.fixture
def temp_persist_dir():
    temp_dir = "./test_faiss_index"
    os.makedirs(temp_dir, exist_ok=True)
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_faiss_index():
    mock_faiss = MagicMock(spec=FAISS)
    mock_faiss.add_documents = MagicMock()
    mock_faiss.save_local = MagicMock()
    mock_faiss.docstore = MagicMock(spec=InMemoryDocstore)
    mock_faiss.docstore._dict = {}
    mock_faiss.index_to_docstore_id = {}
    mock_faiss.as_retriever = MagicMock(return_value=MagicMock())
    mock_faiss.asimilarity_search = AsyncMock(
        return_value=[
            Document(page_content="test doc 1"),
            Document(page_content="test doc 2"),
        ]
    )
    mock_faiss.similarity_search_with_score = MagicMock(
        return_value=[
            (Document(page_content="test doc 1"), 0.8),
            (Document(page_content="test doc 2"), 0.7),
        ]
    )
    return mock_faiss


@pytest.fixture(autouse=True)
def mock_dependencies(temp_persist_dir):
    with (
        patch(
            "medical_graph_rag.nlp.vectorstore.HuggingFaceEmbeddings"
        ) as MockEmbeddings,
        patch("medical_graph_rag.nlp.vectorstore.ChatOpenAI") as MockChatOpenAI,
        patch(
            "medical_graph_rag.nlp.vectorstore.FAISS.load_local"
        ) as MockFAISSLoadLocal,
        patch(
            "medical_graph_rag.nlp.vectorstore.os.path.exists", return_value=False
        ) as mock_os_path_exists,
        patch("medical_graph_rag.nlp.vectorstore.ensure_semantic_cache"),
        patch("medical_graph_rag.nlp.vectorstore.ranker", autospec=True),
        patch(
            "medical_graph_rag.nlp.vectorstore.ContextualCompressionRetriever"
        ) as MockContextualCompressionRetriever,
        patch(
            "medical_graph_rag.nlp.vectorstore.CrossEncoderReranker", autospec=True
        ) as MockCrossEncoderReranker,
        patch(
            "medical_graph_rag.nlp.vectorstore.HuggingFaceCrossEncoder", autospec=True
        ) as MockHuggingFaceCrossEncoder,
        patch(
            "medical_graph_rag.nlp.vectorstore.EmbeddingsFilter", autospec=True
        ) as MockEmbeddingsFilter,
        patch(
            "medical_graph_rag.nlp.vectorstore.EmbeddingsRedundantFilter", autospec=True
        ) as MockEmbeddingsRedundantFilter,
        patch(
            "medical_graph_rag.nlp.vectorstore.FlashrankRerank", autospec=True
        ) as MockFlashrankRerank,
        patch(
            "medical_graph_rag.nlp.vectorstore.LLMChainExtractor", autospec=True
        ) as MockLLMChainExtractor,
        patch(
            "medical_graph_rag.nlp.vectorstore.DocumentCompressorPipeline",
            autospec=True,
        ) as MockDocumentCompressorPipeline,
        patch("medical_graph_rag.nlp.vectorstore.shutil.rmtree") as mock_shutil_rmtree,
    ):
        MockEmbeddings.return_value = DeterministicFakeEmbedding(size=10)
        MockChatOpenAI.return_value = MagicMock()

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.ainvoke = AsyncMock(
            return_value=[
                Document(page_content="reranked doc 1"),
                Document(page_content="reranked doc 2"),
            ]
        )
        mock_retriever_instance.invoke = MagicMock(
            return_value=[
                Document(page_content="reranked doc 1"),
                Document(page_content="reranked doc 2"),
            ]
        )
        MockContextualCompressionRetriever.return_value = mock_retriever_instance

        MockCrossEncoderReranker.return_value = MagicMock()
        MockHuggingFaceCrossEncoder.return_value = MagicMock()
        MockEmbeddingsFilter.return_value = MagicMock()
        MockEmbeddingsRedundantFilter.return_value = MagicMock()
        MockFlashrankRerank.return_value = MagicMock()
        MockLLMChainExtractor.from_llm.return_value = MagicMock()

        MockDocumentCompressorPipeline.return_value = MagicMock()
        MockDocumentCompressorPipeline.return_value.invoke = MagicMock(
            return_value=[Document(page_content="compressed doc from pipeline")]
        )

        yield (
            MockEmbeddings,
            MockChatOpenAI,
            MockFAISSLoadLocal,
            MockContextualCompressionRetriever,
            mock_os_path_exists,
            mock_shutil_rmtree,
        )


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_initialization_no_existing_index(
        self, temp_persist_dir, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, _, _ = mock_dependencies
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        assert vec_store.vector_index is None
        assert not MockFAISSLoadLocal.called
        assert isinstance(vec_store.embeddings, DeterministicFakeEmbedding)
        assert vec_store.semaphore is not None
        assert vec_store.llm is not None

    @pytest.mark.asyncio
    async def test_initialization_with_existing_index(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        assert vec_store.vector_index == mock_faiss_index
        MockFAISSLoadLocal.assert_called_once_with(
            temp_persist_dir, vec_store.embeddings, allow_dangerous_deserialization=True
        )
        assert vec_store.added_doc_hashes == set()

    @pytest.mark.asyncio
    async def test_load_local_index_failure(self, temp_persist_dir, mock_dependencies):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.side_effect = Exception("Failed to load")

        vec_store = VectorStore(persist_directory=temp_persist_dir)
        assert vec_store.vector_index is None
        assert not vec_store.added_doc_hashes
        MockFAISSLoadLocal.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_new_index(self, temp_persist_dir, mock_dependencies):
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        docs = [Document(page_content="Hello world")]

        with patch(
            "medical_graph_rag.nlp.vectorstore.FAISS.from_documents"
        ) as MockFAISSFromDocs:
            MockFAISSFromDocs.return_value = MagicMock(spec=FAISS)
            MockFAISSFromDocs.return_value.save_local = MagicMock()
            await vec_store._create_vector_index(docs)
            MockFAISSFromDocs.assert_called_once()
            assert vec_store.vector_index is not None
            assert vec_store._get_document_hash(docs[0]) in vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_add_documents_to_existing_index(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        docs = [Document(page_content="Another document")]
        await vec_store._create_vector_index(docs)

        mock_faiss_index.add_documents.assert_called_once_with(docs)
        assert vec_store._get_document_hash(docs[0]) in vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_save_local_index_failure(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index

        vec_store = VectorStore(persist_directory=temp_persist_dir)
        vec_store.vector_index.save_local.side_effect = Exception("Save failed")

        vec_store._save_local_index()
        vec_store.vector_index.save_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_batch_and_persist_error(
        self, temp_persist_dir, mock_dependencies
    ):
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        docs = [Document(page_content="error doc")]

        with patch(
            "medical_graph_rag.nlp.vectorstore.FAISS.from_documents"
        ) as MockFAISSFromDocs:
            MockFAISSFromDocs.side_effect = Exception("Batch add error")
            added_count = await vec_store._add_batch_and_persist(docs)
            assert added_count == 0
            MockFAISSFromDocs.assert_called_once()
            assert not vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_similarity_search_without_reranker(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir, use_reranker=False)

        query = "test query"
        results = await vec_store.similarity_search(query, k=2)
        mock_faiss_index.asimilarity_search.assert_called_once_with(query, k=2)
        assert len(results) == 2
        assert results[0].page_content == "test doc 1"

    @pytest.mark.asyncio
    async def test_similarity_search_error(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir, use_reranker=False)

        mock_faiss_index.asimilarity_search.side_effect = Exception("Search error")

        results = await vec_store.similarity_search("query")
        assert results == []
        mock_faiss_index.asimilarity_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_with_score_error(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        mock_faiss_index.similarity_search_with_score.side_effect = Exception(
            "Score search error"
        )

        results = await vec_store.similarity_search_with_score("query")
        assert results == []
        mock_faiss_index.similarity_search_with_score.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_with_reranker(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        (
            _,
            _,
            MockFAISSLoadLocal,
            MockContextualCompressionRetriever,
            mock_os_path_exists,
            _,
        ) = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir, use_reranker=True)
        vec_store.compression_retriever = (
            MockContextualCompressionRetriever.return_value
        )

        query = "test query"
        results = await vec_store.similarity_search(query, k=2)
        MockContextualCompressionRetriever.return_value.ainvoke.assert_called_once_with(
            query
        )
        assert results[0].page_content == "reranked doc 1"
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete_index(self, temp_persist_dir, mock_dependencies):
        _, _, _, _, mock_os_path_exists, mock_shutil_rmtree = mock_dependencies
        mock_os_path_exists.return_value = True

        vec_store = VectorStore(persist_directory=temp_persist_dir)
        os.makedirs(vec_store.persist_directory, exist_ok=True)

        await vec_store.delete_index()
        mock_shutil_rmtree.assert_called_once_with(temp_persist_dir)
        assert vec_store.vector_index is None
        assert not vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_delete_index_failure(self, temp_persist_dir, mock_dependencies):
        _, _, _, _, mock_os_path_exists, mock_shutil_rmtree = mock_dependencies
        mock_os_path_exists.return_value = True
        os.makedirs(temp_persist_dir, exist_ok=True)

        mock_shutil_rmtree.side_effect = Exception("Deletion error")
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        vec_store.vector_index = MagicMock()

        await vec_store.delete_index()
        mock_shutil_rmtree.assert_called_once_with(temp_persist_dir)
        assert vec_store.vector_index is None
        assert not vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_get_document_hash(self, temp_persist_dir):
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        doc = Document(page_content="This is a test document.")
        computed_hash = vec_store._get_document_hash(doc)
        assert len(computed_hash) == 32
        assert all(c in "0123456789abcdef" for c in computed_hash)
        doc_different = Document(page_content="A different document.")
        assert vec_store._get_document_hash(doc_different) != computed_hash

    @pytest.mark.asyncio
    async def test_get_document_hash_empty_content(self, temp_persist_dir):
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        doc = Document(page_content="")
        computed_hash = vec_store._get_document_hash(doc)
        assert computed_hash == ""

    @pytest.mark.asyncio
    async def test_reconstruct_hashes_no_vector_index(self, temp_persist_dir):
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        vec_store.vector_index = None
        vec_store._reconstruct_hashes()
        assert not vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_reconstruct_hashes_no_docstore(
        self, temp_persist_dir, mock_faiss_index
    ):
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        vec_store.vector_index = mock_faiss_index
        vec_store.vector_index.docstore = None
        vec_store._reconstruct_hashes()
        assert not vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_reconstruct_hashes_non_document_in_docstore(
        self, temp_persist_dir, mock_faiss_index
    ):
        vec_store = VectorStore(persist_directory=temp_persist_dir)
        vec_store.vector_index = mock_faiss_index
        vec_store.vector_index.docstore._dict = {"test_id": "not a document"}
        vec_store.vector_index.index_to_docstore_id = {0: "test_id"}
        vec_store._reconstruct_hashes()
        assert not vec_store.added_doc_hashes

    @pytest.mark.asyncio
    async def test_setup_reranker_disabled(self, temp_persist_dir):
        vec_store = VectorStore(persist_directory=temp_persist_dir, use_reranker=False)
        vec_store.vector_index = MagicMock()
        vec_store._setup_reranker()
        assert vec_store.compression_retriever is None

    @pytest.mark.asyncio
    async def test_setup_reranker_no_vector_index(self, temp_persist_dir):
        vec_store = VectorStore(persist_directory=temp_persist_dir, use_reranker=True)
        vec_store.vector_index = None
        vec_store._setup_reranker()
        assert vec_store.compression_retriever is None

    @pytest.mark.asyncio
    async def test_setup_reranker_failure(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, _, _, _, _ = mock_dependencies
        with patch(
            "medical_graph_rag.nlp.vectorstore.HuggingFaceCrossEncoder"
        ) as MockHuggingFaceCrossEncoder:
            MockHuggingFaceCrossEncoder.side_effect = Exception("Reranker init error")
            vec_store = VectorStore(
                persist_directory=temp_persist_dir, use_reranker=True
            )
            vec_store.vector_index = mock_faiss_index

            vec_store._setup_reranker()

            assert vec_store.compression_retriever is None
            assert vec_store.use_reranker is False

    @pytest.mark.asyncio
    async def test_batch_query(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        _, _, MockFAISSLoadLocal, _, mock_os_path_exists, _ = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir, use_reranker=False)

        queries = ["query1", "query2"]
        results = await vec_store.batch_query(queries, k=1)

        assert mock_faiss_index.asimilarity_search.call_count == len(queries)
        assert len(results) == len(queries)
        assert all(len(r) == 2 for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_relevant_documents(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        (
            _,
            _,
            MockFAISSLoadLocal,
            MockContextualCompressionRetriever,
            mock_os_path_exists,
            _,
        ) = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        MockContextualCompressionRetriever.return_value.invoke.return_value = [
            Document(page_content="compressed doc")
        ]

        results = vec_store.retrieve_relevant_documents("test query")
        MockContextualCompressionRetriever.return_value.invoke.assert_called_once_with(
            "test query"
        )
        assert len(results) == 1
        assert results[0].page_content == "compressed doc"

    @pytest.mark.asyncio
    async def test_retrieve_relevant_documents_no_results(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        (
            _,
            _,
            MockFAISSLoadLocal,
            MockContextualCompressionRetriever,
            mock_os_path_exists,
            _,
        ) = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        MockContextualCompressionRetriever.return_value.invoke.return_value = []

        with patch("builtins.print") as mock_print:
            results = vec_store.retrieve_relevant_documents("test query")
            MockContextualCompressionRetriever.return_value.invoke.assert_called_once_with(
                "test query"
            )
            assert results == []
            mock_print.assert_called_once_with(
                "No relevant documents found for the query."
            )

    @pytest.mark.asyncio
    async def test_retrieve_relevant_documents_value_error(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        (
            _,
            _,
            MockFAISSLoadLocal,
            MockContextualCompressionRetriever,
            mock_os_path_exists,
            _,
        ) = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        MockContextualCompressionRetriever.return_value.invoke.side_effect = ValueError(
            "Error: token_type_ids missing from input feed"
        )

        results = vec_store.retrieve_relevant_documents("test query")
        assert results == []
        MockContextualCompressionRetriever.return_value.invoke.assert_called_once_with(
            "test query"
        )

    @pytest.mark.asyncio
    async def test_retrieve_relevant_documents_general_exception(
        self, temp_persist_dir, mock_faiss_index, mock_dependencies
    ):
        (
            _,
            _,
            MockFAISSLoadLocal,
            MockContextualCompressionRetriever,
            mock_os_path_exists,
            _,
        ) = mock_dependencies
        mock_os_path_exists.return_value = True
        MockFAISSLoadLocal.return_value = mock_faiss_index
        vec_store = VectorStore(persist_directory=temp_persist_dir)

        MockContextualCompressionRetriever.return_value.invoke.side_effect = Exception(
            "Generic retrieval error"
        )

        results = vec_store.retrieve_relevant_documents("test query")
        assert results == []
        MockContextualCompressionRetriever.return_value.invoke.assert_called_once_with(
            "test query"
        )

    @pytest.mark.asyncio
    async def test_filter_valid_docs(self):
        vec_store = VectorStore()
        docs = [
            Document(page_content="Valid content"),
            Document(page_content="   "),
            Document(page_content=""),
            Document(page_content="Another valid doc"),
        ]
        filtered_docs = vec_store._filter_valid_docs(docs)
        assert len(filtered_docs) == 2
        assert filtered_docs[0].page_content == "Valid content"
        assert filtered_docs[1].page_content == "Another valid doc"

    @pytest.mark.asyncio
    async def test_is_new_document(self):
        vec_store = VectorStore()
        doc1 = Document(page_content="Unique content")
        doc2 = Document(page_content="Existing content")
        vec_store.added_doc_hashes.add(vec_store._get_document_hash(doc2))

        assert vec_store._is_new_document(doc1) is True
        assert vec_store._is_new_document(doc2) is False
