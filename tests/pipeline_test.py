import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medical_graph_rag.pipeline import Main


def test_initialize_llm_requires_api_key():
    with patch("medical_graph_rag.pipeline.DEEPSEEK_API_KEY", None):
        main = Main.__new__(Main)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            main._initialize_llm()


def test_init_uses_injected_embedding_and_ner():
    with (
        patch("medical_graph_rag.pipeline.Main._initialize_llm", return_value="llm"),
        patch("medical_graph_rag.pipeline.KnowledgeGraph") as kg_cls,
        patch("medical_graph_rag.pipeline.VectorStore") as vs_cls,
        patch("medical_graph_rag.pipeline.GraphVisualizer") as gv_cls,
    ):
        emb = object()
        ner = object()
        main = Main(cache_dir="cache", embedding_model=emb, ner_pipeline=ner)
        assert main.llm == "llm"
        kg_cls.assert_called_once_with(
            cache_dir="cache", embeddings=emb, ner_pipeline=ner
        )
        vs_cls.assert_called_once_with(embeddings=emb)
        gv_cls.assert_called_once()


def test_process_documents_initializes_query_engine():
    main = Main.__new__(Main)
    main.llm = "llm"
    main.knowledge_graph = MagicMock()
    main.vector_store = MagicMock()
    main.vector_store.create_vector_index = AsyncMock()

    with patch("medical_graph_rag.pipeline.QueryEngine", return_value="qe") as qe_cls:
        asyncio.run(main.process_documents([{"content": "a"}]))
        main.knowledge_graph.build_knowledge_graph.assert_called_once()
        main.vector_store.create_vector_index.assert_awaited_once()
        qe_cls.assert_called_once_with(main.vector_store, main.knowledge_graph, "llm")
        assert main.query_engine == "qe"


def test_query_requires_initialized_query_engine():
    main = Main.__new__(Main)
    main.query_engine = None
    with pytest.raises(RuntimeError, match="Query engine not initialized"):
        asyncio.run(main.query("what?"))


def test_query_calls_visualizer_when_traversal_path_present():
    main = Main.__new__(Main)
    main.knowledge_graph = MagicMock()
    main.knowledge_graph.graph = MagicMock()
    main.query_engine = MagicMock()
    main.query_engine.query = AsyncMock(return_value=("answer", [1, 2], {1: "doc"}))
    main.visualizer = MagicMock()
    main.visualizer.visualize_traversal_async = AsyncMock()

    result = asyncio.run(main.query("query"))
    assert result == ("answer", [1, 2], {1: "doc"})
    main.visualizer.visualize_traversal_async.assert_awaited_once()


def test_initialize_llm_success():
    with (
        patch("medical_graph_rag.pipeline.DEEPSEEK_API_KEY", "k"),
        patch("medical_graph_rag.pipeline.ChatDeepSeek", return_value="client") as cls,
    ):
        main = Main.__new__(Main)
        client = main._initialize_llm()
        assert client == "client"
        cls.assert_called_once()


def test_initialize_embeddings_uses_configured_model():
    with patch(
        "medical_graph_rag.pipeline.HuggingFaceEmbeddings", return_value="emb"
    ) as cls:
        main = Main.__new__(Main)
        emb = main._initialize_embeddings()
        assert emb == "emb"
        cls.assert_called_once()


def test_init_logs_and_raises_on_failure():
    with (
        patch(
            "medical_graph_rag.pipeline.Main._initialize_llm",
            side_effect=RuntimeError("boom"),
        ),
        patch("medical_graph_rag.pipeline.logger.error") as log_error,
    ):
        with pytest.raises(RuntimeError):
            Main()
        log_error.assert_called_once()


def test_process_documents_raises_on_failure():
    main = Main.__new__(Main)
    main.knowledge_graph = MagicMock()
    main.knowledge_graph.build_knowledge_graph.side_effect = RuntimeError("fail")
    main.vector_store = MagicMock()
    main.llm = "llm"
    with pytest.raises(RuntimeError):
        asyncio.run(main.process_documents([{"content": "a"}]))


def test_query_with_no_traversal_path_skips_visualization():
    main = Main.__new__(Main)
    main.knowledge_graph = MagicMock()
    main.query_engine = MagicMock()
    main.query_engine.query = AsyncMock(return_value=("answer", [], {}))
    main.visualizer = MagicMock()
    main.visualizer.visualize_traversal_async = AsyncMock()

    result = asyncio.run(main.query("query"))
    assert result == ("answer", [], {})
    main.visualizer.visualize_traversal_async.assert_not_awaited()
