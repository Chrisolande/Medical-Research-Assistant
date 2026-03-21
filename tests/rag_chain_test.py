import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import networkx as nx

from medical_graph_rag.rag_chain import QueryEngine


def _build_engine():
    vector_store = MagicMock()
    knowledge_graph = MagicMock()
    llm = MagicMock()
    engine = QueryEngine(
        vector_store=vector_store, knowledge_graph=knowledge_graph, llm=llm
    )
    return engine, vector_store, knowledge_graph


def test_parse_fallback_response():
    engine, _, _ = _build_engine()
    sufficient, answer = engine._parse_fallback_response(
        "Sufficient: Yes\nSynthesized Answer (if Yes): test answer"
    )
    assert sufficient is True
    assert answer == "test answer"


def test_check_answer_short_context_returns_false():
    engine, _, _ = _build_engine()
    ok, answer = engine._check_answer("q", "short line")
    assert ok is False
    assert answer == ""


def test_analyze_chunk_distribution_empty_returns_zero():
    engine, _, _ = _build_engine()
    avg = asyncio.run(engine._analyze_chunk_distribution([]))
    assert avg == 0.0


def test_initialize_traversal_uses_content_to_node_id():
    engine, vector_store, knowledge_graph = _build_engine()
    doc = SimpleNamespace(page_content="retrieved")
    closest_doc = SimpleNamespace(page_content="matched-node")
    vector_store.similarity_search_with_score = AsyncMock(
        return_value=[(closest_doc, 0.5)]
    )
    knowledge_graph.content_to_node_id = {"matched-node": 7}

    queue, distances = asyncio.run(engine._initialize_traversal([doc]))
    assert queue
    assert distances[7] > 0


def test_query_returns_no_relevant_information_when_no_docs():
    engine, vector_store, _ = _build_engine()
    vector_store.retrieve_relevant_documents = AsyncMock(return_value=[])

    response, traversal, filtered = asyncio.run(engine.query("q"))
    assert response == "No relevant information found."
    assert traversal == []
    assert filtered == {}


def test_check_answer_structured_success():
    engine = QueryEngine.__new__(QueryEngine)
    engine.answer_check_chain = MagicMock()
    engine.answer_check_chain.invoke.return_value = SimpleNamespace(
        is_sufficient=True, synthesized_answer="answer"
    )
    ok, answer = engine._check_answer("q", "line1\nline2\nline3")
    assert ok is True
    assert answer == "answer"


def test_process_node_updates_context_and_visited_concepts():
    engine = QueryEngine.__new__(QueryEngine)
    engine.max_context_length = 1000
    graph = nx.Graph()
    graph.add_node(1, content="abc", concepts=["Diabetes"])
    engine.knowledge_graph = SimpleNamespace(
        graph=graph, _lemmatize_concept=lambda c: c.lower()
    )
    engine._check_answer = MagicMock(return_value=(False, ""))

    expanded, traversal, filtered, ans, done = engine._process_node(
        current_node=1,
        query="q",
        expanded_context="",
        traversal_path=[],
        visited_concepts=set(),
        filtered_content={},
        step=1,
    )
    assert "abc" in expanded
    assert traversal == [1]
    assert filtered[1] == "abc"
    assert ans == ""
    assert done is False


def test_explore_neighbors_pushes_new_nodes():
    engine = QueryEngine.__new__(QueryEngine)
    graph = nx.Graph()
    graph.add_node(1, concepts=["a"])
    graph.add_node(2, concepts=["b"])
    graph.add_edge(1, 2, weight=0.8)
    engine.knowledge_graph = SimpleNamespace(
        graph=graph, _lemmatize_concept=lambda c: c.lower()
    )
    pq = []
    distances = {1: 1.0}
    engine._explore_neighbors(1, 1.0, [], set(), distances, pq)
    assert pq
    assert 2 in distances


def test_query_uses_expand_context_when_docs_found():
    engine, vector_store, _ = _build_engine()
    vector_store.retrieve_relevant_documents = AsyncMock(
        return_value=[SimpleNamespace(page_content="doc")]
    )
    engine._analyze_chunk_distribution = AsyncMock(return_value=1.0)
    engine._expand_context = AsyncMock(
        return_value=("ctx", [1], {1: "c"}, SimpleNamespace(content="final"))
    )
    response, traversal, filtered = asyncio.run(engine.query("q"))
    assert response.content == "final"
    assert traversal == [1]
    assert filtered == {1: "c"}
