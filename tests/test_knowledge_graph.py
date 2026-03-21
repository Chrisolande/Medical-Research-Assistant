from types import SimpleNamespace

import numpy as np

from medical_graph_rag.graph import KnowledgeGraph


def _make_split(text: str):
    return SimpleNamespace(page_content=text)


def _make_kg_for_unit_tests():
    kg = KnowledgeGraph.__new__(KnowledgeGraph)
    import networkx as nx
    from nltk.stem import WordNetLemmatizer

    kg.graph = nx.Graph()
    kg.lemmatizer = WordNetLemmatizer()
    kg.concept_cache = {}
    kg.content_to_node_id = {}
    kg.batch_size = 2
    kg.max_concurrent_calls = 10
    kg.edges_threshold = 0.8
    kg.embeddings_cache = {}
    kg._save_cache = lambda: None
    kg.cache_manager = SimpleNamespace(load_cache=lambda: {})
    kg._load_cache = lambda: None
    return kg


def test_rebuild_replaces_stale_graph():
    kg = _make_kg_for_unit_tests()
    first = [_make_split("doc-a"), _make_split("doc-b"), _make_split("doc-c")]
    second = [_make_split("doc-x"), _make_split("doc-y"), _make_split("doc-z")]
    kg._create_embeddings = lambda splits: [np.array([1.0, 0.0])] * len(splits)
    kg._extract_concepts_batch = lambda splits: [
        kg.graph.nodes[i].update({"concepts": ["c"]}) for i, _ in enumerate(splits)
    ]
    kg._add_edges = lambda embeddings: None

    kg.build_knowledge_graph(first)
    kg.build_knowledge_graph(second)

    contents = {data["content"] for _, data in kg.graph.nodes(data=True)}
    assert {"doc-x", "doc-y", "doc-z"} <= contents
    assert "doc-a" not in contents


def test_duplicate_splits_deduplicated():
    kg = _make_kg_for_unit_tests()
    splits = [_make_split("dup"), _make_split("dup"), _make_split("unique")]
    deduped = kg._add_nodes(splits)

    assert len(deduped) == 2
    assert kg.graph.number_of_nodes() == 2
    assert sorted(kg.content_to_node_id.keys()) == ["dup", "unique"]


def test_concepts_stored_lowercase():
    kg = _make_kg_for_unit_tests()
    splits = [_make_split("doc-one")]
    kg._add_nodes(splits)
    kg._ner_pipeline = lambda _batch: [[{"word": "Diabetes", "score": 0.9}]]

    kg._extract_concepts_batch(splits)

    assert kg.concept_cache["doc-one"] == ["diabetes"]


def test_concept_empty_node_gets_fallback_edge():
    kg = _make_kg_for_unit_tests()
    splits = [_make_split("one"), _make_split("two")]
    kg._add_nodes(splits)
    kg.graph.nodes[0]["concepts"] = []
    kg.graph.nodes[1]["concepts"] = ["topic"]

    embeddings = np.array([[1.0, 0.0], [0.95, 0.05]])
    kg._add_edges(embeddings)

    assert kg.graph.has_edge(0, 1)
    edge = kg.graph[0][1]
    assert edge["shared_concepts"] == []
    assert edge["weight"] < edge["similarity"]


def test_embeddings_none_raises_clearly():
    kg = _make_kg_for_unit_tests()
    splits = [_make_split("a"), _make_split("b")]
    kg.embeddings = SimpleNamespace(
        embed_documents=lambda texts: [np.array([0.1, 0.2])]
    )

    try:
        kg._create_embeddings(splits)
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        assert "Embedding computation failed for" in str(exc)
