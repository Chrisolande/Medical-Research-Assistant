import networkx as nx

from medical_graph_rag.graph import KnowledgeGraph


def test_get_subgraph_stats_scoped_to_traversal_path():
    kg = KnowledgeGraph.__new__(KnowledgeGraph)
    kg.graph = nx.Graph()
    kg.graph.add_edge(0, 1, weight=0.7)
    kg.graph.add_edge(1, 2, weight=0.8)
    kg.graph.add_edge(2, 3, weight=0.9)

    stats = kg.get_subgraph_stats([0, 1, 2])

    assert stats["nodes"] == 3
    assert stats["edges"] == 2
    assert stats["density"] == nx.density(kg.graph.subgraph([0, 1, 2]))
