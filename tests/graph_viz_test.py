import networkx as nx
from matplotlib import pyplot as plt

from medical_graph_rag.graph_viz import GraphVisualizer


def test_prepare_node_labels_covers_nodes_not_in_traversal():
    visualizer = GraphVisualizer()
    graph = nx.Graph()
    graph.add_node(1, concepts=["alpha"])
    graph.add_node(2, concepts=["beta"])
    labels = visualizer._prepare_node_labels(graph, traversal_path=[1])
    assert 1 in labels
    assert 2 in labels


def test_create_traversal_graph_returns_subgraph():
    visualizer = GraphVisualizer()
    graph = nx.Graph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    subgraph = visualizer._create_traversal_graph(graph, traversal_path=[1, 2])
    assert set(subgraph.nodes()) == {1, 2}


def test_draw_and_highlight_helpers_execute():
    visualizer = GraphVisualizer()
    graph = nx.Graph()
    graph.add_node(1, concepts=["a"])
    graph.add_node(2, concepts=["b"])
    graph.add_edge(1, 2, weight=0.7)
    fig, ax = plt.subplots()
    pos = {1: (0, 0), 2: (1, 0)}

    weights = visualizer._draw_base_graph(graph, pos, ax)
    visualizer._draw_traversal_path([1, 2], pos, ax)
    visualizer._highlight_special_nodes(graph, pos, [1, 2], ax)
    visualizer._add_visualization_elements(fig, ax, weights)
    plt.close(fig)


def test_visualize_traversal_returns_buffer():
    visualizer = GraphVisualizer()
    graph = nx.Graph()
    graph.add_node(1, concepts=["a"])
    graph.add_node(2, concepts=["b"])
    graph.add_edge(1, 2, weight=0.7)
    buf = visualizer.visualize_traversal(graph, [1, 2])
    assert buf is not None
