import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio
from config import VisualizationConfig, NodeStyle, EdgeStyle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphVisualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.node_style = NodeStyle()
        self.edge_style = EdgeStyle()

        logger.info("Initialized the GraphVisualizer")

    # === GRAPH TRAVERSAL ===
    def _create_traversal_graph(self, graph: nx.Graph):
        logger.info("Creating the traversal graph ...")
        traversal_graph = nx.DiGraph()
        traversal_graph.add_nodes_from(graph.nodes(data = True))
        traversal_graph.add_edges_from(graph.edges(data = True))
        logger.info(f"Created traversal graph with {len(traversal_graph.nodes)} nodes and {len(traversal_graph.edges)} edges")
        return traversal_graph

    def _generate_optimized_layout(self, graph:nx.Graph):
        logger.info("Generating optimized graph layout")
        return nx.spring_layout(
            graph, 
            k=self.config.layout_k, 
            iterations=self.config.layout_iterations,
            seed = 42
        )

    def _prepare_node_labels(self, graph:nx.Graph, traversal_path):
        logger.info("Preparing node labels ...")
        labels = {}

        # Add Traversal order nodes to the graph
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get(concepts, [])
            concept_text = concepts[0] if concepts else str(node)
            if len(concept_text) > self.config.max_label_length:
                concept_text = concept_text[:self.config.max_label_length] + "..."
            labels[node] = f"{i + 1}. {concept_text}"

        # Add labels for non traversal paths
        for node in graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                concept_text = concepts[0] if concepts else str(node)
                if len(concept_text) > self.config.max_label_length:
                    concept_text = concept_text[:self.config.max_label_length] + "..."
                labels[node] = concept_text
        
        return labels