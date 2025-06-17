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

    def _draw_base_graph(self, graph: nx.Graph, pos: Dict, ax):
        logger.info("Drawing basic graph ...")

        # Draw edges with weight based coloring
        edges = list(graph.edges())
        edge_weights = [graph[u][v].get('weight', 0.5) for u, v in edges]

        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edges,
            edge_color = edge_weights,
            edge_cmap=getattr(plt.cm, self.edge_style.colormap),
            width=self.config.edge_width,
            ax=ax,
            alpha=0.6
        )

        # Draw regular nodes
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=self.node_style.regular,
            node_size=self.config.node_size,
            ax=ax,
            alpha=0.8
        )

        return edge_weights

    def _draw_traversal_path(self, traversal_path, pos, ax):
        logger.info("Drawing traversal path ...")

        for i in range(len(traversal_path) - 1):
            start, end = traversal_path[i], traversal_path[i + 1]
            start_pos, end_pos = pos[start], pos[end]
            arrow = patches.FancyArrowPatch(
                start_pos, end_pos,
                connectionstyle=f"arc3,rad={self.config.curve_radius}",
                color=self.edge_style.traversal_color,
                arrowstyle="->",
                mutation_scale=20,
                linestyle=self.edge_style.traversal_style,
                linewidth=self.config.traversal_edge_width,
                zorder=4,
                alpha=0.8
            )
            ax.add_patch(arrow)

    def _highlight_special_nodes(self, graph: nx.DiGraph, pos: Dict, traversal_path: List[int], ax: plt.Axes) -> None:
        logger.info("Highlighting special nodes ...")
        if not traversal_path:
            return
        
        start_node, end_node = traversal_path[0], traversal_path[-1]
        # Remove the first and the end node if the traversal length is greater then 2
        visited_nodes = set(traversal_path[1:-1]) if len(traversal_path) > 2 else set() 

        # Draw visited nodes
        if visited_nodes:
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=list(visited_nodes),
                node_color=self.node_style.visited,
                node_size=self.config.node_size,
                ax=ax
            )
        
        # Draw start node
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=[start_node],
            node_color=self.node_style.start,
            node_size=self.config.node_size,
            ax=ax
        )
        
        # Draw end node
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=[end_node],
            node_color=self.node_style.end,
            node_size=self.config.node_size,
            ax=ax
        )

    def _add_visualization_elements(self, fig:plt.Figure, ax: plt.Axes,edge_weights):
        """Add colorbar, legend, and title to the visualization."""
        logger.info("Adding visualization elements")

        # Add colorbar for edge weights
        if edge_weights:
            sm = plt.cm.ScalarMappable(
                cmap=getattr(plt.cm, self.edge_style.colormap),
                norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Edge Weight', rotation=270, labelpad=15)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.edge_style.regular_color, linewidth=2, label='Regular Edge'),
            plt.Line2D([0], [0], color=self.edge_style.traversal_color, linewidth=2, 
                      linestyle=self.edge_style.traversal_style, label='Traversal Path'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.node_style.start, 
                      markersize=15, label='Start Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.node_style.visited, 
                      markersize=15, label='Visited Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.node_style.end, 
                      markersize=15, label='End Node')
        ]
        
        legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.9)
        
        ax.set_title("Enhanced Graph Traversal Visualization", fontsize=16, fontweight='bold')
        ax.axis('off')

    # === MAIN VISUALIZATION ===
    async def visualize_traversal_async(self, graph, traversal_path):
        logger.info(f"Starting async visualization for path of length {len(traversal_path)}")
        
        if not traversal_path:
            logger.warning("Empty traversal path provided")
            return

        try:
            # Create the traversal graph
            traversal_graph = self._create_traversal_graph(graph)

            # Generate layout
            pos = self._generate_optimized_layout(traversal_graph)

            # Prepare labels
            labels = self._prepare_node_labels(graph, traversal_path)
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.config.figure_size)

            # Draw base graph
            edge_weights = self._draw_base_graph(traversal_graph, pos, ax)
            
            # Draw traversal path
            self._draw_traversal_path(traversal_path, pos, ax)
            
            # Highlight special nodes
            self._highlight_special_nodes(traversal_graph, pos, traversal_path, ax)
            
            # Draw labels
            nx.draw_networkx_labels(traversal_graph, pos, labels, 
                                  font_size=self.config.font_size, 
                                  font_weight="bold", ax=ax)
            
            # Add visualization elements
            self._add_visualization_elements(fig, ax, edge_weights)
            
            plt.tight_layout()
            plt.show()
            
            logger.info("Visualization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            raise

    def visualize_traversal(self, graph, traversal_path):
        """Synchronous wrapper"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.visualize_traversal_async(graph, traversal_path))
        except Exception as e:
            logger.error(f"Error in synchronous visualization: {str(e)}")
            raise
        finally:
            loop.close()

    @staticmethod
    def print_filtered_content(traversal_path: List[int], filtered_content: Dict[int, str]) -> None:
        """
        Print the filtered content of visited nodes in traversal order.
        """
        logger.info(f"Printing filtered content for {len(traversal_path)} nodes")
        
        if not traversal_path:
            logger.warning("Empty traversal path provided")
            return
        
        print("\n" + "="*80)
        print("FILTERED CONTENT OF VISITED NODES (IN TRAVERSAL ORDER)")
        print("="*80)
        
        for i, node in enumerate(traversal_path):
            content = filtered_content.get(node, 'No filtered content available')
            preview = content[:self.config.content_preview_length]
            if len(content) > self.config.content_preview_length:
                preview += "..."
            
            print(f"\n Step {i + 1} - Node {node}")
            print("-" * 50)
            print(f"Content Preview: {preview}")
            print("-" * 50)
        
        print(f"\n Completed traversal of {len(traversal_path)} nodes")
        logger.info("Content printing completed")

