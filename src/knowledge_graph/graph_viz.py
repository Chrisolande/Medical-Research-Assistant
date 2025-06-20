import asyncio
import logging
from typing import Any, Dict, List, Optional

import matplotlib.patches as patches
import networkx as nx
from matplotlib import pyplot as plt

from src.core.config import EdgeStyle, NodeStyle, VisualizationConfig
from src.core.utils import print_filtered_content

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """GraphVisualizer class."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize init."""
        self.config = config or VisualizationConfig()
        self.node_style = NodeStyle()
        self.edge_style = EdgeStyle()

        logger.info("Initialized the GraphVisualizer")

    # === GRAPH TRAVERSAL ===
    def _create_traversal_graph(
        self, graph: nx.Graph, traversal_path: List[int]
    ) -> nx.Graph:
        """Create traveral graph."""
        logger.info("Creating the traversal subgraph with neighbors...")

        nodes_to_include = set(traversal_path)
        for node in traversal_path:
            nodes_to_include.update(graph.neighbors(node))

        traversal_graph = graph.subgraph(nodes_to_include).copy()

        logger.info(
            f"Created traversal subgraph with {len(traversal_graph.nodes)} nodes and {len(traversal_graph.edges)} edges"
        )
        return traversal_graph

    def _generate_optimized_layout(self, graph: nx.Graph) -> Dict[Any, List[float]]:
        """Generate Optimized Layout method."""
        logger.info("Generating optimized graph layout")
        return nx.spring_layout(
            graph,
            k=self.config.layout_k,
            iterations=self.config.layout_iterations,
            seed=42,
        )

    def _prepare_node_labels(
        self, graph: nx.Graph, traversal_path: List[int]
    ) -> Dict[Any, str]:
        """Prepare node labels."""
        logger.info("Preparing node labels ...")
        labels = {}

        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get("concepts", [])
            concept_text = concepts[0] if concepts else str(node)
            if len(concept_text) > self.config.max_label_length:
                concept_text = concept_text[: self.config.max_label_length] + "..."
            labels[node] = f"{i + 1}. {concept_text}"

        for _node in graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get("concepts", [])
                concept_text = concepts[0] if concepts else str(node)
                if len(concept_text) > self.config.max_label_length:
                    concept_text = concept_text[: self.config.max_label_length] + "..."
                labels[node] = concept_text

        return labels

    def _draw_base_graph(self, graph: nx.Graph, pos: Dict, ax: plt.Axes) -> List[float]:
        """Draw Base Graph method."""
        logger.info("Drawing basic graph ...")

        edges = list(graph.edges())
        edge_weights = [graph[u][v].get("weight", 0.5) for u, v in edges]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges,
            edge_color=edge_weights,
            edge_cmap=getattr(plt.cm, self.edge_style.colormap),
            width=self.config.edge_width,
            ax=ax,
            alpha=0.6,
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=self.node_style.regular,
            node_size=self.config.node_size,
            ax=ax,
            alpha=0.8,
        )

        return edge_weights

    def _draw_traversal_path(
        self, traversal_path: List[int], pos: Dict, ax: plt.Axes
    ) -> None:
        """Draw the traversal path on the graph visualization.

        This method adds arrows between consecutive nodes in the traversal path to visually
        represent the path taken during the traversal. The arrows are drawn with specified
        style and color configurations.

        Args:
            traversal_path (List[int]): The list of node IDs representing the traversal path.
            pos (Dict): A dictionary mapping node IDs to their positions in the graph layout.
            ax (plt.Axes): The matplotlib axes on which to draw the traversal path.
        """
        logger.info("Drawing traversal path ...")

        for i in range(len(traversal_path) - 1):
            start, end = traversal_path[i], traversal_path[i + 1]
            start_pos, end_pos = pos[start], pos[end]
            arrow = patches.FancyArrowPatch(
                start_pos,
                end_pos,
                connectionstyle=f"arc3,rad={self.config.curve_radius}",
                color=self.edge_style.traversal_color,
                arrowstyle="->",
                mutation_scale=20,
                linestyle=self.edge_style.traversal_style,
                linewidth=self.config.traversal_edge_width,
                zorder=4,
                alpha=0.8,
            )
            ax.add_patch(arrow)

    def _highlight_special_nodes(
        self,
        graph: nx.Graph,
        pos,
        traversal_path: List[int],
        ax: plt.Axes,
    ) -> None:
        """Highlight special nodes in the graph visualization.

        This method highlights the start, end, and visited nodes along the traversal path
        using different colors and styles as specified in the node style configuration.

        Args:
            graph (nx.Graph): The graph containing the nodes.
            pos (Dict[int, Tuple[float, float]]): The layout positions of the nodes.
            traversal_path (List[int]): The path of nodes to highlight.
            ax (plt.Axes): The matplotlib axes to draw on.
        """
        logger.info("Highlighting special nodes ...")
        if not traversal_path:
            return

        start_node, end_node = traversal_path[0], traversal_path[-1]
        visited_nodes = set(traversal_path[1:-1]) if len(traversal_path) > 2 else set()

        if visited_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=list(visited_nodes),
                node_color=self.node_style.visited,
                node_size=self.config.node_size,
                ax=ax,
            )

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[start_node],
            node_color=self.node_style.start,
            node_size=self.config.node_size,
            ax=ax,
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[end_node],
            node_color=self.node_style.end,
            node_size=self.config.node_size,
            ax=ax,
        )

    def _add_visualization_elements(
        self, fig: plt.Figure, ax: plt.Axes, edge_weights: List[float]
    ) -> None:
        """Add colorbar, legend, and title to the visualization."""
        logger.info("Adding visualization elements")

        if edge_weights:
            sm = plt.cm.ScalarMappable(
                cmap=getattr(plt.cm, self.edge_style.colormap),
                norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)),
            )
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04
            )
            cbar.set_label("Edge Weight", rotation=270, labelpad=15)

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color=self.edge_style.regular_color,
                linewidth=2,
                label="Regular Edge",
            ),
            plt.Line2D(
                [0],
                [0],
                color=self.edge_style.traversal_color,
                linewidth=2,
                linestyle=self.edge_style.traversal_style,
                label="Traversal Path",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.node_style.start,
                markersize=15,
                label="Start Node",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.node_style.visited,
                markersize=15,
                label="Visited Node",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.node_style.end,
                markersize=15,
                label="End Node",
            ),
        ]

        legend = ax.legend(
            handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1), ncol=2
        )
        legend.get_frame().set_alpha(0.9)

        ax.set_title("Graph Traversal Visualization", fontsize=16, fontweight="bold")
        ax.axis("off")

    # === MAIN VISUALIZATION ===
    async def visualize_traversal_async(
        self, graph: nx.Graph, traversal_path: List[int]
    ) -> None:
        """Asynchronously visualize the traversal of a graph.

        Args:
            graph (nx.Graph): The graph to visualize.
            traversal_path (List[int]): The path of nodes to visualize in traversal order.

        This method creates a subgraph from the traversal path, generates an optimized layout,
        and visualizes the graph with node labels, edge weights, and special node highlights.
        Visualization is displayed using matplotlib.
        """
        logger.info(
            f"Starting async visualization for path of length {len(traversal_path)}"
        )

        if not traversal_path:
            logger.warning("Empty traversal path provided")
            return

        try:
            traversal_graph = self._create_traversal_graph(graph, traversal_path)
            pos = self._generate_optimized_layout(traversal_graph)
            labels = self._prepare_node_labels(traversal_graph, traversal_path)

            fig, ax = plt.subplots(figsize=self.config.figure_size)

            edge_weights = self._draw_base_graph(traversal_graph, pos, ax)
            self._draw_traversal_path(traversal_path, pos, ax)
            self._highlight_special_nodes(traversal_graph, pos, traversal_path, ax)

            nx.draw_networkx_labels(
                traversal_graph,
                pos,
                labels,
                font_size=self.config.font_size,
                font_weight="bold",
                ax=ax,
            )

            self._add_visualization_elements(fig, ax, edge_weights)

            plt.tight_layout()
            plt.show()

            logger.info("Visualization completed successfully")

        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}", exc_info=True)
            raise

    def visualize_traversal(self, graph, traversal_path):
        """Synchronous wrapper to run the async visualization."""
        try:
            # loop = asyncio.get_event_loop()
            asyncio.create_task(self.visualize_traversal_async(graph, traversal_path))

        except Exception as e:
            logger.error(f"Error in synchronous visualization: {str(e)}")
            raise

    def print_filtered_content(
        self, traversal_path: List[int], filtered_content: Dict[int, str]
    ) -> None:
        """Print the filtered content of visited nodes in traversal order."""
        print_filtered_content(
            traversal_path=traversal_path,
            filtered_content=filtered_content,
            content_preview_length=self.config.content_preview_length,
        )
