from typing import Tuple
from dataclasses import dataclass
@dataclass
class VisualizationConfig:
    figure_size:Tuple[int, int] = (16, 12)
    node_size: int = 3000
    edge_width: int = 2
    traversal_edge_width: int = 3
    font_size: int = 8
    curve_radius: float = 0.3
    edge_offset: float = 0.1
    layout_iterations: int = 50
    layout_k: float = 1
    max_label_length: int = 20
    content_preview_length: int = 200

@dataclass
class NodeStyle:
    """Style configuration for different node types."""
    regular: str = 'lightblue'
    start: str = 'lightgreen'
    end: str = 'lightcoral'
    visited: str = 'gold'

@dataclass
class EdgeStyle:
    """Style configuration for different edge types."""
    regular_color: str = 'blue'
    traversal_color: str = 'red'
    traversal_style: str = '--'
    colormap: str = 'Blues'