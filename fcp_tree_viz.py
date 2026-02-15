"""
FCP Tree Visualization Module

Visualizes GP trees with evaluation context, showing:
- Active vs inactive branches
- Intermediate values at each node
- Decision branch selection
- Terminal classification (feature vs constant)
- Decision chain ordering

Requires: networkx, matplotlib (optional for rendering)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
from xml.sax.saxutils import escape as xml_escape

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from deap import gp


class NodeType(Enum):
    """Classification of tree nodes."""

    PRIMITIVE = "primitive"
    TERMINAL_FEATURE = "feature"
    TERMINAL_CONSTANT = "constant"
    CONDITIONAL = "conditional"


@dataclass
class NodeInfo:
    """Information about a single node in the evaluated tree."""

    node_id: int
    label: str  # Original label (ADD, x, etc.)
    node_type: NodeType

    # Evaluation context
    output_value: Optional[float] = None
    is_active: bool = True  # Was this node on the executed path?

    # Contribution to final output (signed, normalized)
    contribution: Optional[float] = None  # Value in [-1, 1] range

    # For conditionals
    condition_result: Optional[bool] = None
    decision_index: Optional[int] = None  # Position in decision chain (1-indexed)
    is_critical: bool = False

    # For display
    display_label: str = ""

    def __post_init__(self):
        if not self.display_label:
            self.display_label = self.label


@dataclass
class TreeVisualization:
    """
    Complete visualization data for an evaluated GP tree.
    """

    nodes: Dict[int, NodeInfo]
    edges: List[Tuple[int, int]]
    root_id: int

    # Metadata
    tree_expression: str
    output_value: float
    num_decisions: int

    def to_networkx(self) -> "nx.DiGraph":
        """Convert to NetworkX DiGraph with node attributes."""
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for graph visualization")

        G = nx.DiGraph()

        # Add nodes with attributes
        for node_id, info in self.nodes.items():
            G.add_node(
                node_id,
                label=info.label,
                display_label=info.display_label,
                node_type=info.node_type.value,
                output_value=info.output_value,
                is_active=info.is_active,
                condition_result=info.condition_result,
                decision_index=info.decision_index,
                is_critical=info.is_critical,
            )

        # Add edges
        for parent, child in self.edges:
            # Determine if this edge is on active path
            parent_active = self.nodes[parent].is_active
            child_active = self.nodes[child].is_active
            is_active_edge = parent_active and child_active

            G.add_edge(parent, child, is_active=is_active_edge)

        return G

    def get_node_colors(self) -> Dict[int, str]:
        """Get color mapping for nodes."""
        colors = {}
        for node_id, info in self.nodes.items():
            if not info.is_active:
                colors[node_id] = "#d1d5db"  # Gray for inactive
            elif info.node_type == NodeType.CONDITIONAL:
                if info.is_critical:
                    colors[node_id] = "#f97316"  # Orange for critical decision
                else:
                    colors[node_id] = "#8b5cf6"  # Purple for decisions
            elif info.node_type == NodeType.TERMINAL_FEATURE:
                colors[node_id] = "#3b82f6"  # Blue for features
            elif info.node_type == NodeType.TERMINAL_CONSTANT:
                colors[node_id] = "#6b7280"  # Dark gray for constants
            else:
                colors[node_id] = "#22c55e"  # Green for active primitives
        return colors

    def get_edge_colors(self) -> Dict[Tuple[int, int], str]:
        """Get color mapping for edges."""
        colors = {}
        for parent, child in self.edges:
            parent_active = self.nodes[parent].is_active
            child_active = self.nodes[child].is_active
            if parent_active and child_active:
                colors[(parent, child)] = "#1f2937"  # Dark for active path
            else:
                colors[(parent, child)] = "#d1d5db"  # Light gray for inactive
        return colors

    def to_dot(self, show_comparison_links: bool = True) -> str:
        """Generate DOT format for Graphviz rendering."""
        lines = ["digraph GPTree {"]
        lines.append("  rankdir=TB;")
        lines.append('  node [shape=box, style=filled, fontname="Helvetica"];')
        lines.append('  edge [fontname="Helvetica"];')
        lines.append("")

        colors = self.get_node_colors()

        # Build children map for finding obs/threshold pairs
        children_map = {}
        for parent, child in self.edges:
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(child)

        # Add nodes
        for node_id, info in self.nodes.items():
            color = colors[node_id]
            fontcolor = (
                "white"
                if color in ["#1f2937", "#8b5cf6", "#3b82f6", "#6b7280"]
                else "black"
            )

            # Build label
            label_parts = [info.display_label]
            if info.output_value is not None:
                label_parts.append(f"= {info.output_value:.4g}")
            if info.decision_index is not None:
                label_parts.append(f"[D{info.decision_index}]")

            label = "\\n".join(label_parts)

            # Border for critical decisions
            penwidth = "3" if info.is_critical else "1"

            lines.append(
                f'  {node_id} [label="{label}", fillcolor="{color}", '
                f'fontcolor="{fontcolor}", penwidth={penwidth}];'
            )

        lines.append("")

        # Add edges
        edge_colors = self.get_edge_colors()
        for parent, child in self.edges:
            color = edge_colors[(parent, child)]
            penwidth = "2" if color == "#1f2937" else "1"
            style = "solid" if color == "#1f2937" else "dashed"

            # Add branch label for conditionals
            parent_info = self.nodes[parent]
            edge_label = ""
            if parent_info.node_type == NodeType.CONDITIONAL:
                # Determine if this is the TRUE or FALSE branch
                children = [c for p, c in self.edges if p == parent]
                if len(children) >= 2:
                    # DEAP order: condition args first, then true_branch, then false_branch
                    # But we need to check the actual structure
                    child_idx = children.index(child)
                    # For IFLTE: args are (obs, threshold, true_val, false_val)
                    # children[0] = obs, children[1] = threshold, children[2] = true, children[3] = false
                    if child_idx == 2:
                        edge_label = "T"
                    elif child_idx == 3:
                        edge_label = "F"

            lines.append(
                f'  {parent} -> {child} [color="{color}", penwidth={penwidth}, '
                f'style={style}, label="{edge_label}"];'
            )

        # Add comparison links between observation and threshold nodes
        if show_comparison_links:
            lines.append("")
            lines.append("  // Comparison links (horizontal)")

            for node_id, info in self.nodes.items():
                if info.node_type == NodeType.CONDITIONAL:
                    children = children_map.get(node_id, [])
                    if len(children) >= 2:
                        obs_node = children[0]
                        thresh_node = children[1]

                        # Determine comparison operator
                        label = info.label.upper()

                        obs_val = self.nodes[obs_node].output_value
                        thresh_val = self.nodes[thresh_node].output_value

                        if label == "CMP":
                            if obs_val is not None and thresh_val is not None:
                                if obs_val < thresh_val:
                                    op_symbol = "< (−1)"
                                elif obs_val > thresh_val:
                                    op_symbol = "> (+1)"
                                else:
                                    op_symbol = "= (0)"
                            else:
                                op_symbol = "⋚"
                        elif label == "MIN":
                            if obs_val is not None and thresh_val is not None:
                                op_symbol = "≤" if obs_val <= thresh_val else ">"
                            else:
                                op_symbol = "⋜"
                        elif label == "MAX":
                            if obs_val is not None and thresh_val is not None:
                                op_symbol = "≥" if obs_val >= thresh_val else "<"
                            else:
                                op_symbol = "⋛"
                        elif "LTE" in label or label == "IFLTE":
                            op_symbol = "≤"
                        elif "LT" in label or label == "IFLT":
                            op_symbol = "<"
                        elif "GTE" in label:
                            op_symbol = "≥"
                        elif "GT" in label:
                            op_symbol = ">"
                        else:
                            op_symbol = "?"

                        # Determine if active
                        is_active = (
                            self.nodes[obs_node].is_active
                            and self.nodes[thresh_node].is_active
                        )
                        if label in ("MIN", "MAX"):
                            is_active = self.nodes[node_id].is_active
                        edge_color = "#8b5cf6" if is_active else "#d1d5db"

                        # Force same rank for obs and threshold
                        lines.append(f"  {{rank=same; {obs_node}; {thresh_node}}}")

                        # Add horizontal edge with constraint=false so it doesn't affect layout
                        lines.append(
                            f'  {obs_node} -> {thresh_node} [label="{op_symbol}", '
                            f'color="{edge_color}", style="dashed", '
                            f'constraint=false, dir=none, fontcolor="{edge_color}", fontsize=14];'
                        )

        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self) -> str:
        """Generate ASCII art representation of the tree."""
        lines = []

        def render_node(node_id: int, prefix: str = "", is_last: bool = True) -> None:
            info = self.nodes[node_id]

            # Connection character
            conn = "└── " if is_last else "├── "

            # Build node string
            parts = []

            # Type indicator
            if info.node_type == NodeType.TERMINAL_FEATURE:
                type_ind = "📊"
            elif info.node_type == NodeType.TERMINAL_CONSTANT:
                type_ind = "🔢"
            elif info.node_type == NodeType.CONDITIONAL:
                type_ind = "❓"
            else:
                type_ind = "⚙️"

            parts.append(f"{type_ind} {info.label}")

            if info.output_value is not None:
                parts.append(f"={info.output_value:.4g}")

            if info.decision_index is not None:
                parts.append(f"[D{info.decision_index}]")

            if info.is_critical:
                parts.append("★")

            if not info.is_active:
                parts.append("[inactive]")

            node_str = " ".join(parts)
            lines.append(f"{prefix}{conn}{node_str}")

            # Find children
            children = [child for parent, child in self.edges if parent == node_id]

            # Recurse
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_id in enumerate(children):
                render_node(child_id, child_prefix, i == len(children) - 1)

        lines.append(f"Tree: {self.tree_expression}")
        lines.append(
            f"Output: {self.output_value:.4g} | Decisions: {self.num_decisions}"
        )
        lines.append("")
        render_node(self.root_id, "", True)

        lines.append("")
        lines.append(
            "Legend: 📊=feature  🔢=constant  ❓=decision  ⚙️=operation  ★=critical"
        )

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter."""
        # Build hierarchical HTML representation
        colors = self.get_node_colors()

        def render_node_html(node_id: int, depth: int = 0) -> str:
            info = self.nodes[node_id]
            color = colors[node_id]

            # Determine text color based on background
            text_color = (
                "white"
                if color in ["#3b82f6", "#6b7280", "#8b5cf6", "#22c55e"]
                else "#374151"
            )

            # Build node badge
            badge_parts = [info.label]
            if info.output_value is not None:
                badge_parts.append(f"={info.output_value:.4g}")

            badge_text = " ".join(badge_parts)

            # Additional markers
            markers = ""
            if info.decision_index:
                markers += f'<span style="background:#1f2937;color:white;padding:1px 4px;border-radius:3px;font-size:10px;margin-left:4px;">D{info.decision_index}</span>'
            if info.is_critical:
                markers += '<span style="margin-left:4px;">⭐</span>'
            if not info.is_active:
                markers += '<span style="color:#9ca3af;margin-left:4px;font-style:italic;">(inactive)</span>'

            # Type icon
            type_icons = {
                NodeType.TERMINAL_FEATURE: "📊",
                NodeType.TERMINAL_CONSTANT: "🔢",
                NodeType.CONDITIONAL: "❓",
                NodeType.PRIMITIVE: "⚙️",
            }
            icon = type_icons.get(info.node_type, "•")

            html = f"""
            <div style="margin-left:{depth*20}px; margin-top:4px;">
                <span style="color:#9ca3af;">{"├── " if depth > 0 else ""}</span>
                <span style="margin-right:4px;">{icon}</span>
                <span style="background:{color}; color:{text_color}; padding:2px 8px; 
                             border-radius:4px; font-family:monospace; font-size:12px;
                             {"border:2px solid #f97316;" if info.is_critical else ""}">{badge_text}</span>
                {markers}
            </div>
            """

            # Render children
            children = [c for p, c in self.edges if p == node_id]
            for child_id in children:
                html += render_node_html(child_id, depth + 1)

            return html

        tree_html = render_node_html(self.root_id)

        return f"""
        <div style="font-family: -apple-system, sans-serif; padding: 15px; 
                    border: 1px solid #d1d5db; border-radius: 8px; background: #ffffff;">
            <div style="margin-bottom: 10px;">
                <b>Tree:</b> <code style="font-size:12px;">{self.tree_expression}</code>
            </div>
            <div style="margin-bottom: 15px;">
                <b>Output:</b> <span style="font-size:18px;font-weight:bold;">{self.output_value:.4g}</span>
                <span style="color:#6b7280;margin-left:15px;">Decisions: {self.num_decisions}</span>
            </div>
            <div style="margin-bottom: 10px; font-size: 11px; color: #6b7280;">
                <span style="margin-right:10px;">📊 feature</span>
                <span style="margin-right:10px;">🔢 constant</span>
                <span style="margin-right:10px;">❓ decision</span>
                <span style="margin-right:10px;">⚙️ operation</span>
                <span>⭐ critical</span>
            </div>
            <div style="background: #f9fafb; padding: 12px; border-radius: 6px; 
                        font-size: 13px; line-height: 1.6; overflow-x: auto;">
                {tree_html}
            </div>
        </div>
        """

    def to_svg(
        self,
        node_height: int = 50,
        min_node_width: int = 50,
        char_width: int = 8,
        level_height: int = 100,
        min_node_spacing: int = 60,
        comparison_spacing: int = 150,
        show_values: bool = True,
        show_decision_labels: bool = True,
        show_comparison_links: bool = True,
        font_size: int = 11,
        padding: int = 12,
    ) -> str:
        """
        Generate an SVG tree diagram.

        Args:
            node_height: Height of node shapes
            min_node_width: Minimum width of node shapes
            char_width: Approximate width per character for sizing
            level_height: Vertical spacing between levels
            min_node_spacing: Minimum horizontal spacing between nodes
            comparison_spacing: Minimum spacing between observation and threshold nodes
                               (to leave room for the comparison operator link)
            show_values: Show computed values below nodes
            show_decision_labels: Show T/F labels on conditional branches
            show_comparison_links: Show horizontal links between observation and threshold
                                   nodes with comparison operator (<=, <)
            font_size: Font size for node labels
            padding: Horizontal padding inside nodes

        Returns:
            SVG string that can be embedded in HTML or saved to file
        """
        # Calculate node dimensions based on label length
        node_dims = {}  # node_id -> (width, height)
        half_height = node_height // 2

        for node_id, info in self.nodes.items():
            label = info.label
            # Calculate width needed for label
            label_width = len(label) * char_width + padding * 2
            width = max(min_node_width, label_width)

            # Diamonds need to be square-ish (width = height for proper diamond)
            if info.node_type == NodeType.CONDITIONAL:
                # For diamonds, use the larger dimension for both
                size = max(width, node_height)
                node_dims[node_id] = (size, size)
            else:
                node_dims[node_id] = (width, node_height)

        # Build adjacency for tree traversal
        children_map = {}
        for parent, child in self.edges:
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(child)

        # Identify observation/threshold node pairs that need extra spacing
        comparison_pairs = set()  # Set of (obs_node, thresh_node) tuples
        if show_comparison_links:
            for node_id, info in self.nodes.items():
                if info.node_type == NodeType.CONDITIONAL:
                    children = children_map.get(node_id, [])
                    if len(children) >= 2:
                        comparison_pairs.add((children[0], children[1]))

        # Calculate positions using a simple tree layout algorithm
        positions = {}  # node_id -> (x, y)

        def get_subtree_width(
            node_id: int,
            parent_id: Optional[int] = None,
            sibling_id: Optional[int] = None,
        ) -> float:
            """Calculate width needed for subtree rooted at node."""
            children = children_map.get(node_id, [])
            node_width = node_dims[node_id][0]

            # Base width for leaf nodes
            if not children:
                base_width = max(min_node_spacing, node_width + 20)
                # Check if this node is part of a comparison pair with its sibling
                if sibling_id is not None:
                    if (node_id, sibling_id) in comparison_pairs or (
                        sibling_id,
                        node_id,
                    ) in comparison_pairs:
                        base_width = max(base_width, comparison_spacing // 2)
                return base_width

            # For internal nodes, sum children widths
            total = 0
            for i, child in enumerate(children):
                # Pass sibling info for comparison pair detection
                sibling = children[i + 1] if i + 1 < len(children) else None
                total += get_subtree_width(child, node_id, sibling)

            return max(total, node_width + 20, min_node_spacing)

        def layout_node(node_id: int, x: float, y: int, available_width: float) -> None:
            """Recursively position nodes."""
            positions[node_id] = (x + available_width / 2, y)

            children = children_map.get(node_id, [])
            if not children:
                return

            # Check if first two children are a comparison pair
            is_conditional = self.nodes[node_id].node_type == NodeType.CONDITIONAL

            # Calculate child widths
            child_widths = []
            for i, child in enumerate(children):
                sibling = children[i + 1] if i + 1 < len(children) else None
                child_widths.append(get_subtree_width(child, node_id, sibling))

            # For conditionals, ensure obs/thresh have enough spacing
            if is_conditional and len(children) >= 2:
                # Calculate minimum width needed for first two children
                min_first_two = comparison_spacing
                current_first_two = child_widths[0] + child_widths[1]

                if current_first_two < min_first_two:
                    # Scale up first two children proportionally
                    scale = min_first_two / current_first_two
                    child_widths[0] = child_widths[0] * scale
                    child_widths[1] = child_widths[1] * scale

            total_width = sum(child_widths)

            # Scale if needed
            if total_width > available_width:
                scale = available_width / total_width
                child_widths = [w * scale for w in child_widths]
            else:
                # Center children
                padding_x = (available_width - total_width) / 2
                x += padding_x

            current_x = x
            for child, width in zip(children, child_widths):
                layout_node(child, current_x, y + level_height, width)
                current_x += width

        # Calculate total width needed
        total_width = get_subtree_width(self.root_id)
        total_width = max(total_width, 400)  # Minimum width

        # Layout the tree
        layout_node(self.root_id, 0, half_height + 30, total_width)

        # Calculate SVG dimensions
        max_y = max(pos[1] for pos in positions.values()) + half_height + 50
        svg_width = total_width + 40
        svg_height = max_y + 20

        # Helper function to draw a diamond shape
        def diamond_path(cx: float, cy: float, w: int, h: int) -> str:
            """Generate SVG path for a diamond centered at (cx, cy)."""
            hw, hh = w // 2, h // 2
            return (
                f"M {cx} {cy - hh} L {cx + hw} {cy} L {cx} {cy + hh} L {cx - hw} {cy} Z"
            )

        # Helper function to draw a rounded rectangle
        def rounded_rect(cx: float, cy: float, w: int, h: int, r: int = 8) -> str:
            """Generate SVG rect attributes for a rounded rectangle centered at (cx, cy)."""
            x = cx - w / 2
            y = cy - h / 2
            return f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}"'

        # Start SVG
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" ',
            f'width="{svg_width}" height="{svg_height}" style="font-family: -apple-system, Arial, sans-serif;">',
            "",
            "  <!-- Background -->",
            f'  <rect width="100%" height="100%" fill="white"/>',
            "",
            "  <!-- Edges -->",
        ]

        # Collect edge labels to draw later (after lines but before nodes)
        edge_labels = []

        # Calculate max contribution for scaling edge thickness
        max_contribution = (
            max(
                abs(info.contribution)
                for info in self.nodes.values()
                if info.contribution is not None
            )
            if any(info.contribution is not None for info in self.nodes.values())
            else 1.0
        )

        if max_contribution == 0:
            max_contribution = 1.0

        # Draw edges first (so nodes appear on top)
        for parent, child in self.edges:
            px, py = positions[parent]
            cx, cy = positions[child]

            parent_info = self.nodes[parent]
            child_info = self.nodes[child]
            parent_w, parent_h = node_dims[parent]
            child_w, child_h = node_dims[child]

            # Determine if edge is active
            is_active = parent_info.is_active and child_info.is_active

            # Get child's contribution for coloring
            child_contribution = (
                child_info.contribution if child_info.contribution is not None else 0.0
            )

            if not is_active:
                # Inactive edges: gray, thin, dashed
                stroke_color = "#d1d5db"
                stroke_width = 1
                stroke_dash = 'stroke-dasharray="4,4"'
            else:
                # Active edges: color and thickness based on contribution
                contrib_magnitude = abs(child_contribution)

                # Thickness: 1.5 to 6 based on contribution magnitude
                stroke_width = 1.5 + (contrib_magnitude * 4.5)
                stroke_dash = ""

                # Color: green for positive, red for negative, gray for near-zero
                if abs(child_contribution) < 0.05:
                    # Near-zero contribution: neutral gray
                    stroke_color = "#9ca3af"
                elif child_contribution > 0:
                    # Positive contribution: green (intensity based on magnitude)
                    # Interpolate from light green to dark green
                    intensity = min(1.0, contrib_magnitude)
                    # #86efac (light) to #16a34a (dark)
                    r = int(134 - intensity * 112)
                    g = int(239 - intensity * 76)
                    b = int(172 - intensity * 98)
                    stroke_color = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    # Negative contribution: red (intensity based on magnitude)
                    intensity = min(1.0, contrib_magnitude)
                    # #fca5a5 (light) to #dc2626 (dark)
                    r = int(252 - intensity * 32)
                    g = int(165 - intensity * 127)
                    b = int(165 - intensity * 127)
                    stroke_color = f"#{r:02x}{g:02x}{b:02x}"

            # Calculate connection points
            start_y = py + parent_h // 2
            end_y = cy - child_h // 2

            # Draw line from bottom of parent to top of child
            svg_parts.append(
                f'  <line x1="{px}" y1="{start_y}" '
                f'x2="{cx}" y2="{end_y}" '
                f'stroke="{stroke_color}" stroke-width="{stroke_width:.1f}" {stroke_dash}/>'
            )

            # Collect T/F labels for conditional branches (to draw with background later)
            if show_decision_labels and parent_info.node_type == NodeType.CONDITIONAL:
                children = children_map.get(parent, [])
                parent_label = parent_info.label.upper()

                if len(children) >= 4:  # IFLTE has 4 children
                    child_idx = children.index(child)
                    if child_idx == 2:  # True branch
                        mid_x = (px + cx) / 2
                        mid_y = (py + cy) / 2
                        edge_labels.append((mid_x, mid_y, "T", "#22c55e"))
                    elif child_idx == 3:  # False branch
                        mid_x = (px + cx) / 2
                        mid_y = (py + cy) / 2
                        edge_labels.append((mid_x, mid_y, "F", "#ef4444"))
                        mid_x = (px + cx) / 2
                        mid_y = (py + cy) / 2
                        edge_labels.append((mid_x, mid_y, "F", "#ef4444"))

        # Draw edge labels with white circle backgrounds
        if edge_labels:
            svg_parts.append("")
            svg_parts.append("  <!-- Edge Labels with Backgrounds -->")
            for lx, ly, label_text, label_color in edge_labels:
                # White circle background
                svg_parts.append(
                    f'  <circle cx="{lx}" cy="{ly}" r="10" fill="white" stroke="{label_color}" stroke-width="1.5"/>'
                )
                # Label text
                svg_parts.append(
                    f'  <text x="{lx}" y="{ly + 4}" text-anchor="middle" '
                    f'fill="{label_color}" font-size="11" font-weight="bold">{label_text}</text>'
                )

        # Draw comparison links between observation and threshold nodes
        if show_comparison_links:
            svg_parts.append("")
            svg_parts.append("  <!-- Comparison Links -->")

            for node_id, info in self.nodes.items():
                if info.node_type == NodeType.CONDITIONAL:
                    children = children_map.get(node_id, [])
                    label_upper = info.label.upper()

                    # Determine comparison operator symbol based on operation type
                    # and whether this operation compares two children or one child to zero

                    # Two-operand comparisons (children[0] vs children[1])
                    if label_upper in ("CMP", "MAX", "MIN") and len(children) >= 2:
                        obs_node = children[0]
                        thresh_node = children[1]

                        # Get actual values to determine comparison result
                        obs_val = self.nodes[obs_node].output_value
                        thresh_val = self.nodes[thresh_node].output_value

                        if label_upper == "CMP":
                            # Show the evaluated comparison: <, =, or >
                            # with the output value (-1, 0, 1) above
                            if obs_val is not None and thresh_val is not None:
                                if obs_val < thresh_val:
                                    op_symbol = "<"
                                    result_label = "−1"
                                elif obs_val > thresh_val:
                                    op_symbol = ">"
                                    result_label = "+1"
                                else:
                                    op_symbol = "="
                                    result_label = "0"
                            else:
                                op_symbol = "⋚"
                                result_label = None
                        elif label_upper == "MIN":
                            # Show which side was selected
                            if obs_val is not None and thresh_val is not None:
                                if obs_val <= thresh_val:
                                    op_symbol = "≤"  # left selected
                                else:
                                    op_symbol = ">"  # right selected
                            else:
                                op_symbol = "⋜"
                            result_label = None
                        elif label_upper == "MAX":
                            if obs_val is not None and thresh_val is not None:
                                if obs_val >= thresh_val:
                                    op_symbol = "≥"  # left selected
                                else:
                                    op_symbol = "<"  # right selected
                            else:
                                op_symbol = "⋛"
                            result_label = None
                        else:
                            op_symbol = "?"
                            result_label = None

                        obs_x, obs_y = positions[obs_node]
                        thresh_x, thresh_y = positions[thresh_node]
                        obs_w, _ = node_dims[obs_node]
                        thresh_w, _ = node_dims[thresh_node]

                        # Only draw if nodes are at same level
                        if abs(obs_y - thresh_y) < 5:
                            line_y = obs_y
                            line_x1 = obs_x + obs_w / 2 + 3
                            line_x2 = thresh_x - thresh_w / 2 - 3
                            mid_x = (line_x1 + line_x2) / 2

                            is_active = (
                                self.nodes[obs_node].is_active
                                and self.nodes[thresh_node].is_active
                            )
                            # For MIN/MAX, at least the parent node should be active
                            if label_upper in ("MIN", "MAX"):
                                is_active = self.nodes[node_id].is_active

                            stroke_color = "#8b5cf6" if is_active else "#d1d5db"

                            gap = 15
                            if line_x2 - line_x1 > gap * 2:
                                svg_parts.append(
                                    f'  <line x1="{line_x1}" y1="{line_y}" '
                                    f'x2="{mid_x - gap}" y2="{line_y}" '
                                    f'stroke="{stroke_color}" stroke-width="2" stroke-dasharray="3,3"/>'
                                )
                                svg_parts.append(
                                    f'  <line x1="{mid_x + gap}" y1="{line_y}" '
                                    f'x2="{line_x2}" y2="{line_y}" '
                                    f'stroke="{stroke_color}" stroke-width="2" stroke-dasharray="3,3"/>'
                                )

                            svg_parts.append(
                                f'  <circle cx="{mid_x}" cy="{line_y}" r="12" '
                                f'fill="white" stroke="{stroke_color}" stroke-width="2"/>'
                            )
                            svg_parts.append(
                                f'  <text x="{mid_x}" y="{line_y + 5}" text-anchor="middle" '
                                f'fill="{stroke_color}" font-size="14" font-weight="bold">{xml_escape(op_symbol)}</text>'
                            )

                            # For CMP: draw output value label above the circle
                            if label_upper == "CMP" and result_label is not None:
                                svg_parts.append(
                                    f'  <text x="{mid_x}" y="{line_y - 16}" text-anchor="middle" '
                                    f'fill="{stroke_color}" font-size="10" font-weight="bold">{result_label}</text>'
                                )

                    # Single-operand comparisons to zero (SIGN, RELU, LRELU)
                    # These don't need a comparison link - the comparison is implicit to 0
                    # But we could add a small "≷0" indicator near the node if desired

                    # Traditional IFLTE-style conditionals with 4 children
                    elif len(children) >= 2 and label_upper not in (
                        "SIGN",
                        "RELU",
                        "LRELU",
                    ):
                        obs_node = children[0]
                        thresh_node = children[1]

                        obs_x, obs_y = positions[obs_node]
                        thresh_x, thresh_y = positions[thresh_node]
                        obs_w, _ = node_dims[obs_node]
                        thresh_w, _ = node_dims[thresh_node]

                        # Only draw if nodes are at same level (they should be)
                        if abs(obs_y - thresh_y) < 5:
                            # Determine comparison operator from label
                            if "LTE" in label_upper or label_upper == "IFLTE":
                                op_symbol = "≤"
                            elif "LT" in label_upper or label_upper == "IFLT":
                                op_symbol = "<"
                            elif "GTE" in label_upper:
                                op_symbol = "≥"
                            elif "GT" in label_upper:
                                op_symbol = ">"
                            else:
                                op_symbol = "?"

                            # Draw horizontal line with operator label
                            line_y = obs_y
                            line_x1 = obs_x + obs_w / 2 + 3
                            line_x2 = thresh_x - thresh_w / 2 - 3
                            mid_x = (line_x1 + line_x2) / 2

                            is_active = (
                                self.nodes[obs_node].is_active
                                and self.nodes[thresh_node].is_active
                            )
                            stroke_color = "#8b5cf6" if is_active else "#d1d5db"

                            gap = 15
                            if line_x2 - line_x1 > gap * 2:
                                svg_parts.append(
                                    f'  <line x1="{line_x1}" y1="{line_y}" '
                                    f'x2="{mid_x - gap}" y2="{line_y}" '
                                    f'stroke="{stroke_color}" stroke-width="2" stroke-dasharray="3,3"/>'
                                )
                                svg_parts.append(
                                    f'  <line x1="{mid_x + gap}" y1="{line_y}" '
                                    f'x2="{line_x2}" y2="{line_y}" '
                                    f'stroke="{stroke_color}" stroke-width="2" stroke-dasharray="3,3"/>'
                                )

                            svg_parts.append(
                                f'  <circle cx="{mid_x}" cy="{line_y}" r="12" '
                                f'fill="white" stroke="{stroke_color}" stroke-width="2"/>'
                            )
                            svg_parts.append(
                                f'  <text x="{mid_x}" y="{line_y + 5}" text-anchor="middle" '
                                f'fill="{stroke_color}" font-size="14" font-weight="bold">{xml_escape(op_symbol)}</text>'
                            )

        svg_parts.append("")
        svg_parts.append("  <!-- Nodes -->")

        # Draw nodes
        colors = self.get_node_colors()
        for node_id, (x, y) in positions.items():
            info = self.nodes[node_id]
            color = colors[node_id]
            node_w, node_h = node_dims[node_id]

            # Determine text color based on background
            text_color = (
                "white"
                if color in ["#3b82f6", "#6b7280", "#8b5cf6", "#22c55e", "#f97316"]
                else "#374151"
            )
            # Lighter/dimmer color for value text
            value_color = (
                "rgba(255,255,255,0.75)"
                if color in ["#3b82f6", "#6b7280", "#8b5cf6", "#22c55e", "#f97316"]
                else "#6b7280"
            )

            # Determine stroke styling
            stroke = "#f97316" if info.is_critical else "#374151"
            stroke_width = 3 if info.is_critical else 1

            # Draw shape based on node type
            if info.node_type == NodeType.CONDITIONAL:
                # Draw diamond for decision/conditional nodes
                path = diamond_path(x, y, node_w, node_h)
                svg_parts.append(
                    f'  <path d="{path}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
                )
            else:
                # Draw rounded rectangle for all other nodes
                rect_attrs = rounded_rect(x, y, node_w, node_h)
                svg_parts.append(
                    f'  <rect {rect_attrs} fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
                )

            # Calculate vertical positions for label and value
            # If showing value, shift label up and put value below it
            has_value = show_values and info.output_value is not None
            if has_value:
                label_y = y - 2  # Shift label up slightly
                value_y = y + 12  # Value below label
            else:
                label_y = y + 4  # Centered vertically

            # Draw label (node name)
            label = info.label
            svg_parts.append(
                f'  <text x="{x}" y="{label_y}" text-anchor="middle" '
                f'fill="{text_color}" font-size="{font_size}" font-weight="600">{label}</text>'
            )

            # Draw value inside node (if enabled and available)
            if has_value:
                val_str = f"{info.output_value:.4g}"
                svg_parts.append(
                    f'  <text x="{x}" y="{value_y}" text-anchor="middle" '
                    f'fill="{value_color}" font-size="9" font-weight="400">{val_str}</text>'
                )

            # Draw decision index badge
            if info.decision_index is not None:
                badge_x = x + node_w // 2 - 5
                badge_y = y - node_h // 2 + 5
                svg_parts.append(
                    f'  <circle cx="{badge_x}" cy="{badge_y}" r="10" fill="#1f2937"/>'
                )
                svg_parts.append(
                    f'  <text x="{badge_x}" y="{badge_y + 3}" text-anchor="middle" '
                    f'fill="white" font-size="9" font-weight="bold">D{info.decision_index}</text>'
                )

            # Draw critical star
            if info.is_critical:
                star_x = x - node_w // 2 + 2
                star_y = y - node_h // 2 + 12
                svg_parts.append(
                    f'  <text x="{star_x}" y="{star_y}" fill="#f97316" font-size="14">⭐</text>'
                )

        # Add legend
        legend_y = svg_height - 15
        svg_parts.append("")
        svg_parts.append("  <!-- Legend -->")
        svg_parts.append(
            f'  <text x="20" y="{legend_y}" fill="#6b7280" font-size="10">'
            f"Nodes: "
            f'<tspan fill="#3b82f6">●</tspan> feature  '
            f'<tspan fill="#6b7280">●</tspan> constant  '
            f'<tspan fill="#8b5cf6">◆</tspan> decision  '
            f'<tspan fill="#22c55e">●</tspan> operation  '
            f'<tspan fill="#d1d5db">●</tspan> inactive  '
            f'<tspan fill="#f97316">⭐</tspan> critical  '
            f"| Edges: "
            f'<tspan fill="#16a34a">━</tspan> +contrib  '
            f'<tspan fill="#dc2626">━</tspan> −contrib'
            f"</text>"
        )

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)

    def to_svg_html(self, **kwargs) -> str:
        """Generate HTML with embedded SVG tree diagram."""
        svg = self.to_svg(**kwargs)
        return f"""
        <div style="font-family: -apple-system, sans-serif; padding: 15px; 
                    border: 1px solid #d1d5db; border-radius: 8px; background: #ffffff;">
            <div style="margin-bottom: 10px;">
                <b>Tree:</b> <code style="font-size: 12px;">{self.tree_expression}</code>
            </div>
            <div style="margin-bottom: 10px;">
                <b>Output:</b> <span style="font-size: 18px; font-weight: bold;">{self.output_value:.4g}</span>
                <span style="color: #6b7280; margin-left: 15px;">Active Decisions: {self.num_decisions}</span>
            </div>
            <div style="overflow-x: auto;">
                {svg}
            </div>
        </div>
        """


def _is_constant_terminal(name: str, constant_names: Optional[Set[str]] = None) -> bool:
    """Check if a terminal is a constant (model parameter) vs a feature."""
    if constant_names and name in constant_names:
        return True

    # Heuristics for constant detection
    upper_name = name.upper()
    if name.isupper() and len(name) > 1:
        return True

    constant_patterns = [
        "WEIGHT",
        "THRESHOLD",
        "BIAS",
        "CONST",
        "COEF",
        "BASE",
        "SCALE",
        "FACTOR",
        "PARAM",
        "LIMIT",
        "ONE",
        "ZERO",
        "TRUE",
        "FALSE",
        "MIN",
        "MAX",
    ]
    for pattern in constant_patterns:
        if pattern in upper_name:
            return True

    if name.startswith("w") and len(name) <= 3 and name[1:].isdigit():
        return True

    return False


def _is_conditional_primitive(name: str) -> bool:
    """
    Check if a primitive is a conditional/decision operation.

    This includes:
    - Explicit conditionals: IFLTE, IFLT, IF, etc.
    - Masked decisions: SIGN, CMP, RELU, MAX, MIN

    These all involve implicit comparisons that determine output.
    """
    upper_name = name.upper()

    # Explicit conditionals
    explicit_conditionals = {"IFLTE", "IFLT", "IFB", "IF", "IFGTE", "IFGT"}

    # Masked decisions - operations that implicitly compare and select
    masked_decisions = {
        "SIGN",  # Compares to 0, outputs ±1
        "CMP",  # Compares two values, outputs -1/0/1
        "RELU",  # Compares to 0, outputs x or 0
        "LRELU",  # Compares to 0, outputs x or 0.01*x
        "MAX",  # Compares two values, selects larger
        "MIN",  # Compares two values, selects smaller
    }

    return (
        upper_name in explicit_conditionals
        or upper_name in masked_decisions
        or name.startswith("if")
    )


def visualize_tree(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
    evaluation_result: Any,
    constant_names: Optional[List[str]] = None,
) -> TreeVisualization:
    """
    Create a visualization of an evaluated GP tree.

    Args:
        tree: The GP tree to visualize
        pset: The primitive set
        terminal_values: Values of all terminals for this evaluation
        evaluation_result: The EvaluationResult from evaluate_with_tracking
        constant_names: Optional list of constant terminal names

    Returns:
        TreeVisualization object with all node/edge information
    """
    # Get basic graph structure from DEAP
    nodes_list, edges, labels = gp.graph(tree)

    # Build constant set
    const_set = set(constant_names) if constant_names else set()

    # Get active decision info
    active_decisions = (
        evaluation_result.active_decisions
        if hasattr(evaluation_result, "active_decisions")
        else evaluation_result.decisions
    )
    # Get ALL decisions (including inactive) for matching inactive conditional nodes
    all_decisions = evaluation_result.decisions

    critical_decision = (
        evaluation_result.get_critical_decision()
        if hasattr(evaluation_result, "get_critical_decision")
        else None
    )

    # We need to evaluate the tree node by node to get intermediate values
    # This requires walking the tree structure
    node_values = _compute_node_values(tree, pset, terminal_values)

    # Determine which nodes are on the active path
    active_nodes = _find_active_path(
        tree, pset, terminal_values, nodes_list, edges, labels
    )

    # Build child map for looking up children of each node
    children_map = {}
    for parent, child in edges:
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(child)

    # Index active decisions - we'll assign indices based on tree position (top-down),
    # not DEAP's evaluation order (inner-first)
    # First, match decisions to nodes, then assign indices by depth (shallowest = D1)

    # Calculate actual depth of each node (distance from root)
    node_depths = {0: 0}  # Root is at depth 0

    def compute_depths(node_id: int, depth: int) -> None:
        node_depths[node_id] = depth
        for child in children_map.get(node_id, []):
            compute_depths(child, depth + 1)

    compute_depths(0, 0)

    # Find all conditional nodes
    conditional_nodes = []
    for node_id in nodes_list:
        label = labels[node_id]
        if _is_conditional_primitive(label):
            conditional_nodes.append((node_id, node_depths.get(node_id, 0)))

    # Sort by depth DESCENDING for matching (deepest first to handle identical values correctly)
    # This ensures inner decisions get matched first when obs/thresh values are the same
    conditional_nodes_for_matching = sorted(conditional_nodes, key=lambda x: -x[1])

    # Match decisions to nodes (deepest first to handle duplicates)
    used_decisions = set()
    node_decision_map = {}  # node_id -> decision

    for node_id, depth in conditional_nodes_for_matching:
        children = children_map.get(node_id, [])
        if len(children) >= 2:
            obs_value = node_values.get(children[0])
            thresh_value = node_values.get(children[1])

            # Find matching decision that hasn't been used yet
            for d in all_decisions:
                if id(d) in used_decisions:
                    continue

                obs_match = (
                    obs_value is not None
                    and abs(d.observation_value - obs_value) < 1e-9
                )
                thresh_match = (
                    thresh_value is not None
                    and abs(d.threshold_value - thresh_value) < 1e-9
                )

                if obs_match and thresh_match:
                    node_decision_map[node_id] = d
                    used_decisions.add(id(d))
                    break

    # Now assign decision indices based on tree order (shallowest = D1)
    # Sort matched conditional nodes by depth ASCENDING (top-down order)
    matched_nodes_by_depth = [
        (node_id, depth)
        for node_id, depth in conditional_nodes
        if node_id in node_decision_map and node_decision_map[node_id].active
    ]
    matched_nodes_by_depth.sort(key=lambda x: x[1])  # Shallowest first

    # Build decision_to_index based on tree position
    decision_to_index = {}
    for i, (node_id, depth) in enumerate(matched_nodes_by_depth):
        d = node_decision_map[node_id]
        decision_to_index[id(d)] = i + 1  # D1, D2, D3...

    # Compute contributions for each node
    # We use the signed_contributions from the evaluation result
    # and propagate them through the tree structure
    node_contributions = _compute_node_contributions(
        tree,
        pset,
        terminal_values,
        nodes_list,
        edges,
        labels,
        evaluation_result,
        active_nodes,
    )

    # Build NodeInfo for each node
    nodes = {}

    for node_id in nodes_list:
        label = labels[node_id]

        # Determine node type
        if label in [p.name for p in pset.primitives[pset.ret]]:
            if _is_conditional_primitive(label):
                node_type = NodeType.CONDITIONAL
            else:
                node_type = NodeType.PRIMITIVE
        else:
            # Terminal
            if _is_constant_terminal(label, const_set):
                node_type = NodeType.TERMINAL_CONSTANT
            else:
                node_type = NodeType.TERMINAL_FEATURE

        # Get output value for this node
        output_value = node_values.get(node_id)

        # Get contribution for this node
        contribution = node_contributions.get(node_id)

        # Check if this is a decision node
        decision_index = None
        is_critical = False
        condition_result = None

        if node_type == NodeType.CONDITIONAL:
            # Look up the pre-matched decision for this node
            matched_decision = node_decision_map.get(node_id)

            if matched_decision:
                condition_result = matched_decision.result
                is_critical = matched_decision is critical_decision

                # Only assign index if this is an active decision
                if matched_decision.active:
                    decision_index = decision_to_index.get(id(matched_decision))

        # Create display label
        if output_value is not None:
            display_label = f"{label}\n({output_value:.4g})"
        else:
            display_label = label

        nodes[node_id] = NodeInfo(
            node_id=node_id,
            label=label,
            node_type=node_type,
            output_value=output_value,
            is_active=node_id in active_nodes,
            contribution=contribution,
            condition_result=condition_result,
            decision_index=decision_index,
            is_critical=is_critical,
            display_label=display_label,
        )

    return TreeVisualization(
        nodes=nodes,
        edges=edges,
        root_id=0,
        tree_expression=str(tree),
        output_value=evaluation_result.output.value,
        num_decisions=len(active_decisions),
    )


def _compute_node_values(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
) -> Dict[int, float]:
    """
    Compute the output value at each node by walking the tree.
    Returns a dict mapping node_id to value.
    """
    # We need to evaluate subtrees. This is a bit complex because
    # DEAP trees are stored as flat lists in prefix order.

    values = {}

    def evaluate_subtree(index: int) -> Tuple[float, int]:
        """
        Recursively evaluate subtree starting at index.
        Returns (value, next_index).
        """
        node = tree[index]

        if isinstance(node, gp.Terminal):
            # It's a terminal
            name = node.name if hasattr(node, "name") else str(node.value)

            if name in terminal_values:
                val = terminal_values[name]
            elif hasattr(node, "value"):
                if hasattr(node.value, "value"):  # TrackedValue
                    val = node.value.value
                elif isinstance(node.value, (int, float)):
                    val = float(node.value)
                else:
                    # It's a string name - look up in terminal_values
                    val = terminal_values.get(str(node.value), 0.0)
            else:
                val = 0.0
            values[index] = val
            return val, index + 1
        else:
            # It's a primitive - evaluate children first
            arity = node.arity
            child_values = []
            next_idx = index + 1

            for _ in range(arity):
                child_val, next_idx = evaluate_subtree(next_idx)
                child_values.append(child_val)

            # Compute this node's value
            # We need to actually apply the primitive
            try:
                if node.name == "ADD":
                    val = child_values[0] + child_values[1]
                elif node.name == "SUB":
                    val = child_values[0] - child_values[1]
                elif node.name == "MUL":
                    val = child_values[0] * child_values[1]
                elif node.name == "DIV" or node.name == "SAFEDIV":
                    val = (
                        child_values[0] / child_values[1] if child_values[1] != 0 else 0
                    )
                elif node.name == "NEG":
                    val = -child_values[0]
                elif node.name == "ABS":
                    val = abs(child_values[0])
                elif node.name in ("IFLTE", "if_less_than_or_eq"):
                    # IFLTE(a, b, true_val, false_val) -> true_val if a <= b else false_val
                    val = (
                        child_values[2]
                        if child_values[0] <= child_values[1]
                        else child_values[3]
                    )
                elif node.name in ("IFLT", "if_less_than"):
                    val = (
                        child_values[2]
                        if child_values[0] < child_values[1]
                        else child_values[3]
                    )
                elif node.name == "MIN":
                    val = min(child_values[0], child_values[1])
                elif node.name == "MAX":
                    val = max(child_values[0], child_values[1])
                elif node.name == "SQRT":
                    val = child_values[0] ** 0.5 if child_values[0] >= 0 else 0
                elif node.name in ("POW2", "SQUARE"):
                    val = child_values[0] ** 2
                elif node.name == "CMP":
                    # Compare: returns -1 if a < b, 0 if a == b, 1 if a > b
                    a, b = child_values[0], child_values[1]
                    if a < b:
                        val = -1.0
                    elif a > b:
                        val = 1.0
                    else:
                        val = 0.0
                elif node.name == "SIGN":
                    # Sign: returns -1 if x < 0, else 1
                    val = 1.0 if child_values[0] >= 0 else -1.0
                elif node.name == "RELU":
                    # ReLU: max(0, x)
                    val = max(0.0, child_values[0])
                elif node.name == "LRELU":
                    # Leaky ReLU: x if x > 0, else 0.01 * x
                    val = (
                        child_values[0]
                        if child_values[0] > 0
                        else 0.01 * child_values[0]
                    )
                elif node.name == "SIG":
                    # Sigmoid
                    import math

                    x = child_values[0]
                    if x >= 0:
                        val = 1 / (1 + math.exp(-min(x, 500)))
                    else:
                        exp_x = math.exp(max(x, -500))
                        val = exp_x / (1 + exp_x)
                elif node.name == "TANH":
                    # Hyperbolic tangent
                    import math

                    val = math.tanh(max(-500, min(500, child_values[0])))
                elif node.name in ("LT", "less_than"):
                    val = 1.0 if child_values[0] < child_values[1] else 0.0
                elif node.name in ("LTE", "less_than_or_eq"):
                    val = 1.0 if child_values[0] <= child_values[1] else 0.0
                elif node.name in ("GT", "greater_than"):
                    val = 1.0 if child_values[0] > child_values[1] else 0.0
                elif node.name in ("GTE", "greater_than_or_eq"):
                    val = 1.0 if child_values[0] >= child_values[1] else 0.0
                elif node.name == "IF":
                    # IF(condition, true_val, false_val)
                    val = child_values[1] if child_values[0] > 0.5 else child_values[2]
                else:
                    # Unknown primitive - try to call it
                    try:
                        val = (
                            node.func(*child_values)
                            if hasattr(node, "func")
                            else sum(child_values)
                        )
                    except:
                        val = sum(child_values)
            except Exception:
                val = 0.0

            values[index] = val
            return val, next_idx

    if len(tree) > 0:
        evaluate_subtree(0)

    return values


def _find_active_path(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
    nodes_list: List[int],
    edges: List[Tuple[int, int]],
    labels: Dict[int, str],
) -> Set[int]:
    """
    Find which nodes are on the active execution path.
    For conditionals, only one branch is active.
    """
    # Start with all nodes active
    active = set(nodes_list)

    # Build adjacency
    children_map = {}
    for parent, child in edges:
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(child)

    # Get node values to determine conditional outcomes
    node_values = _compute_node_values(tree, pset, terminal_values)

    def mark_inactive(node_id: int) -> None:
        """Recursively mark a subtree as inactive."""
        active.discard(node_id)
        for child in children_map.get(node_id, []):
            mark_inactive(child)

    # Find conditional nodes and mark inactive branches
    for node_id in nodes_list:
        label = labels[node_id]
        if _is_conditional_primitive(label):
            children = children_map.get(node_id, [])
            if (
                len(children) == 4
            ):  # IFLTE has 4 children: obs, threshold, true_val, false_val
                # Get observation and threshold values
                obs_val = node_values.get(children[0], 0)
                thresh_val = node_values.get(children[1], 0)

                # Determine which branch is taken
                if label in ("IFLTE", "if_less_than_or_eq"):
                    condition_true = obs_val <= thresh_val
                elif label in ("IFLT", "if_less_than"):
                    condition_true = obs_val < thresh_val
                else:
                    condition_true = obs_val <= thresh_val  # Default

                # Mark inactive branch
                if condition_true:
                    mark_inactive(children[3])  # false branch inactive
                else:
                    mark_inactive(children[2])  # true branch inactive

            # MIN/MAX: mark the non-selected input as inactive
            elif len(children) == 2 and label.upper() in ("MIN", "MAX"):
                left_val = node_values.get(children[0], 0)
                right_val = node_values.get(children[1], 0)

                if label.upper() == "MIN":
                    # MIN selects the smaller value
                    if left_val <= right_val:
                        mark_inactive(children[1])  # right is not selected
                    else:
                        mark_inactive(children[0])  # left is not selected
                else:  # MAX
                    # MAX selects the larger value
                    if left_val >= right_val:
                        mark_inactive(children[1])  # right is not selected
                    else:
                        mark_inactive(children[0])  # left is not selected

    return active


def _compute_node_contributions(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
    nodes_list: List[int],
    edges: List[Tuple[int, int]],
    labels: Dict[int, str],
    evaluation_result: Any,
    active_nodes: Set[int],
) -> Dict[int, float]:
    """
    Compute the signed contribution of each node to the final output.

    For terminals: use signed_contributions from evaluation result
    For operations: depends on operation type
      - Arithmetic (ADD, SUB, MUL): sum of children contributions
      - Fixed-output decisions (SIGN, CMP): 0 (output doesn't depend on input magnitude)
      - Passthrough decisions (MIN, MAX, RELU): contribution from selected/passed input

    Returns dict mapping node_id to contribution value (can be positive or negative).
    Values are normalized relative to the maximum absolute contribution.
    """
    contributions = {}

    # Compute node values once for determining decision outcomes
    node_values = _compute_node_values(tree, pset, terminal_values)

    # Get signed contributions from evaluation result
    signed_contribs = {}
    if hasattr(evaluation_result, "output") and hasattr(
        evaluation_result.output, "signed_contributions"
    ):
        sc = evaluation_result.output.signed_contributions
        if hasattr(sc, "contributions"):
            signed_contribs = sc.contributions

    # Also get regular contributions as fallback
    regular_contribs = {}
    if hasattr(evaluation_result, "output") and hasattr(
        evaluation_result.output, "contributions"
    ):
        regular_contribs = evaluation_result.output.contributions

    # Build parent map
    parent_map = {}
    children_map = {}
    for parent, child in edges:
        parent_map[child] = parent
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(child)

    # First pass: assign contributions to terminal nodes
    for node_id in nodes_list:
        label = labels[node_id]

        # Check if this is a terminal
        is_terminal = (
            node_id not in children_map or len(children_map.get(node_id, [])) == 0
        )

        if is_terminal:
            # Look up contribution by label
            if label in signed_contribs:
                contributions[node_id] = signed_contribs[label]
            elif label in regular_contribs:
                # Use regular contribution, try to determine sign from value
                contrib = regular_contribs[label]
                if label in terminal_values:
                    val = terminal_values[label]
                    # Heuristic: if terminal value is negative, contribution might be negative
                    # This is imperfect but better than nothing
                    contributions[node_id] = contrib
                else:
                    contributions[node_id] = contrib
            else:
                contributions[node_id] = 0.0

    # Second pass: propagate contributions up through operations
    # Process nodes in reverse topological order (leaves to root)
    def get_node_depth(node_id: int) -> int:
        depth = 0
        current = node_id
        while current in parent_map:
            current = parent_map[current]
            depth += 1
        return depth

    # Sort by depth descending (process leaves first)
    nodes_by_depth = sorted(nodes_list, key=get_node_depth, reverse=True)

    for node_id in nodes_by_depth:
        if node_id in contributions:
            continue  # Already assigned (terminal)

        children = children_map.get(node_id, [])
        if not children:
            contributions[node_id] = 0.0
            continue

        label = labels[node_id]

        # For inactive nodes, contribution is 0
        if node_id not in active_nodes:
            contributions[node_id] = 0.0
            continue

        # Aggregate children contributions based on operation type
        child_contribs = [contributions.get(c, 0.0) for c in children]

        # Filter to only active children
        active_child_contribs = [
            contributions.get(c, 0.0) for c in children if c in active_nodes
        ]

        if _is_conditional_primitive(label):
            label_upper = label.upper()

            # Handle different decision types based on their structure
            if len(children) >= 4:
                # Traditional conditionals: IFLTE(obs, thresh, true_branch, false_branch)
                true_contrib = contributions.get(children[2], 0.0)
                false_contrib = contributions.get(children[3], 0.0)

                # Check which branch is active
                if children[2] in active_nodes:
                    contributions[node_id] = true_contrib
                elif children[3] in active_nodes:
                    contributions[node_id] = false_contrib
                else:
                    contributions[node_id] = 0.0

            elif len(children) == 2:
                # Two-input decisions: MIN, MAX, CMP
                left_contrib = contributions.get(children[0], 0.0)
                right_contrib = contributions.get(children[1], 0.0)

                if label_upper == "MAX":
                    # MAX passes through the larger value's contribution
                    left_val = node_values.get(children[0], 0.0)
                    right_val = node_values.get(children[1], 0.0)
                    if left_val >= right_val:
                        contributions[node_id] = left_contrib
                    else:
                        contributions[node_id] = right_contrib

                elif label_upper == "MIN":
                    # MIN passes through the smaller value's contribution
                    left_val = node_values.get(children[0], 0.0)
                    right_val = node_values.get(children[1], 0.0)
                    if left_val <= right_val:
                        contributions[node_id] = left_contrib
                    else:
                        contributions[node_id] = right_contrib

                elif label_upper == "CMP":
                    # CMP outputs a fixed value (-1, 0, 1), so no contribution passes through
                    contributions[node_id] = 0.0

                else:
                    # Other 2-child decisions - sum contributions as fallback
                    contributions[node_id] = left_contrib + right_contrib

            elif len(children) == 1:
                # Single-input decisions: SIGN, RELU
                child_contrib = contributions.get(children[0], 0.0)

                if label_upper in ("SIGN",):
                    # SIGN outputs fixed value, no contribution passes through
                    contributions[node_id] = 0.0
                elif label_upper in ("RELU", "LRELU"):
                    # RELU passes through if positive, otherwise 0
                    child_val = node_values.get(children[0], 0.0)
                    if child_val > 0:
                        contributions[node_id] = child_contrib
                    else:
                        contributions[node_id] = 0.0
                else:
                    contributions[node_id] = child_contrib
            else:
                contributions[node_id] = 0.0
        else:
            # For other operations, sum active children contributions
            contributions[node_id] = sum(active_child_contribs)

    # Normalize contributions to [-1, 1] range
    max_abs = max(abs(v) for v in contributions.values()) if contributions else 1.0
    if max_abs > 0:
        contributions = {k: v / max_abs for k, v in contributions.items()}

    return contributions


def render_tree_matplotlib(
    viz: TreeVisualization,
    figsize: Tuple[int, int] = (12, 8),
    show_values: bool = True,
) -> Any:
    """
    Render tree visualization using matplotlib.
    Returns matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for render_tree_matplotlib")

    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for render_tree_matplotlib")

    G = viz.to_networkx()

    fig, ax = plt.subplots(figsize=figsize)

    # Use graphviz layout if available, otherwise hierarchical
    try:
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G)

    # Get colors
    node_colors = [viz.get_node_colors()[n] for n in G.nodes()]
    edge_colors = [viz.get_edge_colors()[(u, v)] for u, v in G.edges()]

    # Draw
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=False,
        node_size=2000,
        font_size=8,
        arrows=True,
    )

    # Add labels
    labels = {n: viz.nodes[n].display_label for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    ax.set_title(f"Tree: {viz.tree_expression}\nOutput: {viz.output_value:.4g}")

    plt.tight_layout()
    return fig


# Self-test
if __name__ == "__main__":
    print("FCP Tree Visualization - Self Test")
    print("=" * 50)

    # Need the tracker for evaluation
    from fcp_tracker import (
        create_tracked_pset,
        TrackedValue,
        evaluate_with_tracking,
    )

    # Create a test tree
    pset = create_tracked_pset("test", ["x", "y"])
    pset.addTerminal(TrackedValue.from_constant(1.0, "ONE"), name="ONE")
    pset.addTerminal(TrackedValue.from_constant(0.0, "ZERO"), name="ZERO")
    pset.addTerminal(TrackedValue.from_constant(10.0, "TEN"), name="TEN")

    # Simple conditional tree
    tree = gp.PrimitiveTree.from_string("IFLTE(x, y, TEN, ZERO)", pset)

    # Evaluate
    result = evaluate_with_tracking(tree, pset, {"x": 3.0, "y": 5.0})

    print(f"Tree: {tree}")
    print(f"Inputs: x=3.0, y=5.0")
    print(f"Output: {result.output.value}")
    print()

    # Create visualization
    viz = visualize_tree(
        tree,
        pset,
        {"x": 3.0, "y": 5.0, "ONE": 1.0, "ZERO": 0.0, "TEN": 10.0},
        result,
        constant_names=["ONE", "ZERO", "TEN"],
    )

    print("ASCII Visualization:")
    print(viz.to_ascii())
    print()

    print("DOT Output (for Graphviz):")
    print(viz.to_dot())
    print()

    print("✓ Self-test passed!")
