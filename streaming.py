from queue import Queue

import streamlit as st


class StreamingNodeDisplay:
    def __init__(self):
        self.node_queue = Queue()
        self.is_processing = False
        self.container = None

    def start_streaming(self):
        """Initialize the streaming display."""
        self.is_processing = True
        self.container = st.empty()

    def add_node(self, step, node_id, content, concepts):
        """Add a node to the streaming display."""
        if self.is_processing:
            self.node_queue.put(
                {
                    "step": step,
                    "node_id": node_id,
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "concepts": concepts,
                }
            )
            self._update_display()

    def _update_display(self):
        """Update the streaming display with current nodes."""
        if not self.container:
            return

        nodes = []
        temp_queue = Queue()

        while not self.node_queue.empty():
            node = self.node_queue.get()
            nodes.append(node)
            temp_queue.put(node)

        while not temp_queue.empty():
            self.node_queue.put(temp_queue.get())

        if nodes:
            with self.container.container():
                # Show last 5 processed nodes
                for node in nodes[-5:]:
                    with st.expander(
                        f"Step {node['step']} - Node {node['node_id']}", expanded=True
                    ):
                        st.write(f"**Content:** {node['content']}")
                        st.write(f"**Concepts:** {', '.join(node['concepts'][:5])}")

    def stop_streaming(self):
        """Stop the streaming display and clear it."""
        self.is_processing = False
        if self.container:
            self.container.empty()
        while not self.node_queue.empty():
            self.node_queue.get()
