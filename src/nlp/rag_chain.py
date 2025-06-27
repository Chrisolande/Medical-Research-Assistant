import heapq
import logging
from typing import Any

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document

from src.core.config import (
    LLM_MAX_CONTEXT_LENGTH,
    MAX_NODES_TO_TRAVERSE,
    MIN_NODES_TO_TRAVERSE,
    AnswerCheck,
)

# Logging for this specific module/class
logger = logging.getLogger(__name__)


class QueryEngine:
    """Query engine for traversing the knowledge graph to answer a query."""

    def __init__(self, vector_store, knowledge_graph, llm):
        """Initialize the query engine."""
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.min_nodes_to_traverse = MIN_NODES_TO_TRAVERSE
        self.max_nodes_to_traverse = MAX_NODES_TO_TRAVERSE
        self.max_context_length = LLM_MAX_CONTEXT_LENGTH
        self.answer_check_chain = self._create_answer_check_chain()
        logger.info("QueryEngine initialized.")

    def _create_answer_check_chain(self):
        """Create the answer check chain."""
        prompt = ChatPromptTemplate.from_template(
            """
            Given the query: "{query}"
            And the context:
            {context}

            IMPORTANT: Only mark as complete if you have comprehensive, detailed information that fully addresses ALL aspects of the query.
            If there's ANY uncertainty or if more context could provide better insights, mark as incomplete.

            Based on the provided context, is the query fully answerable with comprehensive detail?
            Answerable: [Yes/No]
            Complete Answer (if Yes):
            """
        )

        return prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> tuple[bool, str]:
        """Check if the query is fully answerable with the provided context."""
        # Ensure sufficient context for a meaningful check
        if len(context.split("\n")) < 3:
            logger.debug("Context too short for comprehensive answer check.")
            return False, ""

        try:
            response = self.answer_check_chain.invoke(
                {"query": query, "context": context}
            )
            return response.is_complete, response.answer
        except ValueError as e:
            if "Structured Output response does not have a 'parsed' field" in str(e):
                logger.warning(
                    "LLM returned unstructured response, falling back to text parsing"
                )
                # Fallback: use the chain without structured output
                fallback_prompt = ChatPromptTemplate.from_template(
                    """
                    Given the query: "{query}"
                    And the context:
                    {context}

                    IMPORTANT: Only mark as complete if you have comprehensive, detailed information that fully addresses ALL aspects of the query.
                    If there's ANY uncertainty or if more context could provide better insights, mark as incomplete.

                    Based on the provided context, is the query fully answerable with comprehensive detail?
                    Answerable: [Yes/No]
                    Complete Answer (if Yes):
                    """
                )
                fallback_chain = fallback_prompt | self.llm
                response = fallback_chain.invoke({"query": query, "context": context})

                # Parse the text response
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                is_complete = "answerable: yes" in response_text.lower()

                # Extract answer if marked as complete
                answer = ""
                if is_complete and "complete answer" in response_text.lower():
                    lines = response_text.split("\n")
                    answer_started = False
                    answer_lines = []
                    for line in lines:
                        if "complete answer" in line.lower():
                            answer_started = True
                            # Check if answer is on same line
                            if ":" in line:
                                answer_part = line.split(":", 1)[1].strip()
                                if answer_part:
                                    answer_lines.append(answer_part)
                        elif answer_started and line.strip():
                            answer_lines.append(line.strip())
                    answer = "\n".join(answer_lines)

                return is_complete, answer
            else:
                logger.error(f"Error during answer check: {e}", exc_info=True)
                return False, ""
        except Exception as e:
            logger.error(f"Error during answer check: {e}", exc_info=True)
            return False, ""

    async def _initialize_traversal(
        self, relevant_docs: list[Document]
    ) -> tuple[list[tuple[float, Any]], dict[Any, float]]:
        """Initialize the graph traversal."""
        priority_queue = []
        distances = {}

        for doc in relevant_docs:
            closest_nodes_results = (
                await self.vector_store.similarity_search_with_score(
                    doc.page_content, k=3
                )
            )
            if not closest_nodes_results:
                logger.debug(
                    f"No closest nodes found for document: {doc.page_content[:50]}..."
                )
                continue

            closest_node_content, similarity_score = closest_nodes_results[0]
            # Find the actual node ID from the knowledge graph
            closest_node_id = None
            for n, data in self.knowledge_graph.graph.nodes(data=True):
                if data.get("content") == closest_node_content.page_content:
                    closest_node_id = n
                    break

            if closest_node_id is None:
                logger.warning(
                    f"Closest node content not found in knowledge graph: {closest_node_content.page_content[:50]}..."
                )
                continue

            priority = 1 / (similarity_score + 1e-15)
            heapq.heappush(priority_queue, (priority, closest_node_id))
            distances[closest_node_id] = priority
            logger.debug(
                f"Initialized traversal with node {closest_node_id} (priority: {priority:.2f})"
            )
        return priority_queue, distances

    def _process_node(
        self,
        current_node: Any,
        query: str,
        expanded_context: str,
        traversal_path: list[Any],
        visited_concepts: set,
        filtered_content: dict[Any, str],
        step: int,
    ) -> tuple[str, list[Any], dict[Any, str], str, bool]:
        """Process a node in the graph traversal."""
        node_data = self.knowledge_graph.graph.nodes[current_node]
        node_content = node_data.get("content", "")
        node_concepts = node_data.get("concepts", [])

        # Check if adding this node's content would exceed max context length
        # This is a simple word count check, consider a token count for more accuracy
        if len((expanded_context + node_content).split()) > self.max_context_length:
            logger.info(f"Skipping node {current_node}: context length limit reached.")
            return (
                expanded_context,
                traversal_path,
                filtered_content,
                "",
                False,
            )  # Return current state, not complete

        filtered_content[current_node] = node_content
        expanded_context = (
            (expanded_context + "\n" + node_content)
            if expanded_context
            else node_content
        )
        traversal_path.append(current_node)

        print(f"\nStep {step} - Node {current_node}:")
        print(f"Content: {node_content[:100]}...")
        print(f"Concepts: {', '.join(node_concepts)}")
        print("-" * 50)

        is_complete, answer = self._check_answer(query, expanded_context)
        if is_complete:
            return expanded_context, traversal_path, filtered_content, answer, True

        node_concepts_set = {
            self.knowledge_graph._lemmatize_concept(c) for c in node_concepts
        }
        visited_concepts.update(node_concepts_set)
        return expanded_context, traversal_path, filtered_content, "", False

    def _explore_neighbors(
        self,
        current_node: Any,
        current_priority: float,
        traversal_path: list[Any],
        distances: dict[Any, float],
        priority_queue: list[tuple[float, Any]],
    ) -> None:
        """Explore the neighbors of a node in the graph traversal and update priority
        queue."""
        for neighbor in self.knowledge_graph.graph.neighbors(current_node):
            if neighbor in traversal_path:  # Skip already traversed nodes
                continue

            edge_data = self.knowledge_graph.graph[current_node][neighbor]
            edge_weight = edge_data.get("weight", 0.5)  # Default weight if not present
            distance = current_priority + (
                1 / (edge_weight + 1e-15)
            )  # Lower priority for higher weight
            if distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                logger.debug(
                    f"Added neighbor {neighbor} to queue with distance {distance:.2f}"
                )

    async def _expand_context(
        self, query: str, relevant_docs: list[Document]
    ) -> tuple[str, list[Any], dict[Any, str], str]:
        """Expand the context by traversing the knowledge graph."""
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""
        priority_queue, distances = await self._initialize_traversal(relevant_docs)
        step = 0

        while priority_queue and len(traversal_path) < self.max_nodes_to_traverse:
            current_priority, current_node = heapq.heappop(priority_queue)

            # Check if a shorter path to this node has already been found or if it's already processed
            if (
                current_priority > distances.get(current_node, float("inf"))
                or current_node in traversal_path
            ):
                continue

            step += 1
            logger.debug(f"Processing node {current_node} at step {step}")

            (
                expanded_context,
                traversal_path,
                filtered_content,
                current_node_answer,
                is_complete_now,
            ) = self._process_node(
                current_node,
                query,
                expanded_context,
                traversal_path,
                visited_concepts,
                filtered_content,
                step,
            )

            # Only check for completion after minimum nodes and every 3rd node after that
            should_check_completion = (
                len(traversal_path) >= self.min_nodes_to_traverse
                and (len(traversal_path) - self.min_nodes_to_traverse) % 3 == 0
            )

            if is_complete_now and should_check_completion:
                final_answer = current_node_answer
                logger.info(f"Found complete answer after {len(traversal_path)} nodes.")
                break  # Exit traversal loop if answer is complete

            self._explore_neighbors(
                current_node,
                current_priority,
                traversal_path,
                distances,
                priority_queue,
            )
        else:
            if len(traversal_path) >= self.max_nodes_to_traverse:
                logger.warning(
                    f"Max nodes to traverse ({self.max_nodes_to_traverse}) reached without a complete answer."
                )
            elif not priority_queue:
                logger.info(
                    "Priority queue exhausted without finding a complete answer."
                )

        if not final_answer:  # If no complete answer was found during traversal
            logger.info("Generating final answer from accumulated context.")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:",
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    async def query(self, query: str) -> tuple[str, list[Any], dict[Any, str]]:
        """Queries the knowledge graph to find an answer."""
        logger.info(f"Starting query for: '{query}'")
        # Retrieve initial relevant documents from the vector store
        relevant_docs = self.vector_store.retrieve_relevant_documents(query)
        if not relevant_docs:
            logger.warning("No initial relevant documents found from vector store.")
            return "No relevant information found.", [], {}

        await self._analyze_chunk_distribution(relevant_docs)

        # Expand context by traversing the graph
        _, traversal_path, filtered_content, final_answer = await self._expand_context(
            query, relevant_docs
        )

        logger.info(f"Final Answer: {final_answer.content[:200]}...")

        return final_answer, traversal_path, filtered_content

    async def _analyze_chunk_distribution(self, relevant_docs: list[Document]) -> float:
        """Analyzes chunk sizes to understand potential traversal limitations."""
        if not relevant_docs:
            logger.warning("No relevant documents to analyze chunk distribution.")
            return 0.0

        chunk_lengths = [len(doc.page_content.split()) for doc in relevant_docs]
        avg_chunk_length = (
            sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0.0
        )  # Added check for empty list

        logger.info(
            f"Chunk analysis - Count: {len(relevant_docs)}, "
            f"Avg length: {avg_chunk_length:.1f} words, "
            f"Range: {min(chunk_lengths)}-{max(chunk_lengths)} words"
        )

        large_chunks = [i for i, length in enumerate(chunk_lengths) if length > 300]
        if large_chunks:
            logger.warning(
                f"Found {len(large_chunks)} large chunks (>300 words) that might contain complete answers."
            )

        return avg_chunk_length
