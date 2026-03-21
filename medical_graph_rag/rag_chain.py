import heapq
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from medical_graph_rag.config import (
    ANSWER_CHECK_INTERVAL,
    LLM_MAX_CONTEXT_LENGTH,
    MAX_NODES_TO_TRAVERSE,
    MIN_NODES_TO_TRAVERSE,
)

logger = logging.getLogger(__name__)


class AnswerCheck(BaseModel):
    """Check if a query is answerable with sufficient information from the provided
    context."""

    is_sufficient: bool = Field(
        description="Whether the context provides sufficient information."
    )
    synthesized_answer: str = Field(
        description="The synthesized answer based on the context."
    )


@dataclass
class QueryEngine:
    vector_store: Any
    knowledge_graph: Any
    llm: Any
    min_nodes_to_traverse: int = MIN_NODES_TO_TRAVERSE
    max_nodes_to_traverse: int = MAX_NODES_TO_TRAVERSE
    max_context_length: int = LLM_MAX_CONTEXT_LENGTH
    answer_check_interval: int = ANSWER_CHECK_INTERVAL

    def __post_init__(self):
        self.answer_check_chain = self._create_answer_check_chain()
        logger.info("QueryEngine initialized.")

    def _create_answer_check_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """
            Given the query: "{query}"
            And the context:
            {context}

            You must respond in exactly this format:
            Sufficient: [Yes or No]
            Synthesized Answer (if Yes): [Your answer here, or leave blank if No]
            """
        )
        return prompt | self.llm.with_structured_output(AnswerCheck)

    def _parse_fallback_response(self, raw_text_response: str) -> tuple[bool, str]:
        lines = raw_text_response.strip().split("\n")
        is_sufficient = False
        synthesized_answer = ""

        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("sufficient:"):
                is_sufficient = "yes" in line_lower
            elif line_lower.startswith("synthesized answer"):
                if ":" in line:
                    synthesized_answer = line.split(":", 1)[1].strip()

        return is_sufficient, synthesized_answer

    def _check_answer(self, query: str, context: str) -> tuple[bool, str]:
        logger.debug(f"[_check_answer] Query: {query}\nContext:\n{context}")

        if len(context.split("\n")) < 3:
            logger.debug(
                "[_check_answer] Context too short for comprehensive answer check."
            )
            return False, ""

        try:
            raw_response = self.answer_check_chain.invoke(
                {"query": query, "context": context}
            )
            logger.debug(f"[_check_answer] Raw LLM structured response: {raw_response}")
            return raw_response.is_sufficient, raw_response.synthesized_answer
        except (ValueError, AttributeError) as e:
            logger.warning(
                f"[_check_answer] Structured output failed: {e}. Using fallback."
            )

            fallback_prompt = ChatPromptTemplate.from_template(
                """
                Given the query: "{query}"
                And the context:
                {context}

                You must respond in exactly this format:
                Sufficient: [Yes or No]
                Synthesized Answer (if Yes): [Your answer here, or leave blank if No]
                """
            )
            fallback_chain = fallback_prompt | self.llm
            response_obj = fallback_chain.invoke({"query": query, "context": context})
            raw_text = (
                response_obj.content
                if hasattr(response_obj, "content")
                else str(response_obj)
            )

            return self._parse_fallback_response(raw_text)
        except Exception as e:
            logger.error(f"[_check_answer] Unexpected error: {e}", exc_info=True)
            return False, ""

    async def _initialize_traversal(
        self, relevant_docs: list[Document]
    ) -> tuple[list[tuple[float, Any]], dict[Any, float]]:
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
            closest_node_id = self.knowledge_graph.content_to_node_id.get(
                closest_node_content.page_content
            )

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
        streaming_callback: Callable | None = None,
    ) -> tuple[str, list[Any], dict[Any, str], str, bool]:
        node_data = self.knowledge_graph.graph.nodes[current_node]
        node_content = node_data.get("content", "")
        node_concepts = node_data.get("concepts", [])

        if len((expanded_context + node_content).split()) > self.max_context_length:
            logger.info(f"Skipping node {current_node}: context length limit reached.")
            return expanded_context, traversal_path, filtered_content, "", False

        filtered_content[current_node] = node_content
        expanded_context = (
            (expanded_context + "\n" + node_content)
            if expanded_context
            else node_content
        )
        traversal_path.append(current_node)

        if streaming_callback:
            streaming_callback(step, current_node, node_content[:100], node_concepts)

        logger.debug(f"Step {step} - Node {current_node}")
        logger.debug(f"Content: {node_content[:100]}...")
        logger.debug(f"Concepts: {', '.join(node_concepts)}")

        is_sufficient, synthesized_answer = self._check_answer(query, expanded_context)
        if is_sufficient:
            return (
                expanded_context,
                traversal_path,
                filtered_content,
                synthesized_answer,
                True,
            )

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
        visited_concepts: set,
        distances: dict[Any, float],
        priority_queue: list[tuple[float, Any]],
    ) -> None:
        for neighbor in self.knowledge_graph.graph.neighbors(current_node):
            if neighbor in traversal_path:
                continue

            edge_data = self.knowledge_graph.graph[current_node][neighbor]
            edge_weight = edge_data.get("weight", 0.5)

            neighbor_concepts = {
                self.knowledge_graph._lemmatize_concept(c)
                for c in self.knowledge_graph.graph.nodes[neighbor].get("concepts", [])
            }
            if neighbor_concepts and visited_concepts:
                overlap = len(neighbor_concepts & visited_concepts) / len(
                    neighbor_concepts
                )
            else:
                overlap = 0.0

            novelty_penalty = 1.0 + overlap
            distance = current_priority + (1 / (edge_weight + 1e-15)) * novelty_penalty
            if distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                logger.debug(
                    f"Neighbour {neighbor}: edge_weight={edge_weight:.3f}, "
                    f"overlap={overlap:.2f}, distance={distance:.3f}"
                )

    def _log_traversal_end(self, traversal_path: list, priority_queue: list) -> None:
        """Log the reason traversal ended without a complete answer."""
        if len(traversal_path) >= self.max_nodes_to_traverse:
            logger.warning(
                f"Max nodes to traverse ({self.max_nodes_to_traverse}) "
                "reached without a complete answer."
            )
        elif not priority_queue:
            logger.info("Priority queue exhausted without finding a complete answer.")

    def _generate_fallback_answer(self, query: str, context: str) -> str:
        """Generate an answer from accumulated context when traversal finds none."""
        logger.info(
            "No sufficient answer found during traversal. "
            "Generating final answer from accumulated context."
        )
        response_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are a helpful assistant. Analyze the following context and use it "
                "to provide a comprehensive answer to the query. If the context allows, "
                "try to synthesize information rather than just extracting it. If the "
                "context is insufficient, clearly state that.\n\n"
                "Context: {context}\n\nQuery: {query}\n\nAnswer:"
            ),
        )
        return (response_prompt | self.llm).invoke({"query": query, "context": context})

    async def _expand_context(
        self,
        query: str,
        relevant_docs: list[Document],
        streaming_callback: Callable | None = None,
    ) -> tuple[str, list[Any], dict[Any, str], str]:
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""
        priority_queue, distances = await self._initialize_traversal(relevant_docs)
        step = 0

        while priority_queue and len(traversal_path) < self.max_nodes_to_traverse:
            current_priority, current_node = heapq.heappop(priority_queue)

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
                is_sufficient_now,
            ) = self._process_node(
                current_node,
                query,
                expanded_context,
                traversal_path,
                visited_concepts,
                filtered_content,
                step,
                streaming_callback,
            )

            should_check_completion = (
                len(traversal_path) >= self.min_nodes_to_traverse
                and len(traversal_path) % self.answer_check_interval == 0
            )

            if is_sufficient_now and should_check_completion:
                final_answer = current_node_answer
                logger.info(
                    f"Found sufficient answer after {len(traversal_path)} nodes."
                )
                break

            self._explore_neighbors(
                current_node,
                current_priority,
                traversal_path,
                visited_concepts,
                distances,
                priority_queue,
            )
        else:
            self._log_traversal_end(traversal_path, priority_queue)

        if not final_answer:
            final_answer = self._generate_fallback_answer(query, expanded_context)

        return expanded_context, traversal_path, filtered_content, final_answer

    async def query(
        self, query: str, streaming_callback: Callable | None = None
    ) -> tuple[str, list[Any], dict[Any, str]]:
        logger.info(f"Starting query for: '{query}'")
        relevant_docs = await self.vector_store.retrieve_relevant_documents(query)
        if not relevant_docs:
            logger.warning("No initial relevant documents found from vector store.")
            return "No relevant information found.", [], {}

        await self._analyze_chunk_distribution(relevant_docs)

        _, traversal_path, filtered_content, final_answer = await self._expand_context(
            query, relevant_docs, streaming_callback
        )

        answer_text = (
            final_answer.content
            if hasattr(final_answer, "content")
            else str(final_answer)
        )
        logger.info(f"Final Answer: {answer_text[:200]}...")

        return final_answer, traversal_path, filtered_content

    async def _analyze_chunk_distribution(self, relevant_docs: list[Document]) -> float:
        if not relevant_docs:
            logger.warning("No relevant documents to analyze chunk distribution.")
            return 0.0

        chunk_lengths = [len(doc.page_content.split()) for doc in relevant_docs]
        avg_chunk_length = (
            sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0.0
        )

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
