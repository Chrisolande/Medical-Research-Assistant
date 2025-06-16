from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import heapq
from typing import List, Set, Dict, Tuple
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

class AnswerCheck(BaseModel):
    is_complete: bool = Field(description = "Whether the answer is complete or not")
    answer: str = Field(description = "The current answer based on the context")

class AnswerCheck(BaseModel):
    """Check if a query is fully answerable with the provided context."""

    is_complete: bool = Field(description="Whether the answer is complete or not")
    answer: str = Field(description="The current answer based on the context")


class QueryEngine:
    """Query engine for traversing the knowledge graph to answer a query."""

    def __init__(self, vector_store, knowledge_graph, llm):
        """Initialize the query engine."""
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm

        self.max_context_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """Create the answer check chain."""
        prompt = ChatPromptTemplate.from_template(
            """
            Given the query: "{query}"
            And the context:
            {context}
            Based on the provided context, is the query fully answerable? If yes, extract and provide the complete answer. If no, state "Incomplete Answer."
            Answerable: [Yes/No]
            Complete Answer (if Yes):
            """
        )
        return prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query, context):
        """Check if the query is fully answerable with the provided context."""
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer

    def _initialize_traversal(self, relevant_docs):
        """Initialize the graph traversal."""
        priority_queue = []
        distances = {}

        for doc in relevant_docs:
            closest_nodes_results = self.vector_store.similarity_search_with_score(
                doc.page_content, k=1
            )
            if not closest_nodes_results:
                continue
            closest_node_content, similarity_score = closest_nodes_results[0]
            closest_node = next(
                n for n, data in self.knowledge_graph.graph.nodes(data=True)
                if data["content"] == closest_node_content.page_content
            )

            if not closest_node:
                continue

            priority = 1 / (similarity_score + 1e-15)
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority
        return priority_queue, distances

    def _process_node(self, current_node, current_priority, query, expanded_context, traversal_path, visited_concepts, filtered_content, step):
        """Process a node in the graph traversal."""
        node_content = self.knowledge_graph.nodes[current_node]["content"]
        node_concepts = self.knowledge_graph.nodes[current_node]["concepts"]
        filtered_content[current_node] = node_content
        expanded_context += "\n" + node_content if expanded_context else node_content
        traversal_path.append(current_node)

        print(f"\nStep {step} - Node {current_node}:")
        print(f"Content: {node_content[:100]}...")
        print(f"Concepts: {', '.join(node_concepts)}")
        print("-" * 50)

        is_complete, answer = self._check_answer(query, expanded_context)
        if is_complete:
            return expanded_context, traversal_path, filtered_content, answer, True

        node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
        visited_concepts.update(node_concepts_set)
        return expanded_context, traversal_path, filtered_content, "", False

    def _explore_neighbors(self, current_node, current_priority, query, expanded_context, traversal_path, visited_concepts, filtered_content, distances, priority_queue):
        """Explore the neighbors of a node in the graph traversal."""
        for neighbor in self.knowledge_graph.graph.neighbors[current_node]:
            if neighbor in traversal_path:
                continue

            edge_data = self.knowledge_graph.graph[current_node][neighbor]
            edge_weight = edge_data["weight"]
            distance = current_priority + (1 / (edge_weight + 1e-15))
            if distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
        return expanded_context, filtered_content, "", False

    def _expand_context(self, query, relevant_docs):
        """Expand the context by traversing the knowledge graph."""
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""
        priority_queue, distances = self._initialize_traversal(relevant_docs)
        step = 0

        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)
            if current_priority > distances.get(current_node, float("inf")) or current_node in traversal_path:
                continue
            step += 1
            expanded_context, traversal_path, filtered_content, current_node_answer, is_complete_now = self._process_node(
                current_node, current_priority, query, expanded_context, traversal_path, visited_concepts, filtered_content, step
            )
            if is_complete_now:
                final_answer = current_node_answer
                break
            expanded_context, filtered_content, _, _ = self._explore_neighbors(
                current_node, current_priority, query, expanded_context, traversal_path, visited_concepts,
                filtered_content, distances, priority_queue
            )

        if not final_answer:
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query):
        """Query the knowledge graph."""
        with get_openai_callback() as cb:
            relevant_docs = self._retrieve_relevant_documents(query)
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)
            cb.print(f"Final Answer: {final_answer}")
            cb.print(f"Total Tokens: {cb.total_tokens}")
            cb.print(f"Prompt Tokens: {cb.prompt_tokens}")
            cb.print(f"Completion Tokens: {cb.completion_tokens}")
            cb.print(f"Total Cost (USD): ${cb.total_cost}")
        return final_answer, traversal_path, filtered_content

    def _retrieve_relevant_documents(self, query):
        """Retrieve relevant documents from the vector store."""
        retriever = self.vector_store.vector_index.as_retriever(search_kwargs={"k": 8})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        return compression_retriever.invoke(query)
