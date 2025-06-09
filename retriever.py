from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from typing import List
from langchain_core.documents import Document
import asyncio

class Retriever: 
    """ Handles retrieval operations for the graph rag system"""
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        vector_store: VectorStore
    ):

        """Initialize the retriever"""
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store

    async def structured_retrieval(self, question: str) -> str:
        """ Retrieve information from the knowledge graph using direct graph queries"""
        
        try:
            # Execute a simpler Cypher query that doesn't rely on fulltext search
            primary_query = asyncio.to_thread(
                self.knowledge_graph.query,
                """
                MATCH (n:__Entity__)-[r]-(m:__Entity__)
                WHERE toLower(n.id) CONTAINS $keyword OR toLower(m.id) CONTAINS $keyword
                RETURN n.id + " - " + type(r) + " -> " + m.id AS output
                LIMIT 10
                """,
                {"keyword": question.lower()}
            )

            secondary_query = asyncio.to_thread(
                self.knowledge_graph.query,
                """
                MATCH (n:__Entity__)-[r1]-(bridge:__Entity__)-[r2]-(m:__Entity__)
                WHERE toLower(n.id) CONTAINS $keyword
                RETURN n.id + " - " + type(r1) + " -> " + bridge.id + " - " + type(r2) + " -> " + m.id AS output
                LIMIT 5
                """,
                {"keyword": question.lower()}
            )

            primary_result, secondary_result = await asyncio.gather(
                primary_query, secondary_query, return_exceptions=True
            )
            
            # Combine results
            all_outputs = []
            for result_set in [primary_result, secondary_result]:
                if isinstance(result_set, Exception):
                    continue
                if result_set:
                    all_outputs.extend([el['output'] for el in result_set])
            
            return "\n".join(all_outputs) if all_outputs else "No relevant information found in the knowledge graph."
                
        except Exception as e:
            return f"Error querying knowledge graph: {str(e)}"

    async def vector_retrieval(self, question: str, k: int = 4) -> List[Document]:
        """Retrieve information using vector similarity search."""
        if self.vector_store.vector_index is None:
            raise ValueError("Vector index not initialized. Call create_vector_index or create_hybrid_index first.")
        return await self.vector_store.similarity_search(question, k=k)

    async def hybrid_retrieval(self, question: str, k_graph: int = 1, k_vector: int = 3) -> str:
        """
        Perform hybrid retrieval combining graph and vector search.
        
        """
        # Get structured data from knowledge graph
        structured_task = self.structured_retrieval(question)
        vector_task = self.vector_retrieval(question, k=k_vector)

        # Run both searches concurrently
        structured_data, unstructured_docs = await asyncio.gather(
            structured_task,
            vector_task,
            return_exceptions=True # Skips exceptions whenever its running them asynchronously returns them as part of the result
        )

        # Handle exceptions
        if isinstance(structured_data, Exception):
            structured_data = f"Graph search error: {str(structured_data)}"
        
        if isinstance(unstructured_docs, Exception):
            unstructured_data = [f"Vector search error: {str(unstructured_docs)}"]
        else:
            unstructured_data = [doc.page_content for doc in unstructured_docs]

        # Combine results
        final_data = f"""Structured data:
        {structured_data}

        Unstructured data:
        {"#Document ".join(unstructured_data)}
        """
        return final_data

    @classmethod
    async def create(cls, kg: KnowledgeGraph, vs: VectorStore):
        """Ensure proper initialization"""
        # Ensure vector store is properly set up
        if vs.vector_index is None:
            raise ValueError("VectorStore must have vector_index initialized before creating retriever")
        return cls(kg, vs)
    
    # batch retrieval for multiple queries
    # TODO: Evaluate if I will have to remove the batch queries implemented before
    async def batch_retrieval(self, questions: List[str], k_vector: int = 3) -> List[str]:
        """Process multiple questions concurrently for maximum throughput."""
        tasks = [self.hybrid_retrieval(q, k_vector=k_vector) for q in questions]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_system_stats(self):
        """Get comprehensive stats from both systems"""
        return {
            "knowledge_graph": self.knowledge_graph.get_system_stats(),
            "vector_store": self.vector_store.get_system_stats()
        }