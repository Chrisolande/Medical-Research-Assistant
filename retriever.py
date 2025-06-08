from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from typing import List
from langchain_core.documents import Document

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

    def structured_retrieval(self, question: str) -> str:
        """ Retrieve information from the knowledge graph using direct graph queries"""
        
        # Instead of entity extraction, directly query the graph
        # using the question text for relevant patterns
        result = ""
        
        # directly query the graph for relevant relationships
        try:
            # Execute a simpler Cypher query that doesn't rely on fulltext search
            response = self.knowledge_graph.query(
                """
                MATCH (n:__Entity__)-[r]-(m:__Entity__)
                WHERE n.id CONTAINS $keyword OR m.id CONTAINS $keyword
                RETURN n.id + " - " + type(r) + " -> " + m.id AS output
                LIMIT 20
                """,
                {"keyword": question.lower()}
            )
            
            # If we got results, return them
            if response:
                result = "\n".join([el['output'] for el in response])
            else:
                result = "No relevant information found in the knowledge graph."
                
        except Exception as e:
            result = f"Error querying knowledge graph: {str(e)}"
            
        return result

    def vector_retrieval(self, question: str, k: int = 4) -> List[Document]:
        """Retrieve information using vector similarity search."""
        return self.vector_store.similarity_search(question)

    def hybrid_retrieval(self, question: str, k_graph: int = 1, k_vector: int = 3) -> str:
        """
        Perform hybrid retrieval combining graph and vector search.
        
        """
        # Get structured data from knowledge graph
        structured_data = self.structured_retrieval(question)
        
        # Get unstructured data from vector store
        unstructured_docs = self.vector_retrieval(question, k=k_vector)
        unstructured_data = [doc.page_content for doc in unstructured_docs]
        
        # Combine results
        final_data = f"""Structured data:
            {structured_data}

            Unstructured data:
            {"#Document ".join(unstructured_data)}
            """
        return final_data