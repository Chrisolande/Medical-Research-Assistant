from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import asyncio

class RAGChain:
    """RAG chain with conversation memory using the existing retriever."""
    
    def __init__(self, retriever, llm, max_memory: int = 10):
        self.retriever = retriever
        self.llm = llm
        self.memory: List[Dict[str, str]] = []
        self.max_memory = max_memory
        
        self.prompt = ChatPromptTemplate.from_template("""
        **Your Goal:** Provide a concise, accurate, and helpful answer to the user's question based *only* on the provided context and previous conversation.

        **Instructions:**
        1.  **Prioritize Context:** Use the **Context** provided below to formulate your answer.
        2.  **Refer to History:** Consider the **Previous conversation** for continuity and to understand the user's intent if the current question is ambiguous.
        3.  **Directly Answer:** Formulate a direct answer to the **Current question**.
        4.  **No Outside Knowledge:** Do NOT use any information outside of the given Context and Previous conversation.
        5.  **Uncertainty:** If the answer is not found in the Context or Previous conversation, state clearly: "I cannot find the answer to your question in the provided information." Do NOT make up an answer.

        ---
        **Context:**
        {context}

        ---
        **Previous conversation:**
        {history}

        ---
        **Current question:**
        {question}

        **Answer:**
        """)

    def _format_history(self) -> str:
        """Format conversation history."""
        if not self.memory:
            return "No previous conversation."
        
        formatted = []
        for turn in self.memory[-self.max_memory:]:
            formatted.append(f"Human: {turn['question']}")
            formatted.append(f"Assistant: {turn['answer']}")
        return "\n".join(formatted)

    def _add_to_memory(self, question: str, answer: str):
        """Add Q&A pair to memory with size limit."""
        self.memory.append({"question": question, "answer": answer})
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    async def invoke(self, question: str, k_vector: int = 3) -> str:
        """Process question with retrieval and memory context."""
        
        # Retrieve context
        context = await self.retriever.hybrid_retrieval(question, k_vector=k_vector)
        
        # Format prompt with context and history
        formatted_prompt = self.prompt.format(
            context=context,
            history=self._format_history(),
            question=question
        )
        
        # Generate response
        response = await asyncio.to_thread(self.llm.invoke, formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Update memory
        self._add_to_memory(question, answer)
        
        return answer

    async def batch_invoke(self, questions: List[str], k_vector: int = 3) -> List[str]:
        """Process multiple questions with shared memory context."""
        results = []
        for question in questions:
            result = await self.invoke(question, k_vector=k_vector)
            results.append(result)
        return results

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()

    @classmethod
    async def create(cls, retriever, llm, max_memory: int = 10):
        """Factory method for async initialization."""
        return cls(retriever, llm, max_memory)