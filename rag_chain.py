from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import asyncio
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.runnables.base import RunnableLambda 
import os
from langsmith import traceable
from langchain.callbacks.manager import trace_as_chain_group
from langsmith import Client

os.environ["LANGCHAIN_PROJECT"] = "simple-rag-demo"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


class RAGChain:
    """RAG chain with conversation memory using the existing retriever and PubMed fallback."""

    def __init__(self, retriever, llm, max_memory: int = 10):
        self.retriever = retriever
        self.llm = llm
        self.memory: List[Dict[str, str]] = []
        self.max_memory = max_memory
        self.client = Client()
        # Initialize the PubMed tool
        self.pubmed_tool = PubmedQueryRun()

        # Updated prompt to guide LLM on tool usage and uncertainty
        self.prompt = ChatPromptTemplate.from_template("""
        **Your Goal:** Provide a concise, accurate, and helpful answer to the user's question.

        **Instructions:**
        1.  **Prioritize Context:** First, attempt to answer the **Current question** using *only* the **Context** provided below.
        2.  **Refer to History:** Consider the **Previous conversation** for continuity and to understand the user's intent.
        3.  **Directly Answer:** Formulate a direct answer to the **Current question**.
        4.  **No Outside Knowledge (Initial Pass):** Do NOT use any information outside of the given Context and Previous conversation *in your initial attempt*.
        5.  **Tool Usage (Fallback):**
            * If you **cannot find a relevant answer** in the provided Context or Previous conversation, then indicate this and state that you will attempt to use the PubMed tool.
            * When using the tool, specify the search query clearly. **Format your tool use response like this:
                `TOOL_USE: pubmed_tool.run("<search query>")`**
                Replace `<search query>` with a concise, effective query based on the user's original question.
            * After receiving the tool output, provide an answer based on the tool's results. If the tool also fails to provide a direct answer, state that explicitly.
        6.  **Uncertainty (Final Fallback):** If, even after attempting tool use, you still cannot find the answer, state clearly: "I cannot find the answer to your question in any of the provided information or tools."

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

    @traceable(run_type = "chain", name = "RAGChain.invoke", project_name = "simple-rag-demo")
    async def invoke(self, question: str, k_vector: int = 3) -> str:
        """Process question with retrieval and memory context, with PubMed fallback."""

        # Retrieve context from internal RAG

        with trace_as_chain_group("Retrieval"):
            context = await self.retriever.hybrid_retrieval(question, k_vector=k_vector)

        # Format prompt for the LLM
        formatted_prompt = self.prompt.format(
            context=context,
            history=self._format_history(),
            question=question
        )

        # Initial LLM call
        response = await asyncio.to_thread(self.llm.with_config(tags=["initial_llm_call"]).invoke, formatted_prompt)
        initial_answer = response.content if hasattr(response, 'content') else str(response)

        # Check for tool use signal
        if "TOOL_USE:" in initial_answer:
            try:
                # Extract the tool call
                tool_call_str = initial_answer.strip().replace("TOOL_USE:", "").strip()
                
                # Safely parse the tool call
                if tool_call_str.startswith("pubmed_tool.run("):
                    search_query = tool_call_str[len("pubmed_tool.run("):-1].strip('"')
                    print(f"DEBUG: Attempting PubMed search with query: '{search_query}'")
                    # Execute the PubMed tool asynchronously
                    pubmed_results = await asyncio.to_thread(self.pubmed_tool.run, search_query) 
                    print(f"DEBUG: PubMed results received. Length: {len(pubmed_results)}")

                    # Re-prompt the LLM with PubMed results
                    reprompt_template = ChatPromptTemplate.from_template("""
                    You previously tried to answer a question but needed to use the PubMed tool. Here are the results from your PubMed search:

                    **PubMed Search Results:**
                    {pubmed_results}

                    **Original Context:**
                    {context}

                    **Previous conversation:**
                    {history}

                    **Original question:**
                    {question}

                    Based on these PubMed search results, the original context, and the previous conversation, provide a concise and accurate answer. If the PubMed results also do not contain the answer, state that you cannot find the answer.

                    **Answer:**
                    """)
                    final_prompt_with_pubmed = reprompt_template.format(
                        pubmed_results=pubmed_results,
                        context=context,
                        history=self._format_history(),
                        question=question
                    )
                    final_response = await asyncio.to_thread(self.llm.invoke, final_prompt_with_pubmed)
                    answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
                else:
                    # If tool call format is unexpected, treat initial answer as final
                    answer = initial_answer
            except Exception as e:
                print(f"ERROR: Failed to use PubMed tool: {e}")
                answer = f"An error occurred while trying to use the PubMed tool: {e}\nOriginal attempt: {initial_answer}"
        else:
            answer = initial_answer # No tool use needed, initial answer is final

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