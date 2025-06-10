from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any, Optional, Tuple
from retrieval import Retriever
from config import OPENROUTER_API_KEY, MODEL_NAME

class RagChain:
    """Handle RAG Pipeline for the Graph RAG System"""

    def __init__(
        self, 
        retriever: Retriever,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):

        """Initialize RAG Chain"""
        self.retriever = retriever
        self.model_name = model or MODEL_NAME
        self.api_key = api_key or OPENROUTER_API_KEY

        self.llm = ChatOpenAI(model = self.model_name,
                                api_key = self.api_key,
                                openai_api_base = "https://openrouter.ai/api/v1",
                                temperature = 0,
                                streaming = True)

        self._condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
                in its original language.

                Chat History:
                {chat_history}
                Follow Up Input: {question}
                Standalone question:
                """
        
        self.CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(self._condense_template)
        
        # Create RAG prompt
        self._rag_template = """Answer the question based on the following context.
        If you don't know the answer, just say you don't know. DO NOT make up an answer.
        If the context doesn't contain the answer, just say the context doesn't contain the answer. DO NOT make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:"""

        self.RAG_PROMPT = ChatPromptTemplate.from_template(self._rag_template)

        # Create the chain
        self._create_chain()

    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
        """ Format the chat history for the condense question prompt"""
        
        formatted_history = ""
        for human, ai in chat_history:
            formatted_history += f"Human: {human}\nAI: {ai}\n"
        return formatted_history

    def _create_chain(self):
        """Create the RAG Chain"""
        self.condense_question_chain = (
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | RunnablePassthrough.assign(
                chat_history=lambda x: self._format_chat_history(x["chat_history"])
            )
            | self.CONDENSE_QUESTION_PROMPT
            | self.llm
            | StrOutputParser()
        )

        # Crete the qa chain
        self.qa_chain = (
            {
                "context": lambda x: self.retriever.hybrid_retrieval(x["question"]),
                "question": lambda x: x["question"],
            }
            | self.RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        # Create the full chain
        self.chain = (
            RunnablePassthrough.assign(
                question=lambda x: (
                    self.condense_question_chain.invoke(x)
                    if x.get("chat_history")
                    else x["question"]
                )
            )
            | self.qa_chain
        )

    def invoke(self, inputs: Dict[str, Any]) -> str:
        """Invoke the RAG Chain"""
        return self.chain.invoke(inputs)