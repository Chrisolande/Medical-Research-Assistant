import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.knowledge_graph.knowledge_graph import KnowledgeGraph
from src.nlp.rag_chain import QueryEngine

# from visualization import Visualizer
from src.nlp.vectorstore import VectorStore


class Main:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            streaming=False,
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.knowledge_graph = KnowledgeGraph(cache_dir="../my_cache")
        self.query_engine = None
        # self.visualizer = Visualizer()?

    def process_documents(self, documents):
        self.vector_store = VectorStore()
        self.knowledge_graph.build_knowledge_graph(documents, self.llm)
        self.query_engine = QueryEngine(
            self.vector_store, self.knowledge_graph, self.llm
        )
