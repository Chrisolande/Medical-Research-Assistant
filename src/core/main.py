import logging
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.knowledge_graph.graph_viz import GraphVisualizer
from src.knowledge_graph.knowledge_graph import KnowledgeGraph
from src.nlp.rag_chain import QueryEngine
from src.nlp.vectorstore import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Main:
    def __init__(self, cache_dir: str = "../my_cache"):
        """Initialize the main processing pipeline.

        Args:
            cache_dir: Directory path for knowledge graph caching
        """
        try:
            self.llm = self._initialize_llm()
            self.embedding_model = self._initialize_embeddings()
            self.knowledge_graph = KnowledgeGraph(cache_dir=cache_dir)
            self.vector_store = VectorStore()
            self.query_engine = None
            self.visualizer = GraphVisualizer()
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM with configuration."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        return ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct",
            api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            streaming=False,
            max_retries=3,
            request_timeout=30,
        )

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize the embedding model."""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  # or 'cuda' if available
            encode_kwargs={"normalize_embeddings": True},
        )

    async def process_documents(self, documents: list[dict]):
        """Process documents to build knowledge graph and vector store.

        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
        """
        try:
            logger.info(f"Processing {len(documents)} documents")

            # Build knowledge graph
            await self.knowledge_graph.build_knowledge_graph(documents, self.llm)

            # Initialize query engine
            self.query_engine = QueryEngine(
                self.vector_store, self.knowledge_graph, self.llm
            )

            logger.info("Document processing completed successfully")
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

    async def query(self, query: str) -> tuple[str, list | None, dict | None]:
        """Execute a query against the knowledge graph.

        Args:
            query: Natural language query string

        Returns:
            Tuple of (response, traversal_path, filtered_content)
        """
        try:
            if not self.query_engine:
                raise RuntimeError(
                    "Query engine not initialized - process documents first"
                )

            logger.info(f"Processing query: '{query}'")
            response, traversal_path, filtered_content = await self.query_engine.query(
                query
            )

            if traversal_path:
                await self.visualizer.visualize_traversal_async(
                    self.knowledge_graph.graph, traversal_path
                )
            else:
                logger.info("No traversal path to visualize")

            return response, traversal_path, filtered_content

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise
