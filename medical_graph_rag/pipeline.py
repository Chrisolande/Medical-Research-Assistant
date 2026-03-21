import logging

from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

from medical_graph_rag.config import (
    DEEPSEEK_API_KEY,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
)
from medical_graph_rag.graph import KnowledgeGraph
from medical_graph_rag.graph_viz import GraphVisualizer
from medical_graph_rag.rag_chain import QueryEngine
from medical_graph_rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Main:
    def __init__(
        self,
        cache_dir: str = "my_cache",
        embedding_model: HuggingFaceEmbeddings | None = None,
        ner_pipeline=None,
    ):
        """Initialize the main processing pipeline.

        Args:
            cache_dir: Directory path for knowledge graph caching
        """
        try:
            self.llm = self._initialize_llm()
            self.embedding_model = embedding_model or self._initialize_embeddings()
            self.knowledge_graph = KnowledgeGraph(
                cache_dir=cache_dir,
                embeddings=self.embedding_model,
                ner_pipeline=ner_pipeline,
            )
            self.vector_store = VectorStore(embeddings=self.embedding_model)
            self.query_engine = None
            self.visualizer = GraphVisualizer()
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_llm(self) -> ChatDeepSeek:
        """Initialize the LLM with configuration."""
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        return ChatDeepSeek(
            model=LLM_MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            temperature=0,
            max_retries=3,
        )

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize the embedding model."""
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
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
            self.knowledge_graph.build_knowledge_graph(documents)

            await self.vector_store.create_vector_index(documents)
            logger.info(f"Added {len(documents)} documents to vector store")

            # Initialize query engine
            self.query_engine = QueryEngine(
                self.vector_store, self.knowledge_graph, self.llm
            )

            logger.info("Document processing completed successfully")
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

    async def query(
        self, query: str, streaming_callback=None
    ) -> tuple[str, list | None, dict | None]:
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
                query, streaming_callback=streaming_callback
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
