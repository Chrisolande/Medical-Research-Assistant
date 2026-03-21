import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

import nest_asyncio
import streamlit as st
from langchain.globals import set_llm_cache
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from medical_graph_rag.batch_processor import PMCBatchProcessor
from medical_graph_rag.config import (
    DEFAULT_MAX_DISTANCE_THRESHOLD,
    NCBI_EMAIL,
    PUBMED_DEFAULT_MAX_RESULTS,
    PUBMED_MAX_RESULTS_LIMIT,
)
from medical_graph_rag.document_processor import DocumentProcessor
from medical_graph_rag.pipeline import Main
from medical_graph_rag.utils import ensure_semantic_cache
from streaming import StreamingNodeDisplay

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUGGESTIONS = [
    "What are the effects of the Gaza war on children?",
    "How does Covid-19 affect mental health?",
    "What are the latest treatments for diabetes?",
    "How has machine learning revolutionized health care?",
]


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> HuggingFaceEmbeddings:
    from medical_graph_rag.config import EMBEDDING_MODEL_NAME

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Loading NER pipeline...")
def load_ner_pipeline():
    from transformers import pipeline

    return pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        tokenizer="d4data/biomedical-ner-all",
        aggregation_strategy="first",
    )


@dataclass
class ConversationEntry:
    query: str
    response: str
    timestamp: datetime
    traversal_path: list | None = None
    filtered_content: dict | None = None


@dataclass
class AppState:
    main: Main | None = None
    documents_processed: bool = False
    cache_dir: str = "my_cache"
    default_data_path: str = "data/output/processed_pmc_data/pmc_chunks.json"
    conversation_history: list[ConversationEntry] = field(default_factory=list)
    streaming_display: StreamingNodeDisplay = field(
        default_factory=StreamingNodeDisplay
    )
    use_cache: bool = False
    max_distance_threshold: float = field(default=DEFAULT_MAX_DISTANCE_THRESHOLD)


def get_state() -> AppState:
    """Initialize or retrieve session state."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    return st.session_state.app_state


# File helpers


def _is_chunked_schema(data) -> bool:
    return (
        isinstance(data, dict)
        and "documents" in data
        and isinstance(data["documents"], list)
        and all(
            isinstance(doc, dict) and "content" in doc and "metadata" in doc
            for doc in data["documents"]
        )
    )


def _is_raw_pmc_schema(data) -> bool:
    return isinstance(data, list) and all(
        isinstance(doc, dict) and "abstract" in doc for doc in data
    )


def validate_json_file(file_path) -> tuple[bool, bool, dict, dict]:
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if _is_chunked_schema(data):
            return True, True, data.get("processing_info", {}), data.get("summary", {})

        if _is_raw_pmc_schema(data):
            return True, False, {}, {}

    except Exception as e:
        logger.error(f"Error validating JSON file: {str(e)}")

    return False, False, {}, {}


def load_chunked_docs(file_path, progress_bar) -> list[Document]:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    processed_docs = [
        Document(page_content=doc["content"], metadata=doc["metadata"])
        for doc in data["documents"]
        if doc["content"].strip()
    ]

    st.write(f"Loaded {len(processed_docs)} pre-chunked documents from {file_path}")
    progress_bar.progress(1.0, text="Loaded pre-chunked documents")
    return processed_docs


async def process_raw_docs(
    file_path: str, progress_bar, cache_dir: str
) -> list[Document]:
    """Process raw PMC data."""
    document_processor = DocumentProcessor()
    batch_processor = PMCBatchProcessor(document_processor=document_processor)

    def progress_callback(completed, total, result):
        progress_bar.progress(
            completed / total, text=f"Processing batch {completed}/{total}"
        )
        if result["success"]:
            st.write(
                f"Batch {result['batch_num']}: {result['chunk_count']} chunks "
                f"from {result['original_count']} documents"
            )
        else:
            st.error(f"Batch {result['batch_num']} failed: {result['error']}")

    results = await batch_processor.process_pmc_file_async(
        file_path=file_path, progress_callback=progress_callback
    )

    os.makedirs(cache_dir, exist_ok=True)
    batch_processor.save_results(results, cache_dir)
    st.write("### Processing Summary")
    st.json(results["processing_summary"])
    return results["all_documents"]


async def ingest_file(file_path: str, progress_bar, state: AppState) -> None:
    try:
        is_valid, is_chunked, processing_info, summary = validate_json_file(file_path)
        if not is_valid:
            st.error(
                f"Invalid JSON file structure at {file_path}. Expected a 'documents' "
                "key with a list of objects containing 'content' and 'metadata' or "
                "a list of objects with 'abstract'."
            )
            return

        if is_chunked:
            processed_docs = load_chunked_docs(file_path, progress_bar)
            if processing_info or summary:
                st.write("### File Processing Info")
                if processing_info:
                    st.json(processing_info)
                if summary:
                    st.json(summary)
        else:
            processed_docs = await process_raw_docs(
                file_path, progress_bar, state.cache_dir
            )

        await state.main.process_documents(processed_docs)
        state.documents_processed = True
        st.success(f"Processed {len(processed_docs)} document chunks successfully!")

    except Exception as e:
        st.error(f"Error while processing the json file: {str(e)}")
        logger.error(f"Error while processing the json file: {str(e)}")


# Query


async def run_query(query: str, state: AppState):
    try:
        state.streaming_display.start_streaming()

        def streaming_callback(step, node_id, content, concepts):
            state.streaming_display.add_node(step, node_id, content, concepts)

        response, traversal_path, filtered_content = await state.main.query(
            query, streaming_callback=streaming_callback
        )

        state.streaming_display.stop_streaming()

        conversation_entry = ConversationEntry(
            query=query,
            response=(
                response.content if hasattr(response, "content") else str(response)
            ),
            timestamp=datetime.now(),
            traversal_path=traversal_path,
            filtered_content=filtered_content,
        )
        state.conversation_history.append(conversation_entry)
        return response, traversal_path, filtered_content

    except Exception as e:
        state.streaming_display.stop_streaming()
        st.error(f"Error during query processing: {str(e)}")
        logger.error(f"Error during query processing: {str(e)}")
        return None, None, None


# Sidebar sections


def render_api_section(state: AppState) -> None:
    st.markdown("### :key: API Configuration")

    api_key_exists = bool(os.getenv("DEEPSEEK_API_KEY"))
    if api_key_exists:
        st.success(":white_check_mark: API key found in environment")
    else:
        st.warning(":warning: API key not found in environment")

        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="Enter your DeepSeek API key",
        )

        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            st.success(":white_check_mark: API key set successfully")

            if state.main:
                st.info("Reinitializing the entire system with the new API key")
                try:
                    state.main = Main(
                        cache_dir=state.cache_dir,
                        embedding_model=load_embedding_model(),
                        ner_pipeline=load_ner_pipeline(),
                    )
                    st.success("Pipeline reinitialized with the new API key")
                except Exception as e:
                    st.error(f"Failed to reinitialize: {str(e)}")


def render_pipeline_control(state: AppState) -> None:
    st.markdown("### :gear: Pipeline Control")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize Pipeline", use_container_width=True):
            try:
                state.main = Main(
                    cache_dir=state.cache_dir,
                    embedding_model=load_embedding_model(),
                    ner_pipeline=load_ner_pipeline(),
                )
                st.success("Pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {str(e)}")

    with col2:
        if state.main and st.button("Reset Pipeline", use_container_width=True):
            state.main = None
            state.documents_processed = False
            st.rerun()


def render_cache_settings(state: AppState) -> None:
    st.markdown("### :control_knobs: Settings")
    new_cache = st.toggle(
        ":floppy_disk: Use Cache",
        value=state.use_cache,
        help="Enable/disable caching for faster repeated queries",
    )
    new_threshold = st.slider(
        ":dart: Max Semantic Distance",
        min_value=0.0,
        max_value=3.0,
        value=state.max_distance_threshold,
        step=0.05,
        help="Lower values = stricter semantic cache matching (L2 distance)",
    )

    if new_cache != state.use_cache or new_threshold != state.max_distance_threshold:
        state.use_cache = new_cache
        state.max_distance_threshold = new_threshold

        if state.use_cache:
            ensure_semantic_cache(max_distance_threshold=state.max_distance_threshold)
            st.success(
                f"Semantic cache enabled with max distance {state.max_distance_threshold}"
            )
        else:
            set_llm_cache(None)
            st.success("Semantic cache disabled")

        st.rerun()


def render_load_documents(state: AppState) -> None:
    """Render default file loading section."""
    st.markdown("### :open_file_folder: Load Documents")
    status_color = "🟢 " if state.documents_processed else ":red_circle: "
    st.markdown(
        f"{status_color} **Status:** {'Loaded' if state.documents_processed else 'Not Loaded'}"
    )
    if st.button(":page_with_curl: Load pmc_chunks.json", use_container_width=True):
        if state.main:
            if os.path.exists(state.default_data_path):
                progress_bar = st.progress(0, text="Starting processing...")
                with st.spinner(f"Processing {state.default_data_path}..."):
                    asyncio.run(
                        ingest_file(state.default_data_path, progress_bar, state)
                    )
                progress_bar.empty()
                st.rerun()
            else:
                st.error(f"File not found: {state.default_data_path}")
        else:
            st.warning("Please initialize the pipeline first.")


def render_custom_upload(state: AppState) -> None:
    st.markdown("### :outbox_tray: Upload Custom JSON")
    uploaded_file = st.file_uploader(
        "Upload JSON file with medical documents",
        type=["json"],
        help="Supports both raw PMC data and pre-chunked documents",
    )
    if uploaded_file and state.main:
        temp_file_path = os.path.join(state.cache_dir, uploaded_file.name)
        os.makedirs(state.cache_dir, exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        progress_bar = st.progress(0, text="Starting processing...")
        with st.spinner("Processing uploaded file..."):
            asyncio.run(ingest_file(temp_file_path, progress_bar, state))

        progress_bar.empty()
        os.remove(temp_file_path)
        st.rerun()


async def _fetch_and_ingest_pubmed(
    query: str, max_results: int, email: str, state: AppState
) -> None:
    from medical_graph_rag.pubmed_downloader import PubMedEntrezDownloader

    downloader = PubMedEntrezDownloader(email=email)

    with st.spinner(f"Searching PubMed for '{query}'..."):
        pmids = await downloader.search_pubmed(query=query, max_results=max_results)

    if not pmids:
        st.warning("No results found for that query.")
        return

    st.info(f"Found {len(pmids)} articles. Fetching abstracts...")

    with st.spinner("Fetching article details..."):
        articles = await downloader.fetch_article_details(pmids)

    articles_with_abstract = [a for a in articles if a.get("abstract", "").strip()]

    if not articles_with_abstract:
        st.warning("No articles with abstracts found.")
        return

    st.info(
        f"Fetched {len(articles)} articles, "
        f"{len(articles_with_abstract)} have abstracts."
    )

    os.makedirs(state.cache_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=state.cache_dir,
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(articles_with_abstract, tmp, ensure_ascii=False)
        tmp_path = tmp.name

    bar = st.progress(0, text="Processing articles...")

    def on_batch(completed, total, result):
        bar.progress(
            completed / total,
            text=f"Processing batch {completed}/{total}",
        )
        if not result["success"]:
            st.error(f"Batch {result['batch_num']} failed: {result['error']}")

    try:
        processor = PMCBatchProcessor(document_processor=DocumentProcessor())
        results = await processor.process_pmc_file_async(
            file_path=tmp_path, progress_callback=on_batch
        )
        bar.empty()
        with st.status(
            "Building knowledge graph - this may take a few minutes...",
            expanded=True,
        ) as status:
            st.write(
                f":brain: Embedding {len(results['all_documents'])} document chunks "
                "and extracting biomedical concepts..."
            )
            await state.main.process_documents(results["all_documents"])
            status.update(
                label=":white_check_mark: Knowledge graph built successfully!",
                state="complete",
                expanded=False,
            )
        state.documents_processed = True
        st.success(
            f"Ingested {len(results['all_documents'])} chunks "
            f"from {len(articles_with_abstract)} PubMed articles."
        )
        st.json(results["processing_summary"])
    finally:
        os.remove(tmp_path)


def render_pubmed_search(state: AppState) -> None:
    st.markdown("### :microscope: Search PubMed")

    ncbi_email = NCBI_EMAIL
    if not ncbi_email:
        ncbi_email = st.text_input(
            "NCBI Email",
            placeholder="your@email.com",
            help="Required by NCBI Entrez. Not stored.",
        )

    query = st.text_input(
        "Search query",
        placeholder="e.g. COVID-19 mental health children",
    )
    max_results = st.slider(
        "Max articles",
        min_value=10,
        max_value=PUBMED_MAX_RESULTS_LIMIT,
        value=PUBMED_DEFAULT_MAX_RESULTS,
        step=10,
    )

    if st.button(":mag: Fetch and Ingest", use_container_width=True):
        if not state.main:
            st.warning("Initialize the pipeline first.")
            return
        if not ncbi_email:
            st.error("NCBI email is required.")
            return
        if not query.strip():
            st.error("Enter a search query.")
            return
        asyncio.run(_fetch_and_ingest_pubmed(query, max_results, ncbi_email, state))
        st.rerun()


def render_sidebar(state: AppState) -> None:
    """Render sidebar with configuration and file loading options."""
    with st.sidebar:
        st.markdown("# :dna: Medical RAG Config")
        render_api_section(state)
        st.divider()
        render_pipeline_control(state)
        st.divider()
        render_cache_settings(state)
        st.divider()
        render_load_documents(state)
        st.divider()
        render_custom_upload(state)
        st.divider()
        render_pubmed_search(state)


# Main content


def render_conversation_history(state: AppState) -> None:
    if state.conversation_history:
        with st.expander(
            f":speech_balloon: Conversation History ({len(state.conversation_history)} queries)",
            expanded=False,
        ):
            for i, conv in enumerate(reversed(state.conversation_history[-5:])):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(
                        f"**Q{len(state.conversation_history) - i}:** {conv.query}"
                    )
                    st.write(f"**A:** {conv.response[:200]}...")
                with col2:
                    st.caption(conv.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                st.divider()

            if st.button(":wastebasket: Clear History"):
                state.conversation_history = []
                st.rerun()


def render_query_results(
    response, traversal_path, filtered_content, state: AppState
) -> None:
    st.subheader(":clipboard: Query Response")
    st.write(response.content if hasattr(response, "content") else str(response))

    if traversal_path:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(":world_map: Traversal Path")
            st.write(f"Nodes traversed: {traversal_path}")

            st.subheader(":bar_chart: Traversal Subgraph Statistics")
            stats = state.main.knowledge_graph.get_subgraph_stats(traversal_path)
            st.json(stats)

        with col2:
            st.subheader(":page_with_curl: Relevant Content")
            for node_id, content in filtered_content.items():
                with st.expander(f"Node {node_id}"):
                    st.write(content[:400] + "..." if len(content) > 400 else content)

        st.subheader("🕸️ Graph Visualization")
        try:
            graph_image_buffer = state.main.visualizer.visualize_traversal(
                state.main.knowledge_graph.graph, traversal_path
            )
            if graph_image_buffer:
                st.image(graph_image_buffer, caption="Knowledge Graph Traversal")
            else:
                st.warning("No visualization generated.")
        except Exception as e:
            st.error(f"Failed to visualize graph: {str(e)}")
            logger.error(f"Failed to visualize graph: {str(e)}")


def render_main_content(state: AppState) -> None:
    """Render main content area."""
    if state.main:
        st.header(":mag: Query the Knowledge Graph")
        render_conversation_history(state)

        st.markdown("### :brain: Ask a Question")
        selected_suggestion = st.selectbox(
            ":bulb: Quick suggestions (optional):",
            [""] + SUGGESTIONS,
            help="Select a suggestion or type your own query below",
        )
        query = st.text_input(
            "Enter your query:",
            value=selected_suggestion,
            placeholder="Enter your medical research question ...",
        )

        if query and state.documents_processed:
            with st.spinner(":mag: Processing query..."):
                response, traversal_path, filtered_content = asyncio.run(
                    run_query(query, state)
                )
                if response:
                    render_query_results(
                        response, traversal_path, filtered_content, state
                    )
                else:
                    st.warning(
                        "No response generated. Please check the query or document processing."
                    )
        elif query and not state.documents_processed:
            st.warning("Please load or upload and process documents before querying.")
    else:
        st.info(":point_left: Please initialize the pipeline from the sidebar.")


def main() -> None:
    st.set_page_config(
        page_title="Medical RAG Knowledge Graph",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    state = get_state()
    st.title(":dna: Medical RAG Knowledge Graph Explorer")
    render_sidebar(state)
    render_main_content(state)


if __name__ == "__main__":
    main()
