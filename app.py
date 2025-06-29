import asyncio
import json
import logging
import os

import streamlit as st
from langchain_core.documents import Document

from medical_graph_rag.core.main import Main
from medical_graph_rag.data_processing.batch_processor import PMCBatchProcessor
from medical_graph_rag.data_processing.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="Medical RAG Knowledge Graph", layout="wide")

# Initialize session state
if "main" not in st.session_state:
    st.session_state.main = None
    st.session_state.documents_processed = False
    st.session_state.cache_dir = "/home/olande/Desktop/FinalRAG/my_cache"
    st.session_state.default_data_path = (
        "data/output/processed_pmc_data/pmc_chunks.json"
    )


def initialize_pipeline():
    try:
        st.session_state.main = Main(cache_dir=st.session_state.cache_dir)
        st.success("Pipeline initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")


def validate_json_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Check for pre-chunked structure: "documents" key with "content" and "metadata"
        if (
            isinstance(data, dict)
            and "documents" in data
            and isinstance(data["documents"], list)
        ):
            if all(
                isinstance(doc, dict) and "content" in doc and "metadata" in doc
                for doc in data["documents"]
            ):
                return (
                    True,
                    True,
                    data.get("processing_info", {}),
                    data.get("summary", {}),
                )

        # Check for raw PMC data structure
        if isinstance(data, list) and all(
            isinstance(doc, dict) and "abstract" in doc for doc in data
        ):
            return True, False, {}, {}

    except Exception as e:
        logger.error(f"Error validating JSON file: {str(e)}")
        return False, False, {}, {}


async def process_file(file_path, progress_bar):
    try:
        # Validate file
        is_valid, is_chunked, processing_info, summary = validate_json_file(file_path)
        if not is_valid:
            st.error(
                f"Invalid JSON file structure at {file_path}. Expected a 'documents' key with a list of objects containing 'content' and 'metadata' or a list of objects with 'abstract'."
            )
            return

        if is_chunked:
            # Load pre-chunked data
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            processed_docs = [
                Document(page_content=doc["content"], metadata=doc["metadata"])
                for doc in data["documents"]
                if doc["content"].strip()
            ]

            st.write(
                f"Loaded {len(processed_docs)} pre-chunked documents from {file_path}"
            )
            progress_bar.progress(1.0, text="Loaded pre-chunked documents")

            # Display processing info and summary
            if processing_info or summary:
                st.write("### File Processing Info")
                if processing_info:
                    st.json(processing_info)
                if summary:
                    st.json(summary)
        # Use the PMCBatch processor for raw JSON processing
        else:
            document_processor = DocumentProcessor()
            batch_processor = PMCBatchProcessor(document_processor=document_processor)

            # Progress callback
            def progress_callback(completed, total, result):
                progress = completed / total
                progress_bar.progress(
                    progress, text=f"Processing batch {completed}/{total}"
                )
                if result["success"]:
                    st.write(
                        f"Batch {result['batch_num']}: {result['chunk_count']} chunks from {result['original_count']} documents"
                    )
                else:
                    st.error(f"Batch {result['batch_num']} failed: {result['error']}")

            # Process the file
            results = await batch_processor.process_pmc_file_async(
                file_path=file_path, progress_callback=progress_callback
            )
            processed_docs = results["all_documents"]

            # Save results
            os.makedirs(st.session_state.cache_dir, exist_ok=True)
            batch_processor.save_results(results, st.session_state.cache_dir)
            st.write("### Processing Summary")
            st.json(results["processing_summary"])

        # Build knowledge graph and vector store
        await st.session_state.main.process_documents(processed_docs)
        st.session_state.documents_processed = True
        st.success(f"Processed {len(processed_docs)} document chunks successfully!")

    except Exception as e:
        st.error(f"Error while processing the json file: {str(e)}")
        logger.error(f"Error while processing the json file: {str(e)}")


async def handle_query(query):
    try:
        response, traversal_path, filtered_content = await st.session_state.main.query(
            query
        )
        return response, traversal_path, filtered_content
    except Exception as e:
        st.error(f"Error during query processing: {str(e)}")
        logger.error(f"Error during query processing: {str(e)}")
        return None, None, None


def main():
    st.title("Medical RAG Knowledge Graph Explorer")
    with st.sidebar:
        st.header("Configuration")
        if st.button("Initialize Pipeline"):
            initialize_pipeline()

        st.header("Load documents")
        # Button to load default file
        if st.button("Load pmc_chunks.json"):
            if st.session_state.main:
                default_path = st.session_state.default_data_path
                if os.path.exists(default_path):
                    progress_bar = st.progress(0, text="Starting processing...")
                    with st.spinner(f"Processing {default_path}..."):
                        asyncio.run(process_file(default_path, progress_bar))
                    progress_bar.empty()
                else:
                    st.error(f"File not found: {default_path}")
            else:
                st.warning("Please initialize the pipeline first.")

        # File uploader for other JSON files
        st.header("Upload Custom JSON")
        uploaded_file = st.file_uploader(
            "Upload JSON file with medical documents", type=["json"]
        )
        if uploaded_file and st.session_state.main:
            temp_file_path = os.path.join(
                st.session_state.cache_dir, uploaded_file.name
            )
            os.makedirs(st.session_state.cache_dir, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            progress_bar = st.progress(0, text="Starting processing...")
            with st.spinner("Processing uploaded file..."):
                asyncio.run(process_file(temp_file_path, progress_bar))
            progress_bar.empty()
            os.remove(temp_file_path)

    # Main content area
    if st.session_state.main:
        st.header("Query the Knowledge Graph")
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What are the effects of the Gaza war on children?",
        )

        if query and st.session_state.documents_processed:
            with st.spinner("Processing query..."):
                response, traversal_path, filtered_content = asyncio.run(
                    handle_query(query)
                )

                if response:
                    st.subheader("Query Response")
                    st.write(
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    # Get the traversal path
                    if traversal_path:
                        st.subheader("Traversal Path")
                        st.write(f"Nodes traversed: {traversal_path}")

                        # Display the filtered content
                        st.subheader("Relevant Content")
                        for node_id, content in filtered_content.items():
                            st.write(f"**Node {node_id}**: {content[:200]}...")

                    # Display knowledge graph statistics
                    st.subheader("Knowledge Graph Statistics")
                    stats = st.session_state.main.knowledge_graph.get_stats()
                    st.json(stats)

                    # Visualize graph
                    if traversal_path:
                        st.subheader("Graph Visualization")
                        try:
                            graph_image_buffer = (
                                st.session_state.main.visualizer.visualize_traversal(
                                    st.session_state.main.knowledge_graph.graph,
                                    traversal_path,
                                )
                            )

                            if graph_image_buffer:
                                st.image(
                                    graph_image_buffer,
                                    caption="Knowledge Graph Traversal",
                                )
                            else:
                                st.warning("No visualization generated.")

                        except Exception as e:
                            st.error(f"Failed to visualize graph: {str(e)}")
                            logger.error(f"Failed to visualize graph: {str(e)}")

                else:
                    st.warning(
                        "No response generated. Please check the query or document processing."
                    )
        elif query and not st.session_state.documents_processed:
            st.warning("Please load or upload and process documents before querying.")
    else:
        st.info("Please initialize the pipeline from the sidebar.")


if __name__ == "__main__":
    main()
