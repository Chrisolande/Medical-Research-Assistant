"""Knowledge Graph module."""

import logging

import networkx as nx
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline

from medical_graph_rag.core.utils import (
    CacheManager,
    calculate_edge_weight,
    create_text_hash,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """KnowledgeGraph class."""

    def __init__(
        self, cache_dir="./cache", batch_size: int = 100, max_concurrent_calls=10
    ):
        """Initialize the kg."""
        logger.info("Initializing KnowledgeGraph")
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}

        # Initialize NER pipeline for biomedical entity extraction
        logger.info("Loading biomedical NER pipeline")
        self.ner_pipeline = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            tokenizer="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
        )

        self.edges_threshold = 0.8
        self.batch_size = batch_size
        self.max_concurrent_calls = max_concurrent_calls

        # Use CacheManager
        self.cache_manager = CacheManager(cache_dir)
        self.embeddings_cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load Cache method."""
        logger.info("Loading cache data")
        data = self.cache_manager.load_cache()
        self.concept_cache = data.get("concepts", {})
        self.embeddings_cache = data.get("embeddings", {})
        graph_data = data.get("graph", None)
        if graph_data:
            self.graph = nx.node_link_graph(graph_data)
            logger.info(
                f"Loaded existing graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )

    def _save_cache(self):
        """Save Cache method."""
        logger.debug("Saving cache data")
        cache_data = {
            "concepts": self.concept_cache,
            "embeddings": self.embeddings_cache,
            "graph": nx.node_link_data(self.graph) if self.graph.nodes else None,
        }
        self.cache_manager.save_cache(cache_data)

    def _compute_similarities(self, embeddings):
        """Compute Similarities method."""
        logger.info("Computing cosine similarities")
        return cosine_similarity(np.array(embeddings))

    def _lemmatize_concept(self, concept):
        """Lemmatize Concept method."""
        return " ".join(
            [self.lemmatizer.lemmatize(word) for word in concept.lower().split()]
        )

    def _create_embeddings(
        self,
        splits: list[str],
        embedding_model="abhinand/MedEmbed-small-v0.1",
    ):
        """Create Embeddings."""
        logger.info(f"Creating embeddings for {len(splits)} documents")
        texts = [split.page_content for split in splits]
        embeddings = [None] * len(splits)
        uncached = []

        for i, text in tqdm(
            enumerate(texts), desc="Checking Embedding Cache", total=len(texts)
        ):
            h = create_text_hash(text)
            if h in self.embeddings_cache:
                embeddings[i] = self.embeddings_cache[h]
            else:
                uncached.append((i, text, h))

        if uncached:
            logger.info(f"Computing {len(uncached)} new embeddings")
            model = HuggingFaceEmbeddings(model_name=embedding_model)

            total_batches = (len(uncached) + self.batch_size - 1) // self.batch_size
            with tqdm(total=total_batches, desc="Embedding Batches") as pbar:
                for i in range(0, len(uncached), self.batch_size):
                    batch = uncached[i : i + self.batch_size]
                    batch_texts = [t[1] for t in batch]
                    batch_embs = model.embed_documents(batch_texts)

                    for (idx, _text, h), emb in zip(batch, batch_embs, strict=False):
                        self.embeddings_cache[h] = emb
                        embeddings[idx] = emb

                    self._save_cache()
                    pbar.update(1)
        else:
            logger.info("All embeddings found in cache")

        return embeddings

    def _extract_concepts_batch(self, splits):
        """Extract concepts using transformers NER pipeline with batching."""
        logger.info(f"Extracting concepts from {len(splits)} documents")
        uncached_splits = [
            (i, s)
            for i, s in enumerate(splits)
            if s.page_content not in self.concept_cache
        ]

        if not uncached_splits:
            logger.info("All concepts found in cache")
            for i, split in enumerate(splits):
                self.graph.nodes[i]["concepts"] = self.concept_cache[split.page_content]
            return

        logger.info(f"Processing {len(uncached_splits)} uncached documents")

        # Process in batches
        for batch_start in tqdm(
            range(0, len(uncached_splits), self.batch_size), desc="NER Batches"
        ):
            batch = uncached_splits[batch_start : batch_start + self.batch_size]
            batch_texts = [
                split.page_content for _, split in batch
            ]  # Truncate for NER model

            try:
                # Run NER pipeline on batch
                batch_results = self.ner_pipeline(batch_texts)

                # Process results
                for (_idx, split), ner_result in zip(
                    batch, batch_results, strict=False
                ):
                    entities = []
                    for entity in ner_result:
                        if entity["score"] > 0.8:  # Confidence threshold
                            entities.append(entity["word"])

                    # Remove duplicates and store
                    self.concept_cache[split.page_content] = list(set(entities))

            except Exception as e:
                logger.error(f"Error processing NER batch: {e}")
                # Fallback: store empty concepts for this batch
                for _idx, split in batch:
                    self.concept_cache[split.page_content] = []

            # Save cache periodically
            if batch_start % (self.batch_size * 5) == 0:
                self._save_cache()

        # Final cache save
        self._save_cache()

        # Update graph nodes
        for i, split in enumerate(splits):
            self.graph.nodes[i]["concepts"] = self.concept_cache[split.page_content]

    def _add_edges(self, embeddings):
        """Add Edges method."""
        logger.info("Adding edges based on similarity and shared concepts")
        sim_matrix = self._compute_similarities(embeddings)
        indices = np.where(np.triu(sim_matrix > self.edges_threshold, k=1))

        logger.info(
            f"Found {len(indices[0])} potential edges above threshold {self.edges_threshold}"
        )
        edges_added = 0

        for i, j in tqdm(
            zip(indices[0], indices[1], strict=False),
            desc="Adding edges",
            total=len(indices[0]),
        ):
            shared = set(self.graph.nodes[i]["concepts"]) & set(
                self.graph.nodes[j]["concepts"]
            )
            if shared:
                weight = calculate_edge_weight(
                    sim_matrix[i, j],
                    list(shared),
                    self.graph.nodes[i]["concepts"],
                    self.graph.nodes[j]["concepts"],
                )
                self.graph.add_edge(
                    i,
                    j,
                    weight=weight,
                    similarity=sim_matrix[i, j],
                    shared_concepts=list(shared),
                )
                edges_added += 1

        logger.info(f"Added {edges_added} edges with shared concepts")

    def _add_nodes(self, splits):
        """Add Nodes method."""
        logger.info(f"Adding {len(splits)} nodes to graph")
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def get_stats(self):
        """Get knowledge graph statistics."""
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "components": nx.number_connected_components(self.graph),
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
                if self.graph.nodes
                else 0
            ),
        }

        if self.graph.edges:
            weights = [d["weight"] for _, _, d in self.graph.edges(data=True)]
            stats["avg_edge_weight"] = np.mean(weights)
            stats["max_edge_weight"] = max(weights)

        logger.info(f"Graph stats: {stats}")
        return stats

    def build_knowledge_graph(self, splits):
        """Build knowledge graph with optimized batch processing."""
        logger.info("Building knowledge graph")

        if (
            self.graph.number_of_nodes() == len(splits)
            and self.graph.number_of_edges() > 0
        ):
            logger.info(
                f"Knowledge graph already exists with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )
            return self.graph

        logger.info("Adding nodes...")
        self._add_nodes(splits)

        logger.info("Creating embeddings...")
        embeddings = self._create_embeddings(splits)

        logger.info("Extracting concepts...")
        self._extract_concepts_batch(splits)

        logger.info("Adding edges...")
        self._add_edges(embeddings)

        logger.info("Final cache save...")
        self._save_cache()

        logger.info("Knowledge graph build complete")
        return self.graph
