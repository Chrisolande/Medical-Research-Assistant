"""Knowledge Graph module."""

import logging

import networkx as nx
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from medical_graph_rag.config import GRAPH_EDGE_SIMILARITY_THRESHOLD
from medical_graph_rag.utils import (
    CacheManager,
    calculate_edge_weight,
    create_text_hash,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """KnowledgeGraph class."""

    def __init__(
        self,
        cache_dir="./cache",
        batch_size: int = 100,
        max_concurrent_calls=10,
        embeddings: HuggingFaceEmbeddings | None = None,
        ner_pipeline=None,
    ):
        """Initialize the kg."""
        logger.info("Initializing KnowledgeGraph")
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.content_to_node_id: dict[str, int] = {}

        self._ner_pipeline = ner_pipeline

        self.edges_threshold = GRAPH_EDGE_SIMILARITY_THRESHOLD
        self.batch_size = batch_size
        self.max_concurrent_calls = max_concurrent_calls

        # Use CacheManager
        self.cache_manager = CacheManager(cache_dir)
        self.embeddings_cache = {}
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="abhinand/MedEmbed-small-v0.1"
        )
        self._load_cache()

    @property
    def ner_pipeline(self):
        if self._ner_pipeline is None:
            logger.info("Lazy-loading NER pipeline...")
            from transformers import pipeline

            self._ner_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
            )
        return self._ner_pipeline

    def _load_cache(self):
        """Load Cache method."""
        logger.info("Loading cache data")
        data = self.cache_manager.load_cache()
        self.concept_cache = data.get("concepts", {})
        self.embeddings_cache = data.get("embeddings", {})
        graph_data = data.get("graph", None)
        if graph_data:
            self.graph.clear()
            self.graph = nx.node_link_graph(graph_data)
            self.content_to_node_id = {
                data["content"]: n
                for n, data in self.graph.nodes(data=True)
                if "content" in data
            }
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

    def _create_embeddings(self, splits: list[str]):
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

            total_batches = (len(uncached) + self.batch_size - 1) // self.batch_size
            with tqdm(total=total_batches, desc="Embedding Batches") as pbar:
                for i in range(0, len(uncached), self.batch_size):
                    batch = uncached[i : i + self.batch_size]
                    batch_texts = [t[1] for t in batch]
                    batch_embs = self.embeddings.embed_documents(batch_texts)

                    for (idx, _text, h), emb in zip(batch, batch_embs, strict=False):
                        self.embeddings_cache[h] = emb
                        embeddings[idx] = emb

                    self._save_cache()
                    pbar.update(1)
        else:
            logger.info("All embeddings found in cache")

        missing = [i for i, e in enumerate(embeddings) if e is None]
        if missing:
            raise ValueError(
                f"Embedding computation failed for {len(missing)} documents "
                f"at indices: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
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
                    self.concept_cache[split.page_content] = list(
                        {e["word"].lower() for e in ner_result if e["score"] > 0.8}
                    )

            except Exception as e:
                logger.error(f"Error processing NER batch: {e}")
                # Fallback: store empty concepts for this batch
                for _idx, split in batch:
                    self.concept_cache[split.page_content] = []

            # Save cache periodically
            if batch_start > 0 and batch_start % (self.batch_size * 5) == 0:
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
        fallback_edges_added = 0

        for i, j in tqdm(
            zip(indices[0], indices[1], strict=False),
            desc="Adding edges",
            total=len(indices[0]),
        ):
            concepts_i = self.graph.nodes[i].get("concepts", [])
            concepts_j = self.graph.nodes[j].get("concepts", [])
            shared = set(concepts_i) & set(concepts_j)
            if shared:
                weight = calculate_edge_weight(
                    sim_matrix[i, j],
                    list(shared),
                    concepts_i,
                    concepts_j,
                )
                self.graph.add_edge(
                    i,
                    j,
                    weight=weight,
                    similarity=float(sim_matrix[i, j]),
                    shared_concepts=list(shared),
                )
                edges_added += 1
            elif not concepts_i or not concepts_j:
                fallback_weight = float(sim_matrix[i, j]) * 0.5
                self.graph.add_edge(
                    i,
                    j,
                    weight=fallback_weight,
                    similarity=float(sim_matrix[i, j]),
                    shared_concepts=[],
                )
                fallback_edges_added += 1

        logger.info(
            f"Added {edges_added} edges with shared concepts and "
            f"{fallback_edges_added} fallback similarity-only edges"
        )

    def _add_nodes(self, splits):
        """Add Nodes method."""
        logger.info(f"Adding nodes from {len(splits)} splits")
        seen: set[str] = set()
        self._node_splits: list = []
        self.content_to_node_id = {}
        for split in splits:
            content = split.page_content
            if content in seen:
                continue
            seen.add(content)
            node_id = len(self._node_splits)
            self.graph.add_node(node_id, content=content)
            self.content_to_node_id[content] = node_id
            self._node_splits.append(split)
        duplicates = len(splits) - len(self._node_splits)
        if duplicates:
            logger.warning(f"Skipped {duplicates} duplicate document chunks")
        logger.info(f"Added {len(self._node_splits)} nodes to graph")
        return self._node_splits

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

    def get_subgraph_stats(self, traversal_path: list[int]) -> dict:
        """Return statistics scoped to the nodes visited during a query."""
        if not traversal_path:
            return {"nodes": 0, "edges": 0, "density": 0.0}

        subgraph = self.graph.subgraph(traversal_path)
        stats = {
            "nodes": subgraph.number_of_nodes(),
            "edges": subgraph.number_of_edges(),
            "density": nx.density(subgraph),
        }
        if subgraph.edges:
            weights = [d["weight"] for _, _, d in subgraph.edges(data=True)]
            stats["avg_edge_weight"] = float(np.mean(weights))
            stats["max_edge_weight"] = float(max(weights))
        return stats


def build_knowledge_graph(self, splits):
    logger.info("Building knowledge graph")

    if self.graph.number_of_nodes() > 0:
        logger.info(
            f"Graph already loaded from cache with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges, skipping build"
        )
        return self.graph

    self.graph.clear()

    logger.info("Adding nodes...")
    splits = self._add_nodes(splits)

    logger.info("Creating embeddings...")
    embeddings = self._create_embeddings(splits)

    logger.info("Extracting concepts...")
    self._extract_concepts_batch(splits)

    logger.info("Adding edges...")
    self._add_edges(embeddings)

    logger.info("Final cache save...")
    self._save_cache()

    logger.info(
        f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, "
        f"{self.graph.number_of_edges()} edges"
    )
    return self.graph
