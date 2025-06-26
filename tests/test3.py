from dataclasses import dataclass
import networkx as nx
from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import List
from nltk.stem import WordNetLemmatizer
import spacy

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from spacy.cli import download

import pickle
import hashlib
from pathlib import Path
import logging
import json
import time

@dataclass
class Concepts:
    concepts_list: List[str]

class KnowledgeGraph:
    def __init__(self, cache_dir="./my_cache", batch_size: int = 100, log_level=logging.INFO):
        # Setup logging
        self.logger = self._setup_logging(log_level)
        self.logger.info("Initializing KnowledgeGraph")
        
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8

        # Caching and batch processing
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_cache = {}
        self.batch_size = batch_size
        
        self.logger.info(f"Cache directory: {self.cache_dir}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Edge threshold: {self.edges_threshold}")
        
        self._load_cache()

    def _setup_logging(self, log_level):
        """Setup logging configuration"""
        logger = logging.getLogger('KnowledgeGraph')
        logger.setLevel(log_level)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # File handler
            file_handler = logging.FileHandler('knowledge_graph.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger

    def _load_cache(self):
        cache_file = self.cache_dir / "cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.concept_cache = data.get('concepts', {})
                    self.embeddings_cache = data.get('embeddings', {})
                    self.extraction_progress = data.get('extraction_progress', {})
                
                self.logger.info(f"Loaded cache: {len(self.concept_cache)} concepts, "
                               f"{len(self.embeddings_cache)} embeddings")
            except Exception as e:
                self.logger.error(f"Failed to load cache: {e}")
                self.concept_cache = {}
                self.embeddings_cache = {}
                self.extraction_progress = {}
        else:
            self.logger.info("No existing cache found, starting fresh")
            self.extraction_progress = {}

    def _save_cache(self):
        try:
            with open(self.cache_dir / "cache.pkl", 'wb') as f:
                pickle.dump({
                    'concepts': self.concept_cache, 
                    'embeddings': self.embeddings_cache,
                    'extraction_progress': self.extraction_progress
                }, f)
            self.logger.debug("Cache saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _compute_similarities(self, embeddings):
        self.logger.info(f"Computing similarity matrix for {len(embeddings)} embeddings")
        try:
            similarities = cosine_similarity(np.array(embeddings))
            self.logger.info(f"Similarity matrix shape: {similarities.shape}")
            return similarities
        except Exception as e:
            self.logger.error(f"Failed to compute similarities: {e}")
            raise
    
    def _load_spacy_model(self):
        try:
            self.logger.info("Loading spaCy model 'en_core_web_sm'")
            nlp = spacy.load("en_core_web_sm")
            self.logger.info("SpaCy model loaded successfully")
            return nlp
        except OSError:
            self.logger.warning("SpaCy model not found, downloading...")
            try:
                download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                self.logger.info("SpaCy model downloaded and loaded successfully")
                return nlp
            except Exception as e:
                self.logger.error(f"Failed to download/load spaCy model: {e}")
                raise
    
    def _lemmatize_concept(self, concept):
        try:
            lemmatized = ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])
            self.logger.debug(f"Lemmatized '{concept}' to '{lemmatized}'")
            return lemmatized
        except Exception as e:
            self.logger.error(f"Failed to lemmatize concept '{concept}': {e}")
            return concept.lower()

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts):
        try:
            max_shared = min(len(self.graph.nodes[node1]['concepts']), 
                            len(self.graph.nodes[node2]['concepts']))
            concept_score = len(shared_concepts) / max_shared if max_shared > 0 else 0
            weight = 0.7 * similarity_score + 0.3 * concept_score
            
            self.logger.debug(f"Edge weight between {node1}-{node2}: {weight:.3f} "
                            f"(sim: {similarity_score:.3f}, concept: {concept_score:.3f})")
            return weight
        except Exception as e:
            self.logger.error(f"Failed to calculate edge weight: {e}")
            return similarity_score

    def _create_embeddings(self, splits: List[str], embedding_model = "sentence-transformers/all-MiniLM-L6-v2"):
        self.logger.info(f"Creating embeddings for {len(splits)} documents using {embedding_model}")
        
        texts = [split.page_content for split in splits]
        embeddings = [None] * len(splits)
        uncached = []

        # Check cache
        for i, text in tqdm(enumerate(texts), desc="Checking Embedding Cache", total=len(texts)):
            h = hashlib.md5(text.encode()).hexdigest()
            if h in self.embeddings_cache:
                embeddings[i] = self.embeddings_cache[h]
            else:
                uncached.append((i, text, h))

        self.logger.info(f"Found {len(texts) - len(uncached)} cached embeddings, "
                        f"need to compute {len(uncached)} new ones")

        if uncached:
            try:
                self.logger.info(f"Computing {len(uncached)} new embeddings...")
                model = HuggingFaceEmbeddings(model_name=embedding_model)
                
                total_batches = (len(uncached) + self.batch_size - 1) // self.batch_size
                self.logger.info(f"Processing {total_batches} batches of size {self.batch_size}")
                
                with tqdm(total=total_batches, desc="Embedding Batches") as pbar:
                    for i in range(0, len(uncached), self.batch_size):
                        batch = uncached[i: i + self.batch_size]
                        batch_texts = [t[1] for t in batch]
                        
                        try:
                            batch_embs = model.embed_documents(batch_texts)
                            
                            for (idx, text, h), emb in zip(batch, batch_embs):
                                self.embeddings_cache[h] = emb
                                embeddings[idx] = emb
                            
                            self._save_cache()
                            self.logger.debug(f"Processed batch {i//self.batch_size + 1}/{total_batches}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to process embedding batch {i//self.batch_size + 1}: {e}")
                            # Fill with zeros as fallback
                            for (idx, text, h) in batch:
                                embeddings[idx] = [0.0] * 384  # Default dimension
                        
                        pbar.update(1)
                        
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding model: {e}")
                raise
        else:
            self.logger.info("All embeddings found in cache!")

        self.logger.info(f"Embeddings creation completed. Total: {len(embeddings)}")
        return embeddings

    def _extract_concepts(self, splits, llm):
        self.logger.info(f"Extracting concepts from {len(splits)} documents")
        
        uncached_splits = [(i, s) for i, s in enumerate(splits) if s.page_content not in self.concept_cache]
        self.logger.info(f"Found {len(uncached_splits)} uncached documents for concept extraction")

        if not uncached_splits:
            self.logger.info("All concepts found in cache, populating graph nodes")
            for i, split in enumerate(splits):
                self.graph.nodes[i]['concepts'] = self.concept_cache[split.page_content]
            return

        # Spacy Entities
        all_entities = {}
        uncached_spacy = []
        
        self.logger.info("Processing spaCy entities...")
        for i, split in uncached_splits:
            spacy_key = f"spacy_{hashlib.md5(split.page_content.encode()).hexdigest()}"
            if spacy_key in self.embeddings_cache:
                all_entities[i] = self.embeddings_cache[spacy_key]
            else:
                uncached_spacy.append((i, split, spacy_key))
        
        self.logger.info(f"Processing {len(uncached_spacy)} documents with spaCy")
        
        # Only process uncached spaCy
        for i, split, spacy_key in tqdm(uncached_spacy, desc="Spacy Entities"):
            try:
                doc = self.nlp(split.page_content)
                entities = [e.text for e in doc.ents if e.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]
                self.embeddings_cache[spacy_key] = entities
                all_entities[i] = entities
                
                self.logger.debug(f"Extracted {len(entities)} entities from document {i}")
                
                if len(all_entities) % 100 == 0:
                    self._save_cache()
                    self.logger.debug(f"Saved cache after processing {len(all_entities)} documents")
                    
            except Exception as e:
                self.logger.error(f"Failed to process spaCy entities for document {i}: {e}")
                all_entities[i] = []

        # LLM concepts in batches
        self.logger.info("Processing LLM concept extraction...")
        total_batches = (len(uncached_splits) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total_batches, desc="LLM Concept Batches") as pbar:
            for i in range(0, len(uncached_splits), self.batch_size):
                batch = uncached_splits[i: i + self.batch_size]
                self.logger.debug(f"Processing LLM batch {i//self.batch_size + 1}/{total_batches}")

                # Process the batch with a single LLM Call
                batch_texts = [f"Doc {idx}: {split.page_content}" for idx, (_, split) in enumerate(batch)]
                
                # CHANGE: Modified prompt for clearer output format and robust parsing
                prompt = f"""For each document provided, extract 1-2 key concepts. Present the output as a list, where each item is 'Doc X: concept1, concept2'. Do not include any introductory or concluding sentences, just the list of document concepts.

                {chr(10).join(batch_texts)}

                Output format (example):
                Doc 0: concept A, concept B
                Doc 1: concept C, concept D
                ...
                """

                try:
                    result = llm.invoke(prompt)
                    result_text = result.content if hasattr(result, "content") else str(result)
                    
                    # CHANGE: Robust parsing for comma-separated list per doc
                    batch_concepts = {}
                    lines = result_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Doc ") and ":" in line:
                            try:
                                # Look for "Doc X: "
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    doc_id_str = parts[0].replace("Doc ", "").strip()
                                    concepts_str = parts[1].strip()
                                    
                                    doc_idx = int(doc_id_str)
                                    concepts = [c.strip() for c in concepts_str.split(',') if c.strip()]
                                    batch_concepts[f"doc_{doc_idx}"] = concepts
                                else:
                                    self.logger.warning(f"CHANGE: Could not parse line (unexpected format) '{line}'")
                            except ValueError:
                                self.logger.warning(f"CHANGE: Could not parse document index from line '{line}'")
                            except Exception as parse_e:
                                self.logger.warning(f"CHANGE: An error occurred during parsing line '{line}': {parse_e}")
                        # CHANGE: Ignore lines that don't start with "Doc " explicitly or are empty
                        elif line and not line.startswith("Here are the extracted key concepts"):
                             self.logger.debug(f"CHANGE: Skipping non-concept line: '{line}'")


                    for idx, (i, split) in enumerate(batch):
                        entities = all_entities.get(i, [])
                        llm_concepts = batch_concepts.get(f"doc_{idx}", [])
                        combined_concepts = list(set(entities + llm_concepts))
                        self.concept_cache[split.page_content] = combined_concepts
                        
                        self.logger.debug(f"Document {i}: {len(entities)} entities + "
                                        f"{len(llm_concepts)} LLM concepts = "
                                        f"{len(combined_concepts)} total")
                
                except Exception as e:
                    self.logger.warning(f"Batch LLM processing failed, falling back to individual processing: {e}")
                    
                    for i, split in batch:
                        try:
                            # CHANGE: Individual prompt also returns comma-separated concepts
                            # Emphasize no extra text in the prompt for individual calls too
                            result = llm.invoke(f"Extract key concepts from: '{split.page_content}'. Return only a comma-separated list of 1-2 concepts. Example: concept1, concept2")
                            
                            concepts = [c.strip() for c in str(result).split(',') if c.strip()]
                            combined_concepts = all_entities.get(i, []) + concepts
                            self.concept_cache[split.page_content] = combined_concepts
                            
                            self.logger.debug(f"Individual processing for document {i}: {len(concepts)} concepts")
                            
                        except Exception as individual_e:
                            self.logger.error(f"Individual LLM processing failed for document {i}: {individual_e}")
                            self.concept_cache[split.page_content] = all_entities.get(i, [])
                
                self._save_cache()
                pbar.update(1)

        # Populate graph nodes
        self.logger.info("Populating graph nodes with extracted concepts")
        for i, split in enumerate(splits):
            concepts = self.concept_cache[split.page_content]
            self.graph.nodes[i]['concepts'] = concepts
            self.logger.debug(f"Node {i}: {len(concepts)} concepts")

    def _add_edges(self, embeddings):
        self.logger.info("Computing similarity matrix and adding edges...")
        
        try:
            sim_matrix = self._compute_similarities(embeddings)
            indices = np.where(np.triu(sim_matrix > self.edges_threshold, k=1))

            self.logger.info(f"Found {len(indices[0])} potential edges above threshold {self.edges_threshold}")
            edges_added = 0

            for i, j in tqdm(zip(indices[0], indices[1]), desc="Adding edges", total=len(indices[0])):
                try:
                    shared = set(self.graph.nodes[i]['concepts']) & set(self.graph.nodes[j]['concepts'])
                    if shared:
                        weight = self._calculate_edge_weight(i, j, sim_matrix[i, j], shared)
                        self.graph.add_edge(i, j, weight=weight, 
                                          similarity=sim_matrix[i, j], 
                                          shared_concepts=list(shared))
                        edges_added += 1
                        
                        if edges_added % 1000 == 0:
                            self.logger.debug(f"Added {edges_added} edges so far...")
                            
                except Exception as e:
                    self.logger.error(f"Failed to add edge between nodes {i} and {j}: {e}")
            
            self.logger.info(f"Successfully added {edges_added} edges with shared concepts")
            
        except Exception as e:
            self.logger.error(f"Failed to add edges: {e}")
            raise
        
    def _add_nodes(self, splits):
        self.logger.info(f"Adding {len(splits)} nodes to graph")
        
        try:
            for i, split in enumerate(splits):
                self.graph.add_node(i, content=split.page_content)
                
                if (i + 1) % 1000 == 0:
                    self.logger.debug(f"Added {i + 1} nodes...")
            
            self.logger.info(f"Successfully added {len(splits)} nodes to graph")
            
        except Exception as e:
            self.logger.error(f"Failed to add nodes: {e}")
            raise

    def get_stats(self):
        """Get knowledge graph statistics"""
        self.logger.info("Computing knowledge graph statistics")
        
        try:
            stats = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'components': nx.number_connected_components(self.graph),
                'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.nodes else 0
            }
            
            if self.graph.edges:
                weights = [d['weight'] for _, _, d in self.graph.edges(data=True)]
                stats['avg_edge_weight'] = np.mean(weights)
                stats['max_edge_weight'] = max(weights)
            
            self.logger.info(f"Graph stats: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to compute stats: {e}")
            return {}

    def visualize(self, figsize=(16, 12), sample_nodes=200, show_concepts=True):
        """Visualize knowledge graph with sampling and concept labels"""
        self.logger.info(f"Visualizing knowledge graph (sample_nodes={sample_nodes})")
        
        try:
            if self.graph.number_of_nodes() > sample_nodes:
                # Sample largest connected components
                components = list(nx.connected_components(self.graph))
                components.sort(key=len, reverse=True)
                
                sampled_nodes = set()
                for comp in components[:min(10, len(components))]:
                    sampled_nodes.update(list(comp)[:sample_nodes//10])
                    if len(sampled_nodes) >= sample_nodes:
                        break
                
                subgraph = self.graph.subgraph(sampled_nodes)
                self.logger.info(f"Showing {len(sampled_nodes)} nodes from largest components")
            else:
                subgraph = self.graph
            
            fig, ax = plt.subplots(figsize=figsize)
            pos = nx.spring_layout(subgraph, k=2, iterations=30)
            
            # Draw edges with weight-based coloring
            edges = list(subgraph.edges(data=True))
            edge_weights = [d.get('weight', 0.5) for _, _, d in edges]
            
            nx.draw_networkx_edges(subgraph, pos, 
                                edge_color=edge_weights,
                                edge_cmap=plt.cm.Blues,
                                width=1.5, alpha=0.6, ax=ax)
            
            # Draw nodes
            nx.draw_networkx_nodes(subgraph, pos, 
                                node_color='lightblue',
                                node_size=100, alpha=0.8, ax=ax)
            
            # Add concept labels if requested
            if show_concepts:
                labels = {}
                for node in subgraph.nodes():
                    concepts = self.graph.nodes[node].get('concepts', [])
                    labels[node] = concepts[0][:15] + '...' if concepts and len(concepts[0]) > 15 else (concepts[0] if concepts else str(node))
                
                nx.draw_networkx_labels(subgraph, pos, labels, font_size=6, ax=ax)
            
            # Add colorbar
            if edge_weights:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Edge Weight', rotation=270, labelpad=15)
            
            ax.set_title(f"Knowledge Graph ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
            ax.axis('off')
            plt.tight_layout()
            plt.show()
            
            self.logger.info("Visualization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            raise
            
    def build_knowledge_graph(self, splits, llm):
        """Build knowledge graph with optimized batch processing"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING KNOWLEDGE GRAPH CONSTRUCTION")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            self.logger.info("Step 1: Adding nodes...")
            node_start = time.time()
            self._add_nodes(splits)
            self.logger.info(f"Nodes added in {time.time() - node_start:.2f} seconds")
            
            self.logger.info("Step 2: Creating embeddings...")
            embed_start = time.time()
            embeddings = self._create_embeddings(splits)
            self.logger.info(f"Embeddings created in {time.time() - embed_start:.2f} seconds")
            
            self.logger.info("Step 3: Extracting concepts...")
            concept_start = time.time()
            self._extract_concepts(splits, llm)
            self.logger.info(f"Concepts extracted in {time.time() - concept_start:.2f} seconds")
            
            self.logger.info("Step 4: Adding edges...")
            edge_start = time.time()
            self._add_edges(embeddings)
            self.logger.info(f"Edges added in {time.time() - edge_start:.2f} seconds")
            
            self.logger.info("Step 5: Final cache save...")
            self._save_cache()
            
            total_time = time.time() - start_time
            self.logger.info("=" * 50)
            self.logger.info(f"KNOWLEDGE GRAPH CONSTRUCTION COMPLETED in {total_time:.2f} seconds")
            self.logger.info("=" * 50)
            
            # Log final statistics
            stats = self.get_stats()
            self.logger.info("Final graph statistics:")
            for key, value in stats.items():
                self.logger.info(f"  {key}: {value}")
            
            return self.graph
            
        except Exception as e:
            self.logger.error(f"KNOWLEDGE GRAPH CONSTRUCTION FAILED: {e}")
            raise