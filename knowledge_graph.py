from dataclasses import dataclass
import networkx as nx
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks import get_openai_callback
from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List, Tuple, Dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
import heapq

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from spacy.cli import download
from spacy.lang.en import English

import json
import pickle
import hashlib
from pathlib import Path

@dataclass
class Concepts:
    concepts_list: List[str]

class KnowledgeGraph:
    def __init__(self,  cache_dir="./cache", batch_size: int = 100):
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
        self._load_cache()

    def _load_cache(self):
        cache_file = self.cache_dir / "cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.concept_cache = data.get('concepts', {})
                self.embeddings_cache = data.get('embeddings', {})
                self.extraction_progress = data.get('extraction_progress', {})  # Track which docs are processed
        else:
            self.extraction_progress = {}

    def _save_cache(self):
        with open(self.cache_dir / "cache.pkl", 'wb') as f:
            pickle.dump({
                'concepts': self.concept_cache, 
                'embeddings': self.embeddings_cache,
                'extraction_progress': self.extraction_progress
            }, f)  
    
    def _compute_similarities(self, embeddings):
        return cosine_similarity(np.array(embeddings))
    
    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading the spacy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
    
    def _lemmatize_concept(self, concept):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts):
        max_shared = min(len(self.graph.nodes[node1]['concepts']), 
                        len(self.graph.nodes[node2]['concepts']))
        concept_score = len(shared_concepts) / max_shared if max_shared > 0 else 0
        return 0.7 * similarity_score + 0.3 * concept_score 

    def _create_embeddings(self, splits: List[str], embedding_model = "sentence-transformers/all-MiniLM-L6-v2"):
        texts = [split.page_content for split in splits]
        embeddings = [None] * len(splits)
        uncached = []

        for i, text in tqdm(enumerate(texts), desc="Checking Embedding Cache", total=len(texts)):
            h = hashlib.md5(text.encode()).hexdigest()
            if h in self.embeddings_cache:
                embeddings[i] = self.embeddings_cache[h]
            else:
                uncached.append((i, text, h))

        if uncached:
            print(f"Computing {len(uncached)} new embeddings...")
            model = HuggingFaceEmbeddings(model_name=embedding_model)
            
            total_batches = (len(uncached) + self.batch_size - 1) // self.batch_size
            with tqdm(total=total_batches, desc="Embedding Batches") as pbar:
                for i in range(0, len(uncached), self.batch_size):
                    batch = uncached[i: i + self.batch_size]
                    batch_texts = [t[1] for t in batch]
                    batch_embs = model.embed_documents(batch_texts)
                    
                    for (idx, text, h), emb in zip(batch, batch_embs):
                        self.embeddings_cache[h] = emb
                        embeddings[idx] = emb
                    
                    self._save_cache()
                    pbar.update(1)
        else:
            print("All embeddings found in cache!")

        return embeddings

    def _extract_concepts(self, splits, llm):
        uncached_splits = [(i, s) for i, s in enumerate(splits) if s.page_content not in self.concept_cache]

        if not uncached_splits:
            for i, split in enumerate(splits):
                self.graph.nodes[i]['concepts'] = self.concept_cache[split.page_content]
            return

        # Spacy Entities
        all_entities = {}
        uncached_spacy = []
        
        for i, split in uncached_splits:
            spacy_key = f"spacy_{hashlib.md5(split.page_content.encode()).hexdigest()}"
            if spacy_key in self.embeddings_cache:
                all_entities[i] = self.embeddings_cache[spacy_key]
            else:
                uncached_spacy.append((i, split, spacy_key))
        
        # Only process uncached spaCy
        for i, split, spacy_key in tqdm(uncached_spacy, desc="Spacy Entities"):
            doc = self.nlp(split.page_content)
            entities = [e.text for e in doc.ents if e.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]
            self.embeddings_cache[spacy_key] = entities
            all_entities[i] = entities
            
            if len(all_entities) % 100 == 0:
                self._save_cache()

            
        # LLM concepts in batches
        total_batches = (len(uncached_splits) + self.batch_size - 1) // self.batch_size
        with tqdm(total=total_batches, desc="LLM Concept Batches") as pbar:
            for i in range(0, len(uncached_splits), self.batch_size):
                batch = uncached_splits[i: i + self.batch_size]

                # Process the batch with a single LLM Call
                batch_texts = [f"Doc {idx}: {split.page_content}" for idx, (_, split) in enumerate(batch)]
                prompt = f"""Extract key concepts from each document. Return JSON format:

                {chr(10).join(batch_texts)}

                Return: {{"doc_0": ["concept1", "concept2"], "doc_1": ["concept1", "concept2"], ...}}"""

                try:
                    result = llm.invoke(prompt)
                    result_text = result.content if hasattr(result, "content") else str(result)
                    json_str = result_text[result_text.find('{'):result_text.rfind('}') + 1]
                    batch_concepts = json.loads(json_str)

                    for idx, (i, split) in enumerate(batch):
                        entities = all_entities.get(i, [])
                        llm_concepts = batch_concepts.get(f"doc_{idx}", [])
                        self.concept_cache[split.page_content] = list(set(entities + llm_concepts))
                
                except:
                    for i, split in batch:
                        try:
                            result = llm.invoke(f"Extract key concepts from {split.page_content} \nConcepts: ")
                            concepts = [c.strip() for c in str(result).split(',')[:10]]
                            self.concept_cache[split.page_content] = all_entities.get(i, []) + concepts
                        except:
                            self.concept_cache[split.page_content] = all_entities.get(i, [])
                
                self._save_cache()
                pbar.update(1)

        for i, split in enumerate(splits):
            self.graph.nodes[i]['concepts'] = self.concept_cache[split.page_content]

    def _add_edges(self, embeddings):
        print("Computing similarity matrix...")
        sim_matrix = self._compute_similarities(embeddings)
        indices = np.where(np.triu(sim_matrix > self.edges_threshold, k=1))

        print(f"Found {len(indices[0])} potential edges above threshold {self.edges_threshold}")
        edges_added = 0

        for i, j in tqdm(zip(indices[0], indices[1]), desc="Adding edges", total=len(indices[0])):
            shared = set(self.graph.nodes[i]['concepts']) & set(self.graph.nodes[j]['concepts'])
            if shared:
                weight = self._calculate_edge_weight(i, j, sim_matrix[i, j], shared)
                self.graph.add_edge(i, j, weight=weight, 
                                  similarity=sim_matrix[i, j], 
                                  shared_concepts=list(shared))
                edges_added += 1
        
        print(f"Added {edges_added} edges with shared concepts")
        
    def _add_nodes(self, splits):
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def get_stats(self):
        """Get knowledge graph statistics"""
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
        
        return stats

    def visualize(self, figsize=(16, 12), sample_nodes=200, show_concepts=True):
        """Visualize knowledge graph with sampling and concept labels"""
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
            print(f"Showing {len(sampled_nodes)} nodes from largest components")
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
            
    def build_knowledge_graph(self, splits, llm):
        """Build knowledge graph with optimized batch processing"""
        print("Adding nodes...")
        self._add_nodes(splits)
        
        print("Creating embeddings...")
        embeddings = self._create_embeddings(splits)
        
        print("Extracting concepts...")
        self._extract_concepts(splits, llm)
        
        print("Adding edges...")
        self._add_edges(embeddings)
        
        print("Final cache save...")
        self._save_cache()
        
        return self.graph