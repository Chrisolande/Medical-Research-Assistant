import json
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
from asyncio import Semaphore
import asyncio

from utils import CacheManager, extract_and_parse_json, create_text_hash, clean_concepts, calculate_edge_weight

@dataclass
class Concepts:
    concepts_list: List[str]

class KnowledgeGraph:
    def __init__(self, cache_dir="./cache", batch_size: int = 100, max_concurrent_calls = 10):
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8
        self.batch_size = batch_size
        self.max_concurrent_calls = max_concurrent_calls
        
        # Use CacheManager
        self.cache_manager = CacheManager(cache_dir)
        self.embeddings_cache = {}
        self._load_cache()

        self.max_concurrent_calls = max_concurrent_calls

    def _load_cache(self):
        data = self.cache_manager.load_cache()
        self.concept_cache = data.get('concepts', {})
        self.embeddings_cache = data.get('embeddings', {})
        self.extraction_progress = data.get('extraction_progress', {})
        graph_data = data.get('graph', None)
        if graph_data:
            self.graph = nx.node_link_graph(graph_data)
            print(f"Loaded existing graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def _save_cache(self):
        cache_data = {
            'concepts': self.concept_cache,
            'embeddings': self.embeddings_cache,
            'extraction_progress': self.extraction_progress,
            'graph': nx.node_link_data(self.graph) if self.graph.nodes else None
        }
        self.cache_manager.save_cache(cache_data)
    
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

    def _create_embeddings(self, splits: List[str], embedding_model = "sentence-transformers/all-MiniLM-L6-v2"):
        texts = [split.page_content for split in splits]
        embeddings = [None] * len(splits)
        uncached = []

        for i, text in tqdm(enumerate(texts), desc="Checking Embedding Cache", total=len(texts)):
            h = create_text_hash(text)
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

    async def _extract_concepts(self, splits, llm):
        uncached_splits = [(i, s) for i, s in enumerate(splits) if s.page_content not in self.concept_cache]

        if not uncached_splits:
            for i, split in enumerate(splits):
                self.graph.nodes[i]['concepts'] = self.concept_cache[split.page_content]
            return

        # Spacy Entities
        all_entities = {}
        uncached_spacy = []
        
        for i, split in uncached_splits:
            spacy_key = f"spacy_{create_text_hash(split.page_content)}"
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

        # Process batches asynchronously
        async def process_batches():
            semaphore = asyncio.Semaphore(self.max_concurrent_calls)  # Limit concurrent requests
            
            async def process_single_batch(batch_data):
                i, batch = batch_data
                async with semaphore:
                    batch_success = False
                    
                    if len(batch) > 1:  # Only try batch if more than 1 item
                        try:
                            batch_prompts = [f"Document {idx}: {split.page_content[:500]}..." 
                                        for idx, (_, split) in enumerate(batch)]
                            
                            prompt = f"""Extract 1-3 key concepts from each document. Return ONLY valid JSON in this exact format:
                            {{"0": ["concept1", "concept2"], "1": ["concept1", "concept2"]}}

                            Rules:
                            - Use double quotes only
                            - No trailing commas
                            - No extra text or explanations
                            - Start response with {{ and end with }}

                            Documents:
                            {chr(10).join(batch_prompts)}

                            JSON:"""
                            
                            result = await llm.ainvoke(prompt)
                            result_text = result.content if hasattr(result, "content") else str(result)
                            
                            batch_concepts = extract_and_parse_json(result_text)
                            if batch_concepts:
                                for idx, (i, split) in enumerate(batch):
                                    entities = all_entities.get(i, [])
                                    llm_concepts = batch_concepts.get(str(idx), [])
                                    self.concept_cache[split.page_content] = list(set(entities + llm_concepts))
                                
                                batch_success = True
                                
                        except Exception as e:
                            print(f"Batch failed: {str(e)[:50]}... falling back")
                    
                    # Fall back to individual processing
                    if not batch_success:
                        for i, split in batch:
                            try:
                                prompt = f"Extract 1-3 key concepts from this text as a comma-separated list:\n{split.page_content[:500]}...\n\nConcepts:"
                                result = await llm.ainvoke(prompt)
                                result_text = result.content if hasattr(result, "content") else str(result)
                                concepts = [c.strip() for c in result_text.replace('\n', ',').split(',')[:5] if c.strip()]
                                self.concept_cache[split.page_content] = list(set(all_entities.get(i, []) + concepts))
                            except Exception as e:
                                print(f"Individual extraction failed for doc {i}: {str(e)[:50]}")
                                self.concept_cache[split.page_content] = all_entities.get(i, [])
            
            # Create batch tasks
            batches = [(i, uncached_splits[i:i + self.batch_size]) 
                    for i in range(0, len(uncached_splits), self.batch_size)]
            
            # Process with progress bar
            tasks = [process_single_batch(batch_data) for batch_data in batches]
            
            with tqdm(total=len(tasks), desc="LLM Concept Batches") as pbar:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    self._save_cache()
                    pbar.update(1)

        # Run async processing
        await process_batches()

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
                weight = calculate_edge_weight(sim_matrix[i, j], list(shared), 
                              self.graph.nodes[i]['concepts'], 
                              self.graph.nodes[j]['concepts'])
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

    async def build_knowledge_graph(self, splits, llm):
        """Build knowledge graph with optimized batch processing"""
        if (self.graph.number_of_nodes() == len(splits) and self.graph.number_of_edges() > 0):
            print(f"Knowledge graph already exists with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return self.graph
        
        print("Adding nodes...")
        self._add_nodes(splits)
        
        print("Creating embeddings...")
        embeddings = self._create_embeddings(splits)
        
        print("Extracting concepts...")
        await self._extract_concepts(splits, llm)
        
        print("Adding edges...")
        self._add_edges(embeddings)
        
        print("Final cache save...")
        self._save_cache()
        
        return self.graph