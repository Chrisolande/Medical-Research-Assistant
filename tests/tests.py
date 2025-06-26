import asyncio
import heapq
import hashlib
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any, Protocol, AsyncGenerator
from collections import defaultdict, deque

import networkx as nx
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks import get_openai_callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Configuration and Models
@dataclass
class QueryEngineConfig:
    max_context_length: int = 4000
    confidence_threshold: float = 0.8
    max_traversal_nodes: int = 50
    top_neighbors_count: int = 3
    similarity_search_k: int = 5
    cache_size: int = 1000
    learning_rate: float = 0.01
    enable_async: bool = True
    enable_caching: bool = True
    enable_learning: bool = True


@dataclass
class QueryMetrics:
    nodes_visited: int = 0
    concepts_explored: int = 0
    cache_hits: int = 0
    total_time: float = 0.0
    confidence_score: float = 0.0
    quality_score: float = 0.0


@dataclass
class AnswerQuality:
    completeness: float = 0.0
    accuracy: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    confidence: float = 0.0
    
    @property
    def overall_score(self) -> float:
        return np.mean([self.completeness, self.accuracy, self.relevance, 
                       self.coherence, self.confidence])


class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the current context provides a complete answer")
    answer: str = Field(description="The current answer based on the context")
    confidence: float = Field(description="Confidence score between 0 and 1", default=0.0)


class AgentType(Enum):
    RETRIEVER = "retriever"
    REASONER = "reasoner"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"


# Core Components
class CacheManager:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_order = deque(maxlen=max_size)
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, query: str, context: str) -> str:
        return hashlib.md5(f"{query}:{context}".encode()).hexdigest()
    
    def get(self, query: str, context: str) -> Optional[Tuple[bool, str]]:
        key = self._get_key(query, context)
        if key in self._cache:
            self.hits += 1
            self._access_order.append(key)
            return self._cache[key]
        self.misses += 1
        return None
    
    def set(self, query: str, context: str, value: Tuple[bool, str]):
        key = self._get_key(query, context)
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Remove least recently used
            oldest = self._access_order.popleft()
            self._cache.pop(oldest, None)
        
        self._cache[key] = value
        self._access_order.append(key)
xccxcxvccv
class ContextManager:
    def __init__(self, max_length: int = 4000):
        self.max_length = max_length
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def optimize_context(self, context: str, query: str) -> str:
        if len(context) <= self.max_length:
            return context
        
        # Split into chunks and rank by relevance to query
        chunks = self._split_into_chunks(context)
        if len(chunks) <= 1:
            return context[:self.max_length]
        
        # Score chunks by relevance
        chunk_scores = self._score_chunks(chunks, query)
        
        # Select top chunks that fit within limit
        selected_chunks = []
        current_length = 0
        
        for chunk, score in sorted(zip(chunks, chunk_scores), 
                                 key=lambda x: x[1], reverse=True):
            if current_length + len(chunk) <= self.max_length:
                selected_chunks.append(chunk)
                current_length += len(chunk)
            else:
                break
        
        return '\n'.join(selected_chunks)
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def _score_chunks(self, chunks: List[str], query: str) -> List[float]:
        if len(chunks) <= 1:
            return [1.0]
        
        try:
            all_texts = chunks + [query]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1]
            chunk_vectors = tfidf_matrix[:-1]
            
            # Compute cosine similarity
            similarities = (chunk_vectors * query_vector.T).toarray().flatten()
            return similarities.tolist()
        except:
            # Fallback to uniform scoring
            return [1.0] * len(chunks)


class OnlineLearningEngine:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.query_patterns = defaultdict(lambda: deque(maxlen=50))
        self.success_patterns = defaultdict(float)
        self.path_success = defaultdict(float)
    
    def learn_from_interaction(self, query: str, path: List[int], 
                             success_score: float, user_feedback: float = 1.0):
        pattern = self._extract_pattern(query)
        path_sig = self._create_path_signature(path)
        
        # Update pattern success
        current = self.success_patterns[pattern]
        self.success_patterns[pattern] = (
            current * (1 - self.learning_rate) + 
            (success_score * user_feedback) * self.learning_rate
        )
        
        # Store successful patterns
        if success_score > 0.7:
            self.query_patterns[pattern].append(path_sig)
            self.path_success[path_sig] = success_score
    
    def predict_optimal_path(self, query: str) -> Optional[List[int]]:
        pattern = self._extract_pattern(query)
        if pattern in self.query_patterns and self.query_patterns[pattern]:
            # Get best performing path for this pattern
            best_path = max(self.query_patterns[pattern], 
                          key=lambda p: self.path_success.get(p, 0))
            return self._decode_path_signature(best_path)
        return None
    
    def _extract_pattern(self, query: str) -> str:
        # Simple pattern extraction - could be more sophisticated
        words = query.lower().split()
        key_words = [w for w in words if len(w) > 3][:3]
        return '_'.join(sorted(key_words))
    
    def _create_path_signature(self, path: List[int]) -> str:
        return ','.join(map(str, path[:10]))  # Limit signature length
    
    def _decode_path_signature(self, signature: str) -> List[int]:
        return [int(x) for x in signature.split(',') if x.isdigit()]


class QualityAssessmentEngine:
    def __init__(self, llm):
        self.llm = llm
        self.assessment_prompt = PromptTemplate(
            input_variables=["query", "answer", "context"],
            template="""Assess the quality of this answer on a scale of 0-1 for each dimension:

Query: {query}
Answer: {answer}
Context: {context}

Rate each dimension (0.0 to 1.0):
Completeness: [How well does the answer address all aspects of the query?]
Accuracy: [How factually correct is the answer based on the context?]
Relevance: [How relevant is the answer to the query?]
Coherence: [How well-structured and logical is the answer?]
Confidence: [How confident can we be in this answer?]

Provide scores as: completeness:X.X accuracy:X.X relevance:X.X coherence:X.X confidence:X.X"""
        )
    
    async def assess_quality(self, query: str, answer: str, context: str) -> AnswerQuality:
        try:
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(
                    self.assessment_prompt.format(query=query, answer=answer, context=context)
                )
            else:
                # Fallback to sync
                response = self.llm.invoke(
                    self.assessment_prompt.format(query=query, answer=answer, context=context)
                )
            
            return self._parse_quality_scores(response)
        except Exception as e:
            print(f"Quality assessment failed: {e}")
            return AnswerQuality(0.5, 0.5, 0.5, 0.5, 0.5)  # Default scores

    def _parse_quality_scores(self, response: str) -> AnswerQuality:
        try:
            scores = {}
            for line in response.split('\n'):
                for dimension in ['completeness', 'accuracy', 'relevance', 'coherence', 'confidence']:
                    if dimension in line.lower():
                        # Extract score
                        parts = line.split(':')
                        if len(parts) > 1:
                            score_str = parts[1].strip().split()[0]
                            scores[dimension] = float(score_str)
                        break
            
            return AnswerQuality(
                completeness=scores.get('completeness', 0.5),
                accuracy=scores.get('accuracy', 0.5),
                relevance=scores.get('relevance', 0.5),
                coherence=scores.get('coherence', 0.5),
                confidence=scores.get('confidence', 0.5)
            )
        except:
            return AnswerQuality(0.5, 0.5, 0.5, 0.5, 0.5)


class EnhancedGraphTraversal:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.node_importance = self._compute_node_importance()
    
    def _compute_node_importance(self) -> Dict[int, float]:
        try:
            # Combine PageRank with degree centrality
            pagerank = nx.pagerank(self.kg.graph, weight='weight')
            degree_centrality = nx.degree_centrality(self.kg.graph)
            
            return {node: 0.7 * pagerank[node] + 0.3 * degree_centrality[node] 
                   for node in self.kg.graph.nodes}
        except:
            # Fallback to uniform importance
            return {node: 1.0 for node in self.kg.graph.nodes}
    
    def get_priority_neighbors(self, node: int, visited: set, top_k: int = 3) -> List[int]:
        neighbors = []
        for neighbor in self.kg.graph.neighbors(node):
            if neighbor not in visited:
                importance = self.node_importance.get(neighbor, 0.5)
                edge_weight = self.kg.graph[node][neighbor].get('weight', 1.0)
                priority = importance * edge_weight
                neighbors.append((neighbor, priority))
        
        # Return top-k neighbors by priority
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n[0] for n in neighbors[:top_k]]


# Main QueryEngine Class
class EnhancedQueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm, config: QueryEngineConfig = None):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.config = config or QueryEngineConfig()
        
        # Initialize components
        self.cache = CacheManager(self.config.cache_size) if self.config.enable_caching else None
        self.context_manager = ContextManager(self.config.max_context_length)
        self.learning_engine = OnlineLearningEngine(self.config.learning_rate) if self.config.enable_learning else None
        self.quality_engine = QualityAssessmentEngine(llm)
        self.graph_traversal = EnhancedGraphTraversal(knowledge_graph)
        
        # Initialize answer checking
        self.answer_check_chain = self._create_answer_check_chain()
        
        # Metrics
        self.metrics = QueryMetrics()
        self._start_time = 0
    
    def _create_answer_check_chain(self):
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Given the query: '{query}'

Context:
{context}

Analyze if this context provides a complete answer. Respond with:
1. is_complete: true/false
2. answer: the answer if complete, or "incomplete" if not
3. confidence: score from 0.0 to 1.0

Format: is_complete:true/false answer:your_answer confidence:0.X"""
        )
        return prompt | self.llm.with_structured_output(AnswerCheck)
    
    async def _check_answer_async(self, query: str, context: str) -> Tuple[bool, str, float]:
        # Check cache first
        if self.cache:
            cached = self.cache.get(query, context)
            if cached:
                return cached[0], cached[1], getattr(cached, 'confidence', 0.7)
        
        # Optimize context
        optimized_context = self.context_manager.optimize_context(context, query)
        
        try:
            if self.config.enable_async and hasattr(self.answer_check_chain, 'ainvoke'):
                response = await self.answer_check_chain.ainvoke({
                    "query": query, 
                    "context": optimized_context
                })
            else:
                # Fallback to sync
                response = self.answer_check_chain.invoke({
                    "query": query, 
                    "context": optimized_context
                })
            
            result = (response.is_complete, response.answer, response.confidence)
            
            # Cache result
            if self.cache:
                self.cache.set(query, context, (response.is_complete, response.answer))
            
            return result
        except Exception as e:
            print(f"Answer check failed: {e}")
            return False, "Error in answer checking", 0.0
    
    def _check_answer(self, query: str, context: str) -> Tuple[bool, str, float]:
        # Sync wrapper for async method
        if self.config.enable_async:
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._check_answer_async(query, context))
            except:
                pass
        
        # Fallback sync implementation
        if self.cache:
            cached = self.cache.get(query, context)
            if cached:
                return cached[0], cached[1], 0.7
        
        optimized_context = self.context_manager.optimize_context(context, query)
        
        try:
            response = self.answer_check_chain.invoke({
                "query": query, 
                "context": optimized_context
            })
            
            if self.cache:
                self.cache.set(query, context, (response.is_complete, response.answer))
            
            return response.is_complete, response.answer, getattr(response, 'confidence', 0.7)
        except Exception as e:
            print(f"Answer check failed: {e}")
            return False, "Error in answer checking", 0.0

    async def _expand_context_async(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        # Check for learned optimal path
        predicted_path = None
        if self.learning_engine:
            predicted_path = self.learning_engine.predict_optimal_path(query)
        
        if predicted_path:
            print("Using learned optimal path")
            return await self._follow_predicted_path(query, predicted_path)
        
        return await self._dijkstra_traversal(query, relevant_docs)
    
    async def _dijkstra_traversal(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""
        
        priority_queue = []
        distances = {}
        visited_nodes = set()
        
        print(f"\nStarting enhanced graph traversal for: {query}")
        
        # Initialize with best nodes from relevant docs
        for doc in relevant_docs:
            try:
                closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
                if closest_nodes:
                    closest_node_content, similarity_score = closest_nodes[0]
                    
                    # Find corresponding node in knowledge graph
                    for node_id in self.knowledge_graph.graph.nodes:
                        if self.knowledge_graph.graph.nodes[node_id]['content'] == closest_node_content.page_content:
                            priority = 1 / max(similarity_score, 0.001)
                            heapq.heappush(priority_queue, (priority, node_id))
                            distances[node_id] = priority
                            break
            except Exception as e:
                print(f"Error initializing from doc: {e}")
                continue
        
        step = 0
        while priority_queue and len(traversal_path) < self.config.max_traversal_nodes:
            current_priority, current_node = heapq.heappop(priority_queue)
            
            if current_node in visited_nodes:
                continue
            
            visited_nodes.add(current_node)
            
            if current_priority > distances.get(current_node, float('inf')):
                continue
            
            step += 1
            traversal_path.append(current_node)
            
            try:
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node].get('concepts', [])
                
                filtered_content[current_node] = node_content
                expanded_context += f"\n{node_content}" if expanded_context else node_content
                
                print(f"Step {step} - Node {current_node}: {node_content[:100]}...")
                
                # Check for complete answer
                is_complete, answer, confidence = self._check_answer(query, expanded_context)
                if is_complete and confidence > self.config.confidence_threshold:
                    final_answer = answer
                    print(f"Complete answer found with confidence {confidence:.2f}")
                    break
                
                # Update visited concepts
                new_concepts = set(node_concepts) - visited_concepts
                if new_concepts:
                    visited_concepts.update(new_concepts)
                    
                    # Get priority neighbors
                    priority_neighbors = self.graph_traversal.get_priority_neighbors(
                        current_node, visited_nodes, self.config.top_neighbors_count
                    )
                    
                    for neighbor in priority_neighbors:
                        if neighbor not in visited_nodes:
                            edge_weight = self.knowledge_graph.graph[current_node][neighbor].get('weight', 1.0)
                            distance = current_priority + (1 / max(edge_weight, 0.001))
                            
                            if distance < distances.get(neighbor, float('inf')):
                                distances[neighbor] = distance
                                heapq.heappush(priority_queue, (distance, neighbor))
                
            except Exception as e:
                print(f"Error processing node {current_node}: {e}")
                continue
        
        # Generate final answer if not found
        if not final_answer:
            final_answer = await self._generate_final_answer(query, expanded_context)
        
        return expanded_context, traversal_path, filtered_content, final_answer
    
    async def _generate_final_answer(self, query: str, context: str) -> str:
        response_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Based on the following context, provide a comprehensive answer to the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
        )
        
        try:
            if self.config.enable_async and hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(response_prompt.format(query=query, context=context))
            else:
                response = self.llm.invoke(response_prompt.format(query=query, context=context))
            
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"Error generating final answer: {e}")
            return "Unable to generate answer due to processing error."
    
    async def _follow_predicted_path(self, query: str, path: List[int]) -> Tuple[str, List[int], Dict[int, str], str]:
        expanded_context = ""
        filtered_content = {}
        
        for node_id in path:
            if node_id in self.knowledge_graph.graph.nodes:
                content = self.knowledge_graph.graph.nodes[node_id]['content']
                filtered_content[node_id] = content
                expanded_context += f"\n{content}" if expanded_context else content
        
        final_answer = await self._generate_final_answer(query, expanded_context)
        return expanded_context, path, filtered_content, final_answer
    
    def _retrieve_relevant_documents(self, query: str):
        print("Retrieving relevant documents...")
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": self.config.similarity_search_k}
            )
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=retriever
            )
            return compression_retriever.invoke(query)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            # Fallback to simple similarity search
            return self.vector_store.similarity_search(query, k=self.config.similarity_search_k)
    
    async def query_async(self, query: str) -> Tuple[str, List[int], Dict[int, str], QueryMetrics, AnswerQuality]:
        self._start_time = time.time()
        self.metrics = QueryMetrics()
        
        try:
            with get_openai_callback() as cb:
                print(f"\nProcessing query: {query}")
                
                # Retrieve relevant documents
                relevant_docs = self._retrieve_relevant_documents(query)
                
                # Expand context using enhanced traversal
                expanded_context, traversal_path, filtered_content, final_answer = await self._expand_context_async(query, relevant_docs)
                
                # Assess answer quality
                quality = await self.quality_engine.assess_quality(query, final_answer, expanded_context)
                
                # Update metrics
                self.metrics.nodes_visited = len(traversal_path)
                self.metrics.cache_hits = self.cache.hits if self.cache else 0
                self.metrics.total_time = time.time() - self._start_time
                self.metrics.confidence_score = quality.confidence
                self.metrics.quality_score = quality.overall_score
                
                # Learn from interaction if quality is good
                if self.learning_engine and quality.overall_score > 0.7:
                    self.learning_engine.learn_from_interaction(query, traversal_path, quality.overall_score)
                
                print(f"\nFinal Answer: {final_answer}")
                print(f"Quality Score: {quality.overall_score:.2f}")
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                
                return final_answer, traversal_path, filtered_content, self.metrics, quality
                
        except Exception as e:
            print(f"Error processing query: {e}")
            fallback_answer = f"Error processing query: {str(e)}"
            return fallback_answer, [], {}, self.metrics, AnswerQuality()
    
    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """Synchronous query method for backward compatibility"""
        try:
            if self.config.enable_async:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(self.query_async(query))
                return result[0], result[1], result[2]  # Return only the first 3 elements
            else:
                return self._sync_query(query)
        except Exception as e:
            print(f"Query failed: {e}")
            return f"Error: {str(e)}", [], {}
    
    def _sync_query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """Fallback synchronous implementation"""
        self._start_time = time.time()
        
        with get_openai_callback() as cb:
            print(f"\nProcessing query (sync): {query}")
            relevant_docs = self._retrieve_relevant_documents(query)
            
            # Simplified sync traversal
            expanded_context, traversal_path, filtered_content, final_answer = self._sync_expand_context(query, relevant_docs)
            
            print(f"\nFinal Answer: {final_answer}")
            print(f"Total Tokens: {cb.total_tokens}")
            
            return final_answer, traversal_path, filtered_content
    
    def _sync_expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """Simplified synchronous context expansion"""
        expanded_context = ""
        traversal_path = []
        filtered_content = {}
        
        # Process only the first few relevant docs to keep it simple
        for i, doc in enumerate(relevant_docs[:3]):
            traversal_path.append(i)
            filtered_content[i] = doc.page_content
            expanded_context += f"\n{doc.page_content}" if expanded_context else doc.page_content
        
        # Generate answer
        final_answer = self.llm.invoke(f"Based on this context: {expanded_context}\n\nAnswer this query: {query}")
        return expanded_context, traversal_path, filtered_content, str(final_answer)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            "cache_hit_rate": self.cache.hits / max(self.cache.hits + self.cache.misses, 1) if self.cache else 0,
            "avg_nodes_visited": self.metrics.nodes_visited,
            "avg_processing_time": self.metrics.total_time,
            "avg_confidence": self.metrics.confidence_score,
            "avg_quality": self.metrics.quality_score
        }
        
        if self.learning_engine:
            stats["learned_patterns"] = len(self.learning_engine.success_patterns)
            stats["successful_paths"] = len(self.learning_engine.path_success)
        
        return stats