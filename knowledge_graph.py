from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional, Set, Tuple
import os
from langchain_core.documents import Document
from tqdm.notebook import tqdm
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from collections import defaultdict
from dotenv import load_dotenv
import asyncio
from tqdm.asyncio import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
load_dotenv()

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class KnowledgeGraph:
    def __init__(self, batch_size: int = 10, entity_batch_size: int = 500, rel_batch_size: int = 200, 
                 url: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, 
                 model_name: Optional[str] = None, max_concurrent: int = 15):  # Increased defaults + concurrency limit
        self.batch_size = batch_size
        self.entity_batch_size = entity_batch_size
        self.rel_batch_size = rel_batch_size
        self.max_concurrent = max_concurrent  # Limit concurrent LLM calls
        self.url = url or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.model_name = model_name or MODEL_NAME
        self.llm = ChatOpenAI(
            model = self.model_name,
            temperature = 0, 
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            streaming = False,  # Disable streaming for batch processing
            max_retries=2,      # Reduce retries
            request_timeout=30  # Add timeout
        )

        self.graph = Neo4jGraph()
        self._graph_loaded = False
        
        # Compiled regex patterns for faster text cleaning
        self._relation_patterns = [
                        (re.compile(r'[\s\-\.]+'), '_'),  # Replace one or more spaces, hyphens, or periods with a single underscore
                        (re.compile(r'[^A-Z0-9_]'), ''),  # Remove any character that is not an uppercase letter, digit, or underscore
                        (re.compile(r'_+'), '_')          # Replace multiple consecutive underscores with a single underscore
                    ]


    def check_graph_exists(self) -> bool:
        """Check if the graph database contains any data."""
        try:
            result = self.graph.query("MATCH (n) RETURN count(n) as count LIMIT 1")
            node_count = result[0]['count'] if result else 0
            return node_count > 0
        except Exception as e:
            print(f"Error checking graph existence: {e}")
            return False

    def load_existing_graph(self, force_reload: bool = False) -> bool:
        """Load existing graph data and set internal state."""
        if self._graph_loaded and not force_reload:
            print("Graph already loaded. Use force_reload=True to reload.")
            return True
            
        if not self.check_graph_exists():
            print("No existing graph found in the database.")
            return False
        
        try:
            stats = self.get_graph_stats()
            print(f"Loading existing graph...")
            print(f"Found {stats['nodes']} nodes and {stats['relationships']} relationships")
            
            if stats.get('entity_types'):
                print(f"Entity types: {list(stats['entity_types'].keys())}")
            if stats.get('relationship_types'):
                print(f"Relationship types: {list(stats['relationship_types'].keys())}")
            
            self._graph_loaded = True
            print("Graph loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading existing graph: {e}")
            return False

    async def load_or_create_graph(self, documents: Optional[List[Document]] = None, 
                       method: str = "custom") -> bool:
        """Load existing graph if it exists, otherwise create new one from documents."""
        if self.load_existing_graph():
            return True
        
        if documents:
            print("No existing graph found. Creating new graph from documents...")
            if method == "custom":
                await self.create_graph_from_documents(documents)  
            else:
                self.create_graph(documents)  
            self._graph_loaded = True
            return True
        
        print("No existing graph found and no documents provided for creation.")
        return False

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a detailed summary of the current graph."""
        if not self.check_graph_exists():
            return {"status": "empty", "message": "No graph data found"}
        
        stats = self.get_graph_stats()
        
        # Get sample nodes
        sample_nodes = self.graph.query("""
            MATCH (n:__Entity__) 
            RETURN n.id as name, n.type as type 
            LIMIT 10
        """)
        
        # Get sample relationships
        sample_rels = self.graph.query("""
            MATCH (a)-[r]->(b) 
            RETURN a.id as source, type(r) as relationship, b.id as target 
            LIMIT 10
        """)
        
        return {
            "status": "loaded",
            "statistics": stats,
            "sample_nodes": sample_nodes,
            "sample_relationships": sample_rels,
            "is_loaded": self._graph_loaded
        }

    def reset_graph_state(self):
        """Reset the internal graph state"""
        self._graph_loaded = False

    def clear_database(self):
        """Clear the database and reset internal state."""
        self.graph.query("MATCH (n) DETACH DELETE n")
        self.reset_graph_state()
        print("Database cleared successfully.")

    def _clean_relationship_name(self, relation: str) -> str:
        """Relationship name cleaning with compiled regex."""
        clean = relation.upper()
        for pattern, replacement in self._relation_patterns:
            clean = pattern.sub(replacement, clean)
        clean = clean.strip('_')
        if not clean or clean[0].isdigit():
            clean = 'RELATED_TO'
        return clean

    def _create_entities_batch(self, entities: Set[Tuple[str, str]]):
        """Batch entity creation."""
        if not entities:
            return
            
        entity_list = list(entities)
        batch_size = min(self.entity_batch_size, 1000)  
        
        # Single query for all entities using UNWIND
        batch_data = [{"name": name, "type": etype} for name, etype in entity_list]
        
        # Process in chunks
        for i in tqdm(range(0, len(batch_data), batch_size), desc="Creating entities"):
            chunk = batch_data[i:i + batch_size]
            query = """
            UNWIND $batch_data AS entity_data
            MERGE (e:__Entity__ {id: entity_data.name})
            SET e.type = entity_data.type
            """
            try:
                self.graph.query(query, {"batch_data": chunk})
            except Exception as e:
                print(f"Batch failed, using individual creation: {e}")
                for entity_data in chunk:
                    try:
                        self.graph.query(
                            "MERGE (e:__Entity__ {id: $name}) SET e.type = $type",
                            {"name": entity_data["name"], "type": entity_data["type"]}
                        )
                    except:
                        continue

    def _create_relationships_batch(self, relationships: List[Tuple[str, str, str]]):
        """Batch relationship creation."""
        if not relationships:
            return
            
        # Group by relationship type for better performance
        rel_groups = defaultdict(list)
        for e1, rel, e2 in relationships:
            clean_rel = self._clean_relationship_name(rel)
            rel_groups[clean_rel].append((e1, e2))
        
        # Process each relationship type
        for rel_type, pairs in rel_groups.items():
            batch_size = min(self.rel_batch_size, 500)
            
            for i in tqdm(range(0, len(pairs), batch_size), 
                         desc=f"Creating {rel_type}", leave=False):
                chunk = pairs[i:i + batch_size]
                batch_data = [{"e1": e1, "e2": e2} for e1, e2 in chunk]
                
                # Optimized single query
                query = f"""
                UNWIND $batch_data AS rel
                MERGE (e1:__Entity__ {{id: rel.e1}})
                MERGE (e2:__Entity__ {{id: rel.e2}})
                MERGE (e1)-[:{rel_type}]->(e2)
                """
                
                try:
                    self.graph.query(query, {"batch_data": batch_data})
                except Exception as e:
                    print(f"Batch relationship creation failed: {e}")
                    # Fallback to individual creation
                    for e1, e2 in chunk:
                        try:
                            self.graph.query(f"""
                                MERGE (e1:__Entity__ {{id: $e1}})
                                MERGE (e2:__Entity__ {{id: $e2}})
                                MERGE (e1)-[:{rel_type}]->(e2)
                            """, {"e1": e1, "e2": e2})
                        except:
                            continue

    async def create_graph_from_documents(self, documents: List[Document]):
        """async processing with controlled concurrency."""
        print(f"Processing {len(documents)} documents with max {self.max_concurrent} concurrent calls...")
        
        # Create semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_doc_with_semaphore(doc):
            async with semaphore:
                return await self._fast_extract_async(doc.page_content)
        
        # Process documents in batches to manage memory
        all_entities = set()
        all_relationships = []
        
        doc_batches = [documents[i:i+50] for i in range(0, len(documents), 50)]  # Process 50 docs at a time
        
        for batch_idx, doc_batch in enumerate(doc_batches):
            print(f"Processing batch {batch_idx + 1}/{len(doc_batches)}")
            
            # Create tasks for this batch
            tasks = [process_doc_with_semaphore(doc) for doc in doc_batch]
            
            # Execute with progress tracking
            batch_results = []
            for coro in tqdm.as_completed(tasks, desc=f"Batch {batch_idx + 1}"):
                try:
                    result = await coro
                    batch_results.append(result)
                except Exception as e:
                    print(f"Document processing failed: {e}")
                    continue
            
            # Combine batch results
            for entities, relationships in batch_results:
                all_entities.update(entities)
                all_relationships.extend(relationships)
            
            # Periodic database write to manage memory
            if (batch_idx + 1) % 5 == 0:  # Every 5 batches
                print(f"Writing intermediate results to database...")
                if all_entities:
                    self._create_entities_batch(all_entities)
                    all_entities.clear()
                if all_relationships:
                    self._create_relationships_batch(all_relationships)
                    all_relationships.clear()
        
        # Final write
        print(f"Final write: {len(all_entities)} entities, {len(all_relationships)} relationships")
        if all_entities:
            self._create_entities_batch(all_entities)
        if all_relationships:
            self._create_relationships_batch(all_relationships)

    async def _fast_extract_async(self, text: str) -> tuple:
        # Skip very short texts
        if len(text.strip()) < 50:
            return set(), []
            
        # Truncate very long texts to save on tokens
        if len(text) > 8000:
            text = text[:8000] + "..."
            
        prompt = f"""Extract medical/scientific entities and relationships from this text. Be precise and focus on key information.

            ENTITY TYPES: Researchers, Institutions, Medical_Conditions, Anatomy, Procedures, Medications, Equipment, Methods, Demographics, Locations, Journals, Time_Periods

            RELATIONSHIP TYPES: authored_by, conducted_at, published_in, diagnosed_with, treated_with, associated_with, affects, used_for, measures, occurs_in, validated_using

            TEXT: {text}

            OUTPUT FORMAT (exactly as shown):
            ENTITIES: Entity1|Type1, Entity2|Type2, Entity3|Type3
            RELATIONSHIPS: Entity1→relationship→Entity2, Entity3→relationship→Entity4

            Keep it concise and medically relevant."""

        try:
            result = await self.llm.ainvoke(prompt) # Async call the llm
            content = result.content.strip()
            
            entities = set()
            relationships = []
            
            # Fast parsing
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('ENTITIES:'):
                    entity_part = line[9:].strip()
                    for item in entity_part.split(','):
                        item = item.strip()
                        if '|' in item:
                            name, etype = item.split('|', 1)
                            entities.add((name.strip(), etype.strip()))
                
                elif line.startswith('RELATIONSHIPS:'):
                    rel_part = line[14:].strip()
                    for rel_item in rel_part.split(','):
                        rel_item = rel_item.strip()
                        if '→' in rel_item:
                            parts = [p.strip() for p in rel_item.split('→')]
                            if len(parts) == 3:
                                relationships.append(tuple(parts))
            
            return entities, relationships
            
        except Exception as e:
            print(f"Extraction failed for text chunk: {e}")
            return set(), []
    
    def create_graph(self, documents: List[Document]):
        """LangChain transformer approach."""
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        
        # Batch size not less than 20
        batch_size = min(self.batch_size, 20)
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing document batches"):
            batch = documents[i:i + batch_size]
            try:
                graph_documents = llm_transformer.convert_to_graph_documents(batch)
                self.graph.add_graph_documents(
                    graph_documents,
                    baseEntityLabel=True,
                    include_source=True
                )
            except Exception as e:
                print(f"Batch {i//batch_size + 1} failed: {e}")
                # Process individually as fallback
                for doc in batch:
                    try:
                        graph_docs = llm_transformer.convert_to_graph_documents([doc])
                        self.graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
                    except:
                        continue

    def visualize_graph(self, cypher_query="MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"):
        try:
            from neo4j import GraphDatabase
            from yfiles_jupyter_graphs import GraphWidget
            driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
            session = driver.session()
            widget = GraphWidget(graph=session.run(cypher_query).graph())
            widget.node_label_mapping = 'id'
            return widget
        except ImportError as e:
            print(f"Error importing required packages: {e}")
            print("Install: pip install yfiles_jupyter_graphs")
        except Exception as e:
            print(f"Error visualizing graph: {str(e)}")

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the graph."""
        return self.graph.query(query, params)

    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the current graph."""
        stats = {}
        
        # Count nodes
        node_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        stats['nodes'] = node_count
        
        # Count relationships
        rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        stats['relationships'] = rel_count
        
        # Count relationship types
        rel_types = self.graph.query("""
            MATCH ()-[r]->() 
            RETURN type(r) as rel_type, count(r) as count 
            ORDER BY count DESC
        """)
        stats['relationship_types'] = {rt['rel_type']: rt['count'] for rt in rel_types}
        
        # Count entity types
        entity_types = self.graph.query("""
            MATCH (n:__Entity__) 
            WHERE n.type IS NOT NULL
            RETURN n.type as entity_type, count(n) as count 
            ORDER BY count DESC
        """)
        stats['entity_types'] = {et['entity_type']: et['count'] for et in entity_types}
        
        return stats