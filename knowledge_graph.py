from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from tqdm.notebook import tqdm
from tqdm.asyncio import tqdm
import asyncio
import os
import re
import pandas as pd

load_dotenv()

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class KnowledgeGraph:
    def __init__(self, batch_size: int = 10, entity_batch_size: int = 500, rel_batch_size: int = 200, 
                 url: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, 
                 model_name: Optional[str] = None, max_concurrent: int = 15):
        self.batch_size = batch_size
        self.entity_batch_size = entity_batch_size
        self.rel_batch_size = rel_batch_size
        self.max_concurrent = max_concurrent
        self.url = url or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.model_name = model_name or MODEL_NAME
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0, 
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            streaming=False,
            max_retries=2,
            request_timeout=30
        )

        self.graph = Neo4jGraph()
        self._graph_loaded = False
        
        # Compiled regex patterns for relationship cleaning
        self._relation_patterns = [
            (re.compile(r'[\s\-\.]+'), '_'),
            (re.compile(r'[^A-Z0-9_]'), ''),
            (re.compile(r'_+'), '_')
        ]

    def _get_node_count(self) -> int:
        """Get total node count - unified method for graph existence checks."""
        try:
            result = self.graph.query("MATCH (n) RETURN count(n) as count LIMIT 1")
            return result[0]['count'] if result else 0
        except Exception as e:
            print(f"Error getting node count: {e}")
            return 0

    def check_graph_exists(self) -> bool:
        """Check if the graph database contains any data."""
        return self._get_node_count() > 0

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
            print(f"Loading existing graph with {stats['nodes']} nodes and {stats['relationships']} relationships")
            
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
        
        # Get sample data in one query
        samples = self.graph.query("""
            MATCH (n:__Entity__) 
            WITH n LIMIT 10
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n.id as node_name, n.type as node_type, 
                   type(r) as relationship, m.id as target_name
            LIMIT 10
        """)
        
        sample_nodes = [{"name": s["node_name"], "type": s["node_type"]} 
                       for s in samples if s["node_name"]]
        sample_rels = [{"source": s["node_name"], "relationship": s["relationship"], "target": s["target_name"]} 
                      for s in samples if s["relationship"]]
        
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
        """Clean relationship name using compiled regex patterns."""
        clean = relation.upper()
        for pattern, replacement in self._relation_patterns:
            clean = pattern.sub(replacement, clean)
        clean = clean.strip('_')
        return clean if clean and not clean[0].isdigit() else 'RELATED_TO'

    def _execute_batch_query(self, query: str, batch_data: List[Dict], 
                           batch_size: int, desc: str = "Processing") -> None:
        """Unified batch execution with fallback to individual processing."""
        for i in tqdm(range(0, len(batch_data), batch_size), desc=desc):
            chunk = batch_data[i:i + batch_size]
            try:
                self.graph.query(query, {"batch_data": chunk})
            except Exception as e:
                print(f"Batch failed ({e}), using individual processing")
                self._execute_individual_fallback(query, chunk)

    def _execute_individual_fallback(self, query: str, chunk: List[Dict]):
        """Fallback to individual query execution."""
        for item in chunk:
            try:
                # Convert query to work with single item
                single_query = query.replace("UNWIND $batch_data AS", "WITH $batch_data AS")
                self.graph.query(single_query, {"batch_data": item})
            except:
                continue

    def _create_entities_batch(self, entities: Set[Tuple[str, str]]):
        """Batch entity creation with unified error handling."""
        if not entities:
            return
            
        batch_data = [{"name": name, "type": etype} for name, etype in entities]
        query = """
        UNWIND $batch_data AS entity_data
        MERGE (e:__Entity__ {id: entity_data.name})
        SET e.type = entity_data.type
        """
        
        self._execute_batch_query(query, batch_data, 
                                min(self.entity_batch_size, 1000), "Creating entities")

    def _create_relationships_batch(self, relationships: List[Tuple[str, str, str]]):
        """Batch relationship creation with unified error handling."""
        if not relationships:
            return
        
        # Group by relationship type
        rel_groups = defaultdict(list)
        for e1, rel, e2 in relationships:
            clean_rel = self._clean_relationship_name(rel)
            rel_groups[clean_rel].append({"e1": e1, "e2": e2})
        
        # Process each relationship type
        for rel_type, pairs in rel_groups.items():
            query = f"""
            UNWIND $batch_data AS rel
            MERGE (e1:__Entity__ {{id: rel.e1}})
            MERGE (e2:__Entity__ {{id: rel.e2}})
            MERGE (e1)-[:{rel_type}]->(e2)
            """
            
            self._execute_batch_query(query, pairs, 
                                    min(self.rel_batch_size, 500), f"Creating {rel_type}")

    async def create_graph_from_documents(self, documents: List[Document]):
        """Async processing with controlled concurrency."""
        print(f"Processing {len(documents)} documents with max {self.max_concurrent} concurrent calls...")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_doc_with_semaphore(doc):
            async with semaphore:
                return await self._fast_extract_async(doc.page_content)
        
        all_entities = set()
        all_relationships = []
        
        # Process in batches to manage memory
        doc_batches = [documents[i:i+50] for i in range(0, len(documents), 50)]
        
        for batch_idx, doc_batch in enumerate(doc_batches):
            print(f"Processing batch {batch_idx + 1}/{len(doc_batches)}")
            
            tasks = [process_doc_with_semaphore(doc) for doc in doc_batch]
            
            batch_results = []
            for coro in tqdm.as_completed(tasks, desc=f"Batch {batch_idx + 1}"):
                try:
                    result = await coro
                    batch_results.append(result)
                except Exception as e:
                    print(f"Document processing failed: {e}")
                    continue
            
            # Combine results
            for entities, relationships in batch_results:
                all_entities.update(entities)
                all_relationships.extend(relationships)
            
            # Periodic database write
            if (batch_idx + 1) % 5 == 0:
                print(f"Writing intermediate results...")
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

    async def _fast_extract_async(self, text: str) -> Tuple[Set[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Fast entity and relationship extraction."""
        if len(text.strip()) < 50:
            return set(), []
            
        # Truncate very long texts
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
            result = await self.llm.ainvoke(prompt)
            return self._parse_extraction_result(result.content.strip())
        except Exception as e:
            print(f"Extraction failed: {e}")
            return set(), []

    def _parse_extraction_result(self, content: str) -> Tuple[Set[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Parse LLM extraction results."""
        entities = set()
        relationships = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('ENTITIES:'):
                for item in line[9:].strip().split(','):
                    item = item.strip()
                    if '|' in item:
                        name, etype = item.split('|', 1)
                        entities.add((name.strip(), etype.strip()))
            
            elif line.startswith('RELATIONSHIPS:'):
                for rel_item in line[14:].strip().split(','):
                    rel_item = rel_item.strip()
                    if '→' in rel_item:
                        parts = [p.strip() for p in rel_item.split('→')]
                        if len(parts) == 3:
                            relationships.append(tuple(parts))
        
        return entities, relationships

    def _create_document_nodes(self, documents: List[Document], source_name: str):
        """Create document nodes for source tracking."""
        doc_data = [{
            "doc_id": f"{source_name}_{i}",
            "source_name": source_name,
            "content": doc.page_content[:1000],
            "added_timestamp": doc.metadata.get('added_timestamp', str(pd.Timestamp.now())),
            "original_metadata": str(doc.metadata)
        } for i, doc in enumerate(documents)]

        query = """
        UNWIND $batch_data AS doc
        CREATE (d:Document {
            id: doc.doc_id,
            source_name: doc.source_name,
            content: doc.content,
            added_timestamp: doc.added_timestamp,
            metadata: doc.original_metadata
        })
        """
        
        self._execute_batch_query(query, doc_data, 100, "Creating document nodes")

    async def incremental_update_with_source_tracking(self, documents: List[Document], source_name: str) -> Dict[str, Any]:
        """Add documents with source tracking."""
        print(f"Adding {len(documents)} documents from source: {source_name}")

        # Add metadata
        timestamp = str(pd.Timestamp.now())
        for doc in documents:
            doc.metadata.update({"source_name": source_name, "added_timestamp": timestamp})

        # Check for existing source
        existing_count = self.graph.query(
            "MATCH (d:Document {source_name: $source_name}) RETURN count(d) as count",
            {"source_name": source_name}
        )[0]['count']

        if existing_count > 0:
            print(f"Warning: Source '{source_name}' already exists with {existing_count} documents.")

        # Get initial stats and process
        initial_stats = self.get_graph_stats()
        await self.create_graph_from_documents(documents)
        self._create_document_nodes(documents, source_name)
        final_stats = self.get_graph_stats()

        return {
            "nodes_added": final_stats['nodes'] - initial_stats['nodes'],
            "relationships_added": final_stats['relationships'] - initial_stats['relationships'],
            "documents_processed": len(documents),
            "source_name": source_name,
            "status": "success"
        }

    async def replace_source_documents(self, documents: List[Document], source_name: str):
        """Replace all documents from a source."""
        print(f"Replacing documents from source: {source_name}")

        # Remove existing documents
        removed_count = self.graph.query(
            "MATCH (d:Document {source_name: $source_name}) DETACH DELETE d RETURN count(d) as removed",
            {"source_name": source_name}
        )[0]['removed']
        
        print(f"Removed {removed_count} existing documents")
        
        # Add new documents
        result = await self.incremental_update_with_source_tracking(documents, source_name)
        result['documents_replaced'] = removed_count
        return result

    def get_source_info(self, source_name: str = None) -> Dict[str, Any]:
        """Get information about document sources."""
        if source_name:
            result = self.graph.query("""
                MATCH (d:Document {source_name: $source_name})
                RETURN 
                    d.source_name as source,
                    count(d) as document_count,
                    min(d.added_timestamp) as first_added,
                    max(d.added_timestamp) as last_added
            """, {"source_name": source_name})
            
            return result[0] if result else {"error": f"Source '{source_name}' not found"}
        else:
            all_sources = self.graph.query("""
                MATCH (d:Document)
                RETURN 
                    d.source_name as source,
                    count(d) as document_count,
                    min(d.added_timestamp) as first_added,
                    max(d.added_timestamp) as last_added
                ORDER BY last_added DESC
            """)
            
            return {"sources": all_sources, "total_sources": len(all_sources)}

    def remove_source(self, source_name: str) -> Dict[str, Any]:
        """Remove all documents and related data from a specific source."""
        print(f"Removing all data from source: {source_name}")
        
        # Get stats before removal
        stats = self.graph.query("""
            MATCH (d:Document {source_name: $source_name})
            RETURN count(d) as docs
        """, {"source_name": source_name})[0]
        
        # Remove documents
        self.graph.query("MATCH (d:Document {source_name: $source_name}) DETACH DELETE d", 
                        {"source_name": source_name})
        
        # Clean up orphaned entities
        orphaned = self.graph.query("""
            MATCH (e:__Entity__)
            WHERE NOT EXISTS((e)-[:EXTRACTED_FROM]-(:Document))
            DETACH DELETE e
            RETURN count(e) as orphaned_removed
        """)[0]
        
        return {
            "source_name": source_name,
            "documents_removed": stats["docs"],
            "orphaned_entities_cleaned": orphaned["orphaned_removed"],
            "status": "success"
        }

    def create_graph(self, documents: List[Document]):
        """LangChain transformer approach."""
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        batch_size = min(self.batch_size, 20)
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing document batches"):
            batch = documents[i:i + batch_size]
            try:
                graph_documents = llm_transformer.convert_to_graph_documents(batch)
                self.graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
            except Exception as e:
                print(f"Batch {i//batch_size + 1} failed: {e}")
                # Individual fallback
                for doc in batch:
                    try:
                        graph_docs = llm_transformer.convert_to_graph_documents([doc])
                        self.graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
                    except:
                        continue

    def visualize_graph(self, limit: int = 50):
        """Visualize the graph using yfiles."""
        try:
            from neo4j import GraphDatabase
            from yfiles_jupyter_graphs import GraphWidget
            
            driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
            session = driver.session()
            
            query = f"MATCH (s)-[r]->(t) RETURN s,r,t LIMIT {limit}"
            widget = GraphWidget(graph=session.run(query).graph())
            widget.node_label_mapping = 'id'
            return widget
        except ImportError:
            print("Install required packages: pip install yfiles_jupyter_graphs")
        except Exception as e:
            print(f"Error visualizing graph: {e}")

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the graph."""
        return self.graph.query(query, params)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the current graph."""
        # Single query to get all stats
        stats_query = """
        MATCH (n) 
        OPTIONAL MATCH (n)-[r]->()
        OPTIONAL MATCH (e:__Entity__)
        RETURN 
            count(DISTINCT n) as nodes,
            count(DISTINCT r) as relationships,
            collect(DISTINCT type(r)) as rel_types,
            collect(DISTINCT e.type) as entity_types
        """
        
        result = self.graph.query(stats_query)[0]
        
        # Get detailed counts
        rel_type_counts = {}
        if result['rel_types']:
            for rel_type in result['rel_types']:
                if rel_type:  # Skip null values
                    count = self.graph.query(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")[0]['count']
                    rel_type_counts[rel_type] = count
        
        entity_type_counts = {}
        if result['entity_types']:
            for entity_type in result['entity_types']:
                if entity_type:  # Skip null values
                    count = self.graph.query(
                        "MATCH (n:__Entity__ {type: $type}) RETURN count(n) as count", 
                        {"type": entity_type}
                    )[0]['count']
                    entity_type_counts[entity_type] = count
        
        return {
            'nodes': result['nodes'],
            'relationships': result['relationships'],
            'relationship_types': rel_type_counts,
            'entity_types': entity_type_counts
        }