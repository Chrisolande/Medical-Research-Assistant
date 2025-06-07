from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from tqdm.notebook import tqdm
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
    def __init__(self, max_concurrent: int = 15):
        self.max_concurrent = max_concurrent
        self.url = NEO4J_URI
        self.username = NEO4J_USERNAME
        self.password = NEO4J_PASSWORD
        self.model_name = MODEL_NAME
        
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
        
        # Compiled regex patterns for relationship cleaning
        self._relation_patterns = [
            (re.compile(r'[\s\-\.]+'), '_'),
            (re.compile(r'[^A-Z0-9_]'), ''),
            (re.compile(r'_+'), '_')
        ]

    def check_graph_exists(self) -> bool:
        """Check if the graph database contains any data."""
        try:
            result = self.graph.query("MATCH (n) RETURN count(n) as count LIMIT 1")
            return result[0]['count'] > 0 if result else False
        except Exception:
            return False

    def clear_database(self):
        """Clear the database."""
        self.graph.query("MATCH (n) DETACH DELETE n")
        print("Database cleared successfully.")

    def _clean_relationship_name(self, relation: str) -> str:
        """Clean relationship name using compiled regex patterns."""
        clean = relation.upper()
        for pattern, replacement in self._relation_patterns:
            clean = pattern.sub(replacement, clean)
        clean = clean.strip('_')
        return clean if clean and not clean[0].isdigit() else 'RELATED_TO'

    def _create_entities(self, entities: Set[Tuple[str, str]], doc_id: str):
        """Create entities one by one to avoid batch issues."""
        for name, etype in entities:
            try:
                self.graph.query("""
                    MERGE (e:__Entity__ {id: $name})
                    SET e.type = $type
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (e)-[:EXTRACTED_FROM]->(d)
                """, {"name": name, "type": etype, "doc_id": doc_id})
            except Exception as e:
                print(f"Failed to create entity {name}: {e}")

    def _create_relationships(self, relationships: List[Tuple[str, str, str]], doc_id: str):
        """Create relationships one by one to avoid batch issues."""
        for e1, rel, e2 in relationships:
            try:
                clean_rel = self._clean_relationship_name(rel)
                # Fixed query - removed the problematic relationship tracking
                query = f"""
                    MERGE (e1:__Entity__ {{id: $e1}})
                    MERGE (e2:__Entity__ {{id: $e2}})
                    MERGE (e1)-[:{clean_rel}]->(e2)
                """
                self.graph.query(query, {"e1": e1, "e2": e2, "doc_id": doc_id})
                    
            except Exception as e:
                print(f"Failed to create relationship {e1}-{rel}-{e2}: {e}")

    async def add_documents(self, documents: List[Document], source_name: str = None):
        """Add documents to the graph (incremental updates supported)."""
        if not source_name:
            source_name = self._extract_source_name(documents)
        
        print(f"Processing {len(documents)} documents from source '{source_name}'...")
        
        # Add source metadata to documents
        timestamp = str(pd.Timestamp.now())
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source_name": source_name, 
                "added_timestamp": timestamp,
                "doc_id": f"{source_name}_{i}_{timestamp}"
            })
        
        # Create document nodes
        self._create_document_nodes(documents)

        # Process documents with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_doc_with_semaphore(doc):
            async with semaphore:
                return await self._extract_entities_relationships(doc.page_content)
        
        # Process documents in smaller batches to manage memory
        for i in range(0, len(documents), 100):
            batch = documents[i:i+100]
            print(f"Processing batch {i//100 + 1}/{(len(documents)-1)//100 + 1}")
            
            tasks = [process_doc_with_semaphore(doc) for doc in batch]
            
            for j, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting")):
                try:
                    entities, relationships = await task
                    doc_id = batch[j].metadata["doc_id"]
                    
                    if entities:
                        self._create_entities(entities, doc_id)
                    if relationships:
                        self._create_relationships(relationships, doc_id)
                        
                except Exception as e:
                    print(f"Document processing failed: {e}")
        
        print("Processing complete!")
        
    def _extract_source_name(self, documents: List[Document]) -> str:
        """Extract source name from document metadata."""
        if documents and 'source' in documents[0].metadata:
            source = documents[0].metadata['source']
            # Clean up file extensions and paths
            if '/' in source:
                source = source.split('/')[-1]
            if '\\' in source:
                source = source.split('\\')[-1]
            if '.' in source:
                source = source.rsplit('.', 1)[0]
            return source
        
        return f"documents_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    async def _extract_entities_relationships(self, text: str) -> Tuple[Set[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Extract entities and relationships from text."""
        if len(text.strip()) < 50:
            return set(), []
            
        # Truncate very long texts
        if len(text) > 8000:
            text = text[:8000] + "..."

        prompt = f"""Extract medical/scientific entities and relationships from text using these examples:

        Example 1:
        Text: "Dr. Smith from Harvard Medical School published a study on diabetes treatment using metformin in the Journal of Medicine."
        ENTITIES: Dr. Smith|Researcher, Harvard Medical School|Institution, diabetes|Medical_Condition, metformin|Medication, Journal of Medicine|Journal
        RELATIONSHIPS: Dr. Smith→affiliated_with→Harvard Medical School, Dr. Smith→published_in→Journal of Medicine, metformin→treats→diabetes

        Example 2:
        Text: "The MRI scan revealed lesions in the brain cortex of patients with multiple sclerosis."
        ENTITIES: MRI scan|Equipment, brain cortex|Anatomy, multiple sclerosis|Medical_Condition, lesions|Medical_Condition
        RELATIONSHIPS: MRI scan→reveals→lesions, lesions→located_in→brain cortex, lesions→associated_with→multiple sclerosis

        Now extract from this text:
        {text}

        OUTPUT FORMAT:
        ENTITIES: Entity1|Type1, Entity2|Type2, Entity3|Type3
        RELATIONSHIPS: Entity1→relationship→Entity2, Entity3→relationship→Entity4"""

        try:
            result = await self.llm.ainvoke(prompt)
            return self._parse_extraction_result(result.content.strip())
        except Exception as e:
            print(f"Extraction failed: {e}")
            return set(), []

    def _parse_extraction_result(self, content: str):
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

    def _create_document_nodes(self, documents: List[Document]):
        """Create document nodes for source tracking."""
        for doc in documents:
            try:
                self.graph.query("""
                    CREATE (d:Document {
                        id: $doc_id,
                        source_name: $source_name,
                        content: $content,
                        added_timestamp: $added_timestamp,
                        metadata: $metadata
                    })
                """, {
                    "doc_id": doc.metadata["doc_id"],
                    "source_name": doc.metadata["source_name"],
                    "content": doc.page_content[:1000],
                    "added_timestamp": doc.metadata["added_timestamp"],
                    "metadata": str(doc.metadata)
                })
            except Exception as e:
                print(f"Failed to create document node: {e}")

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
        
        # Remove documents and their relationships
        self.graph.query("""
            MATCH (d:Document {source_name: $source_name})
            DETACH DELETE d
        """, {"source_name": source_name})
        
        # Clean up orphaned entities and relationship metadata
        orphaned = self.graph.query("""
            MATCH (e:__Entity__)
            WHERE NOT EXISTS((e)-[:EXTRACTED_FROM]-(:Document))
            DETACH DELETE e
            RETURN count(e) as orphaned_removed
        """)[0]
        
        # Clean up orphaned relationship metadata
        self.graph.query("""
            MATCH (meta:RelationshipMeta)
            WHERE NOT EXISTS((meta)-[:DOCUMENTS_RELATIONSHIP]-(:Document))
            DELETE meta
        """)
        
        return {
            "source_name": source_name,
            "documents_removed": stats["docs"],
            "orphaned_entities_cleaned": orphaned["orphaned_removed"],
            "status": "success"
        }

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the graph."""
        return self.graph.query(query, params)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the current graph."""
        stats = self.graph.query("""
            MATCH (n) 
            OPTIONAL MATCH ()-[r]->()
            RETURN 
                count(DISTINCT n) as nodes,
                count(DISTINCT r) as relationships
        """)[0]
        
        # Get entity types
        entity_types = self.graph.query("""
            MATCH (e:__Entity__)
            RETURN e.type as type, count(e) as count
            ORDER BY count DESC
        """)
        
        # Get relationship types
        rel_types = self.graph.query("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
        """)
        
        return {
            'nodes': stats['nodes'],
            'relationships': stats['relationships'],
            'entity_types': {item['type']: item['count'] for item in entity_types if item['type']},
            'relationship_types': {item['type']: item['count'] for item in rel_types if item['type']}
        }

    def get_relationship_sources(self, entity1: str, entity2: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Get source documents for specific relationships."""
        if relationship_type:
            query = """
            MATCH (meta:RelationshipMeta {source: $e1, target: $e2, type: $rel_type})
            MATCH (meta)-[:DOCUMENTS_RELATIONSHIP]->(d:Document)
            RETURN d.source_name as source, d.id as doc_id, d.added_timestamp as timestamp
            """
            params = {"e1": entity1, "e2": entity2, "rel_type": relationship_type}
        else:
            query = """
            MATCH (meta:RelationshipMeta {source: $e1, target: $e2})
            MATCH (meta)-[:DOCUMENTS_RELATIONSHIP]->(d:Document)
            RETURN meta.type as relationship_type, d.source_name as source, d.id as doc_id, d.added_timestamp as timestamp
            """
            params = {"e1": entity1, "e2": entity2}
        
        return self.graph.query(query, params)

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