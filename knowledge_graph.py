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
import json

load_dotenv()

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class KnowledgeGraph:
    def __init__(self, max_concurrent: int = 50):  
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
            max_retries=1,  
            request_timeout=15, 
            max_tokens=1000  
        )
        self.graph = Neo4jGraph()
        
        # Compiled regex patterns for relationship cleaning
        self._relation_patterns = [
            (re.compile(r'[\s\-\.]+'), '_'),
            (re.compile(r'[^A-Z0-9_]'), ''),
            (re.compile(r'_+'), '_')
        ]
        
        # Batch processing
        self.entity_batch = []
        self.relationship_batch = []
        self.batch_size = 1000  

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

    def _batch_create_entities(self, entities_batch):
        """Create entities in batches."""
        if not entities_batch:
            return
            
        try:
            query = """
            UNWIND $entities as entity
            MERGE (e:__Entity__ {id: entity.name})
            SET e.type = entity.type
            WITH e, entity
            MATCH (d:Document {id: entity.doc_id})
            MERGE (e)-[:EXTRACTED_FROM]->(d)
            """
            
            batch_data = [
                {"name": name, "type": etype, "doc_id": doc_id}
                for name, etype, doc_id in entities_batch
            ]
            
            self.graph.query(query, {"entities": batch_data})
            
        except Exception as e:
            print(f"Failed to create entity batch: {e}")
            # Fallback to individual creation for failed batch
            for name, etype, doc_id in entities_batch:
                try:
                    self.graph.query("""
                        MERGE (e:__Entity__ {id: $name})
                        SET e.type = $type
                        WITH e
                        MATCH (d:Document {id: $doc_id})
                        MERGE (e)-[:EXTRACTED_FROM]->(d)
                    """, {"name": name, "type": etype, "doc_id": doc_id})
                except Exception as inner_e:
                    print(f"Failed to create entity {name}: {inner_e}")

    def _batch_create_relationships(self, relationships_batch):
        """Create relationships in batches."""
        if not relationships_batch:
            return
            
        # Group by relationship type 
        rel_groups = defaultdict(list)
        for e1, rel, e2, doc_id in relationships_batch:
            clean_rel = self._clean_relationship_name(rel)
            rel_groups[clean_rel].append((e1, e2, doc_id))
        
        for rel_type, relations in rel_groups.items():
            try:
                query = f"""
                UNWIND $relations as rel
                MERGE (e1:__Entity__ {{id: rel.e1}})
                MERGE (e2:__Entity__ {{id: rel.e2}})
                MERGE (e1)-[:{rel_type}]->(e2)
                """
                
                batch_data = [
                    {"e1": e1, "e2": e2, "doc_id": doc_id}
                    for e1, e2, doc_id in relations
                ]
                
                self.graph.query(query, {"relations": batch_data})
                
            except Exception as e:
                print(f"Failed to create relationship batch for {rel_type}: {e}")
                # Fallback to individual creation
                for e1, e2, doc_id in relations:
                    try:
                        fallback_query = f"""
                        MERGE (e1:__Entity__ {{id: $e1}})
                        MERGE (e2:__Entity__ {{id: $e2}})
                        MERGE (e1)-[:{rel_type}]->(e2)
                        """
                        self.graph.query(fallback_query, {"e1": e1, "e2": e2, "doc_id": doc_id})
                    except Exception as inner_e:
                        print(f"Failed to create relationship {e1}-{rel_type}-{e2}: {inner_e}")

    def _flush_batches(self):
        """Flush accumulated batches to database."""
        if self.entity_batch:
            self._batch_create_entities(self.entity_batch)
            self.entity_batch.clear()
            
        if self.relationship_batch:
            self._batch_create_relationships(self.relationship_batch)
            self.relationship_batch.clear()

    async def add_documents(self, documents: List[Document], source_name: str = None):
        """Add documents to the graph with batch processing."""
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
        
        # Create document nodes in batch
        self._batch_create_document_nodes(documents)

        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_doc_with_semaphore(doc):
            async with semaphore:
                return await self._extract_entities_relationships(doc.page_content, doc.metadata["doc_id"])
        
        batch_size = 200  
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            tasks = [process_doc_with_semaphore(doc) for doc in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Document processing failed: {result}")
                    continue
                    
                entities, relationships = result
                doc_id = batch[j].metadata["doc_id"]
                
                # Add to batch containers   
                for name, etype in entities:
                    self.entity_batch.append((name, etype, doc_id))
                
                for e1, rel, e2 in relationships:
                    self.relationship_batch.append((e1, rel, e2, doc_id))
                
                # Flush batches when they reach size limit
                if len(self.entity_batch) >= self.batch_size:
                    self._flush_batches()
        
        # Flush any remaining batches
        self._flush_batches()
        print("Processing complete!")

    def _batch_create_document_nodes(self, documents: List[Document]):
        """Create document nodes in batch."""
        try:
            query = """
            UNWIND $docs as doc
            CREATE (d:Document {
                id: doc.doc_id,
                source_name: doc.source_name,
                content: doc.content,
                added_timestamp: doc.added_timestamp,
                metadata: doc.metadata
            })
            """
            
            batch_data = [
                {
                    "doc_id": doc.metadata["doc_id"],
                    "source_name": doc.metadata["source_name"],
                    "content": doc.page_content[:1000],
                    "added_timestamp": doc.metadata["added_timestamp"],
                    "metadata": json.dumps(doc.metadata)  # Proper JSON serialization
                }
                for doc in documents
            ]
            
            self.graph.query(query, {"docs": batch_data})
            
        except Exception as e:
            print(f"Failed to create document nodes in batch: {e}")
            # Fallback to individual creation
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
                        "metadata": json.dumps(doc.metadata)
                    })
                except Exception as inner_e:
                    print(f"Failed to create document node: {inner_e}")
        
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

    async def _extract_entities_relationships(self, text: str, doc_id: str):
        """Extract entities and relationships from text."""
        if len(text.strip()) < 30: 
            return set(), []
            
        if len(text) > 4000: 
            text = text[:4000] + "..."

        prompt = f"""Extract key medical/scientific entities and relationships. Be concise.

        Text: {text}

        Format:
        ENTITIES: Name1|Type1, Name2|Type2
        RELATIONSHIPS: Entity1→relation→Entity2

        Types: Person, Institution, Medical_Condition, Medication, Equipment, Anatomy, Journal, Procedure
        Relations: treats, causes, located_in, published_in, affiliated_with, associated_with, reveals"""

        try:
            result = await self.llm.ainvoke(prompt)
            return self._parse_extraction_result(result.content.strip())
        except Exception as e:
            print(f"Extraction failed for doc {doc_id}: {e}")
            return set(), []

    def _parse_extraction_result(self, content: str):
        """Parse extraction result."""
        entities = set()
        relationships = []
        
        try:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('ENTITIES:'):
                    entity_text = line[9:].strip()
                    if entity_text:
                        for item in entity_text.split(','):
                            item = item.strip()
                            if '|' in item and len(item.split('|')) == 2:
                                name, etype = item.split('|', 1)
                                name, etype = name.strip(), etype.strip()
                                if name and etype:
                                    entities.add((name, etype))
                
                elif line.startswith('RELATIONSHIPS:'):
                    rel_text = line[14:].strip()
                    if rel_text:
                        for rel_item in rel_text.split(','):
                            rel_item = rel_item.strip()
                            if '→' in rel_item:
                                parts = [p.strip() for p in rel_item.split('→')]
                                if len(parts) == 3 and all(parts):
                                    relationships.append(tuple(parts))
        
        except Exception as e:
            print(f"Parsing error: {e}")
        
        return entities, relationships

    def get_source_info(self, source_name: str = None):
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

    def query(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Execute a Cypher query on the graph."""
        return self.graph.query(query, params)

    def get_graph_stats(self):
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