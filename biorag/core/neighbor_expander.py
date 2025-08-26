# Neighbor Expansion - Get adjacent chunks for distributed details
# Fixes: Fe2+ + ascorbate experiments spread across pages/captions

from typing import List, Dict, Any
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class NeighborExpander:
    """
    Expands retrieved chunks with their neighbors (±2 chunks)
    Critical for scientific papers where experimental details are split
    """
    
    def __init__(self, vector_db):
        """
        Args:
            vector_db: ChromaDB vector database instance
        """
        self.vector_db = vector_db
    
    def expand_with_neighbors(self, docs: List[Document], window: int = 2) -> List[Document]:
        """
        For each retrieved chunk, also fetch neighboring chunks by index
        
        Args:
            docs: Initial retrieved documents
            window: How many neighbors to fetch each side (±window)
        """
        try:
            expanded_docs = []
            seen_keys = set()
            
            logger.info(f"Expanding {len(docs)} docs with ±{window} neighbors")
            
            for doc in docs:
                # Add original document
                doc_key = self._get_doc_key(doc)
                if doc_key not in seen_keys:
                    expanded_docs.append(doc)
                    seen_keys.add(doc_key)
                
                # Try to get neighbors
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_id = doc.metadata.get('doc_id')
                    chunk_idx = doc.metadata.get('chunk_idx')
                    
                    if doc_id is not None and chunk_idx is not None:
                        try:
                            chunk_idx = int(chunk_idx)
                            
                            # Fetch neighboring chunks
                            for offset in range(-window, window + 1):
                                if offset == 0:  # Skip original
                                    continue
                                    
                                neighbor_idx = chunk_idx + offset
                                if neighbor_idx < 0:  # No negative indices
                                    continue
                                
                                # Try to fetch neighbor
                                neighbor_key = f"{doc_id}_{neighbor_idx}"
                                if neighbor_key in seen_keys:
                                    continue
                                
                                neighbor_doc = self._fetch_by_id(doc_id, neighbor_idx)
                                if neighbor_doc:
                                    expanded_docs.append(neighbor_doc)
                                    seen_keys.add(neighbor_key)
                                    
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not parse chunk_idx: {chunk_idx}")
            
            logger.info(f"✅ Expanded to {len(expanded_docs)} docs (including neighbors)")
            return expanded_docs
            
        except Exception as e:
            logger.error(f"Neighbor expansion failed: {str(e)}")
            return docs  # Return original if expansion fails
    
    def _get_doc_key(self, doc: Document) -> str:
        """Create unique key for document"""
        if hasattr(doc, 'metadata') and doc.metadata:
            doc_id = doc.metadata.get('doc_id', '')
            chunk_idx = doc.metadata.get('chunk_idx', '')
            if doc_id and chunk_idx:
                return f"{doc_id}_{chunk_idx}"
        
        return str(hash(doc.page_content[:200]))
    
    def _fetch_by_id(self, doc_id: str, chunk_idx: int) -> Document:
        """
        Fetch document by doc_id and chunk_idx from ChromaDB
        """
        try:
            # Try to get from collection
            collection = self.vector_db._collection
            
            # Try different ways to query the collection
            # First try exact metadata match
            results = None
            try:
                results = collection.get(
                    where={"$and": [
                        {"doc_id": {"$eq": doc_id}}, 
                        {"chunk_idx": {"$eq": str(chunk_idx)}}
                    ]},
                    include=["documents", "metadatas"]
                )
            except Exception:
                # Fallback: try simpler where clause
                try:
                    results = collection.get(
                        where={"chunk_idx": str(chunk_idx)},
                        include=["documents", "metadatas"]
                    )
                except Exception:
                    # Final fallback: get all and filter
                    all_results = collection.get(include=["documents", "metadatas"])
                    if all_results and all_results.get('metadatas'):
                        for i, meta in enumerate(all_results['metadatas']):
                            if (meta.get('doc_id') == doc_id and 
                                str(meta.get('chunk_idx')) == str(chunk_idx)):
                                results = {
                                    'documents': [all_results['documents'][i]],
                                    'metadatas': [meta]
                                }
                                break
            
            if results and results.get('documents') and len(results['documents']) > 0:
                # Convert back to Document
                content = results['documents'][0]
                metadata = results['metadatas'][0] if results.get('metadatas') else {}
                
                return Document(
                    page_content=content,
                    metadata=metadata
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch neighbor {doc_id}_{chunk_idx}: {str(e)}")
            return None