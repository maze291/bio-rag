# Ensemble Retriever - BM25 + Sentence Transformers
# Critical fix for numeric/scientific term retrieval

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
import logging

logger = logging.getLogger(__name__)

class EnsembleRetriever:
    """
    Combines BM25 (exact term matching) with dense vector search
    Essential for scientific papers with numbers, units, wavelengths
    """
    
    def __init__(self, dense_retriever, documents: List[Document]):
        """
        Args:
            dense_retriever: Existing sentence-transformers retriever 
            documents: Same documents used for vector index
        """
        self.dense_retriever = dense_retriever
        
        # Create BM25 retriever from same documents
        logger.info("Creating BM25 retriever for exact term matching...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 25  # Get more candidates for fusion
        
        logger.info("✅ Ensemble retriever ready (BM25 + Dense)")
    
    def get_relevant_documents(self, query: str, k: int = 15) -> List[Document]:
        """
        Retrieve using both BM25 and dense, then fuse results
        """
        try:
            # BM25 retrieval (great for exact terms like "41-fold", "μM h⁻¹")
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)
            logger.info(f"BM25 retrieved {len(bm25_docs)} documents")
            
            # Dense retrieval (great for semantic similarity)  
            dense_docs = self.dense_retriever.get_relevant_documents(query)
            logger.info(f"Dense retrieved {len(dense_docs)} documents")
            
            # Simple rank fusion
            doc_scores = {}
            
            # BM25 ranking (0-based, lower is better)
            for rank, doc in enumerate(bm25_docs):
                doc_key = self._get_doc_key(doc)
                doc_scores[doc_key] = doc_scores.get(doc_key, 0) + (1.0 / (rank + 1))
            
            # Dense ranking  
            for rank, doc in enumerate(dense_docs):
                doc_key = self._get_doc_key(doc)
                doc_scores[doc_key] = doc_scores.get(doc_key, 0) + (1.0 / (rank + 1))
            
            # Combine and deduplicate
            all_docs = {self._get_doc_key(d): d for d in bm25_docs + dense_docs}
            
            # Sort by fused scores
            fused_docs = sorted(
                all_docs.values(),
                key=lambda d: doc_scores.get(self._get_doc_key(d), 0),
                reverse=True
            )
            
            result = fused_docs[:k]
            logger.info(f"✅ Ensemble returned {len(result)} fused documents")
            return result
            
        except Exception as e:
            logger.error(f"Ensemble retrieval failed: {str(e)}")
            # Fallback to dense only
            return self.dense_retriever.get_relevant_documents(query)
    
    def _get_doc_key(self, doc: Document) -> str:
        """Create unique key for document deduplication"""
        # Use metadata if available, otherwise content hash
        if hasattr(doc, 'metadata') and doc.metadata:
            doc_id = doc.metadata.get('doc_id', '')
            chunk_idx = doc.metadata.get('chunk_idx', '')
            if doc_id and chunk_idx:
                return f"{doc_id}_{chunk_idx}"
        
        # Fallback to content hash
        return str(hash(doc.page_content[:200]))