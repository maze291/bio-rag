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
        logger.info(f"🔄 ENSEMBLE FUSION: Starting retrieval for query: '{query}' (k={k})")
        
        try:
            # BM25 retrieval (great for exact terms like "41-fold", "μM h⁻¹")
            logger.info(f"🔄 Step 1: Running BM25 retrieval...")
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)
            logger.info(f"🔄 BM25 retrieved {len(bm25_docs)} documents")
            
            # Log top BM25 results
            for i, doc in enumerate(bm25_docs[:3]):
                preview = doc.page_content[:100].replace('\n', ' ')
                logger.info(f"🔄 BM25 #{i+1}: {preview}...")
            
            # Dense retrieval (great for semantic similarity)  
            logger.info(f"🔄 Step 2: Running Dense vector retrieval...")
            dense_docs = self.dense_retriever.get_relevant_documents(query)
            logger.info(f"🔄 Dense retrieved {len(dense_docs)} documents")
            
            # Log top Dense results
            for i, doc in enumerate(dense_docs[:3]):
                preview = doc.page_content[:100].replace('\n', ' ')
                logger.info(f"🔄 Dense #{i+1}: {preview}...")
            
            # Simple rank fusion
            logger.info(f"🔄 Step 3: Computing rank fusion scores...")
            doc_scores = {}
            
            # BM25 ranking (0-based, lower is better)
            logger.info(f"🔄 Processing BM25 rankings...")
            for rank, doc in enumerate(bm25_docs):
                doc_key = self._get_doc_key(doc)
                score = 1.0 / (rank + 1)
                doc_scores[doc_key] = doc_scores.get(doc_key, 0) + score
                logger.info(f"🔄 BM25 doc {doc_key[:20]}... rank={rank+1}, score={score:.3f}")
            
            # Dense ranking  
            logger.info(f"🔄 Processing Dense rankings...")
            for rank, doc in enumerate(dense_docs):
                doc_key = self._get_doc_key(doc)
                score = 1.0 / (rank + 1)
                old_score = doc_scores.get(doc_key, 0)
                new_score = old_score + score
                doc_scores[doc_key] = new_score
                overlap = "🔄 OVERLAP!" if old_score > 0 else "🔄 New doc"
                logger.info(f"🔄 Dense doc {doc_key[:20]}... rank={rank+1}, score={score:.3f}, total={new_score:.3f} {overlap}")
            
            # Combine and deduplicate
            logger.info(f"🔄 Step 4: Combining and deduplicating documents...")
            all_docs = {self._get_doc_key(d): d for d in bm25_docs + dense_docs}
            logger.info(f"🔄 Total unique documents before ranking: {len(all_docs)}")
            
            # Sort by fused scores
            logger.info(f"🔄 Step 5: Sorting by fused scores...")
            fused_docs = sorted(
                all_docs.values(),
                key=lambda d: doc_scores.get(self._get_doc_key(d), 0),
                reverse=True
            )
            
            # Log top fused results
            logger.info(f"🔄 Top 5 fused results:")
            for i, doc in enumerate(fused_docs[:5]):
                doc_key = self._get_doc_key(doc)
                final_score = doc_scores.get(doc_key, 0)
                preview = doc.page_content[:80].replace('\n', ' ')
                logger.info(f"🔄 #{i+1} (score={final_score:.3f}): {preview}...")
            
            result = fused_docs[:k]
            logger.info(f"✅ ENSEMBLE COMPLETE: Returned {len(result)} fused documents (requested {k})")
            return result
            
        except Exception as e:
            logger.error(f"💥 Ensemble retrieval failed: {str(e)}")
            logger.info(f"🔄 Falling back to dense-only retrieval...")
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