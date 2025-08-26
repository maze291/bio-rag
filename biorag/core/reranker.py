# Cross-Encoder Reranker for better relevance ranking
# Uses cross-encoder/ms-marco-MiniLM-L6-v2 for fast, accurate reranking

from typing import List, Tuple, Optional
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model for better relevance
    Essential after ensemble fusion for scientific queries
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: HuggingFace cross-encoder model name
        """
        self.model_name = model_name
        self.model = None
        
        try:
            logger.info(f"Loading cross-encoder model: {model_name}")
            self.model = CrossEncoder(model_name)
            logger.info("✅ Cross-encoder reranker loaded successfully")
        except Exception as e:
            logger.error(f"💥 Failed to load cross-encoder: {str(e)}")
            logger.info("🔄 Reranker will be disabled - using original rankings")
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Rerank documents by relevance to query
        
        Args:
            query: User query
            documents: Documents to rerank  
            top_k: Return top K documents (None = return all, reranked)
        
        Returns:
            Documents reranked by relevance score
        """
        if not self.model or not documents:
            logger.info("🔄 Cross-encoder not available or no documents - returning original order")
            return documents[:top_k] if top_k else documents
        
        if len(documents) == 1:
            logger.info("🔄 Only one document - skipping reranking")
            return documents
            
        logger.info(f"🎯 RERANKING: Starting rerank of {len(documents)} documents")
        logger.info(f"🎯 Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Prepare query-document pairs
            pairs = []
            doc_previews = []
            
            for i, doc in enumerate(documents):
                # Truncate content to avoid token limits (cross-encoders have ~512 token limit)
                content = doc.page_content[:2000]  # Keep more context than preview
                pairs.append([query, content])
                
                # Log preview
                preview = content[:100].replace('\n', ' ')
                doc_previews.append(f"Doc {i+1}: {preview}...")
                logger.info(f"🎯 Doc {i+1}: {preview}...")
            
            # Get relevance scores
            logger.info(f"🎯 Computing relevance scores...")
            scores = self.model.predict(pairs)
            
            # Convert to numpy for easier handling
            scores = np.array(scores)
            
            # Log scores
            logger.info(f"🎯 Relevance scores:")
            for i, (doc, score) in enumerate(zip(documents, scores)):
                preview = doc.page_content[:80].replace('\n', ' ')
                logger.info(f"🎯 #{i+1} (score={score:.4f}): {preview}...")
            
            # Sort by relevance score (descending)
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Extract reranked documents and update metadata
            reranked_docs = []
            logger.info(f"🎯 Final rankings after reranking:")
            
            for rank, (doc, score) in enumerate(scored_docs):
                # Update metadata with rerank score
                doc.metadata['rerank_score'] = float(score)
                doc.metadata['rerank_position'] = rank + 1
                
                preview = doc.page_content[:80].replace('\n', ' ')
                logger.info(f"🎯 #{rank+1} (score={score:.4f}): {preview}...")
                
                reranked_docs.append(doc)
            
            # Return top-k if specified
            result = reranked_docs[:top_k] if top_k else reranked_docs
            
            logger.info(f"✅ RERANKING COMPLETE: Returned {len(result)} documents")
            return result
            
        except Exception as e:
            logger.error(f"💥 Reranking failed: {str(e)}")
            logger.info(f"🔄 Falling back to original document order")
            return documents[:top_k] if top_k else documents
    
    def is_available(self) -> bool:
        """Check if reranker is available"""
        return self.model is not None