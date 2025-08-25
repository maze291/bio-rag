"""
BioRAG Vector Database Manager
Handles embeddings and vector storage with multiple model support
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import torch
import logging
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    Manages vector database operations with support for:
    - Multiple embedding models (SciBERT, BioBERT, OpenAI, etc.)
    - ChromaDB for persistence
    - Efficient similarity search
    - Hybrid search capabilities
    """

    def __init__(self,
                 embedding_model: str = "scibert",  # Now maps to sentence-transformers model
                 persist_directory: str = "./vector_db",
                 collection_name: str = "biorag_docs"):
        """
        Initialize Vector DB Manager

        Args:
            embedding_model: Model to use for embeddings
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the collection
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize embeddings
        self.embeddings = self._init_embeddings(embedding_model)

        # Store embedding dimensions
        self._embedding_dimensions = None

        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

    def _init_embeddings(self, model_name: str):
        """
        Initialize embedding model based on name

        Args:
            model_name: Name or path of the model

        Returns:
            Embeddings instance
        """
        # Special handling for OpenAI request
        if model_name == "openai":
            if os.getenv("OPENAI_API_KEY"):
                logger.info("Using OpenAI embeddings")
                return OpenAIEmbeddings(model="text-embedding-3-small")
            else:
                logger.warning("OpenAI requested but no API key found, falling back to default")
                model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Biomedical models mapping
        bio_models = {
            # PROPER sentence-transformers models (fine-tuned for similarity)
            "scibert": "sentence-transformers/all-mpnet-base-v2",  # GPT's recommended baseline
            "scibert-sentence": "pritamdeka/S-Scibert-snli-multinli-stsb",
            "biobert-sentence": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", 
            "mpnet": "sentence-transformers/all-mpnet-base-v2",  # GPT's recommended baseline
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            # Legacy BERT models (not recommended for retrieval)
            "scibert-legacy": "allenai/scibert_scivocab_uncased",
            "biobert": "dmis-lab/biobert-v1.1", 
            "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
            "default": "sentence-transformers/all-mpnet-base-v2"  # GPT's recommended baseline
        }

        # Resolve model name
        if model_name in bio_models:
            model_name = bio_models[model_name]
        elif not model_name:
            model_name = bio_models["scibert"]

        logger.info(f"Loading embedding model: {model_name}")

        # Configure model kwargs
        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs['device'] = 'cuda'
            logger.info("Using GPU for embeddings")
        else:
            model_kwargs['device'] = 'cpu'
            logger.info("Using CPU for embeddings")

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}
        )

        # Get embedding dimensions
        try:
            test_embedding = embeddings.embed_query("test")
            self._embedding_dimensions = len(test_embedding)
            logger.info(f"Embedding dimensions: {self._embedding_dimensions}")
        except:
            self._embedding_dimensions = 768  # Default assumption

        return embeddings

    def create_db(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector database from documents

        Args:
            documents: List of documents to embed

        Returns:
            Chroma vector store instance
        """
        if not documents:
            raise ValueError("No documents provided to create database")

        logger.info(f"Creating vector database with {len(documents)} documents")

        # Save metadata about the database
        self._save_metadata()

        # Create Chroma instance
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # Persist to disk
        vectordb.persist()

        logger.info(f"Vector database created and persisted to {self.persist_directory}")

        return vectordb

    def load_db(self, persist_directory: Optional[str] = None) -> Chroma:
        """
        Load existing vector database

        Args:
            persist_directory: Directory containing the database

        Returns:
            Chroma vector store instance
        """
        persist_dir = persist_directory or self.persist_directory

        # Check if database exists
        if not Path(persist_dir).exists():
            raise FileNotFoundError(f"No database found at {persist_dir}")

        # Check metadata compatibility
        if not self._check_compatibility(persist_dir):
            logger.warning("Database was created with different settings. This may cause issues.")

        logger.info(f"Loading vector database from {persist_dir}")

        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        # Get collection info
        try:
            collection = vectordb._collection
            count = collection.count()
            logger.info(f"Loaded vector database with {count} documents")
        except Exception as e:
            logger.warning(f"Could not get document count: {str(e)}")

        return vectordb

    def add_documents(self, vectordb: Chroma, documents: List[Document]) -> None:
        """
        Add new documents to existing database

        Args:
            vectordb: Existing vector database
            documents: Documents to add
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector database")

        # Add documents
        vectordb.add_documents(documents)

        # Persist changes
        vectordb.persist()

        # Update metadata
        self._update_metadata(len(documents))

        logger.info("Documents added and persisted")

    def similarity_search(self,
                          vectordb: Chroma,
                          query: str,
                          k: int = 5,
                          filter: Optional[Dict] = None) -> List[Document]:
        """
        Perform similarity search

        Args:
            vectordb: Vector database instance
            query: Query text
            k: Number of results
            filter: Metadata filter

        Returns:
            List of similar documents
        """
        results = vectordb.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

        return results

    def similarity_search_with_score(self,
                                     vectordb: Chroma,
                                     query: str,
                                     k: int = 5,
                                     filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores

        Args:
            vectordb: Vector database instance
            query: Query text
            k: Number of results
            filter: Metadata filter

        Returns:
            List of (document, score) tuples
        """
        results = vectordb.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter
        )

        return results

    def mmr_search(self,
                   vectordb: Chroma,
                   query: str,
                   k: int = 5,
                   fetch_k: int = 15,
                   lambda_mult: float = 0.7) -> List[Document]:
        """
        Perform Maximum Marginal Relevance search

        Args:
            vectordb: Vector database instance
            query: Query text
            k: Number of results to return
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity parameter (0-1)

        Returns:
            List of diverse relevant documents
        """
        results = vectordb.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

        return results

    def hybrid_search(self,
                      vectordb: Chroma,
                      query: str,
                      k: int = 5,
                      alpha: float = 0.7) -> List[Document]:
        """
        Perform hybrid search combining similarity and keyword matching

        Args:
            vectordb: Vector database instance
            query: Query text
            k: Number of results
            alpha: Weight for similarity vs keyword (0-1)

        Returns:
            List of documents
        """
        # For large databases, use a more efficient approach
        collection_size = self._get_collection_size(vectordb)

        if collection_size > 10000:
            # For large collections, use only similarity search with keyword filter
            logger.info("Large collection detected, using filtered similarity search")

            # Extract key terms from query
            keywords = set(query.lower().split())
            important_keywords = [kw for kw in keywords if len(kw) > 3][:3]

            # Create filter for important keywords
            if important_keywords:
                # Use metadata filtering if available
                results = self.similarity_search_with_score(
                    vectordb, query, k=k * 2
                )
            else:
                results = self.similarity_search_with_score(
                    vectordb, query, k=k
                )

            # Filter and score based on keyword presence
            final_results = []
            for doc, sim_score in results:
                content_lower = doc.page_content.lower()
                keyword_score = sum(1 for kw in keywords if kw in content_lower) / len(keywords)
                combined_score = alpha * sim_score + (1 - alpha) * keyword_score
                final_results.append((doc, combined_score))

            # Sort by combined score
            final_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in final_results[:k]]

        else:
            # For smaller collections, use the original comprehensive approach
            # Get similarity results
            similarity_results = self.similarity_search_with_score(vectordb, query, k=k * 2)

            # Get keyword results
            keywords = query.lower().split()

            # Get sample of documents for keyword matching
            try:
                # Use pagination for efficiency
                all_docs = vectordb.get(limit=min(collection_size, 1000))

                keyword_results = []
                for i, content in enumerate(all_docs.get('documents', [])):
                    if content:
                        content_lower = content.lower()
                        score = sum(1 for keyword in keywords if keyword in content_lower)
                        if score > 0:
                            doc = Document(
                                page_content=content,
                                metadata=all_docs['metadatas'][i] if all_docs.get('metadatas') else {}
                            )
                            keyword_results.append((doc, score))

                # Sort keyword results by score
                keyword_results.sort(key=lambda x: x[1], reverse=True)
                keyword_results = keyword_results[:k]

            except Exception as e:
                logger.warning(f"Keyword search failed: {str(e)}")
                keyword_results = []

            # Combine results
            combined_results = {}

            # Add similarity results
            for doc, score in similarity_results:
                doc_id = self._get_doc_id(doc)
                combined_results[doc_id] = {
                    'doc': doc,
                    'similarity_score': score,
                    'keyword_score': 0
                }

            # Add keyword results
            for doc, score in keyword_results:
                doc_id = self._get_doc_id(doc)
                if doc_id in combined_results:
                    combined_results[doc_id]['keyword_score'] = score / len(keywords)
                else:
                    combined_results[doc_id] = {
                        'doc': doc,
                        'similarity_score': 0,
                        'keyword_score': score / len(keywords)
                    }

            # Calculate combined scores
            final_results = []
            for doc_id, scores in combined_results.items():
                combined_score = (alpha * scores['similarity_score'] +
                                  (1 - alpha) * scores['keyword_score'])
                final_results.append((scores['doc'], combined_score))

            # Sort by combined score
            final_results.sort(key=lambda x: x[1], reverse=True)

            # Return top k documents
            return [doc for doc, _ in final_results[:k]]

    def get_retriever(self,
                      vectordb: Chroma,
                      search_type: str = "similarity",
                      search_kwargs: Optional[Dict] = None):
        """
        Get a retriever instance for the vector database

        Args:
            vectordb: Vector database instance
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters

        Returns:
            Retriever instance
        """
        kwargs = search_kwargs or {"k": 5}

        return vectordb.as_retriever(
            search_type=search_type,
            search_kwargs=kwargs
        )

    def update_embeddings(self, model_name: str):
        """
        Update the embedding model

        Args:
            model_name: New model name
        """
        logger.info(f"Updating embedding model to {model_name}")
        old_model = self.embedding_model_name

        self.embedding_model_name = model_name
        self.embeddings = self._init_embeddings(model_name)

        logger.warning(f"Embedding model changed from {old_model} to {model_name}. "
                       "Existing vector database may be incompatible.")

    def delete_collection(self):
        """Delete the current collection safely"""
        try:
            # First try to load the database
            vectordb = self.load_db()

            # Delete the collection
            client = vectordb._client
            client.delete_collection(name=self.collection_name)

            # Remove metadata
            metadata_path = Path(self.persist_directory) / ".biorag_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()

            logger.info(f"Deleted collection {self.collection_name}")

        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            # Try to remove directory as fallback
            import shutil
            if Path(self.persist_directory).exists():
                shutil.rmtree(self.persist_directory)
                logger.info(f"Removed persist directory {self.persist_directory}")

    def get_collection_stats(self, vectordb: Chroma) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Args:
            vectordb: Vector database instance

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_documents": 0,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory,
            "embedding_dimensions": self._embedding_dimensions or "unknown"
        }

        try:
            # Get document count
            collection = vectordb._collection
            stats["total_documents"] = collection.count()

            # Get sample of metadata fields
            sample = collection.get(limit=1)
            if sample and sample.get('metadatas'):
                stats["metadata_fields"] = list(sample['metadatas'][0].keys())

            # Get actual embedding dimensions if not set
            if not self._embedding_dimensions and sample and sample.get('embeddings'):
                stats["embedding_dimensions"] = len(sample['embeddings'][0])
                self._embedding_dimensions = stats["embedding_dimensions"]

        except Exception as e:
            logger.warning(f"Could not get all collection stats: {str(e)}")

        return stats

    def _get_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document"""
        # Use first 100 chars of content + metadata hash
        content_part = doc.page_content[:100]
        metadata_str = json.dumps(doc.metadata, sort_keys=True)
        return hashlib.md5(f"{content_part}{metadata_str}".encode()).hexdigest()

    def _get_collection_size(self, vectordb: Chroma) -> int:
        """Get the size of the collection efficiently"""
        try:
            return vectordb._collection.count()
        except:
            return 0

    def _save_metadata(self):
        """Save metadata about the database"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "embedding_model": self.embedding_model_name,
            "embedding_dimensions": self._embedding_dimensions,
            "collection_name": self.collection_name,
            "version": "1.0"
        }

        metadata_path = Path(self.persist_directory) / ".biorag_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata: {str(e)}")

    def _update_metadata(self, docs_added: int = 0):
        """Update metadata file"""
        metadata_path = Path(self.persist_directory) / ".biorag_metadata.json"

        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            metadata.update({
                "last_updated": datetime.now().isoformat(),
                "last_operation": f"Added {docs_added} documents" if docs_added else "Updated"
            })

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not update metadata: {str(e)}")

    def _check_compatibility(self, persist_dir: str) -> bool:
        """Check if existing database is compatible"""
        metadata_path = Path(persist_dir) / ".biorag_metadata.json"

        if not metadata_path.exists():
            return True  # No metadata, assume compatible

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check embedding model
            if metadata.get("embedding_model") != self.embedding_model_name:
                logger.warning(f"Embedding model mismatch: database has {metadata.get('embedding_model')}, "
                               f"current is {self.embedding_model_name}")
                return False

            # Check dimensions if available
            if (metadata.get("embedding_dimensions") and
                    self._embedding_dimensions and
                    metadata["embedding_dimensions"] != self._embedding_dimensions):
                logger.warning(f"Embedding dimension mismatch: database has {metadata['embedding_dimensions']}, "
                               f"current is {self._embedding_dimensions}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Could not check compatibility: {str(e)}")
            return True  # Assume compatible on error