"""
BioRAG Central Configuration
Contains model names, API endpoints, and other configurable parameters
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class ModelConfig:
    """Configuration for models used in BioRAG"""
    
    # LLM Models
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3"
    
    # Embedding Models
    default_embedding: str = "scibert"  # Maps to sentence-transformers/all-mpnet-base-v2
    fallback_embedding: str = "minilm"  # Maps to sentence-transformers/all-MiniLM-L6-v2
    
    # Cross-encoder for reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Model mappings for biomedical embeddings
    embedding_models: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize model mappings"""
        if self.embedding_models is None:
            self.embedding_models = {
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


@dataclass
class APIConfig:
    """Configuration for API endpoints and limits"""
    
    # Rate limits
    requests_per_minute: int = 60
    max_concurrent_requests: int = 10
    
    # Timeouts
    request_timeout: int = 30
    llm_timeout: int = 120
    
    # Content limits
    max_content_size_mb: int = 50
    max_items_per_rss: int = 20
    
    # Security
    default_allowed_domains: Optional[List[str]] = None
    block_internal_ips: bool = True


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters - moved from rag_chain.py"""
    # Standard retrieval
    similarity_k: int = 40
    mmr_k: int = 15
    mmr_fetch_k: int = 40
    
    # Ensemble retrieval
    ensemble_k: int = 25
    
    # Neighbor expansion
    neighbor_window: int = 2
    
    # Reranking
    rerank_top_n: int = 15
    
    # Fast mode (reduced parameters for speed)
    fast_mode: bool = False
    
    def __post_init__(self):
        """Apply fast mode settings if enabled"""
        if self.fast_mode:
            self.similarity_k = 15
            self.mmr_k = 8
            self.mmr_fetch_k = 20
            self.ensemble_k = 12
            self.neighbor_window = 1
            self.rerank_top_n = 8


@dataclass
class BioRAGConfig:
    """Main configuration class combining all settings"""
    
    models: ModelConfig
    api: APIConfig
    retrieval: RetrievalConfig
    
    # Paths
    data_dir: str = "./data"
    vector_db_dir: str = "./vector_db"
    cache_dir: str = "./cache"
    
    # Logging
    log_level: str = "INFO"
    debug_retrieval: bool = False
    
    def __init__(self, 
                 models: Optional[ModelConfig] = None,
                 api: Optional[APIConfig] = None,
                 retrieval: Optional[RetrievalConfig] = None):
        """Initialize with optional custom configurations"""
        self.models = models or ModelConfig()
        self.api = api or APIConfig()
        self.retrieval = retrieval or RetrievalConfig()
        
        # Override with environment variables if present
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        # Model overrides
        if os.getenv("BIORAG_OPENAI_MODEL"):
            self.models.openai_model = os.getenv("BIORAG_OPENAI_MODEL")
        
        if os.getenv("BIORAG_OLLAMA_MODEL"):
            self.models.ollama_model = os.getenv("BIORAG_OLLAMA_MODEL")
        
        if os.getenv("BIORAG_EMBEDDING_MODEL"):
            self.models.default_embedding = os.getenv("BIORAG_EMBEDDING_MODEL")
        
        # Path overrides
        if os.getenv("BIORAG_DATA_DIR"):
            self.data_dir = os.getenv("BIORAG_DATA_DIR")
        
        if os.getenv("BIORAG_VECTOR_DB_DIR"):
            self.vector_db_dir = os.getenv("BIORAG_VECTOR_DB_DIR")
        
        # Fast mode override
        if os.getenv("BIORAG_FAST_MODE", "").lower() in ("true", "1", "yes"):
            self.retrieval.fast_mode = True
        
        # Debug override
        if os.getenv("BIORAG_DEBUG", "").lower() in ("true", "1", "yes"):
            self.debug_retrieval = True
            self.log_level = "DEBUG"
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'BioRAGConfig':
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            models = ModelConfig(**config_data.get('models', {}))
            api = APIConfig(**config_data.get('api', {}))
            retrieval = RetrievalConfig(**config_data.get('retrieval', {}))
            
            config = cls(models=models, api=api, retrieval=retrieval)
            
            # Override other settings
            for key, value in config_data.items():
                if key not in ['models', 'api', 'retrieval'] and hasattr(config, key):
                    setattr(config, key, value)
            
            return config
            
        except ImportError:
            raise ImportError("PyYAML is required to load config from file. Install with: pip install pyyaml")
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        try:
            import yaml
            
            config_data = {
                'models': {
                    'openai_model': self.models.openai_model,
                    'ollama_model': self.models.ollama_model,
                    'default_embedding': self.models.default_embedding,
                    'fallback_embedding': self.models.fallback_embedding,
                    'reranker_model': self.models.reranker_model
                },
                'api': {
                    'requests_per_minute': self.api.requests_per_minute,
                    'max_concurrent_requests': self.api.max_concurrent_requests,
                    'request_timeout': self.api.request_timeout,
                    'llm_timeout': self.api.llm_timeout,
                    'max_content_size_mb': self.api.max_content_size_mb,
                    'max_items_per_rss': self.api.max_items_per_rss,
                    'block_internal_ips': self.api.block_internal_ips
                },
                'retrieval': {
                    'similarity_k': self.retrieval.similarity_k,
                    'mmr_k': self.retrieval.mmr_k,
                    'mmr_fetch_k': self.retrieval.mmr_fetch_k,
                    'ensemble_k': self.retrieval.ensemble_k,
                    'neighbor_window': self.retrieval.neighbor_window,
                    'rerank_top_n': self.retrieval.rerank_top_n,
                    'fast_mode': self.retrieval.fast_mode
                },
                'data_dir': self.data_dir,
                'vector_db_dir': self.vector_db_dir,
                'cache_dir': self.cache_dir,
                'log_level': self.log_level,
                'debug_retrieval': self.debug_retrieval
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
        except ImportError:
            raise ImportError("PyYAML is required to save config to file. Install with: pip install pyyaml")


# Default global configuration instance
default_config = BioRAGConfig()