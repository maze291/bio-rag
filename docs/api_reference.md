# 📚 BioRAG API Reference

Complete API documentation for developers integrating with or extending BioRAG.

## 📦 Package Structure

```python
biorag/
├── core/
│   ├── ingest.py       # Document processing
│   ├── vectordb.py     # Vector storage and retrieval  
│   ├── linker.py       # Entity recognition and linking
│   ├── glossary.py     # Jargon simplification
│   └── rag_chain.py    # RAG orchestration
├── cli.py              # Command-line interface
└── main.py             # Streamlit web interface
```

## 🔧 Core Classes

### `IngestPipeline`

Document ingestion and preprocessing pipeline.

```python
from biorag.core.ingest import IngestPipeline

class IngestPipeline:
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None,
                 enable_ocr: bool = True)
```

#### **Methods**

##### `ingest_file(file_path: Union[str, Path]) -> List[Document]`
Process a single file and return document chunks.

**Parameters:**
- `file_path`: Path to file (PDF, HTML, TXT, MD)

**Returns:**
- List of LangChain Document objects with metadata

**Example:**
```python
ingester = IngestPipeline(chunk_size=500, chunk_overlap=50)
documents = ingester.ingest_file("research_paper.pdf")

for doc in documents:
    print(f"Chunk {doc.metadata['chunk_id']}: {doc.page_content[:100]}...")
```

##### `ingest_url(url: str) -> List[Document]`
Ingest content from a web URL.

**Parameters:**
- `url`: Web page URL

**Returns:**
- List of Document objects

**Example:**
```python
docs = ingester.ingest_url("https://www.nature.com/articles/s41586-020-2012-7")
```

##### `ingest_rss(rss_url: str, max_items: int = 20) -> List[Document]`
Ingest articles from RSS feed.

**Parameters:**
- `rss_url`: RSS feed URL
- `max_items`: Maximum articles to fetch

**Returns:**
- List of Document objects

**Example:**
```python
docs = ingester.ingest_rss("https://pubmed.ncbi.nlm.nih.gov/rss/search/cancer/", max_items=10)
```

##### `set_chunk_params(chunk_size: int, chunk_overlap: int)`
Update chunking parameters and clear cache.

**Parameters:**
- `chunk_size`: Target size for text chunks
- `chunk_overlap`: Overlap between chunks

**Example:**
```python
ingester.set_chunk_params(chunk_size=800, chunk_overlap=100)
```

---

### `VectorDBManager`

Vector database operations with ChromaDB backend.

```python
from biorag.core.vectordb import VectorDBManager

class VectorDBManager:
    def __init__(self,
                 embedding_model: str = "allenai/scibert_scivocab_uncased",
                 persist_directory: str = "./vector_db",
                 collection_name: str = "biorag_docs")
```

#### **Methods**

##### `create_db(documents: List[Document]) -> Chroma`
Create new vector database from documents.

**Parameters:**
- `documents`: List of Document objects to embed

**Returns:**
- ChromaDB instance

**Example:**
```python
db_manager = VectorDBManager(embedding_model="scibert")
vectordb = db_manager.create_db(documents)
```

##### `load_db(persist_directory: Optional[str] = None) -> Chroma`
Load existing vector database.

**Parameters:**
- `persist_directory`: Database directory path

**Returns:**
- ChromaDB instance

**Example:**
```python
vectordb = db_manager.load_db("./my_vector_db")
```

##### `similarity_search(vectordb: Chroma, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]`
Perform similarity search.

**Parameters:**
- `vectordb`: Vector database instance
- `query`: Search query
- `k`: Number of results
- `filter`: Metadata filter

**Returns:**
- List of similar documents

**Example:**
```python
results = db_manager.similarity_search(vectordb, "BRCA1 mutations", k=3)
```

##### `mmr_search(vectordb: Chroma, query: str, k: int = 5, fetch_k: int = 15, lambda_mult: float = 0.7) -> List[Document]`
Maximum Marginal Relevance search for diverse results.

**Parameters:**
- `vectordb`: Vector database instance
- `query`: Search query
- `k`: Number of results to return
- `fetch_k`: Number of candidates to fetch
- `lambda_mult`: Diversity parameter (0-1)

**Returns:**
- List of diverse relevant documents

**Example:**
```python
diverse_results = db_manager.mmr_search(vectordb, "cancer treatment", k=5, lambda_mult=0.8)
```

##### `get_collection_stats(vectordb: Chroma) -> Dict[str, Any]`
Get database statistics.

**Returns:**
- Dictionary with collection metadata

**Example:**
```python
stats = db_manager.get_collection_stats(vectordb)
print(f"Total documents: {stats['total_documents']}")
```

---

### `EntityLinker`

Biomedical entity recognition and linking.

```python
from biorag.core.linker import EntityLinker, Entity

class EntityLinker:
    def __init__(self)
```

#### **Entity Class**
```python
@dataclass
class Entity:
    text: str                    # Entity text
    label: str                   # Entity type (gene, protein, chemical, etc.)
    start: int                   # Start position in text
    end: int                     # End position in text
    kb_id: Optional[str] = None  # Knowledge base ID
    url: Optional[str] = None    # External database URL
    description: Optional[str] = None  # Entity description
    aliases: List[str] = field(default_factory=list)  # Alternative names
```

#### **Methods**

##### `extract_entities(text: str) -> List[Entity]`
Extract named entities from text.

**Parameters:**
- `text`: Input text

**Returns:**
- List of Entity objects

**Example:**
```python
entity_linker = EntityLinker()
entities = entity_linker.extract_entities("BRCA1 mutations cause breast cancer")

for entity in entities:
    print(f"{entity.text} ({entity.label}) -> {entity.url}")
```

##### `create_linked_html(text: str, entities: List[Entity]) -> str`
Create HTML with clickable entity links.

**Parameters:**
- `text`: Original text
- `entities`: List of entities

**Returns:**
- HTML string with linked entities

**Example:**
```python
html = entity_linker.create_linked_html(text, entities)
# Returns: BRCA1 <a href="https://uniprot.org/..." class="bio-entity">mutations</a>...
```

##### `get_entity_summary(entities: List[Entity]) -> Dict[str, List[Dict]]`
Group entities by type.

**Parameters:**
- `entities`: List of entities

**Returns:**
- Dictionary grouped by entity type

**Example:**
```python
summary = entity_linker.get_entity_summary(entities)
print(f"Found {len(summary['genes_proteins'])} genes/proteins")
```

---

### `GlossaryManager`

Jargon simplification and tooltip generation.

```python
from biorag.core.glossary import GlossaryManager

class GlossaryManager:
    def __init__(self, custom_glossary: Optional[Dict[str, str]] = None)
```

#### **Methods**

##### `simplify_text(text: str, format: str = "html") -> str`
Add tooltips to jargon terms.

**Parameters:**
- `text`: Input text
- `format`: Output format ("html" or "markdown")

**Returns:**
- Enhanced text with tooltips

**Example:**
```python
glossary_mgr = GlossaryManager()
enhanced = glossary_mgr.simplify_text("Apoptosis is important in cancer", format="html")
# Returns: <span class="jargon" title="programmed cell death">Apoptosis</span>...
```

##### `extract_jargon(text: str) -> List[Tuple[str, str]]`
Find all jargon terms in text.

**Parameters:**
- `text`: Input text

**Returns:**
- List of (term, definition) tuples

**Example:**
```python
jargon = glossary_mgr.extract_jargon("DNA methylation affects gene expression")
# Returns: [("methylation", "chemical modification..."), ...]
```

##### `add_terms(new_terms: Dict[str, str])`
Add custom terms to glossary.

**Parameters:**
- `new_terms`: Dictionary of term -> definition

**Example:**
```python
custom_terms = {
    "my_gene": "A novel gene I discovered",
    "my_process": "A specialized cellular process"
}
glossary_mgr.add_terms(custom_terms)
```

##### `search_terms(query: str) -> List[Tuple[str, str]]`
Search glossary terms.

**Parameters:**
- `query`: Search query

**Returns:**
- List of matching (term, definition) tuples

**Example:**
```python
results = glossary_mgr.search_terms("cell death")
# Returns terms related to cell death (apoptosis, necrosis, etc.)
```

---

### `RAGChain`

Complete RAG pipeline orchestration.

```python
from biorag.core.rag_chain import RAGChain

class RAGChain:
    def __init__(self,
                 vector_db,
                 entity_linker,
                 glossary_manager,
                 llm_model: Optional[str] = None)
```

#### **Methods**

##### `query(question: str, enable_hyde: bool = True, enable_decomposition: bool = True, enable_mmr: bool = True) -> Dict[str, Any]`
Process complete query through RAG pipeline.

**Parameters:**
- `question`: User's question
- `enable_hyde`: Use Hypothetical Document Embeddings
- `enable_decomposition`: Decompose complex queries
- `enable_mmr`: Use MMR for diverse retrieval

**Returns:**
- Dictionary with answer and metadata

**Example:**
```python
rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr)
result = rag_chain.query("How does BRCA1 affect DNA repair?")

print(f"Answer: {result['answer']}")
print(f"Entities found: {len(result['entities'])}")
print(f"Confidence: {result['confidence']}")
```

**Return Structure:**
```python
{
    "answer": str,                    # Plain text answer
    "enhanced_answer": str,           # HTML with entity links and tooltips
    "source_docs": List[Document],    # Supporting documents
    "entities": List[Dict],           # Detected entities
    "sub_queries": List[str],         # Sub-queries (if decomposed)
    "confidence": float               # Confidence score (0-1)
}
```

##### `explain_retrieval(question: str) -> Dict[str, Any]`
Explain the retrieval process for transparency.

**Parameters:**
- `question`: Query to explain

**Returns:**
- Dictionary explaining each processing step

**Example:**
```python
explanation = rag_chain.explain_retrieval("Complex query about cancer genetics")
for step in explanation['steps']:
    print(f"{step['step']}: {step['description']}")
```

---

## 🔌 Integration Examples

### **Basic Document Processing**
```python
from biorag.core.ingest import IngestPipeline
from biorag.core.vectordb import VectorDBManager

# Initialize components
ingester = IngestPipeline(chunk_size=500)
db_manager = VectorDBManager(embedding_model="scibert")

# Process documents
documents = ingester.ingest_file("research_paper.pdf")
vectordb = db_manager.create_db(documents)

# Search
results = db_manager.similarity_search(vectordb, "BRCA1 function", k=3)
```

### **Complete RAG Pipeline**
```python
from biorag.core.ingest import IngestPipeline
from biorag.core.vectordb import VectorDBManager
from biorag.core.linker import EntityLinker
from biorag.core.glossary import GlossaryManager
from biorag.core.rag_chain import RAGChain

# Initialize all components
ingester = IngestPipeline()
db_manager = VectorDBManager()
entity_linker = EntityLinker()
glossary_mgr = GlossaryManager()

# Process documents
documents = ingester.ingest_file("paper.pdf")
vectordb = db_manager.create_db(documents)

# Create RAG chain
rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr)

# Query
result = rag_chain.query("Explain BRCA1's role in cancer")
print(result['enhanced_answer'])
```

### **Custom Entity Knowledge Base**
```python
# Extend entity knowledge base
custom_entities = {
    "MY_GENE": {
        "type": "gene",
        "name": "My Novel Gene",
        "ncbi_gene_id": "12345",
        "aliases": ["NOVEL1", "NEWGENE"]
    }
}

# Add to entity linker's knowledge base
entity_linker.entity_kb.update(custom_entities)
entity_linker._build_alias_lookup()
```

### **Custom Glossary Terms**
```python
# Add domain-specific terms
custom_terms = {
    "crispr_cas9": "gene editing technology using RNA-guided nuclease",
    "organoid": "3D cell culture model resembling organ structure",
    "single_cell_sequencing": "method to analyze gene expression in individual cells"
}

glossary_mgr.add_terms(custom_terms)
```

### **Batch Processing**
```python
import os
from pathlib import Path

# Process multiple files
file_extensions = ['.pdf', '.txt', '.html']
documents = []

for file_path in Path("documents/").iterdir():
    if file_path.suffix in file_extensions:
        try:
            docs = ingester.ingest_file(str(file_path))
            documents.extend(docs)
            print(f"Processed {file_path.name}: {len(docs)} chunks")
        except Exception as e:
            print(f"Error with {file_path.name}: {e}")

# Create database from all documents
vectordb = db_manager.create_db(documents)
```

### **RSS Feed Monitoring**
```python
import schedule
import time

def update_knowledge_base():
    """Periodic RSS feed updates"""
    rss_feeds = [
        "https://pubmed.ncbi.nlm.nih.gov/rss/search/cancer/",
        "https://www.nature.com/nm.rss"
    ]
    
    new_docs = []
    for feed_url in rss_feeds:
        try:
            docs = ingester.ingest_rss(feed_url, max_items=5)
            new_docs.extend(docs)
        except Exception as e:
            print(f"Error with feed {feed_url}: {e}")
    
    if new_docs:
        db_manager.add_documents(vectordb, new_docs)
        print(f"Added {len(new_docs)} new documents")

# Schedule updates
schedule.every(6).hours.do(update_knowledge_base)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## 🛠️ Advanced Configuration

### **Custom Embedding Models**
```python
# Use different embedding models
models = {
    "biobert": "dmis-lab/biobert-v1.1",
    "clinical": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmed": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
}

db_manager = VectorDBManager(embedding_model=models["clinical"])
```

### **Custom Text Splitters**
```python
# Custom chunking strategy
custom_separators = [
    "\n\n## ",      # Section headers
    "\n\n### ",     # Subsection headers
    "\n\nFigure ",  # Figure captions
    "\n\nTable ",   # Table captions
    "\n\n",         # Paragraphs
    ". ",           # Sentences
    " "             # Words
]

ingester = IngestPipeline(
    chunk_size=800,
    chunk_overlap=100,
    separators=custom_separators
)
```

### **Custom LLM Integration**
```python
from langchain.llms.base import LLM

class CustomBiomedicalLLM(LLM):
    """Custom LLM wrapper for biomedical models"""
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Your custom LLM implementation
        return "Custom response"
    
    @property
    def _llm_type(self) -> str:
        return "custom_biomedical"

# Use with RAG chain
custom_llm = CustomBiomedicalLLM()
rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr, llm_model=custom_llm)
```

## 🔍 Error Handling

### **Common Exceptions**
```python
from biorag.core.ingest import IngestPipeline

try:
    documents = ingester.ingest_file("nonexistent.pdf")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Unsupported file type: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### **Validation Helpers**
```python
def validate_documents(documents: List[Document]) -> bool:
    """Validate document structure"""
    if not documents:
        return False
    
    for doc in documents:
        if not hasattr(doc, 'page_content') or not doc.page_content.strip():
            return False
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
            return False
    
    return True

def validate_vectordb(vectordb) -> bool:
    """Validate vector database"""
    try:
        count = vectordb._collection.count()
        return count > 0
    except:
        return False
```

## 📊 Performance Monitoring

### **Timing Decorators**
```python
import time
from functools import wraps

def time_it(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

# Usage
@time_it
def process_large_document(file_path):
    return ingester.ingest_file(file_path)
```

### **Memory Usage Tracking**
```python
import psutil
import os

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return f"{memory_mb:.1f} MB"

print(f"Memory before processing: {get_memory_usage()}")
documents = ingester.ingest_file("large_file.pdf")
print(f"Memory after processing: {get_memory_usage()}")
```

---

## 📞 Support

- **API Questions**: [GitHub Discussions](https://github.com/yourusername/biorag/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/biorag/issues)
- **Feature Requests**: [GitHub Issues](https://github.com/yourusername/biorag/issues)

For more examples, see the [Examples Documentation](examples.md).