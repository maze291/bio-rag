#!/usr/bin/env python3
"""
BioRAG REST API Server
Provides HTTP endpoints for the React frontend to communicate with BioRAG backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import tempfile
import os
import requests
from urllib.parse import urlparse

try:
    import feedparser
except ImportError:
    feedparser = None

# Add biorag modules to path
sys.path.append(str(Path(__file__).parent / 'biorag'))

try:
    from biorag.core.ingest import IngestPipeline
    from biorag.core.vectordb import VectorDBManager
    from biorag.core.linker import EntityLinker
    from biorag.core.glossary import GlossaryManager
    from biorag.core.rag_chain import RAGChain
except ImportError as e:
    print(f"Failed to import BioRAG modules: {e}")
    print("Make sure you're running from the bio-rag directory")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global components (initialized once)
ingester = None
db_manager = None
entity_linker = None
glossary_mgr = None
vector_db = None
rag_chain = None

def initialize_components():
    """Initialize BioRAG components"""
    global ingester, db_manager, entity_linker, glossary_mgr, vector_db, rag_chain
    
    try:
        print("Initializing BioRAG components...")
        ingester = IngestPipeline()
        print("IngestPipeline loaded")
        
        db_manager = VectorDBManager()
        print("VectorDBManager loaded")
        
        entity_linker = EntityLinker()
        print("EntityLinker loaded")
        
        glossary_mgr = GlossaryManager()
        print("GlossaryManager loaded")
        
        # Try to load existing vector DB or create empty one
        try:
            vector_db = db_manager.load_db()
            print("Loaded existing vector database")
        except:
            print("No existing vector database found")
        
        if vector_db:
            rag_chain = RAGChain(vector_db, entity_linker, glossary_mgr)
            print("RAGChain initialized")
        
        print("BioRAG API server ready!")
        return True
        
    except Exception as e:
        print(f"Failed to initialize BioRAG: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "BioRAG API is running"})

@app.route('/api/query', methods=['POST'])
def query():
    """Process a user query"""
    global rag_chain
    
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        enable_hyde = data.get('enable_hyde', True)
        enable_decomposition = data.get('enable_decomposition', True)
        temperature = data.get('temperature', 0.7)
        
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        if not rag_chain:
            return jsonify({
                "enhanced_answer": f"BioRAG is not fully initialized yet. Please upload some documents first to build the knowledge base.\n\nYour question: \"{query_text}\"",
                "entities": [],
                "source_docs": [],
                "confidence_score": 0.0
            })
        
        # Process query with BioRAG
        result = rag_chain.query(
            query_text,
            enable_hyde=enable_hyde,
            enable_decomposition=enable_decomposition
        )
        
        # Convert source_docs to serializable format
        source_docs = []
        raw_source_docs = result.get("source_docs", [])
        for doc in raw_source_docs:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # This is a Document object, extract serializable data
                source_docs.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown")
                })
            elif isinstance(doc, dict):
                # Already serializable
                source_docs.append(doc)
        
        return jsonify({
            "enhanced_answer": result.get("enhanced_answer", result.get("answer", query_text)),
            "entities": result.get("entities", []),
            "source_docs": source_docs,
            "confidence_score": result.get("confidence_score", 0.8)
        })
        
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({
            "enhanced_answer": f"I encountered an error processing your question: {str(e)}\n\nYour question: \"{query_text if 'query_text' in locals() else 'unknown'}\"",
            "entities": [],
            "source_docs": []
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload and process files"""
    global ingester, db_manager, vector_db, rag_chain
    
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided"}), 400
        
        # Check file sizes (50MB per file limit)
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
        for file in files:
            if file.filename == '':
                continue
            # Get file size by seeking to end
            file.seek(0, 2)  # Seek to end
            size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if size > MAX_FILE_SIZE:
                return jsonify({
                    "error": f"File '{file.filename}' exceeds 50MB limit ({size / (1024*1024):.1f}MB)"
                }), 413
        
        all_docs = []
        processed_files = 0
        
        for file in files:
            if file.filename == '':
                continue
                
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # Process file
                docs = ingester.ingest_file(tmp_path)
                all_docs.extend(docs)
                processed_files += 1
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        
        # Update vector database
        if all_docs:
            if vector_db is None:
                vector_db = db_manager.create_db(all_docs)
            else:
                db_manager.add_documents(vector_db, all_docs)
            
            # Rebuild RAG chain
            rag_chain = RAGChain(vector_db, entity_linker, glossary_mgr)
        
        return jsonify({
            "success": True,
            "message": f"Successfully processed {processed_files} files",
            "documents_processed": processed_files,
            "document_count": len(all_docs)
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({
            "success": False,
            "message": f"Upload failed: {str(e)}",
            "documents_processed": 0,
            "document_count": 0
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        doc_count = 0
        if vector_db:
            # Try to get document count from vector DB
            try:
                doc_count = vector_db._collection.count()
            except:
                doc_count = 0
        
        return jsonify({
            "document_count": doc_count,
            "entity_count": len(entity_linker._entity_db) if entity_linker else 0,
            "model_info": "BioRAG Local"
        })
        
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({
            "document_count": 0,
            "entity_count": 0,
            "model_info": "BioRAG (Error)"
        }), 500

@app.route('/api/rss', methods=['POST'])
def add_rss():
    """
    Body: { "rss_url": "https://example.com/feed.xml" }
    Returns: { "success": bool, "added": int, "skipped": int, "errors": [str] }
    """
    global ingester, db_manager, vector_db, rag_chain
    
    if feedparser is None:
        return jsonify({
            "success": False, 
            "error": "feedparser not installed. Run: pip install feedparser"
        }), 500
    
    try:
        data = request.get_json(silent=True) or {}
        rss_url = data.get("rss_url")
        
        if not rss_url:
            return jsonify({
                "success": False, 
                "error": "Missing rss_url"
            }), 400
        
        print(f"Processing RSS feed: {rss_url}")
        parsed = feedparser.parse(rss_url)
        
        if parsed.get("bozo"):
            return jsonify({
                "success": False, 
                "error": f"Invalid RSS: {parsed.get('bozo_exception')}"
            }), 400
        
        entries = parsed.get("entries", [])[:10]  # Limit to 10 articles
        added = 0
        skipped = 0
        errors = []
        docs_all = []
        
        print(f"Found {len(entries)} RSS entries to process")
        
        for i, entry in enumerate(entries):
            url = entry.get("link")
            title = entry.get("title", "Untitled")
            
            if not url:
                skipped += 1
                continue
            
            print(f"Processing article {i+1}/{len(entries)}: {title[:50]}...")
            
            try:
                # Download content
                resp = requests.get(url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                resp.raise_for_status()
                
                content_type = resp.headers.get("content-type", "").lower()
                
                # Handle different content types
                if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                    # Save PDF to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(resp.content)
                        tmp_path = tmp_file.name
                    
                    try:
                        docs = ingester.ingest_file(tmp_path)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                        
                else:
                    # Save HTML to temporary file
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as tmp_file:
                        tmp_file.write(resp.text)
                        tmp_path = tmp_file.name
                    
                    try:
                        docs = ingester.ingest_file(tmp_path)
                        # Add RSS metadata
                        for doc in docs:
                            doc.metadata.update({
                                'source': url,
                                'title': title,
                                'rss_feed': rss_url
                            })
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                
                if docs:
                    docs_all.extend(docs)
                    added += 1
                    print(f"  Added {len(docs)} documents from {title[:30]}...")
                else:
                    skipped += 1
                    print(f"  Warning: No content extracted from {title[:30]}")
                    
            except Exception as ex:
                error_msg = f"{title[:30]}: {str(ex)}"
                errors.append(error_msg)
                print(f"  Error: {error_msg}")
        
        # Update vector database
        if docs_all:
            if vector_db is None:
                vector_db = db_manager.create_db(docs_all)
            else:
                db_manager.add_documents(vector_db, docs_all)
            
            # Rebuild RAG chain
            rag_chain = RAGChain(vector_db, entity_linker, glossary_mgr)
            
            print(f"RSS processing complete. Added {added} articles, {len(docs_all)} total documents")
        
        return jsonify({
            "success": True,
            "added": added,
            "skipped": skipped,
            "errors": errors,
            "total_documents": len(docs_all)
        })
        
    except Exception as e:
        print(f"RSS processing error: {e}")
        return jsonify({
            "success": False,
            "error": f"RSS processing failed: {str(e)}"
        }), 500

@app.route('/api/selftest', methods=['POST'])
def self_test():
    """Run self-test"""
    try:
        # Create test document
        test_text = "BRCA1 mutations are associated with breast cancer. Tamoxifen treatment shows efficacy."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_text)
            test_file = f.name
        
        try:
            # Test ingestion
            docs = ingester.ingest_file(test_file)
            
            # Create test DB
            test_db = db_manager.create_db(docs)
            test_chain = RAGChain(test_db, entity_linker, glossary_mgr)
            
            # Test query
            result = test_chain.query("What genes are mentioned?")
            
            return jsonify({
                "success": True,
                "message": "Self-test completed successfully!",
                "details": {
                    "documents_processed": len(docs),
                    "entities_found": len(result.get('entities', [])),
                    "test_query_response": result.get('enhanced_answer', result.get('answer', ''))[:100]
                }
            })
            
        finally:
            # Cleanup
            Path(test_file).unlink(missing_ok=True)
            
    except Exception as e:
        print(f"Self-test error: {e}")
        return jsonify({
            "success": False,
            "message": f"Self-test failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("Starting BioRAG API server...")
    
    if initialize_components():
        print("\nStarting Flask server on http://localhost:8000")
        print("React frontend should connect automatically")
        print("Press Ctrl+C to stop\n")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        print("Failed to start BioRAG API server")
        sys.exit(1)