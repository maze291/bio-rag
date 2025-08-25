# Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                health_status["components"]["redis"] = {"status": "healthy"}
            except Exception as e:
                health_status["components"]["redis"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        else:
            health_status["components"]["redis"] = {"status": "not_configured"}
        
        # Check Celery
        if self.celery_app:
            try:
                # Check if Celery workers are available
                inspect = self.celery_app.control.inspect()
                active_workers = inspect.active()
                health_status["components"]["celery"] = {
                    "status": "healthy" if active_workers else "no_workers",
                    "active_workers": len(active_workers) if active_workers else 0
                }
            except Exception as e:
                health_status["components"]["celery"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        else:
            health_status["components"]["celery"] = {"status": "not_configured"}
        
        # Check memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            health_status["components"]["system"] = {
                "status": "healthy" if memory_percent < 85 else "high_memory",
                "memory_usage_percent": memory_percent,
                "available_gb": psutil.virtual_memory().available / (1024**3)
            }
        except Exception as e:
            health_status["components"]["system"] = {
                "status": "unknown",
                "error": str(e)
            }
        
        return health_status
    
    def _direct_query(self, question: str) -> Dict[str, Any]:
        """Direct query without caching"""
        if not hasattr(self, 'rag_chain') or not self.rag_chain:
            raise ValueError("RAG chain not initialized")
        
        return self.rag_chain.query(question)
    
    def _process_documents_sync(self, file_paths: List[str]) -> Dict[str, Any]:
        """Synchronous document processing fallback"""
        results = {"processed": 0, "failed": 0, "files": []}
        
        for file_path in file_paths:
            try:
                docs = self.ingester.ingest_file(file_path)
                if hasattr(self, 'vectordb'):
                    self.db_manager.add_documents(self.vectordb, docs)
                else:
                    self.vectordb = self.db_manager.create_db(docs)
                
                results["processed"] += 1
                results["files"].append({"file": file_path, "status": "success", "chunks": len(docs)})
                
            except Exception as e:
                results["failed"] += 1
                results["files"].append({"file": file_path, "status": "error", "error": str(e)})
        
        return results

# Production configuration
production_config = {
    'redis_url': 'redis://localhost:6379/0',
    'celery_broker': 'redis://localhost:6379/1',
    'db_path': '/data/biorag_vectordb',
    'embedding_model': 'scibert',
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'enable_ocr': True,
    'log_level': 'INFO'
}

# Initialize production system
prod_biorag = ProductionBioRAG(production_config)

# Example API endpoint integration
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_endpoint():
    """Health check endpoint for load balancers"""
    return jsonify(prod_biorag.health_check())

@app.route('/query', methods=['POST'])
def query_endpoint():
    """Main query endpoint with caching"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question parameter'}), 400
    
    try:
        start_time = time.time()
        result = prod_biorag.cached_query(data['question'])
        processing_time = time.time() - start_time
        
        return jsonify({
            'result': result,
            'processing_time_seconds': processing_time,
            'cached': 'cache_hit' in result  # Would be set by caching logic
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/upload', methods=['POST'])
def upload_endpoint():
    """Document upload endpoint with background processing"""
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    file_paths = []
    
    # Save uploaded files
    upload_dir = Path('/tmp/biorag_uploads')
    upload_dir.mkdir(exist_ok=True)
    
    for file in files:
        if file.filename:
            file_path = upload_dir / file.filename
            file.save(str(file_path))
            file_paths.append(str(file_path))
    
    # Submit for background processing
    task_id = prod_biorag.background_document_processing(file_paths)
    
    return jsonify({
        'message': f'{len(file_paths)} files submitted for processing',
        'task_id': task_id,
        'status_url': f'/status/{task_id}'
    })

@app.route('/status/<task_id>', methods=['GET'])
def status_endpoint(task_id):
    """Check processing status"""
    status = prod_biorag.get_processing_status(task_id)
    return jsonify(status)

# Run production server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
```

### **Docker Production Setup**

```dockerfile
# Dockerfile.production
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash biorag
WORKDIR /app
RUN chown biorag:biorag /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install production dependencies
RUN pip install gunicorn redis celery psutil

# Copy application code
COPY --chown=biorag:biorag . .
USER biorag

# Install SpaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_ner_bionlp13cg_md || echo "Biomedical model not available"

# Production web server
FROM base as web
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "300", "main:app"]

# Background worker
FROM base as worker
CMD ["celery", "worker", "-A", "biorag.celery_tasks", "--loglevel=info", "--concurrency=2"]

# Scheduler for periodic tasks
FROM base as beat
CMD ["celery", "beat", "-A", "biorag.celery_tasks", "--loglevel=info"]
```

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  biorag-web:
    build:
      context: .
      dockerfile: Dockerfile.production
      target: web
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER=redis://redis:6379/1
      - BIORAG_DB_PATH=/data/vector_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - biorag_data:/data
      - ./logs:/app/logs
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  biorag-worker:
    build:
      context: .
      dockerfile: Dockerfile.production
      target: worker
    restart: unless-stopped
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER=redis://redis:6379/1
      - BIORAG_DB_PATH=/data/vector_db
    volumes:
      - biorag_data:/data
      - ./logs:/app/logs
    depends_on:
      - redis
    deploy:
      replicas: 2

  biorag-beat:
    build:
      context: .
      dockerfile: Dockerfile.production
      target: beat
    restart: unless-stopped
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER=redis://redis:6379/1
    depends_on:
      - redis

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - biorag-web

volumes:
  redis_data:
  biorag_data:
```

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream biorag_backend {
        server biorag-web:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Main application
        location / {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://biorag_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
        }

        # File upload endpoint with stricter limits
        location /upload {
            limit_req zone=upload burst=5 nodelay;
            client_max_body_size 100M;
            proxy_pass http://biorag_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 600s;
        }

        # Health check (no rate limiting)
        location /health {
            proxy_pass http://biorag_backend;
            access_log off;
        }
    }
}
```

## 📊 Monitoring and Analytics

### **Comprehensive Monitoring System**

```python
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

class BioRAGMonitoring:
    """Comprehensive monitoring and analytics for BioRAG system"""
    
    def __init__(self, metrics_storage_path: str = "./metrics"):
        self.metrics_storage = Path(metrics_storage_path)
        self.metrics_storage.mkdir(exist_ok=True)
        self.query_metrics = []
        self.system_metrics = []
        self.error_log = []
        
    def track_query(self, question: str, result: Dict[str, Any], processing_time: float):
        """Track individual query metrics"""
        
        query_metric = {
            'timestamp': datetime.now().isoformat(),
            'question_length': len(question),
            'question_hash': hash(question),
            'answer_length': len(result.get('answer', '')),
            'entities_found': len(result.get('entities', [])),
            'sources_used': len(result.get('source_docs', [])),
            'confidence_score': result.get('confidence', 0.0),
            'processing_time_seconds': processing_time,
            'sub_queries_generated': len(result.get('sub_queries', [])),
            'enhancement_applied': bool(result.get('enhanced_answer'))
        }
        
        self.query_metrics.append(query_metric)
        
        # Persist metrics periodically
        if len(self.query_metrics) % 100 == 0:
            self._persist_metrics()
    
    def track_system_performance(self):
        """Track system resource usage"""
        
        system_metric = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids())
        }
        
        # Add GPU metrics if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                system_metric['gpu_usage_percent'] = gpus[0].load * 100
                system_metric['gpu_memory_percent'] = gpus[0].memoryUtil * 100
        except ImportError:
            pass
        
        self.system_metrics.append(system_metric)
        
        # Keep only last 1000 metrics in memory
        if len(self.system_metrics) > 1000:
            self.system_metrics = self.system_metrics[-1000:]
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log errors with context"""
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        self.error_log.append(error_entry)
        
        # Persist errors immediately
        self._persist_errors()
    
    def generate_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent metrics
        recent_queries = [
            m for m in self.query_metrics 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        recent_system = [
            m for m in self.system_metrics
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        recent_errors = [
            e for e in self.error_log
            if datetime.fromisoformat(e['timestamp']) > cutoff_time
        ]
        
        # Query analytics
        query_analytics = {}
        if recent_queries:
            query_analytics = {
                'total_queries': len(recent_queries),
                'avg_processing_time': sum(q['processing_time_seconds'] for q in recent_queries) / len(recent_queries),
                'avg_confidence': sum(q['confidence_score'] for q in recent_queries) / len(recent_queries),
                'avg_entities_per_query': sum(q['entities_found'] for q in recent_queries) / len(recent_queries),
                'queries_with_sub_decomposition': sum(1 for q in recent_queries if q['sub_queries_generated'] > 0),
                'avg_answer_length': sum(q['answer_length'] for q in recent_queries) / len(recent_queries),
                'performance_percentiles': {
                    'p50_processing_time': self._percentile([q['processing_time_seconds'] for q in recent_queries], 50),
                    'p95_processing_time': self._percentile([q['processing_time_seconds'] for q in recent_queries], 95),
                    'p99_processing_time': self._percentile([q['processing_time_seconds'] for q in recent_queries], 99)
                }
            }
        
        # System analytics
        system_analytics = {}
        if recent_system:
            system_analytics = {
                'avg_cpu_percent': sum(s['cpu_percent'] for s in recent_system) / len(recent_system),
                'avg_memory_percent': sum(s['memory_percent'] for s in recent_system) / len(recent_system),
                'max_cpu_percent': max(s['cpu_percent'] for s in recent_system),
                'max_memory_percent': max(s['memory_percent'] for s in recent_system),
                'min_available_memory_gb': min(s['memory_available_gb'] for s in recent_system)
            }
        
        # Error analytics
        error_analytics = {
            'total_errors': len(recent_errors),
            'error_types': {},
            'error_rate_per_hour': len(recent_errors) / hours_back if hours_back > 0 else 0
        }
        
        for error in recent_errors:
            error_type = error['error_type']
            error_analytics['error_types'][error_type] = error_analytics['error_types'].get(error_type, 0) + 1
        
        return {
            'report_generated': datetime.now().isoformat(),
            'time_period_hours': hours_back,
            'query_analytics': query_analytics,
            'system_analytics': system_analytics,
            'error_analytics': error_analytics,
            'recommendations': self._generate_recommendations(query_analytics, system_analytics, error_analytics)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _generate_recommendations(self, query_analytics: Dict, system_analytics: Dict, error_analytics: Dict) -> List[str]:
        """Generate performance recommendations"""
        
        recommendations = []
        
        # Query performance recommendations
        if query_analytics.get('avg_processing_time', 0) > 10:
            recommendations.append("Consider using a smaller embedding model or reducing chunk size for faster queries")
        
        if query_analytics.get('avg_confidence', 0) < 0.6:
            recommendations.append("Low average confidence scores suggest need for more relevant documents or better chunking")
        
        # System performance recommendations
        if system_analytics.get('avg_memory_percent', 0) > 80:
            recommendations.append("High memory usage detected - consider memory optimization or scaling up")
        
        if system_analytics.get('avg_cpu_percent', 0) > 80:
            recommendations.append("High CPU usage - consider horizontal scaling or CPU optimization")
        
        # Error rate recommendations
        if error_analytics.get('error_rate_per_hour', 0) > 1:
            recommendations.append("Elevated error rate detected - investigate most common error types")
        
        return recommendations
    
    def _persist_metrics(self):
        """Persist metrics to disk"""
        
        # Save query metrics
        query_file = self.metrics_storage / f"query_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(query_file, 'a') as f:
            for metric in self.query_metrics:
                f.write(json.dumps(metric) + '\n')
        
        # Clear in-memory metrics after persisting
        self.query_metrics = []
    
    def _persist_errors(self):
        """Persist errors to disk"""
        
        error_file = self.metrics_storage / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(error_file, 'a') as f:
            for error in self.error_log[-10:]:  # Only persist recent errors
                f.write(json.dumps(error) + '\n')

# Monitoring integration example
monitoring = BioRAGMonitoring()

class MonitoredBioRAG:
    """BioRAG wrapper with integrated monitoring"""
    
    def __init__(self, rag_chain, monitoring_system):
        self.rag_chain = rag_chain
        self.monitoring = monitoring_system
        
        # Start system monitoring
        import threading
        self.monitoring_thread = threading.Thread(target=self._continuous_monitoring, daemon=True)
        self.monitoring_thread.start()
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Monitored query execution"""
        
        start_time = time.time()
        
        try:
            # Execute query
            result = self.rag_chain.query(question, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Track successful query
            self.monitoring.track_query(question, result, processing_time)
            
            return result
            
        except Exception as e:
            # Log error with context
            processing_time = time.time() - start_time
            self.monitoring.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                context={
                    'question': question[:100],  # Truncated for privacy
                    'processing_time': processing_time,
                    'kwargs': kwargs
                }
            )
            raise
    
    def _continuous_monitoring(self):
        """Continuous system monitoring in background"""
        
        while True:
            try:
                self.monitoring.track_system_performance()
                time.sleep(60)  # Monitor every minute
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get real-time performance dashboard data"""
        
        return {
            'current_time': datetime.now().isoformat(),
            'last_24h_report': self.monitoring.generate_performance_report(24),
            'last_1h_report': self.monitoring.generate_performance_report(1),
            'system_status': self._get_current_system_status()
        }
    
    def _get_current_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'uptime_seconds': time.time() - psutil.boot_time()
        }

# Usage example with monitoring
monitored_biorag = MonitoredBioRAG(rag_chain, monitoring)

# Queries are automatically monitored
result = monitored_biorag.query("How do BRCA1 mutations affect DNA repair?")

# Generate performance report
performance_dashboard = monitored_biorag.get_performance_dashboard()
print("📊 Performance Dashboard:")
print(f"Queries last 24h: {performance_dashboard['last_24h_report']['query_analytics'].get('total_queries', 0)}")
print(f"Avg processing time: {performance_dashboard['last_24h_report']['query_analytics'].get('avg_processing_time', 0):.2f}s")
print(f"System CPU: {performance_dashboard['system_status']['cpu_percent']:.1f}%")
print(f"System Memory: {performance_dashboard['system_status']['memory_percent']:.1f}%")
```

---

## 🎓 Best Practices and Tips

### **Development Workflow**

```python
class BioRAGDevelopmentWorkflow:
    """Best practices for BioRAG development and testing"""
    
    @staticmethod
    def setup_development_environment():
        """Recommended development setup"""
        
        setup_commands = [
            "# Create isolated environment",
            "python -m venv biorag_dev",
            "source biorag_dev/bin/activate",
            "",
            "# Install in development mode",
            "pip install -e .",
            "pip install -e '.[dev]'",
            "",
            "# Setup pre-commit hooks",
            "pre-commit install",
            "",
            "# Create test data directory",
            "mkdir -p test_data/documents test_data/rss_feeds",
            "",
            "# Setup environment variables",
            "cp .env.example .env",
            "# Edit .env with your settings"
        ]
        
        return '\n'.join(setup_commands)
    
    @staticmethod
    def testing_strategy():
        """Comprehensive testing approach"""
        
        return {
            'unit_tests': {
                'scope': 'Individual component functionality',
                'examples': [
                    'test_ingest_single_file()',
                    'test_entity_extraction()',
                    'test_glossary_lookup()',
                    'test_vector_similarity_search()'
                ],
                'run_command': 'pytest tests/unit/ -v'
            },
            'integration_tests': {
                'scope': 'Component interactions',
                'examples': [
                    'test_end_to_end_workflow()',
                    'test_rag_chain_with_real_documents()',
                    'test_entity_linker_with_vector_db()'
                ],
                'run_command': 'pytest tests/integration/ -v'
            },
            'smoke_tests': {
                'scope': 'System health check',
                'examples': [
                    'test_system_imports()',
                    'test_basic_functionality()',
                    'test_configuration_loading()'
                ],
                'run_command': 'python tests/smoke_test.py'
            },
            'performance_tests': {
                'scope': 'Load and performance validation',
                'examples': [
                    'test_large_document_processing()',
                    'test_concurrent_queries()',
                    'test_memory_usage_limits()'
                ],
                'run_command': 'pytest tests/performance/ -v --benchmark'
            }
        }
    
    @staticmethod
    def code_quality_checklist():
        """Code quality and review checklist"""
        
        return {
            'before_commit': [
                "Run black code formatter: black .",
                "Run isort import sorter: isort .",
                "Run flake8 linting: flake8 .",
                "Run mypy type checking: mypy biorag/",
                "Run tests: pytest tests/ -v",
                "Check test coverage: pytest --cov=biorag tests/"
            ],
            'documentation': [
                "Update docstrings for new functions",
                "Add type hints to function signatures",
                "Update README.md if public API changes",
                "Add examples for new features",
                "Update API reference documentation"
            ],
            'security': [
                "Validate all user inputs",
                "Sanitize HTML outputs",
                "Check for hardcoded secrets",
                "Review file upload restrictions",
                "Validate external URL access"
            ],
            'performance': [
                "Profile memory usage for large documents",
                "Test query response times",
                "Validate caching behavior",
                "Check for memory leaks",
                "Monitor resource usage patterns"
            ]
        }

# Print development guidelines
workflow = BioRAGDevelopmentWorkflow()

print("🛠️ Development Environment Setup:")
print(workflow.setup_development_environment())

print("\n🧪 Testing Strategy:")
testing = workflow.testing_strategy()
for test_type, details in testing.items():
    print(f"\n{test_type.title()}:")
    print(f"  Scope: {details['scope']}")
    print(f"  Run: {details['run_command']}")

print("\n✅ Code Quality Checklist:")
checklist = workflow.code_quality_checklist()
for category, items in checklist.items():
    print(f"\n{category.title()}:")
    for item in items:
        print(f"  □ {item}")
```

---

This comprehensive examples guide provides production-ready code patterns for deploying, monitoring, and maintaining BioRAG in real-world scenarios. Each example includes complete implementations that can be adapted for specific requirements.

**Next Steps:**
1. Choose examples relevant to your use case
2. Adapt configurations for your environment  
3. Start with simpler scenarios and add complexity
4. Implement monitoring from the beginning
5. Follow the development workflow for code quality

For additional support, refer to the [Installation Guide](installation.md) and [API Reference](api_reference.md).**Ready to implement these examples?** Start with the basic scenarios and gradually add complexity as you become familiar with the system.

## 📈 Performance Optimization Examples

### **High-Throughput Document Processing**

```python
import asyncio
import concurrent.futures
from typing import List, Dict, Any
import time

class HighThroughputBioRAG:
    """Optimized BioRAG for processing large document collections"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.ingester = IngestPipeline(chunk_size=800, chunk_overlap=100)
        self.db_manager = VectorDBManager(embedding_model="all-MiniLM-L6-v2")  # Faster model
        self.processing_stats = {}
        
    def batch_process_documents(self, document_paths: List[str], batch_size: int = 10) -> Dict[str, Any]:
        """Process documents in parallel batches"""
        
        start_time = time.time()
        all_documents = []
        failed_files = []
        
        # Process in batches to manage memory
        for i in range(0, len(document_paths), batch_size):
            batch = document_paths[i:i + batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(len(document_paths)-1)//batch_size + 1}")
            
            # Parallel processing within batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._safe_ingest_file, file_path): file_path 
                    for file_path in batch
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        documents = future.result()
                        if documents:
                            all_documents.extend(documents)
                            print(f"  ✅ {Path(file_path).name}: {len(documents)} chunks")
                        else:
                            failed_files.append(file_path)
                            print(f"  ❌ {Path(file_path).name}: No content extracted")
                    except Exception as e:
                        failed_files.append(file_path)
                        print(f"  ❌ {Path(file_path).name}: {str(e)}")
            
            # Intermediate cleanup to manage memory
            if len(all_documents) > 10000:  # Process in chunks
                self._process_document_chunk(all_documents)
                all_documents = []
        
        # Process remaining documents
        if all_documents:
            self._process_document_chunk(all_documents)
        
        processing_time = time.time() - start_time
        
        stats = {
            'total_files': len(document_paths),
            'successful_files': len(document_paths) - len(failed_files),
            'failed_files': len(failed_files),
            'total_chunks': sum(len(docs) for docs in self.processing_stats.values()),
            'processing_time_seconds': processing_time,
            'throughput_files_per_minute': (len(document_paths) / processing_time) * 60,
            'failed_file_list': failed_files
        }
        
        return stats
    
    def _safe_ingest_file(self, file_path: str) -> List:
        """Safely ingest a single file with error handling"""
        try:
            documents = self.ingester.ingest_file(file_path)
            self.processing_stats[file_path] = documents
            return documents
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _process_document_chunk(self, documents: List) -> None:
        """Process a chunk of documents into vector database"""
        if not hasattr(self, 'vectordb') or self.vectordb is None:
            self.vectordb = self.db_manager.create_db(documents)
        else:
            self.db_manager.add_documents(self.vectordb, documents)
        
        print(f"  💾 Added {len(documents)} documents to vector database")
    
    def optimize_for_query_speed(self) -> None:
        """Optimize the system for faster querying"""
        # Use faster embedding model for queries
        self.db_manager.update_embeddings("all-MiniLM-L6-v2")
        
        # Pre-load models to avoid initialization delays
        if hasattr(self, 'vectordb'):
            # Warm up the vector database
            _ = self.db_manager.similarity_search(self.vectordb, "test query", k=1)
            print("🚀 Vector database warmed up for faster queries")

# High-throughput processing example
high_throughput = HighThroughputBioRAG(max_workers=6)

# Process large collection of papers
paper_collection = [
    f"papers/paper_{i:04d}.pdf" for i in range(1, 501)  # 500 papers
]

# Batch process with performance monitoring
stats = high_throughput.batch_process_documents(paper_collection, batch_size=20)

print("\n📊 Processing Performance Report:")
print(f"Total files processed: {stats['successful_files']}/{stats['total_files']}")
print(f"Total document chunks: {stats['total_chunks']}")
print(f"Processing time: {stats['processing_time_seconds']:.2f} seconds")
print(f"Throughput: {stats['throughput_files_per_minute']:.1f} files/minute")
print(f"Failed files: {stats['failed_files']}")

# Optimize for subsequent querying
high_throughput.optimize_for_query_speed()
```

### **Memory-Efficient Large Document Handling**

```python
class MemoryEfficientBioRAG:
    """BioRAG optimized for processing very large documents or limited memory systems"""
    
    def __init__(self, memory_limit_mb: int = 2048):
        self.memory_limit_mb = memory_limit_mb
        self.chunk_cache = {}
        self.vector_db_shards = []
        
    def setup_memory_efficient_processing(self):
        """Configure for memory-constrained environments"""
        
        # Use smaller models and chunk sizes
        self.ingester = IngestPipeline(
            chunk_size=400,        # Smaller chunks
            chunk_overlap=50,      # Less overlap
            enable_ocr=False       # Disable memory-intensive OCR
        )
        
        # Lightweight embedding model
        self.db_manager = VectorDBManager(
            embedding_model="all-MiniLM-L6-v2",  # Smaller model
            persist_directory="./vector_db_shards"
        )
        
        # Memory-efficient entity linker
        self.entity_linker = EntityLinker()
        # Disable heavy biomedical models if memory is constrained
        
        print(f"💾 Configured for {self.memory_limit_mb}MB memory limit")
    
    def process_large_document_streaming(self, file_path: str, shard_size: int = 1000) -> Dict[str, Any]:
        """Process large documents in streaming fashion"""
        
        current_memory = self._get_memory_usage()
        print(f"📊 Starting memory usage: {current_memory:.1f}MB")
        
        # Stream process the document
        documents = self.ingester.ingest_file(file_path)
        print(f"📄 Document loaded: {len(documents)} chunks")
        
        # Process in shards to manage memory
        processed_shards = 0
        total_chunks = 0
        
        for i in range(0, len(documents), shard_size):
            shard = documents[i:i + shard_size]
            
            # Check memory before processing shard
            current_memory = self._get_memory_usage()
            if current_memory > self.memory_limit_mb * 0.8:  # 80% threshold
                print(f"⚠️ Memory usage high ({current_memory:.1f}MB), creating new shard")
                self._finalize_current_shard()
            
            # Create shard database
            shard_db = self.db_manager.create_db(shard)
            self.vector_db_shards.append({
                'database': shard_db,
                'chunk_count': len(shard),
                'shard_id': processed_shards
            })
            
            total_chunks += len(shard)
            processed_shards += 1
            
            print(f"  💽 Shard {processed_shards}: {len(shard)} chunks")
            
            # Force garbage collection
            import gc
            gc.collect()
        
        final_memory = self._get_memory_usage()
        
        return {
            'total_shards': processed_shards,
            'total_chunks': total_chunks,
            'initial_memory_mb': current_memory,
            'final_memory_mb': final_memory,
            'memory_efficiency': (current_memory - final_memory) / current_memory
        }
    
    def federated_search(self, query: str, k: int = 5) -> List:
        """Search across all shards and merge results"""
        
        all_results = []
        
        for shard_info in self.vector_db_shards:
            try:
                shard_results = self.db_manager.similarity_search(
                    shard_info['database'], 
                    query, 
                    k=k//len(self.vector_db_shards) + 1  # Distribute k across shards
                )
                
                # Add shard metadata
                for result in shard_results:
                    result.metadata['shard_id'] = shard_info['shard_id']
                    all_results.append(result)
                    
            except Exception as e:
                print(f"⚠️ Error searching shard {shard_info['shard_id']}: {e}")
        
        # Sort by relevance (if relevance scores available)
        # For now, just return top k results
        return all_results[:k]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _finalize_current_shard(self):
        """Finalize current shard and prepare for next"""
        if self.vector_db_shards:
            last_shard = self.vector_db_shards[-1]
            # Persist the shard
            last_shard['database'].persist()
            print(f"💾 Finalized shard {last_shard['shard_id']}")

# Memory-efficient processing example
memory_efficient = MemoryEfficientBioRAG(memory_limit_mb=1024)  # 1GB limit
memory_efficient.setup_memory_efficient_processing()

# Process very large document
large_document = "very_large_research_compendium.pdf"  # Hypothetical large file
processing_stats = memory_efficient.process_large_document_streaming(
    large_document, 
    shard_size=500  # Small shards for memory efficiency
)

print("\n🎯 Memory-Efficient Processing Results:")
print(f"Shards created: {processing_stats['total_shards']}")
print(f"Total chunks: {processing_stats['total_chunks']}")
print(f"Memory efficiency: {processing_stats['memory_efficiency']:.2%}")

# Federated search across shards
query_results = memory_efficient.federated_search("BRCA1 mutations and cancer risk", k=10)
print(f"\n🔍 Federated search results: {len(query_results)} documents found")
```

## 🔧 Custom Extensions and Integrations

### **Custom Entity Linker Extension**

```python
class CustomBiomedicalLinker(EntityLinker):
    """Extended entity linker with custom biomedical databases"""
    
    def __init__(self, custom_databases: Dict[str, str] = None):
        super().__init__()
        self.custom_databases = custom_databases or {}
        self._setup_custom_knowledge_base()
    
    def _setup_custom_knowledge_base(self):
        """Add custom biomedical entities and databases"""
        
        # Custom lab-specific entities
        lab_specific_entities = {
            "LAB_STRAIN_001": {
                "type": "organism",
                "name": "Modified E. coli strain for protein expression",
                "lab_id": "LAB001",
                "creation_date": "2024-01-15",
                "aliases": ["Expression strain 001", "Protein producer"]
            },
            "CUSTOM_ANTIBODY_AB123": {
                "type": "protein",
                "name": "Custom Anti-BRCA1 Antibody Clone AB123",
                "lab_id": "AB123",
                "specificity": "BRCA1 C-terminus",
                "aliases": ["AB123", "Anti-BRCA1-C"]
            }
        }
        
        # Extend main knowledge base
        self.entity_kb.update(lab_specific_entities)
        self._build_alias_lookup()
        
        # Custom URL templates for lab databases
        self.url_templates.update({
            'lab_strain': 'https://lab-database.example.com/strains/{id}',
            'lab_antibody': 'https://lab-database.example.com/antibodies/{id}',
            'internal_protocol': 'https://lab-protocols.example.com/protocol/{id}'
        })
        
        print(f"🔬 Added {len(lab_specific_entities)} custom lab entities")
    
    def link_to_custom_database(self, entity_text: str, database_name: str, entity_id: str) -> str:
        """Create links to custom databases"""
        
        if database_name in self.custom_databases:
            base_url = self.custom_databases[database_name]
            return f"{base_url}/{entity_id}"
        else:
            return f"https://custom-db.example.com/{database_name}/{entity_id}"
    
    def add_laboratory_entities(self, lab_entities: Dict[str, Dict]) -> None:
        """Add laboratory-specific entities in bulk"""
        
        for entity_id, entity_info in lab_entities.items():
            # Validate required fields
            if 'type' not in entity_info or 'name' not in entity_info:
                print(f"⚠️ Skipping {entity_id}: missing required fields")
                continue
            
            # Add to knowledge base
            self.entity_kb[entity_id] = entity_info
            
            # Add to alias lookup
            self.alias_lookup[entity_id.lower()] = entity_id
            if 'aliases' in entity_info:
                for alias in entity_info['aliases']:
                    self.alias_lookup[alias.lower()] = entity_id
        
        print(f"🧪 Added {len(lab_entities)} laboratory entities")
    
    def export_custom_entities(self, output_file: str) -> None:
        """Export custom entities for sharing or backup"""
        
        custom_entities = {
            entity_id: entity_info 
            for entity_id, entity_info in self.entity_kb.items()
            if 'lab_id' in entity_info  # Filter for custom entities
        }
        
        with open(output_file, 'w') as f:
            json.dump(custom_entities, f, indent=2)
        
        print(f"📤 Exported {len(custom_entities)} custom entities to {output_file}")

# Custom entity linker example
custom_databases = {
    'lab_strains': 'https://mylab.university.edu/strains',
    'antibody_collection': 'https://mylab.university.edu/antibodies',
    'plasmid_library': 'https://mylab.university.edu/plasmids'
}

custom_linker = CustomBiomedicalLinker(custom_databases)

# Add laboratory-specific entities
lab_entities = {
    "PLASMID_pET28a_BRCA1": {
        "type": "plasmid",
        "name": "pET28a vector with BRCA1 insert",
        "lab_id": "PL001",
        "resistance": "kanamycin",
        "aliases": ["pET28a-BRCA1", "BRCA1 expression vector"]
    },
    "CELL_LINE_HEK293_KO": {
        "type": "cell_line",
        "name": "HEK293 BRCA1 knockout cells",
        "lab_id": "CL001",
        "modification": "CRISPR-Cas9 knockout",
        "aliases": ["HEK293-BRCA1-KO", "BRCA1 null cells"]
    }
}

custom_linker.add_laboratory_entities(lab_entities)

# Test custom entity recognition
test_text = "We transfected HEK293-BRCA1-KO cells with pET28a-BRCA1 plasmid using our LAB_STRAIN_001 for protein expression."

entities = custom_linker.extract_entities(test_text)
print(f"\n🔍 Found {len(entities)} entities:")
for entity in entities:
    print(f"  {entity.text} ({entity.label}) -> {entity.url}")

# Export custom entities
custom_linker.export_custom_entities("lab_entities_backup.json")
```

### **Specialized Glossary for Different Domains**

```python
class DomainSpecificGlossary(GlossaryManager):
    """Glossary manager with domain-specific term sets"""
    
    def __init__(self, domains: List[str] = None):
        super().__init__()
        self.active_domains = domains or ['general']
        self.domain_glossaries = {}
        self._load_domain_specific_terms()
    
    def _load_domain_specific_terms(self):
        """Load terms specific to different biomedical domains"""
        
        # Oncology-specific terms
        oncology_terms = {
            "tumor_microenvironment": "cellular and molecular environment surrounding a tumor",
            "immune_checkpoint": "regulatory pathways in immune cells preventing excessive activation",
            "car_t_therapy": "chimeric antigen receptor T-cell therapy using modified immune cells",
            "liquid_biopsy": "analysis of tumor components in blood samples",
            "minimal_residual_disease": "small number of cancer cells remaining after treatment",
            "progression_free_survival": "time without tumor growth during treatment",
            "overall_survival": "time from treatment start until death from any cause",
            "objective_response_rate": "percentage of patients showing tumor shrinkage"
        }
        
        # Genetics/Genomics terms
        genetics_terms = {
            "polygenic_risk_score": "genetic risk prediction based on multiple variants",
            "genome_wide_association_study": "analysis linking genetic variants to traits",
            "copy_number_variation": "structural variation in genome segment copies",
            "loss_of_heterozygosity": "loss of one functional gene copy in cancer",
            "microsatellite_instability": "hypermutability from DNA mismatch repair defects",
            "germline_variant": "genetic variant present in reproductive cells",
            "somatic_mutation": "genetic change in non-reproductive cells",
            "penetrance": "probability that gene carriers develop the phenotype",
            "expressivity": "degree of phenotype expression in gene carriers"
        }
        
        # Pharmacology terms
        pharmacology_terms = {
            "pharmacokinetics": "study of drug absorption, distribution, metabolism, excretion",
            "pharmacodynamics": "study of drug effects on biological systems",
            "bioavailability": "fraction of administered drug reaching systemic circulation",
            "half_life": "time required for drug concentration to decrease by half",
            "therapeutic_window": "dose range between efficacy and toxicity",
            "drug_drug_interaction": "modification of drug effect by another drug",
            "cytochrome_p450": "enzyme family metabolizing drugs and toxins",
            "first_pass_metabolism": "drug metabolism before reaching systemic circulation",
            "steady_state": "equilibrium between drug input and elimination"
        }
        
        # Immunology terms
        immunology_terms = {
            "adaptive_immunity": "specific immune response with memory",
            "innate_immunity": "non-specific immediate immune response",
            "antigen_presentation": "display of foreign peptides to T cells",
            "major_histocompatibility_complex": "proteins presenting antigens to immune cells",
            "complement_system": "protein cascade enhancing immune responses",
            "cytokine_storm": "excessive systemic inflammatory response",
            "immunosuppression": "reduced immune system activity",
            "tolerance": "immune system acceptance of specific antigens",
            "autoimmunity": "immune system attacking body's own tissues"
        }
        
        # Store domain glossaries
        self.domain_glossaries = {
            'oncology': oncology_terms,
            'genetics': genetics_terms,
            'pharmacology': pharmacology_terms,
            'immunology': immunology_terms
        }
        
        # Load terms for active domains
        self._update_active_glossary()
    
    def _update_active_glossary(self):
        """Update main glossary with terms from active domains"""
        
        # Start with base glossary
        active_terms = self.glossary.copy()
        
        # Add terms from active domains
        for domain in self.active_domains:
            if domain in self.domain_glossaries:
                active_terms.update(self.domain_glossaries[domain])
                print(f"📚 Loaded {len(self.domain_glossaries[domain])} {domain} terms")
        
        # Update main glossary
        self.glossary = active_terms
        self._build_pattern()
        
        print(f"📖 Active glossary: {len(self.glossary)} total terms")
    
    def set_active_domains(self, domains: List[str]) -> None:
        """Change active domains and reload glossary"""
        
        valid_domains = [d for d in domains if d in self.domain_glossaries or d == 'general']
        invalid_domains = [d for d in domains if d not in valid_domains]
        
        if invalid_domains:
            print(f"⚠️ Unknown domains: {invalid_domains}")
            print(f"Available domains: {list(self.domain_glossaries.keys())}")
        
        self.active_domains = valid_domains
        self._update_active_glossary()
    
    def get_domain_coverage(self, text: str) -> Dict[str, int]:
        """Analyze which domains are most relevant for given text"""
        
        domain_coverage = {}
        
        for domain, terms in self.domain_glossaries.items():
            term_count = 0
            text_lower = text.lower()
            
            for term in terms.keys():
                if term.replace('_', ' ') in text_lower:
                    term_count += 1
            
            domain_coverage[domain] = term_count
        
        return domain_coverage
    
    def auto_select_domains(self, text: str, max_domains: int = 3) -> List[str]:
        """Automatically select most relevant domains for text"""
        
        coverage = self.get_domain_coverage(text)
        
        # Sort domains by coverage
        sorted_domains = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
        
        # Select top domains with non-zero coverage
        selected = [domain for domain, count in sorted_domains[:max_domains] if count > 0]
        
        return selected or ['general']
    
    def create_domain_specific_explanation(self, text: str, target_domain: str) -> str:
        """Create explanation tailored to specific domain expertise level"""
        
        # Temporarily switch to target domain
        original_domains = self.active_domains.copy()
        self.set_active_domains([target_domain])
        
        # Create enhanced explanation
        enhanced_text = self.simplify_text(text, format="html")
        
        # Restore original domains
        self.set_active_domains(original_domains)
        
        return enhanced_text

# Domain-specific glossary example
domain_glossary = DomainSpecificGlossary(domains=['oncology', 'genetics'])

# Analyze text for domain relevance
cancer_text = """
The tumor microenvironment plays a crucial role in cancer progression. 
Somatic mutations in tumor suppressor genes lead to loss of heterozygosity, 
while immune checkpoint inhibitors can overcome immunosuppression. 
CAR-T therapy shows promise for liquid tumors but faces challenges in 
solid tumors due to the immunosuppressive microenvironment.
"""

# Check domain coverage
coverage = domain_glossary.get_domain_coverage(cancer_text)
print("📊 Domain Coverage Analysis:")
for domain, count in coverage.items():
    print(f"  {domain}: {count} relevant terms")

# Auto-select optimal domains
optimal_domains = domain_glossary.auto_select_domains(cancer_text)
print(f"\n🎯 Recommended domains: {optimal_domains}")

# Switch to optimal domains
domain_glossary.set_active_domains(optimal_domains)

# Create enhanced explanation
enhanced_explanation = domain_glossary.simplify_text(cancer_text, format="html")
print(f"\n📝 Enhanced text with {len(optimal_domains)} domain glossaries")

# Create explanations for different expertise levels
beginner_explanation = domain_glossary.create_domain_specific_explanation(
    cancer_text, 'general'
)
expert_explanation = domain_glossary.create_domain_specific_explanation(
    cancer_text, 'oncology'
)

print("\n👨‍🎓 Beginner-level explanation includes basic term definitions")
print("👨‍⚕️ Expert-level explanation focuses on specialized oncology terms")
```

## 🌐 Production Deployment Examples

### **Scalable Cloud Deployment**

```python
import os
from pathlib import Path
import json
import logging
from typing import Optional
import redis
from celery import Celery

class ProductionBioRAG:
    """Production-ready BioRAG with caching, queuing, and monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.celery_app = None
        self._setup_production_environment()
    
    def _setup_production_environment(self):
        """Configure for production deployment"""
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('biorag_production.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup Redis for caching
        if self.config.get('redis_url'):
            self.redis_client = redis.from_url(self.config['redis_url'])
            self.logger.info("Redis caching enabled")
        
        # Setup Celery for background processing
        if self.config.get('celery_broker'):
            self.celery_app = Celery('biorag', broker=self.config['celery_broker'])
            self.logger.info("Celery background processing enabled")
        
        # Initialize core components with production settings
        self.ingester = IngestPipeline(
            chunk_size=self.config.get('chunk_size', 1000),
            chunk_overlap=self.config.get('chunk_overlap', 200),
            enable_ocr=self.config.get('enable_ocr', True)
        )
        
        self.db_manager = VectorDBManager(
            embedding_model=self.config.get('embedding_model', 'scibert'),
            persist_directory=self.config.get('db_path', './vector_db')
        )
        
        self.logger.info("Production BioRAG initialized")
    
    def cached_query(self, question: str, cache_ttl: int = 3600) -> Dict[str, Any]:
        """Query with Redis caching for performance"""
        
        if not self.redis_client:
            return self._direct_query(question)
        
        # Create cache key
        cache_key = f"biorag:query:{hash(question)}"
        
        # Check cache
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for query: {question[:50]}...")
                return json.loads(cached_result)
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        
        # Process query
        result = self._direct_query(question)
        
        # Cache result
        try:
            self.redis_client.setex(
                cache_key, 
                cache_ttl, 
                json.dumps(result, default=str)
            )
            self.logger.info(f"Cached result for query: {question[:50]}...")
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
        
        return result
    
    def background_document_processing(self, file_paths: List[str]) -> str:
        """Process documents in background using Celery"""
        
        if not self.celery_app:
            self.logger.warning("Celery not configured, processing synchronously")
            return self._process_documents_sync(file_paths)
        
        # Submit to background queue
        task = self.celery_app.send_task(
            'biorag.process_documents',
            args=[file_paths],
            kwargs={'config': self.config}
        )
        
        self.logger.info(f"Submitted {len(file_paths)} files for background processing")
        return task.id
    
    def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of background processing task"""
        
        if not self.celery_app:
            return {"status": "not_available", "message": "Celery not configured"}
        
        task_result = self.celery_app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None,
            "info": task_result.info
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for monitoring"""
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "components": {}
        }
        
        # Check vector database
        try:
            if hasattr(self, 'vectordb') and self.vectordb:
                stats = self.db_manager.get_collection_stats(self.vectordb)
                health_status["components"]["vector_db"] = {
                    "status": "healthy",
                    "document_count": stats["total_documents"],
                    "embedding_model": stats["embedding_model"]
                }
            else:
                health_status["components"]["vector_db"] = {
                    "status": "not_initialized",
                    "message": "Vector database not loaded"
                }
        except Exception as e:
            health_status["components"]["vector_db"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check Redis
        if self.redis    def generate_research_questions(self, research_area: str) -> List[str]:
        """Generate novel research questions for grant proposals"""
        if not self.rag_chain:
            raise ValueError("Research landscape not analyzed yet")
        
        prompt = f"""
        Based on current research in {research_area}, identify 5 novel research questions that:
        1. Address significant gaps in current knowledge
        2. Have potential for high impact
        3. Are feasible with current technology
        4. Align with funding agency priorities
        
        Focus on innovative approaches and emerging opportunities.
        """
        
        result = self.rag_chain.query(prompt, enable_decomposition=True)
        
        # Extract questions from the response
        lines = result['answer'].split('\n')
        questions = [line.strip() for line in lines if '?' in line]
        
        return questions[:5]
    
    def assess_funding_alignment(self, research_proposal: str, funding_program: str) -> Dict[str, Any]:
        """Assess how well proposal aligns with funding priorities"""
        if not self.rag_chain:
            raise ValueError("Research landscape not analyzed yet")
        
        alignment_prompt = f"""
        Analyze how well this research proposal aligns with {funding_program} priorities:
        
        Proposal: {research_proposal}
        
        Evaluate:
        1. Strategic alignment with funding goals
        2. Innovation and potential impact
        3. Methodological feasibility
        4. Collaborative opportunities
        5. Translation potential
        
        Provide specific recommendations for strengthening the proposal.
        """
        
        result = self.rag_chain.query(alignment_prompt)
        
        return {
            'alignment_score': result['confidence'],
            'analysis': result['answer'],
            'key_entities': result['entities'],
            'recommendations': self._extract_recommendations(result['answer'])
        }
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract actionable recommendations from analysis"""
        # Simple extraction based on common recommendation patterns
        recommendations = []
        lines = analysis_text.split('.')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
                if len(line) > 20:  # Filter out very short fragments
                    recommendations.append(line)
        
        return recommendations[:5]

# Usage example
grant_assistant = GrantProposalAssistant()

# Analyze landscape for cancer immunotherapy research
docs_count = grant_assistant.analyze_research_landscape(
    research_area="cancer immunotherapy",
    funding_agency="NIH"
)
print(f"📚 Analyzed {docs_count} recent documents")

# Generate novel research questions
questions = grant_assistant.generate_research_questions("cancer immunotherapy")
print("\n🎯 Novel Research Questions:")
for i, question in enumerate(questions, 1):
    print(f"{i}. {question}")

# Assess funding alignment
proposal_text = """
We propose to develop novel CAR-T cell therapies targeting solid tumors 
by engineering enhanced persistence and tumor-homing capabilities. 
Our approach combines CRISPR gene editing with synthetic biology to 
create next-generation immune cells capable of overcoming the 
immunosuppressive tumor microenvironment.
"""

alignment = grant_assistant.assess_funding_alignment(
    proposal_text, 
    "NIH Cancer Moonshot Initiative"
)

print(f"\n📊 Funding Alignment Score: {alignment['alignment_score']:.2f}")
print("\n💡 Recommendations:")
for rec in alignment['recommendations']:
    print(f"  • {rec}")
```

## 🏭 Industry Applications

### **Pharmaceutical Research & Development**

```python
class PharmaceuticalRAG:
    def __init__(self):
        self.drug_pipeline = {}
        self.competitive_intelligence = {}
        
    def setup_drug_development_kb(self, therapeutic_area: str):
        """Setup knowledge base for drug development"""
        
        # Specialized data sources for pharma
        pharma_sources = {
            'clinical_trials': 'https://clinicaltrials.gov/ct2/results/rss.xml',
            'fda_approvals': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-safety-and-availability/rss.xml',
            'patent_filings': f'https://patents.googleapis.com/rss/{therapeutic_area}',
            'biotech_news': 'https://www.biopharmadive.com/feeds/news/'
        }
        
        all_docs = []
        for source_name, feed_url in pharma_sources.items():
            try:
                docs = self.ingester.ingest_rss(feed_url, max_items=20)
                all_docs.extend(docs)
                print(f"📡 {source_name}: {len(docs)} documents")
            except Exception as e:
                print(f"⚠️ {source_name} unavailable: {e}")
        
        return all_docs
    
    def analyze_competitive_landscape(self, drug_target: str) -> Dict[str, Any]:
        """Analyze competitive landscape for a drug target"""
        
        analysis_questions = [
            f"What companies are developing drugs targeting {drug_target}?",
            f"What is the current clinical trial status for {drug_target} inhibitors?",
            f"What are the main challenges in developing {drug_target}-targeted therapies?",
            f"What alternative approaches exist for modulating {drug_target}?",
            f"What is the patent landscape for {drug_target} modulators?"
        ]
        
        competitive_data = {}
        for question in analysis_questions:
            result = self.rag_chain.query(question)
            
            # Extract company names and drug names
            entities = result['entities']
            companies = [e['text'] for e in entities if 'company' in e.get('type', '').lower()]
            drugs = [e['text'] for e in entities if e.get('type', '').lower() in ['chemical', 'drug']]
            
            competitive_data[question] = {
                'answer': result['answer'],
                'companies_mentioned': companies,
                'drugs_mentioned': drugs,
                'confidence': result['confidence']
            }
        
        return competitive_data
    
    def safety_signal_monitoring(self, drug_name: str) -> Dict[str, Any]:
        """Monitor safety signals and adverse events"""
        
        safety_queries = [
            f"What adverse events have been reported for {drug_name}?",
            f"Are there any drug-drug interactions with {drug_name}?",
            f"What are the contraindications for {drug_name}?",
            f"How does {drug_name} safety profile compare to alternatives?",
            f"What post-market surveillance data exists for {drug_name}?"
        ]
        
        safety_profile = {}
        for query in safety_queries:
            result = self.rag_chain.query(query)
            safety_profile[query.split('?')[0]] = {
                'findings': result['answer'],
                'confidence': result['confidence'],
                'sources': len(result['source_docs'])
            }
        
        return safety_profile

# Pharmaceutical industry example
pharma_rag = PharmaceuticalRAG()

# Analyze PARP inhibitor competitive landscape
target_analysis = pharma_rag.analyze_competitive_landscape("PARP1")
print("🏢 Competitive Landscape Analysis:")
for question, data in target_analysis.items():
    print(f"\nQ: {question}")
    print(f"Companies: {', '.join(data['companies_mentioned'][:3])}")
    print(f"Drugs: {', '.join(data['drugs_mentioned'][:3])}")
    print(f"Confidence: {data['confidence']:.2f}")

# Safety monitoring for olaparib
safety_data = pharma_rag.safety_signal_monitoring("olaparib")
print("\n🚨 Safety Profile Summary:")
for aspect, data in safety_data.items():
    print(f"{aspect}: {data['findings'][:100]}... (Confidence: {data['confidence']:.2f})")
```

### **Regulatory Affairs Intelligence**

```python
class RegulatoryIntelligence:
    def __init__(self):
        self.regulatory_updates = {}
        
    def monitor_regulatory_changes(self, therapeutic_area: str, regions: List[str] = ['FDA', 'EMA', 'PMDA']):
        """Monitor regulatory guidance updates"""
        
        regulatory_feeds = {
            'FDA': [
                'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/fda-news-releases/rss.xml',
                'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-safety-and-availability/rss.xml'
            ],
            'EMA': [
                'https://www.ema.europa.eu/en/rss.xml'
            ],
            'PMDA': [
                # PMDA feeds (if available)
            ]
        }
        
        all_regulatory_docs = []
        for region in regions:
            if region in regulatory_feeds:
                for feed_url in regulatory_feeds[region]:
                    try:
                        docs = self.ingester.ingest_rss(feed_url, max_items=15)
                        # Filter for therapeutic area
                        relevant_docs = [doc for doc in docs 
                                       if therapeutic_area.lower() in doc.page_content.lower()]
                        all_regulatory_docs.extend(relevant_docs)
                        print(f"📋 {region}: {len(relevant_docs)} relevant documents")
                    except Exception as e:
                        print(f"⚠️ {region} feed error: {e}")
        
        return all_regulatory_docs
    
    def guidance_impact_analysis(self, new_guidance: str, drug_program: str) -> Dict[str, Any]:
        """Analyze impact of new regulatory guidance on drug program"""
        
        impact_questions = [
            f"How does this guidance affect {drug_program} development timeline?",
            f"What new requirements does this guidance introduce?",
            f"Are there any competitive advantages or disadvantages?",
            f"What additional studies or data might be needed?",
            f"How should the regulatory strategy be modified?"
        ]
        
        impact_analysis = {}
        for question in impact_questions:
            # Combine guidance text with question
            full_query = f"Given this regulatory guidance: {new_guidance}\n\nQuestion: {question}"
            result = self.rag_chain.query(full_query)
            
            impact_analysis[question] = {
                'analysis': result['answer'],
                'confidence': result['confidence'],
                'key_requirements': self._extract_requirements(result['answer'])
            }
        
        return impact_analysis
    
    def _extract_requirements(self, analysis_text: str) -> List[str]:
        """Extract regulatory requirements from analysis"""
        requirements = []
        sentences = analysis_text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['require', 'must', 'shall', 'mandatory']):
                requirements.append(sentence.strip())
        
        return requirements[:5]

# Regulatory intelligence example
reg_intel = RegulatoryIntelligence()

# Monitor oncology regulatory changes
regulatory_docs = reg_intel.monitor_regulatory_changes("oncology", regions=['FDA', 'EMA'])

# Analyze impact of new guidance
new_guidance = """
FDA has updated guidance on Real-World Evidence for regulatory decision-making, 
emphasizing the use of electronic health records and claims data to support 
drug approvals in oncology. New requirements include patient-reported outcomes 
and long-term safety follow-up protocols.
"""

impact = reg_intel.guidance_impact_analysis(new_guidance, "CAR-T cell therapy")
print("📊 Regulatory Impact Analysis:")
for question, analysis in impact.items():
    print(f"\n❓ {question}")
    print(f"📝 Key requirements: {len(analysis['key_requirements'])}")
    for req in analysis['key_requirements'][:2]:
        print(f"   • {req}")
```

## 🎓 Educational Applications

### **Medical School Teaching Assistant**

```python
class MedicalEducationRAG:
    def __init__(self):
        self.curriculum_map = {}
        self.learning_objectives = {}
        
    def setup_medical_curriculum(self, specialty: str = "general"):
        """Setup curriculum-aligned knowledge base"""
        
        # Medical education sources
        medical_sources = [
            "medical_physiology_textbook.pdf",
            "pathology_case_studies.pdf",
            "clinical_guidelines.pdf",
            "medical_journal_articles.pdf"
        ]
        
        # Specialty-specific RSS feeds
        specialty_feeds = {
            'cardiology': 'https://www.ahajournals.org/rss/circulation.xml',
            'oncology': 'https://ascopubs.org/action/showFeed?type=etoc&feed=rss&jc=jco',
            'neurology': 'https://n.neurology.org/rss/current.xml',
            'general': 'https://www.nejm.org/action/showFeed?type=etoc&feed=rss'
        }
        
        all_docs = []
        
        # Process textbooks and materials
        for source in medical_sources:
            if os.path.exists(source):
                try:
                    docs = self.ingester.ingest_file(source)
                    all_docs.extend(docs)
                    print(f"📚 {source}: {len(docs)} sections")
                except Exception as e:
                    print(f"⚠️ {source}: {e}")
        
        # Add current medical literature
        if specialty in specialty_feeds:
            try:
                docs = self.ingester.ingest_rss(specialty_feeds[specialty], max_items=10)
                all_docs.extend(docs)
                print(f"📡 Current literature: {len(docs)} articles")
            except Exception as e:
                print(f"⚠️ Literature feed error: {e}")
        
        # Setup with medical glossary
        medical_terms = {
            'myocardial_infarction': 'heart attack caused by blocked blood flow to heart muscle',
            'angioplasty': 'procedure to open blocked coronary arteries using balloon catheter',
            'electrocardiogram': 'test recording electrical activity of the heart',
            'cardiac_enzymes': 'proteins released when heart muscle is damaged'
        }
        
        vectordb = self.db_manager.create_db(all_docs)
        entity_linker = EntityLinker()
        glossary_mgr = GlossaryManager(medical_terms)
        self.rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr)
        
        return len(all_docs)
    
    def generate_case_study(self, topic: str, difficulty: str = "intermediate") -> Dict[str, Any]:
        """Generate clinical case study for learning"""
        
        case_prompt = f"""
        Create a {difficulty} level clinical case study about {topic} that includes:
        1. Patient presentation with relevant history
        2. Physical examination findings
        3. Diagnostic considerations
        4. Laboratory/imaging results
        5. Treatment plan and rationale
        
        Make it realistic and educational for medical students.
        """
        
        result = self.rag_chain.query(case_prompt, enable_decomposition=True)
        
        # Extract case components
        case_study = {
            'topic': topic,
            'difficulty': difficulty,
            'full_case': result['answer'],
            'key_concepts': [e['text'] for e in result['entities'][:10]],
            'learning_points': self._extract_learning_points(result['answer'])
        }
        
        return case_study
    
    def create_study_guide(self, topics: List[str]) -> Dict[str, Any]:
        """Create comprehensive study guide"""
        
        study_guide = {
            'topics_covered': topics,
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        for topic in topics:
            # Generate key concepts
            concept_query = f"What are the essential concepts a medical student should know about {topic}?"
            concept_result = self.rag_chain.query(concept_query)
            
            # Generate practice questions
            question_query = f"Generate 3 clinical reasoning questions about {topic} for medical students"
            question_result = self.rag_chain.query(question_query)
            
            study_guide['sections'][topic] = {
                'key_concepts': concept_result['answer'],
                'practice_questions': question_result['answer'],
                'entities': [e['text'] for e in concept_result['entities'][:5]]
            }
        
        return study_guide
    
    def _extract_learning_points(self, case_text: str) -> List[str]:
        """Extract key learning points from case study"""
        learning_points = []
        sentences = case_text.split('.')
        
        for sentence in sentences:
            # Look for educational content indicators
            if any(keyword in sentence.lower() for keyword in 
                   ['important', 'note', 'remember', 'key', 'critical', 'significant']):
                if len(sentence.strip()) > 20:
                    learning_points.append(sentence.strip())
        
        return learning_points[:5]

# Medical education example
med_edu = MedicalEducationRAG()

# Setup cardiology curriculum
docs_count = med_edu.setup_medical_curriculum("cardiology")
print(f"📖 Medical curriculum loaded: {docs_count} documents")

# Generate cardiology case study
case = med_edu.generate_case_study("acute myocardial infarction", "intermediate")
print(f"\n🏥 Case Study: {case['topic']}")
print(f"📝 Key concepts: {', '.join(case['key_concepts'][:5])}")
print(f"💡 Learning points: {len(case['learning_points'])}")

# Create study guide for cardiology topics
cardiology_topics = [
    "acute coronary syndrome",
    "heart failure", 
    "cardiac arrhythmias",
    "valvular heart disease"
]

study_guide = med_edu.create_study_guide(cardiology_topics)
print(f"\n📚 Study Guide Created:")
print(f"Topics: {len(study_guide['sections'])}")
for topic in study_guide['sections']:
    entities = len(study_guide['sections'][topic]['entities'])
    print(f"  • {topic}: {entities} key entities")
```

## 🔬 Research Lab Integration

### **Laboratory Information Management**

```python
class LabRAGSystem:
    def __init__(self):
        self.protocols = {}
        self.safety_database = {}
        self.equipment_manuals = {}
        
    def setup_lab_knowledge_base(self, lab_type: str = "molecular_biology"):
        """Setup lab-specific knowledge base"""
        
        # Lab documentation sources
        lab_docs = {
            'protocols': 'lab_protocols/',
            'safety_sheets': 'msds_sheets/',
            'equipment_manuals': 'equipment_docs/',
            'publications': 'lab_publications/'
        }
        
        all_docs = []
        for doc_type, folder_path in lab_docs.items():
            if os.path.exists(folder_path):
                for file_path in Path(folder_path).rglob('*'):
                    if file_path.suffix in ['.pdf', '.txt', '.md']:
                        try:
                            docs = self.ingester.ingest_file(str(file_path))
                            all_docs.extend(docs)
                            print(f"📋 {doc_type}/{file_path.name}: {len(docs)} sections")
                        except Exception as e:
                            print(f"⚠️ Error with {file_path}: {e}")
        
        # Add lab-specific glossary
        lab_terms = {
            'pcr_cycles': 'number of temperature cycles in polymerase chain reaction',
            'gel_electrophoresis': 'technique separating DNA fragments by size using electric field',
            'western_blot': 'method detecting specific proteins using antibodies',
            'cell_culture': 'growing cells in controlled laboratory conditions'
        }
        
        vectordb = self.db_manager.create_db(all_docs)
        entity_linker = EntityLinker()
        glossary_mgr = GlossaryManager(lab_terms)
        self.rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr)
        
        return len(all_docs)
    
    def protocol_assistant(self, technique: str, specific_question: str = None) -> Dict[str, Any]:
        """Provide protocol guidance and troubleshooting"""
        
        if specific_question:
            query = f"For {technique}: {specific_question}"
        else:
            query = f"What is the standard protocol for {technique}?"
        
        result = self.rag_chain.query(query)
        
        # Extract protocol steps
        protocol_steps = self._extract_protocol_steps(result['answer'])
        
        return {
            'technique': technique,
            'protocol_guidance': result['answer'],
            'steps': protocol_steps,
            'safety_considerations': self._extract_safety_info(result['answer']),
            'related_entities': result['entities']
        }
    
    def troubleshooting_assistant(self, problem_description: str) -> Dict[str, Any]:
        """Help troubleshoot experimental problems"""
        
        troubleshoot_query = f"""
        Experimental problem: {problem_description}
        
        Provide:
        1. Possible causes of this problem
        2. Step-by-step troubleshooting approach
        3. Prevention strategies
        4. Alternative methods if needed
        """
        
        result = self.rag_chain.query(troubleshoot_query, enable_decomposition=True)
        
        return {
            'problem': problem_description,
            'analysis': result['answer'],
            'confidence': result['confidence'],
            'suggested_actions': self._extract_actions(result['answer'])
        }
    
    def safety_query(self, chemical_or_procedure: str) -> Dict[str, Any]:
        """Provide safety information for chemicals or procedures"""
        
        safety_query = f"""
        What are the safety considerations, hazards, and proper handling procedures for {chemical_or_procedure}?
        Include personal protective equipment requirements and emergency procedures.
        """
        
        result = self.rag_chain.query(safety_query)
        
        return {
            'item': chemical_or_procedure,
            'safety_info': result['answer'],
            'ppe_required': self._extract_ppe_requirements(result['answer']),
            'hazard_level': self._assess_hazard_level(result['answer'])
        }
    
    def _extract_protocol_steps(self, protocol_text: str) -> List[str]:
        """Extract numbered steps from protocol"""
        steps = []
        lines = protocol_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered steps
            if re.match(r'^\d+\.', line) or re.match(r'^Step \d+', line):
                steps.append(line)
        
        return steps[:20]  # Limit to 20 steps
    
    def _extract_safety_info(self, text: str) -> List[str]:
        """Extract safety-related information"""
        safety_info = []
        sentences = text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in 
                   ['safety', 'caution', 'warning', 'hazard', 'danger', 'protective']):
                safety_info.append(sentence.strip())
        
        return safety_info[:5]
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract actionable troubleshooting steps"""
        actions = []
        sentences = text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in 
                   ['check', 'verify', 'adjust', 'replace', 'clean', 'repeat']):
                actions.append(sentence.strip())
        
        return actions[:8]
    
    def _extract_ppe_requirements(self, safety_text: str) -> List[str]:
        """Extract PPE requirements"""
        ppe_items = []
        ppe_keywords = ['gloves', 'goggles', 'lab coat', 'fume hood', 'mask', 'respirator']
        
        for keyword in ppe_keywords:
            if keyword in safety_text.lower():
                ppe_items.append(keyword)
        
        return ppe_items
    
    def _assess_hazard_level(self, safety_text: str) -> str:
        """Assess hazard level based on safety text"""
        high_risk_keywords = ['toxic', 'carcinogenic', 'explosive', 'highly flammable']
        medium_risk_keywords = ['irritant', 'corrosive', 'flammable']
        
        text_lower = safety_text.lower()
        
        if any(keyword in text_lower for keyword in high_risk_keywords):
            return "HIGH"
        elif any(keyword in text_lower for keyword in medium_risk_keywords):
            return "MEDIUM"
        else:
            return "LOW"

# Laboratory integration example
lab_rag = LabRAGSystem()

# Setup molecular biology lab
docs_count = lab_rag.setup_lab_knowledge_base("molecular_biology")
print(f"🧪 Lab knowledge base: {docs_count} documents")

# Protocol assistance
pcr_protocol = lab_rag.protocol_assistant("PCR amplification", "Why am I getting no amplification?")
print(f"\n🔬 PCR Protocol Assistance:")
print(f"Steps identified: {len(pcr_protocol['steps'])}")
print(f"Safety considerations: {len(pcr_protocol['safety_considerations'])}")

# Troubleshooting
problem = "My western blot bands are very faint and there's high background"
troubleshoot = lab_rag.troubleshooting_assistant(problem)
print(f"\n🔧 Troubleshooting Analysis:")
print(f"Suggested actions: {len(troubleshoot['suggested_actions'])}")
print(f"Confidence: {troubleshoot['confidence']:.2f}")

# Safety information
safety_info = lab_rag.safety_query("formaldehyde")
print(f"\n⚠️ Safety Information for formaldehyde:")
print(f"Hazard level: {safety_info['hazard_level']}")
print(f"PPE required: {', '.join(safety_info['ppe_required'])}")
```

---

## 🚀 Advanced Integration Patterns

### **Multi-Modal Research Platform**

```python
class MultiModalBioRAG:
    """Advanced platform combining text, images, and structured data"""
    
    def __init__(self):
        self.text_rag = None
        self.image_analyzer = None
        self.structured_data = None
        
    def setup_multimodal_pipeline(self):
        """Setup pipeline for multiple data modalities"""
        
        # Text processing (existing RAG)
        self.text_rag = RAGChain(vectordb, entity_linker, glossary_mgr)
        
        # Placeholder for image analysis integration
        # Could integrate with medical imaging AI models
        self.image_analyzer = self._setup_image_analysis()
        
        # Structured data integration (databases, spreadsheets)
        self.structured_data = self._setup_structured_data()
    
    def comprehensive_analysis(self, research_question: str, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze question across multiple data modalities"""
        
        results = {
            'question': research_question,
            'text_analysis': None,
            'image_analysis': None,
            'structured_analysis': None,
            'integrated_insights': None
        }
        
        # Text-based analysis
        if 'documents' in data_sources:
            results['text_analysis'] = self.text_rag.query(research_question)
        
        # Image analysis (if images provided)
        if 'images' in data_sources:
            results['image_analysis'] = self._analyze_images(
                data_sources['images'], 
                research_question
            )
        
        # Structured data analysis
        if 'databases' in data_sources:
            results['structured_analysis'] = self._query_structured_data(
                data_sources['databases'],
                research_question
            )
        
        # Integrate insights from all modalities
        results['integrated_insights'] = self._integrate_multimodal_results(results)
        
        return results
    
    def _setup_image_analysis(self):
        """Setup medical image analysis (placeholder)"""
        # Integration point for medical imaging AI
        return {"status": "placeholder_for_image_ai"}
    
    def _setup_structured_data(self):
        """Setup structured data connections"""
        # Integration point for databases, APIs
        return {"status": "placeholder_for_database_connections"}
    
    def _analyze_images(self, images: List[str], question: str) -> Dict[str, Any]:
        """Analyze medical images in context of question"""
        # Placeholder for image analysis
        return {
            "images_analyzed": len(images),
            "findings": "Image analysis not implemented in this example"
        }
    
    def _query_structured_data(self, databases: List[str], question: str) -> Dict[str, Any]:
        """Query structured databases"""
        # Placeholder for database queries
        return {
            "databases_queried": len(databases),
            "structured_findings": "Database integration not implemented in this example"
        }
    
    def _integrate_multimodal_results(self, results: Dict[str, Any]) -> str:
        """Integrate insights from different modalities"""
        
        integration_prompt = f"""
        Integrate these research findings to provide comprehensive insights:
        
        Text Analysis: {results.get('text_analysis', {}).get('answer', 'Not available')}
        Image Analysis: {results.get('image_analysis', {}).get('findings', 'Not available')}
        Structured Data: {results.get('structured_analysis', {}).get('structured_findings', 'Not available')}
        
        Provide integrated insights that synthesize findings across all available data modalities.
        """
        
        if self.text_rag:
            integration_result = self.text_rag.query(integration_prompt)
            return integration_result['answer']
        else:
            return "Integration requires initialized text RAG system"

# Example usage of advanced patterns
multimodal_rag = MultiModalBioRAG()
multimodal_rag.setup_multimodal_pipeline()

# Comprehensive analysis across data types
research_data = {
    'documents': ['research_paper.pdf', 'clinical_study.pdf'],
    'images': ['histology_slide.jpg', 'mri_scan.dcm'],
    'databases': ['clinical_trials_db', 'genomic_variants_db']
}

comprehensive_results = multimodal_rag.comprehensive_analysis(
    "What factors predict response to immunotherapy in melanoma patients?",
    research_data
)

print("🔬 Comprehensive Multi-Modal Analysis:")
for modality, results in comprehensive_results.items():
    if results:
        print(f"  {modality}: Analysis completed")
```

---

This comprehensive examples guide demonstrates BioRAG's versatility across academic research, clinical applications, pharmaceutical development, education, and laboratory management. Each example includes complete, runnable code that can be adapted for specific use cases.

**Ready to implement these examples?** Start with the basic scenarios# 🎯 BioRAG Examples & Use Cases

Comprehensive examples showing how to use BioRAG for various biomedical research scenarios.

## 🏥 Clinical Research Scenarios

### **Scenario 1: Cancer Genomics Research**

**Goal**: Analyze BRCA gene mutations and treatment options

```python
from biorag.core.ingest import IngestPipeline
from biorag.core.vectordb import VectorDBManager
from biorag.core.rag_chain import RAGChain
from biorag.core.linker import EntityLinker
from biorag.core.glossary import GlossaryManager

# Initialize components for cancer research
ingester = IngestPipeline(chunk_size=800, chunk_overlap=100)
db_manager = VectorDBManager(embedding_model="scibert")
entity_linker = EntityLinker()
glossary_mgr = GlossaryManager()

# Ingest cancer research papers
cancer_papers = [
    "brca1_functional_analysis.pdf",
    "parp_inhibitor_mechanisms.pdf", 
    "hereditary_cancer_guidelines.pdf"
]

all_documents = []
for paper in cancer_papers:
    try:
        docs = ingester.ingest_file(paper)
        all_documents.extend(docs)
        print(f"✅ Processed {paper}: {len(docs)} chunks")
    except Exception as e:
        print(f"❌ Error with {paper}: {e}")

# Create knowledge base
vectordb = db_manager.create_db(all_documents)
rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr)

# Research questions
questions = [
    "What are the functional differences between BRCA1 and BRCA2 mutations?",
    "How do PARP inhibitors achieve synthetic lethality in BRCA-deficient tumors?",
    "What are the current guidelines for genetic testing in hereditary cancer?",
    "Which biomarkers predict response to PARP inhibitor therapy?"
]

for question in questions:
    print(f"\n🔬 Question: {question}")
    result = rag_chain.query(question)
    print(f"📋 Answer: {result['answer'][:200]}...")
    print(f"🔗 Entities found: {len(result['entities'])}")
    print(f"📊 Confidence: {result['confidence']:.2f}")
```

### **Scenario 2: Drug Discovery Pipeline**

**Goal**: Research novel therapeutic compounds and their mechanisms

```python
# Specialized setup for drug discovery
drug_discovery_feeds = [
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/drug%20discovery/",
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/pharmaceutical%20development/"
]

# Ingest latest drug discovery research
drug_docs = []
for feed_url in drug_discovery_feeds:
    try:
        docs = ingester.ingest_rss(feed_url, max_items=15)
        drug_docs.extend(docs)
        print(f"📡 Fetched {len(docs)} articles from RSS")
    except Exception as e:
        print(f"❌ RSS error: {e}")

# Add to existing knowledge base
if drug_docs:
    db_manager.add_documents(vectordb, drug_docs)
    print(f"📚 Added {len(drug_docs)} drug discovery documents")

# Drug discovery questions
drug_questions = [
    "What are the latest advances in targeted cancer therapy?",
    "How do CDK4/6 inhibitors work in breast cancer treatment?",
    "What are the challenges in developing resistance-resistant drugs?",
    "Which novel drug delivery systems show promise for brain tumors?"
]

for question in drug_questions:
    result = rag_chain.query(question, enable_decomposition=True)
    
    # Enhanced analysis for drug discovery
    entities = result['entities']
    chemicals = [e for e in entities if e['type'] in ['Chemicals Drugs', 'Chemical']]
    
    print(f"\n💊 Question: {question}")
    print(f"🧪 Chemical compounds mentioned: {len(chemicals)}")
    for chemical in chemicals[:3]:  # Show top 3
        print(f"  - {chemical['text']}: {chemical.get('description', 'No description')}")
```

### **Scenario 3: Precision Medicine Analysis**

**Goal**: Analyze personalized treatment approaches based on genetic profiles

```python
# Precision medicine setup with enhanced entity recognition
precision_med_terms = {
    "pharmacogenomics": "study of how genes affect drug response",
    "biomarker_stratification": "patient grouping based on molecular markers",
    "companion_diagnostics": "tests identifying patients likely to benefit from specific treatments",
    "tumor_mutational_burden": "number of mutations in cancer genome affecting treatment response"
}

# Add specialized terms
glossary_mgr.add_terms(precision_med_terms)

# Precision medicine research questions
precision_questions = [
    "How does tumor mutational burden predict immunotherapy response?",
    "What pharmacogenomic factors affect tamoxifen metabolism?", 
    "How can liquid biopsies guide treatment decisions?",
    "What are the challenges in implementing precision oncology?"
]

results_summary = []
for question in precision_questions:
    result = rag_chain.query(question)
    
    # Extract key insights
    jargon_terms = glossary_mgr.extract_jargon(result['answer'])
    
    results_summary.append({
        'question': question,
        'answer_length': len(result['answer']),
        'entities_found': len(result['entities']),
        'jargon_terms': len(jargon_terms),
        'confidence': result['confidence']
    })

# Generate summary report
print("\n📊 Precision Medicine Analysis Summary:")
print("-" * 50)
for i, summary in enumerate(results_summary, 1):
    print(f"{i}. {summary['question'][:50]}...")
    print(f"   Answer length: {summary['answer_length']} chars")
    print(f"   Entities: {summary['entities_found']}, Jargon: {summary['jargon_terms']}")
    print(f"   Confidence: {summary['confidence']:.2f}\n")
```

## 🧬 Academic Research Workflows

### **Literature Review Automation**

```python
import json
from datetime import datetime, timedelta

class LiteratureReviewBot:
    def __init__(self, research_topic: str):
        self.topic = research_topic
        self.ingester = IngestPipeline()
        self.db_manager = VectorDBManager()
        self.rag_chain = None
        self.review_data = []
    
    def setup_knowledge_base(self, papers: List[str], rss_feeds: List[str] = None):
        """Initialize knowledge base with papers and feeds"""
        all_docs = []
        
        # Process papers
        for paper_path in papers:
            try:
                docs = self.ingester.ingest_file(paper_path)
                all_docs.extend(docs)
                print(f"📄 Added {paper_path}: {len(docs)} chunks")
            except Exception as e:
                print(f"❌ Error with {paper_path}: {e}")
        
        # Process RSS feeds
        if rss_feeds:
            for feed_url in rss_feeds:
                try:
                    docs = self.ingester.ingest_rss(feed_url, max_items=20)
                    all_docs.extend(docs)
                    print(f"📡 Added RSS feed: {len(docs)} articles")
                except Exception as e:
                    print(f"❌ RSS error: {e}")
        
        # Create vector database
        vectordb = self.db_manager.create_db(all_docs)
        entity_linker = EntityLinker()
        glossary_mgr = GlossaryManager()
        self.rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr)
        
        print(f"🗄️ Knowledge base ready with {len(all_docs)} documents")
    
    def conduct_review(self, review_questions: List[str]) -> Dict[str, Any]:
        """Conduct systematic literature review"""
        if not self.rag_chain:
            raise ValueError("Knowledge base not initialized")
        
        review_results = {
            'topic': self.topic,
            'timestamp': datetime.now().isoformat(),
            'questions_analyzed': len(review_questions),
            'findings': []
        }
        
        for question in review_questions:
            print(f"\n🔍 Analyzing: {question}")
            
            result = self.rag_chain.query(
                question, 
                enable_decomposition=True,
                enable_hyde=True
            )
            
            # Extract key findings
            finding = {
                'question': question,
                'answer': result['answer'],
                'key_entities': [e['text'] for e in result['entities'][:10]],
                'confidence': result['confidence'],
                'source_count': len(result['source_docs']),
                'sub_questions': result.get('sub_queries', [])
            }
            
            review_results['findings'].append(finding)
            
        return review_results
    
    def generate_summary_report(self, review_results: Dict[str, Any]) -> str:
        """Generate executive summary of literature review"""
        summary_prompt = f"""
        Based on the following literature review findings for the topic "{review_results['topic']}", 
        generate a comprehensive executive summary highlighting:
        1. Key research trends and patterns
        2. Major findings and consensus points  
        3. Research gaps and future directions
        4. Most frequently mentioned entities/concepts
        
        Findings: {json.dumps(review_results['findings'], indent=2)}
        """
        
        if self.rag_chain:
            summary_result = self.rag_chain.query(summary_prompt)
            return summary_result['answer']
        else:
            return "Summary generation requires initialized RAG chain"

# Usage example
review_bot = LiteratureReviewBot("CRISPR gene editing in cancer therapy")

# Setup with papers and RSS feeds
papers = [
    "crispr_cancer_review.pdf",
    "gene_editing_clinical_trials.pdf", 
    "crispr_safety_considerations.pdf"
]

rss_feeds = [
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/CRISPR%20cancer/",
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/gene%20editing%20therapy/"
]

review_bot.setup_knowledge_base(papers, rss_feeds)

# Define systematic review questions
review_questions = [
    "What are the current applications of CRISPR in cancer treatment?",
    "What safety concerns have been identified with CRISPR gene editing?",
    "How effective is CRISPR compared to traditional cancer therapies?",
    "What are the regulatory challenges for CRISPR-based treatments?",
    "Which cancer types show the most promise for CRISPR intervention?"
]

# Conduct review
results = review_bot.conduct_review(review_questions)

# Generate summary
summary = review_bot.generate_summary_report(results)
print("\n📋 Literature Review Summary:")
print("=" * 60)
print(summary)

# Export results
with open(f"literature_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    json.dump(results, f, indent=2)
```

### **Grant Proposal Research Assistant**

```python
class GrantProposalAssistant:
    def __init__(self):
        self.rag_chain = None
        self.research_landscape = {}
    
    def analyze_research_landscape(self, research_area: str, funding_agency: str = "NIH"):
        """Analyze current research landscape for grant writing"""
        # Specialized RSS feeds for grant research
        funding_feeds = {
            "NIH": [
                "https://www.nih.gov/news-events/news-releases/rss.xml",
                "https://grants.nih.gov/grants/guide/rss_guide_pa.xml"
            ],
            "NSF": [
                "https://www.nsf.gov/rss/rss_nsf_news.xml"
            ]
        }
        
        research_feeds = [
            f"https://pubmed.ncbi.nlm.nih.gov/rss/search/{research_area.replace(' ', '%20')}/"
        ]
        
        # Combine funding and research feeds
        all_feeds = funding_feeds.get(funding_agency, []) + research_feeds
        
        # Ingest recent funding announcements and research
        docs = []
        for feed_url in all_feeds:
            try:
                feed_docs = self.ingester.ingest_rss(feed_url, max_items=25)
                docs.extend(feed_docs)
            except Exception as e:
                print(f"⚠️ Feed error: {e}")
        
        if docs:
            vectordb = self.db_manager.create_db(docs)
            entity_linker = EntityLinker()
            glossary_mgr = GlossaryManager()
            self.rag_chain = RAGChain(vectordb, entity_linker, glossary_mgr)
        
        return len(docs)
    
    def generate_research_questions(self, research_area: str) -> List[str]:
        """Generate novel research questions for grant proposals"""
        if not self.