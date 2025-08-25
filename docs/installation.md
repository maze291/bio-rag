# 🛠️ BioRAG Installation Guide

Comprehensive installation instructions for all platforms and deployment scenarios.

## 📋 Prerequisites

### **System Requirements**
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 5GB free space for models and data
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### **Optional Dependencies**
- **Tesseract OCR**: For scanned PDF processing
- **CUDA/GPU**: For faster embedding generation
- **OpenAI API Key**: For GPT-4 integration

## 🚀 Quick Install (Recommended)

### **Option 1: Automated Script**
```bash
# Clone the repository
git clone https://github.com/yourusername/biorag.git
cd biorag

# Run automated installer
chmod +x install.sh
./install.sh

# Test installation
python cli.py --selftest
```

### **Option 2: Manual pip Install**
```bash
# Clone and install
git clone https://github.com/yourusername/biorag.git
cd biorag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SpaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bionlp13cg_md  # Optional

# Test installation
python cli.py --selftest
```

## 🖥️ Platform-Specific Instructions

### **Windows**

#### **Using Windows Subsystem for Linux (WSL) - Recommended**
```bash
# Install WSL2 with Ubuntu
wsl --install

# Inside WSL, follow the Linux instructions below
```

#### **Native Windows Installation**
```powershell
# Install Python from python.org or Microsoft Store
# Install Git for Windows

# Clone repository
git clone https://github.com/yourusername/biorag.git
cd biorag

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract (optional)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# Install SpaCy models
python -m spacy download en_core_web_sm

# Test
python cli.py --selftest
```

### **macOS**

#### **Using Homebrew (Recommended)**
```bash
# Install prerequisites
brew install python tesseract git

# Clone repository
git clone https://github.com/yourusername/biorag.git
cd biorag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SpaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bionlp13cg_md

# Test installation
python cli.py --selftest
```

#### **Using MacPorts**
```bash
# Install prerequisites
sudo port install python310 tesseract git

# Follow similar steps as above, using python3.10
```

### **Linux**

#### **Ubuntu/Debian**
```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv git
sudo apt install -y tesseract-ocr tesseract-ocr-eng
sudo apt install -y build-essential python3-dev

# Clone repository
git clone https://github.com/yourusername/biorag.git
cd biorag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SpaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bionlp13cg_md

# Test installation
python cli.py --selftest
```

#### **CentOS/RHEL/Fedora**
```bash
# Install system dependencies
sudo dnf install -y python3 python3-pip python3-devel git
sudo dnf install -y tesseract tesseract-langpack-eng
sudo dnf install -y gcc gcc-c++

# Follow similar steps as Ubuntu
```

#### **Arch Linux**
```bash
# Install system dependencies
sudo pacman -S python python-pip git tesseract tesseract-data-eng

# Follow similar steps as Ubuntu
```

## 🐳 Docker Installation

### **Using Docker Compose (Recommended)**
```yaml
# docker-compose.yml
version: '3.8'
services:
  biorag:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./vector_db:/app/vector_db
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8501
```

### **Manual Docker Build**
```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SpaCy models
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
```

```bash
# Build and run
docker build -t biorag .
docker run -p 8501:8501 biorag
```

## ☁️ Cloud Deployment

### **Google Cloud Platform**

#### **Cloud Run**
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy biorag \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### **App Engine**
```yaml
# app.yaml
runtime: python310

env_variables:
  OPENAI_API_KEY: "your-key-here"

automatic_scaling:
  min_instances: 0
  max_instances: 10
```

```bash
gcloud app deploy
```

### **AWS**

#### **Elastic Beanstalk**
```bash
# Install EB CLI
pip install awsebcli

# Initialize and deploy
eb init biorag
eb create biorag-env
eb deploy
```

#### **ECS/Fargate**
```bash
# Build and push to ECR
aws ecr create-repository --repository-name biorag
docker tag biorag:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/biorag:latest
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/biorag:latest

# Deploy using ECS console or CloudFormation
```

### **Azure**

#### **Container Instances**
```bash
# Create resource group
az group create --name biorag-rg --location eastus

# Deploy container
az container create \
  --resource-group biorag-rg \
  --name biorag-app \
  --image biorag:latest \
  --dns-name-label biorag-unique \
  --ports 8501
```

### **Heroku**
```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run main.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-biorag-app
git push heroku main
```

## 🔧 Advanced Installation Options

### **Development Installation**
```bash
# Clone with development dependencies
git clone https://github.com/yourusername/biorag.git
cd biorag

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### **GPU Acceleration**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# BioRAG will automatically use GPU for embeddings
```

### **Custom Model Installation**
```bash
# Install additional SpaCy models
python -m spacy download en_ner_bc5cdr_md  # Chemistry/Disease NER
python -m spacy download en_core_sci_sm    # General scientific model

# Download custom models
mkdir models
cd models
wget https://example.com/custom-biomedical-model.tar.gz
tar -xzf custom-biomedical-model.tar.gz
```

## 🔑 Configuration

### **Environment Variables**
Create `.env` file:
```bash
# Copy template
cp .env.example .env

# Edit with your settings
nano .env
```

```bash
# .env file contents
OPENAI_API_KEY=your_openai_api_key_here
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
BIORAG_MAX_DOCS=50000
BIORAG_CHUNK_SIZE=1000
BIORAG_CHUNK_OVERLAP=200
BIORAG_DB_PATH=./vector_db
BIORAG_EMBEDDING_MODEL=scibert
BIORAG_LLM_MODEL=ollama
BIORAG_LOG_LEVEL=INFO
```

### **Model Configuration**
```python
# config.py
EMBEDDING_MODELS = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "biobert": "dmis-lab/biobert-v1.1", 
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "clinical": "emilyalsentzer/Bio_ClinicalBERT"
}

LLM_MODELS = {
    "gpt4": "openai",
    "llama": "ollama",
    "mixtral": "ollama"
}
```

## 🧪 Verification

### **Self-Test**
```bash
# Run comprehensive self-test
python cli.py --selftest

# Run smoke test
python tests/smoke_test.py

# Run unit tests
pytest tests/ -v
```

### **Manual Verification**
```bash
# Test CLI
python cli.py --ingest examples/sample_biomedical_text.txt
python cli.py --query "What is BRCA1?"

# Test web interface
streamlit run main.py
# Open http://localhost:8501

# Test RSS ingestion
python cli.py --rss "https://www.nature.com/nm.rss" --query "Latest research"
```

## 🔧 Troubleshooting

### **Common Issues**

#### **SpaCy Model Not Found**
```bash
# Solution: Install missing models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bionlp13cg_md
```

#### **Tesseract Not Found**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### **CUDA/GPU Issues**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Memory Issues**
```bash
# Reduce model size in .env
BIORAG_EMBEDDING_MODEL=all-MiniLM-L6-v2  # Smaller model
BIORAG_CHUNK_SIZE=500  # Smaller chunks
```

#### **Permission Issues**
```bash
# Fix pip permissions
pip install --user -r requirements.txt

# Fix file permissions
chmod +x install.sh
```

### **Performance Optimization**

#### **For Low-Memory Systems**
```bash
# Use lightweight models
export BIORAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export BIORAG_CHUNK_SIZE=300
```

#### **For High-Performance Systems**
```bash
# Use full-size models
export BIORAG_EMBEDDING_MODEL="scibert"
export BIORAG_CHUNK_SIZE=1500
export BIORAG_USE_GPU=true
```

## 📞 Getting Help

### **Log Collection**
```bash
# Enable debug logging
export BIORAG_LOG_LEVEL=DEBUG

# Run with verbose output
python cli.py --selftest --verbose

# Check logs
tail -f ~/.biorag/logs/biorag.log
```

### **System Information**
```bash
# Collect system info for bug reports
python -c "
import sys, platform, torch
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### **Support Channels**
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/biorag/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/yourusername/biorag/discussions)
- 📧 **Email**: your.email@domain.com
- 📖 **Documentation**: [GitHub Wiki](https://github.com/yourusername/biorag/wiki)

---

**Installation successful?** 🎉 Head to the [Examples Guide](examples.md) to start exploring BioRAG!