#!/bin/bash
# BioRAG Installation Script
# Handles system dependencies and Python packages

set -e  # Exit on any error

echo "🧬 BioRAG Installation Script"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Found Python $PYTHON_VERSION"
        
        # Check if version is >= 3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (Ubuntu/Debian)
        if command -v apt-get &> /dev/null; then
            print_status "Detected Ubuntu/Debian system"
            sudo apt-get update
            sudo apt-get install -y \
                tesseract-ocr \
                tesseract-ocr-eng \
                python3-pip \
                python3-venv \
                python3-dev \
                build-essential
            print_success "System dependencies installed"
        elif command -v yum &> /dev/null; then
            print_status "Detected RHEL/CentOS system"
            sudo yum install -y \
                tesseract \
                tesseract-langpack-eng \
                python3-pip \
                python3-devel \
                gcc \
                gcc-c++
            print_success "System dependencies installed"
        else
            print_warning "Unknown Linux distribution. Please install tesseract-ocr manually."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        print_status "Detected macOS system"
        if command -v brew &> /dev/null; then
            brew install tesseract
            print_success "Tesseract installed via Homebrew"
        else
            print_warning "Homebrew not found. Please install tesseract manually or install Homebrew."
        fi
    else
        print_warning "Unknown operating system. Please install tesseract-ocr manually."
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Make sure we're in the virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Not in virtual environment, activating..."
        source .venv/bin/activate
    fi
    
    # Install requirements
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

# Install SpaCy models
install_spacy_models() {
    print_status "Installing SpaCy models..."
    
    # Basic English model (required)
    python -m spacy download en_core_web_sm
    print_success "Basic English model installed"
    
    # Try to install biomedical models (optional)
    print_status "Attempting to install biomedical NER models..."
    
    if python -m spacy download en_ner_bionlp13cg_md; then
        print_success "Biomedical NER model installed"
    else
        print_warning "Could not install biomedical NER model (this is optional)"
    fi
    
    if python -m spacy download en_ner_bc5cdr_md; then
        print_success "Chemical/Disease NER model installed"
    else
        print_warning "Could not install chemical/disease NER model (this is optional)"
    fi
}

# Run self-test
run_selftest() {
    print_status "Running self-test..."
    
    if python cli.py --selftest; then
        print_success "Self-test passed!"
    else
        print_error "Self-test failed. Please check the installation."
        return 1
    fi
}

# Create .env file from example
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ] && [ -f ".env.example" ]; then
        cp .env.example .env
        print_success "Created .env file from example"
        print_warning "Please edit .env file to add your API keys and configure settings"
    else
        print_warning ".env file already exists or .env.example not found"
    fi
}

# Main installation flow
main() {
    echo
    print_status "Starting BioRAG installation..."
    echo
    
    # Check if we're in the right directory
    if [ ! -f "cli.py" ] || [ ! -f "main.py" ]; then
        print_error "Installation script must be run from the BioRAG project directory"
        exit 1
    fi
    
    # Installation steps
    check_python
    install_system_deps
    create_venv
    install_python_deps
    install_spacy_models
    setup_env
    
    echo
    print_status "Running self-test to verify installation..."
    if run_selftest; then
        echo
        print_success "🎉 BioRAG installation completed successfully!"
        echo
        echo "Next steps:"
        echo "1. Activate the virtual environment: source .venv/bin/activate"
        echo "2. Edit .env file to add your API keys (optional)"
        echo "3. Start the web interface: streamlit run main.py"
        echo "4. Or use the CLI: python cli.py --help"
        echo
    else
        echo
        print_error "Installation completed but self-test failed"
        echo "Please check the error messages above and try manual testing"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --system-only)
        print_status "Installing only system dependencies..."
        check_python
        install_system_deps
        print_success "System dependencies installed"
        ;;
    --python-only)
        print_status "Installing only Python dependencies..."
        check_python
        create_venv
        install_python_deps
        install_spacy_models
        print_success "Python dependencies installed"
        ;;
    --help|-h)
        echo "BioRAG Installation Script"
        echo
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --system-only    Install only system dependencies"
        echo "  --python-only    Install only Python dependencies" 
        echo "  --help, -h       Show this help message"
        echo
        echo "Run without arguments to perform full installation"
        ;;
    *)
        main
        ;;
esac