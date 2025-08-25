#!/usr/bin/env python3
"""
BioRAG Complete System - Streamlit UI Entry Point
A comprehensive RAG system with entity linking and jargon simplification for biomedical literature
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import tempfile
import hashlib
from urllib.parse import urlparse
import re  # Needed by sanitize_html
import concurrent.futures  # Allows background thread for heavy ingestion

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

from core.ingest import IngestPipeline
from core.vectordb import VectorDBManager
from core.linker import EntityLinker
from core.glossary import GlossaryManager
from core.rag_chain import RAGChain

# Page configuration
st.set_page_config(
    page_title="BioRAG Knowledge Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Security: Domain allowlist for URL ingestion
ALLOWED_DOMAINS = {
    'pubmed.ncbi.nlm.nih.gov',
    'arxiv.org',
    'biorxiv.org',
    'medrxiv.org',
    'nature.com',
    'science.org',
    'cell.com',
    'plos.org',
    'biomedcentral.com',
    'elifesciences.org'
}

# ChatGPT-like UI styling
st.markdown("""
<style>
    /* Reset and base styles */
    .stApp {
        background-color: #ffffff;
        color: #374151;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main content area - full width like ChatGPT */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Sidebar styling - minimal like ChatGPT */
    .css-1d391kg {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e5e7;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e5e7;
        width: 260px !important;
    }
    
    /* Chat message styling - exactly like ChatGPT */
    .stChatMessage {
        background-color: transparent;
        border: none;
        border-radius: 0;
        padding: 24px 0;
        margin: 0;
        max-width: 768px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* User message background */
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #f7f7f8;
    }
    
    /* Assistant message background */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #ffffff;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        max-width: 768px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    .stChatInput > div {
        border: 1px solid #d1d5db;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stChatInput input {
        border: none;
        padding: 12px 16px;
        font-size: 16px;
        line-height: 1.5;
    }
    
    /* Title styling */
    .main h1, .main h2 {
        color: #374151;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Button styling - ChatGPT style */
    .stButton > button {
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #0d8b6d;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        background-color: #f9fafb;
        text-align: center;
        padding: 2rem;
    }
    
    /* Entity links - subtle green like ChatGPT links */
    .bio-entity {
        color: #10a37f;
        text-decoration: underline;
        text-decoration-thickness: 1px;
        text-underline-offset: 2px;
        font-weight: 500;
    }
    
    .bio-entity:hover {
        color: #0d8b6d;
    }
    
    /* Jargon tooltips */
    .jargon {
        color: #6366f1;
        border-bottom: 1px dashed #6366f1;
        cursor: help;
    }
    
    /* Source expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #374151;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        color: #9ca3af;
    }
    
    .loading-dots span {
        animation: loading-dots 1.4s ease-in-out infinite both;
    }
    
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes loading-dots {
        0%, 80%, 100% { opacity: 0.3; }
        40% { opacity: 1; }
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    /* Warning/error styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 14px;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        font-size: 14px;
        color: #374151;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        color: #10a37f;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with proper defaults
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'detected_entities' not in st.session_state:
    st.session_state.detected_entities = []
if 'document_count' not in st.session_state:
    st.session_state.document_count = 0
if 'message_limit' not in st.session_state:
    st.session_state.message_limit = 100  # Prevent memory bloat
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"


# Helper functions
def sanitize_html(text: str) -> str:
    """Basic HTML sanitization to prevent XSS"""
    # Remove script tags and dangerous attributes
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'on\w+\s*=\s*["\'].*?["\']',
        r'javascript:',
        r'data:text/html'
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    return text


def validate_url(url: str) -> bool:
    """Validate URL against allowlist"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check if domain or any parent domain is in allowlist
        for allowed in ALLOWED_DOMAINS:
            if domain == allowed or domain.endswith('.' + allowed):
                return True

        return False
    except:
        return False


def trim_message_history():
    """Trim message history to prevent memory bloat"""
    if len(st.session_state.messages) > st.session_state.message_limit:
        # Keep first message and last N messages
        st.session_state.messages = (
                st.session_state.messages[:1] +
                st.session_state.messages[-(st.session_state.message_limit - 1):]
        )


# Initialize components with proper error handling
#@st.cache_resource
def init_components():
    """Initialize all system components"""
    try:
        st.write("Loading IngestPipeline...")
        ingester = IngestPipeline()
        st.write("✅ IngestPipeline loaded")

        st.write("Loading VectorDBManager...")
        db_manager = VectorDBManager()
        st.write("✅ VectorDBManager loaded")

        st.write("Loading EntityLinker...")
        entity_linker = EntityLinker()
        st.write("✅ EntityLinker loaded")

        st.write("Loading GlossaryManager...")
        glossary_mgr = GlossaryManager()
        st.write("✅ GlossaryManager loaded")

        return ingester, db_manager, entity_linker, glossary_mgr
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Function to rebuild RAG chain when settings change
def rebuild_rag_chain():
    """Rebuild RAG chain with current settings"""
    if st.session_state.vector_db is not None:
        st.session_state.rag_chain = RAGChain(
            st.session_state.vector_db,
            entity_linker,
            glossary_mgr,
            llm_model="openai" if st.session_state.llm_model == "openai" else None
        )
# Initialize core components at the module level so they are always in scope
ingester, db_manager, entity_linker, glossary_mgr = init_components()


# Sidebar
with st.sidebar:
    st.markdown("# 🧬 BioRAG Assistant")
    st.markdown("### Knowledge Base Management")

    # File upload with better error handling
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'html'],
        accept_multiple_files=True,
        help="Upload PDFs, text files, or HTML documents (max 200MB each)"
    )

    # RSS Feed input
    with st.expander("📡 Add RSS Feeds", expanded=False):
        rss_url = st.text_input("RSS Feed URL")
        if st.button("Fetch RSS", use_container_width=True):
            with st.spinner("Fetching RSS feed..."):
                try:
                    docs = ingester.ingest_rss(rss_url)
                    st.success(f"Fetched {len(docs)} articles!")
                except Exception as e:
                    st.error(f"Error fetching RSS feed. Please check the URL and try again.")

    # URL input with security validation
    with st.expander("🌐 Add Web Page", expanded=False):
        web_url = st.text_input("Web Page URL")
        st.caption(f"Allowed domains: {', '.join(sorted(ALLOWED_DOMAINS)[:5])}...")

        if st.button("Fetch Page", use_container_width=True):
            if not web_url:
                st.error("Please enter a URL")
            elif not validate_url(web_url):
                st.error("URL domain not in allowlist. Only scientific sources are allowed.")
            else:
                with st.spinner("Fetching web page..."):
                    try:
                        docs = ingester.ingest_url(web_url)
                        st.success(f"Fetched page successfully!")
                    except Exception as e:
                        st.error(f"Error fetching page. Please check the URL and try again.")

    # Process uploaded files with proper async handling
    if uploaded_files:
        # Check file sizes
        total_size = sum(file.size for file in uploaded_files)
        if total_size > 200 * 1024 * 1024:  # 200MB total
            st.error("Total file size exceeds 200MB. Please upload smaller files.")
        else:
            if st.button("Process Documents", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_docs = []
                for i, file in enumerate(uploaded_files):
                    if file.size > 50 * 1024 * 1024:  # 50MB per file
                        status_text.text(f"Skipping {file.name} - too large")
                        continue

                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))

                    try:
                        print(f"DEBUG: Processing file {file.name}")
                        # Use proper temp file handling
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                            tmp_file.write(file.read())
                            tmp_path = Path(tmp_file.name)

                        print(f"DEBUG: Created temp file {tmp_path}")

                        # Ingest
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            docs = pool.submit(ingester.ingest_file, str(tmp_path)).result()

                        print(f"DEBUG: Got {len(docs)} docs")
                        all_docs.extend(docs)

                        # Cleanup
                        tmp_path.unlink()

                    except Exception as e:
                        print(f"DEBUG ERROR: {e}")
                        print(f"DEBUG ERROR TYPE: {type(e)}")
                        import traceback

                        print(traceback.format_exc())
                        st.error(f"Error processing {file.name}: {str(e)}")
                        if 'tmp_path' in locals() and tmp_path.exists():
                            tmp_path.unlink()

                # Build/update vector DB
                if all_docs:
                    status_text.text("Building vector database...")
                    try:
                        if st.session_state.vector_db is None:
                            st.session_state.vector_db = db_manager.create_db(all_docs)
                        else:
                            db_manager.add_documents(st.session_state.vector_db, all_docs)

                        # Rebuild RAG chain with current settings
                        rebuild_rag_chain()

                        st.session_state.document_count += len(all_docs)
                        status_text.text(f"✅ Processed {len(all_docs)} documents!")
                    except Exception as e:
                        st.error("Error building vector database. Please try again.")

                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()

    st.divider()

    # System settings with state sync
    st.markdown("### ⚙️ Settings")

    # LLM selection with proper state management
    use_openai = st.checkbox(
        "Use OpenAI GPT-4",
        value=st.session_state.llm_model == "openai",
        key="use_openai_checkbox"
    )

    # Handle LLM change
    if use_openai and st.session_state.llm_model != "openai":
        if os.getenv("OPENAI_API_KEY"):
            st.session_state.llm_model = "openai"
            rebuild_rag_chain()
        else:
            st.warning("OpenAI API key not found. Please enter it below.")
    elif not use_openai and st.session_state.llm_model != "ollama":
        st.session_state.llm_model = "ollama"
        rebuild_rag_chain()

    if use_openai and not os.getenv("OPENAI_API_KEY"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.session_state.llm_model = "openai"
            rebuild_rag_chain()
            st.rerun()

    enable_hyde = st.checkbox("Enable HyDE", value=True, help="Hypothetical Document Embeddings")
    enable_decomposition = st.checkbox("Enable Query Decomposition", value=True)

    # Message history limit
    message_limit = st.slider(
        "Message History Limit",
        min_value=10,
        max_value=500,
        value=st.session_state.message_limit,
        help="Limit message history to prevent memory issues"
    )
    if message_limit != st.session_state.message_limit:
        st.session_state.message_limit = message_limit
        trim_message_history()

    st.divider()

    # Knowledge base stats
    st.markdown("### 📊 Knowledge Base Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", st.session_state.document_count)
    with col2:
        st.metric("Entities", len(st.session_state.detected_entities))

# Main chat interface - ChatGPT style header
st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><h1 style="margin-bottom: 0.5rem;">🧬 BioRAG</h1><p style="color: #6b7280; font-size: 16px;">Intelligent biomedical document analysis</p></div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Sanitize HTML content before rendering
        sanitized_content = sanitize_html(message["content"])
        st.markdown(sanitized_content, unsafe_allow_html=True)

# Chat input with ChatGPT-style placeholder
if prompt := st.chat_input("Message BioRAG..."):
    # Trim message history if needed
    trim_message_history()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.rag_chain is None:
            st.warning("⚠️ Please upload documents first to build the knowledge base.")
        else:
            # Show thinking animation - ChatGPT style
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(
                '<div class="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></div>',
                unsafe_allow_html=True
            )

            try:
                # Get response with all enhancements
                response_data = st.session_state.rag_chain.query(
                    prompt,
                    enable_hyde=enable_hyde,
                    enable_decomposition=enable_decomposition
                )

                # Clear thinking animation
                thinking_placeholder.empty()

                # Sanitize and display enhanced response
                sanitized_answer = sanitize_html(response_data["enhanced_answer"])
                st.markdown(sanitized_answer, unsafe_allow_html=True)

                # Update detected entities
                st.session_state.detected_entities = response_data["entities"]

                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": sanitized_answer
                })

                # Show source documents
                with st.expander("📚 Sources", expanded=False):
                    for i, doc in enumerate(response_data["source_docs"]):
                        st.markdown(f"**Source {i + 1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(f"*Relevance Score: {doc.metadata.get('score', 'N/A')}*")
                        st.text(doc.page_content[:200] + "...")

            except Exception as e:
                thinking_placeholder.empty()
                st.error("Error generating response. Please try again.")

# Right panel - Detected Entities with pagination
if st.session_state.detected_entities:
    with st.container():
        st.markdown("### 🔬 Detected Entities")

        # Add "Show All" toggle
        show_all = st.checkbox("Show all entities", value=False)

        st.markdown('<div class="entity-panel">', unsafe_allow_html=True)

        # Group entities by type
        entities_by_type = {}
        for entity in st.session_state.detected_entities:
            entity_type = entity.get("type", "unknown")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Display entities
        for entity_type, entities in entities_by_type.items():
            st.markdown(f"#### {entity_type.title()}")

            # Determine how many to show
            display_limit = len(entities) if show_all else min(5, len(entities))

            for entity in entities[:display_limit]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f'<div class="entity-item">'
                        f'<span class="entity-type {entity_type.lower()}">{entity_type}</span>'
                        f'<a href="{entity["url"]}" target="_blank" class="bio-entity">{entity["text"]}</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Show count if not showing all
            if not show_all and len(entities) > 5:
                st.caption(f"...and {len(entities) - 5} more")

        st.markdown('</div>', unsafe_allow_html=True)

# Footer with quick actions
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.detected_entities = []
        st.rerun()

with col2:
    if st.button("💾 Export Chat", use_container_width=True):
        chat_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "Download JSON",
            chat_json,
            f"biorag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col3:
    if st.button("🧪 Run Self-Test", use_container_width=True):
        with st.spinner("Running self-test..."):
            try:
                # Create test document
                test_text = "BRCA1 mutations are associated with breast cancer. Tamoxifen treatment shows efficacy."

                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(test_text)
                    test_file = f.name

                # Test ingestion
                docs = ingester.ingest_file(test_file)

                # Create test DB
                test_db = db_manager.create_db(docs)
                test_chain = RAGChain(test_db, entity_linker, glossary_mgr)

                # Test query
                result = test_chain.query("What genes are mentioned?")

                # Cleanup
                Path(test_file).unlink()

                st.success("✅ Self-test completed successfully!")
                st.info(f"Found {len(result['entities'])} entities in test")

            except Exception as e:
                st.error(f"Self-test failed: Please check your installation")