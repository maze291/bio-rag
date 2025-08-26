"""
BioRAG Ingestion Pipeline
Handles document loading, parsing, and chunking from multiple sources
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib
import re
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import PyPDF2
import pdfplumber
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF for better PDF handling

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    Universal document ingestion pipeline with support for:
    - PDFs (with OCR fallback)
    - HTML/Web pages
    - Plain text
    - RSS feeds
    - Multiple chunking strategies
    """

    def __init__(self,
                 chunk_size: int = 1500,  # Larger chunks for scientific papers
                 chunk_overlap: int = 300,  # More overlap to preserve context
                 separators: Optional[List[str]] = None,
                 enable_ocr: bool = True):
        """
        Initialize ingestion pipeline

        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between chunks for context
            separators: Custom separators for splitting
            enable_ocr: Whether to enable OCR for scanned PDFs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr

        # Default separators optimized for scientific text
        self.separators = separators or [
            "\n[SECTION:",  # Section headers  
            "\n[TABLE]",    # Table markers
            "\n[FIGURE CAPTION]",  # Figure captions
            "\n\n\n",  # Multiple newlines (section breaks)
            "\n\n",  # Paragraph breaks
            "\nResults",  # Results section
            "\nDiscussion",  # Discussion section
            "\nMethods",  # Methods section
            "\nConclusion",  # Conclusion section
            "\n",  # Line breaks
            ". ",  # Sentence ends
            "! ",
            "? ",
            "; ",  # Semicolons (common in scientific text)
            ", ",  # Commas
            " ",  # Spaces
            ""  # Characters
        ]

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )

        # Cache for processed documents
        self.doc_cache = {}

        # Check OCR availability
        if self.enable_ocr:
            try:
                pytesseract.get_tesseract_version()
                logger.info("OCR enabled with Tesseract")
            except Exception:
                logger.warning("Tesseract not found. OCR disabled.")
                self.enable_ocr = False

    def ingest_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Ingest a single file and return chunked documents

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects with chunks
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate cache key including chunk parameters
        cache_key = self._generate_cache_key(file_path)
        if cache_key in self.doc_cache:
            logger.info(f"Using cached version of {file_path.name}")
            return self.doc_cache[cache_key]

        # Determine file type and process accordingly
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                documents = self._ingest_pdf(file_path)
            elif suffix in ['.html', '.htm']:
                documents = self._ingest_html(file_path)
            elif suffix in ['.txt', '.md']:
                documents = self._ingest_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            # Cache the results
            self.doc_cache[cache_key] = documents

            return documents

        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {str(e)}")
            raise

    def ingest_url(self, url: str) -> List[Document]:
        """
        Ingest content from a web URL

        Args:
            url: Web page URL

        Returns:
            List of Document objects
        """
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; BioRAG/1.0)'
            })
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text content while preserving structure
            # Get text with newlines between different elements
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
                text = element.get_text(strip=True)
                if text:
                    text_parts.append(text)

            text = '\n\n'.join(text_parts)

            # Clean up text without losing structure
            text = self._clean_text(text, preserve_paragraphs=True)

            # Create metadata
            metadata = {
                'source': url,
                'title': soup.title.string if soup.title else urlparse(url).netloc,
                'ingested_at': datetime.now().isoformat(),
                'type': 'web_page'
            }

            # Split into chunks
            documents = self._create_chunks(text, metadata)

            return documents

        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {str(e)}")
            raise

    def ingest_rss(self, rss_url: str, max_items: int = 20) -> List[Document]:
        """
        Ingest articles from an RSS feed

        Args:
            rss_url: RSS feed URL
            max_items: Maximum number of items to fetch

        Returns:
            List of Document objects
        """
        try:
            # Parse RSS feed
            feed = feedparser.parse(rss_url)

            if feed.bozo:
                raise ValueError(f"Error parsing RSS feed: {feed.bozo_exception}")

            all_documents = []

            # Process each entry
            for i, entry in enumerate(feed.entries[:max_items]):
                # Extract content
                content = ""

                # Try different content fields
                if hasattr(entry, 'content'):
                    content = entry.content[0].value
                elif hasattr(entry, 'summary'):
                    content = entry.summary
                elif hasattr(entry, 'description'):
                    content = entry.description

                if not content:
                    continue

                # Clean HTML if present
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                text = self._clean_text(text, preserve_paragraphs=True)

                # Add title
                if hasattr(entry, 'title'):
                    text = f"Title: {entry.title}\n\n{text}"

                # Create metadata safely
                metadata = {
                    'source': entry.link if hasattr(entry, 'link') else rss_url,
                    'title': entry.title if hasattr(entry, 'title') else f"Article {i + 1}",
                    'published': entry.published if hasattr(entry, 'published') else None,
                    'ingested_at': datetime.now().isoformat(),
                    'type': 'rss_article',
                    'feed_title': feed.feed.title if hasattr(feed.feed, 'title') else None
                }

                # Handle authors safely
                if hasattr(entry, 'authors') and entry.authors:
                    try:
                        author_names = []
                        for author in entry.authors:
                            if hasattr(author, 'name'):
                                author_names.append(author.name)
                            elif isinstance(author, str):
                                author_names.append(author)
                        if author_names:
                            metadata['authors'] = ', '.join(author_names)
                    except:
                        pass

                # Remove None values from metadata
                metadata = {k: v for k, v in metadata.items() if v is not None}

                # Create chunks
                documents = self._create_chunks(text, metadata)
                all_documents.extend(documents)

            logger.info(f"Ingested {len(all_documents)} chunks from {len(feed.entries[:max_items])} RSS articles")
            return all_documents

        except Exception as e:
            logger.error(f"Error ingesting RSS feed {rss_url}: {str(e)}")
            raise

    def _ingest_pdf(self, file_path: Path) -> List[Document]:
        """Ingest PDF file with multiple fallback strategies"""
        text = ""
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'ingested_at': datetime.now().isoformat(),
            'type': 'pdf'
        }

        # Try method 1: Unstructured (best for complex PDFs)
        try:
            elements = partition_pdf(
                filename=str(file_path),
                strategy="hi_res",  # High resolution processing
                infer_table_structure=True,
                include_page_breaks=True,
                extract_images_in_pdf=True,  # Extract images for figures
                extract_image_block_types=["Image", "Table"],  # Focus on tables and figures
                chunking_strategy="by_title",  # Keep related content together
                languages=['eng'],  # Specify language for better OCR
                ocr_languages='eng'  # Enable OCR with English language
            )

            # Extract text and metadata with better structure preservation
            for element in elements:
                if hasattr(element, 'text') and element.text.strip():
                    # Add element category for better context
                    if hasattr(element, 'category'):
                        if element.category == "Table":
                            text += f"\n[TABLE]\n{element.text}\n"
                        elif element.category == "FigureCaption":
                            text += f"\n[FIGURE CAPTION]\n{element.text}\n"
                        elif element.category == "Header":
                            text += f"\n[SECTION: {element.text}]\n"
                        elif element.category == "Title":
                            text += f"\n[TITLE: {element.text}]\n"
                            if 'title' not in metadata:
                                metadata['title'] = element.text
                        else:
                            text += element.text + "\n"
                    else:
                        text += element.text + "\n"

            if text.strip():
                logger.info(f"Successfully parsed PDF with Unstructured: {file_path.name}")
                text = self._clean_text(text)
                return self._create_chunks(text, metadata)

        except Exception as e:
            logger.warning(f"Unstructured failed, trying pdfplumber: {str(e)}")

        # Try method 2: pdfplumber (good for tables)
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                if pdf.metadata:
                    metadata.update({
                        'title': pdf.metadata.get('Title', ''),
                        'author': pdf.metadata.get('Author', ''),
                        'subject': pdf.metadata.get('Subject', ''),
                        'creator': pdf.metadata.get('Creator', '')
                    })

                # Extract text from each page
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}\n"

                    # Extract tables safely
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table:  # Check if table is not None
                                text += "\n[TABLE]\n"
                                for row in table:
                                    if row:  # Check if row is not None
                                        text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                                text += "\n"

            if text.strip():
                logger.info(f"Successfully parsed PDF with pdfplumber: {file_path.name}")
                text = self._clean_text(text)
                return self._create_chunks(text, metadata)

        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {str(e)}")

        # Try method 3: PyPDF2 (basic fallback)
        try:
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                # Extract metadata
                if reader.metadata:
                    metadata.update({
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', '')
                    })

                # Extract text
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}\n"

            if text.strip():
                logger.info(f"Successfully parsed PDF with PyPDF2: {file_path.name}")
                text = self._clean_text(text)
                return self._create_chunks(text, metadata)

        except Exception as e:
            logger.warning(f"PyPDF2 failed: {str(e)}")

        # Try method 4: PyMuPDF with OCR fallback
        if self.enable_ocr:
            try:
                text = ""
                doc = fitz.open(file_path)

                for page_num, page in enumerate(doc):
                    # Try text extraction first
                    page_text = page.get_text()

                    # If no text, try OCR
                    if not page_text.strip() and self.enable_ocr:
                        logger.info(f"Page {page_num + 1} appears to be scanned, attempting OCR...")

                        # Convert page to image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))

                        # Run OCR
                        try:
                            page_text = pytesseract.image_to_string(img)
                        except Exception as ocr_error:
                            logger.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
                            continue

                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}\n"

                doc.close()

                if text.strip():
                    logger.info(f"Successfully parsed PDF with PyMuPDF/OCR: {file_path.name}")
                    text = self._clean_text(text)
                    return self._create_chunks(text, metadata)

            except Exception as e:
                logger.error(f"PyMuPDF/OCR failed: {str(e)}")

        # If all methods failed
        raise ValueError(f"Could not extract text from PDF: {file_path.name}")

    def _ingest_html(self, file_path: Path) -> List[Document]:
        """Ingest HTML file"""
        try:
            # Use Unstructured for HTML
            elements = partition_html(filename=str(file_path))

            text = ""
            metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'ingested_at': datetime.now().isoformat(),
                'type': 'html'
            }

            for element in elements:
                if hasattr(element, 'text'):
                    text += element.text + "\n"

                if hasattr(element, 'category') and element.category == "Title" and 'title' not in metadata:
                    metadata['title'] = element.text

            # Clean text
            text = self._clean_text(text, preserve_paragraphs=True)

            return self._create_chunks(text, metadata)

        except Exception as e:
            logger.error(f"Error parsing HTML file {file_path}: {str(e)}")
            raise

    def _ingest_text(self, file_path: Path) -> List[Document]:
        """Ingest plain text file"""
        try:
            # Read file with encoding detection
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            text = None

            for encoding in encodings:
                try:
                    text = file_path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                raise ValueError(f"Could not decode file {file_path}")

            # Clean text
            text = self._clean_text(text, preserve_paragraphs=True)

            metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'ingested_at': datetime.now().isoformat(),
                'type': 'text'
            }

            return self._create_chunks(text, metadata)

        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            raise

    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Create document chunks with metadata

        Args:
            text: Full text content
            metadata: Document metadata

        Returns:
            List of Document chunks
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Create Document objects
        documents = []
        
        # Generate document ID if not present
        doc_id = metadata.get('doc_id') or self._generate_doc_id(metadata)
        
        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue

            # Determine section based on content
            section = self._extract_section(chunk)

            # Create comprehensive chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'doc_id': doc_id,           # Required for neighbor expansion
                'chunk_idx': i,             # Required for neighbor expansion  
                'section': section,         # Required for section-aware retrieval
                'chunk_id': i,              # Legacy compatibility
                'chunk_total': len(chunks),
                'chunk_size': len(chunk)
            })

            # Create document
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)

        return documents

    def _clean_text(self, text: str, preserve_paragraphs: bool = False) -> str:
        """
        Clean and normalize text with special handling for scientific notation

        Args:
            text: Raw text
            preserve_paragraphs: Whether to preserve paragraph structure

        Returns:
            Cleaned text
        """
        # Fix common encoding issues
        text = text.replace('\u2019', "'")
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '--')
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        
        # Fix scientific notation and chemical formulas
        scientific_replacements = {
            # Subscripts
            '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
            '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
            # Superscripts  
            '\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3', '\u2074': '4',
            '\u2075': '5', '\u2076': '6', '\u2077': '7', '\u2078': '8', '\u2079': '9',
            # Common scientific symbols - keep original symbols for search
            '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b3': 'gamma', '\u03b4': 'delta',
            '\u03bc': 'μ', '\u00b0': '°', '\u2103': '°C',  # Keep symbols, don't convert to words
            # Fix common OCR errors in chemical formulas
            'CHa': 'CH4', 'CHsz': 'CH4', 'C2Hs': 'C2H6', 'C2Ho': 'C2H6',
            'Fe**': 'Fe2+', 'Fe***': 'Fe3+', 'SO4**': 'SO4 2-',
            # Fix corrupted micro symbol variations 
            'microM': 'μM', 'micro M': 'μM', 'uM': 'μM',
            # Fix degree symbol variations
            'degreesC': '°C', 'degrees C': '°C', 'degC': '°C',
            # Fix special Unicode dashes and arrows
            '\u2212': '-', '\u2192': '→', '\u2190': '←',
        }
        
        for old, new in scientific_replacements.items():
            text = text.replace(old, new)
        
        # Use regex to fix more complex scientific notation patterns
        import re
        
        # Fix patterns like "CH4" when they appear as "CHx" where x is corrupted
        text = re.sub(r'\bCH[a-z]\b', 'CH4', text)  # CH + any lowercase letter -> CH4
        text = re.sub(r'\bC2H[a-z]\b', 'C2H6', text)  # C2H + any lowercase letter -> C2H6
        
        # Fix ionic charges
        text = re.sub(r'Fe\*{2,}', 'Fe2+', text)  # Fe** -> Fe2+
        text = re.sub(r'Fe\*{3,}', 'Fe3+', text)  # Fe*** -> Fe3+
        
        # Fix degrees and chemical symbols
        text = re.sub(r'(\d+)\s*[°o]\s*C', r'\1°C', text)  # Fix temperature notation
        
        # Fix unit patterns that got corrupted
        text = re.sub(r'(\d+\.?\d*)\s*microM\s*h[\-−]1', r'\1 μM·h⁻¹', text)
        text = re.sub(r'(\d+\.?\d*)\s*μM\s*h[\-−]1', r'\1 μM·h⁻¹', text)
        text = re.sub(r'(\d+\.?\d*)\s*uM\s*h[\-−]1', r'\1 μM·h⁻¹', text)
        
        # Fix fold increase patterns
        text = re.sub(r'(\d+)[\-−]\s*fold', r'\1-fold', text)
        text = re.sub(r'(\d+)\s*×\s*fold', r'\1× fold', text)
        
        # Fix arrow notation for increases
        text = re.sub(r'(\d+\.?\d*)\s*→\s*(\d+\.?\d*)', r'\1 → \2', text)

        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        if preserve_paragraphs:
            # Preserve paragraph structure
            # Replace multiple spaces with single space
            text = re.sub(r'[ \t]+', ' ', text)
            # Replace 3+ newlines with 2
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Remove trailing/leading spaces on each line
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)
        else:
            # Aggressive cleaning
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)

        # Final trim
        text = text.strip()

        return text

    def _generate_cache_key(self, file_path: Path) -> str:
        """Generate cache key including file hash and chunking params"""
        # Get file hash
        file_hash = self._hash_file(file_path)

        # Include chunking parameters in cache key
        param_str = f"{self.chunk_size}_{self.chunk_overlap}_{'_'.join(self.separators[:3])}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"{file_hash}_{param_hash}"

    def _hash_file(self, file_path: Path) -> str:
        """Generate hash for file content"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def set_chunk_params(self, chunk_size: int, chunk_overlap: int):
        """Update chunking parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Recreate text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )

        # Clear cache since parameters changed
        self.doc_cache.clear()
        logger.info(f"Updated chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
    
    def _generate_doc_id(self, metadata: Dict[str, Any]) -> str:
        """Generate a unique document ID from metadata"""
        # Use filename or source if available
        if 'filename' in metadata:
            base = Path(metadata['filename']).stem
        elif 'source' in metadata:
            base = Path(metadata['source']).stem
        elif 'title' in metadata:
            base = re.sub(r'[^\w\-_]', '_', metadata['title'][:50])
        else:
            base = "doc"
        
        # Add hash for uniqueness
        content_hash = hashlib.md5(str(metadata).encode()).hexdigest()[:8]
        return f"{base}_{content_hash}"
    
    def _extract_section(self, chunk: str) -> str:
        """Extract section type from chunk content"""
        chunk_lower = chunk.lower().strip()
        
        # Check for section markers added during PDF parsing
        if chunk.startswith('[FIGURE CAPTION]'):
            return "figure_caption"
        elif chunk.startswith('[TABLE]'):
            return "table"
        elif chunk.startswith('[SECTION:'):
            return "header"
        elif chunk.startswith('[TITLE:'):
            return "title"
        
        # Check for common section headers
        first_lines = '\n'.join(chunk_lower.split('\n')[:3])
        
        if any(keyword in first_lines for keyword in ['abstract', 'summary']):
            return "abstract"
        elif any(keyword in first_lines for keyword in ['introduction', 'background']):
            return "introduction"
        elif any(keyword in first_lines for keyword in ['method', 'experimental', 'procedure']):
            return "methods"
        elif any(keyword in first_lines for keyword in ['result', 'findings', 'observation']):
            return "results"
        elif any(keyword in first_lines for keyword in ['discussion', 'interpretation']):
            return "discussion"
        elif any(keyword in first_lines for keyword in ['conclusion', 'summary', 'implications']):
            return "conclusion"
        elif any(keyword in first_lines for keyword in ['reference', 'bibliography', 'citation']):
            return "references"
        elif any(keyword in first_lines for keyword in ['figure', 'fig.', 'caption']):
            return "figure_caption"
        elif any(keyword in first_lines for keyword in ['table', 'tab.']):
            return "table"
        else:
            return "content"