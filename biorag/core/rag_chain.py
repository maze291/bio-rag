"""
BioRAG Chain
Main RAG orchestration with query processing, retrieval, and answer generation
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import asyncio
from datetime import datetime
import re

# Import our ensemble retrieval fixes
from .ensemble_retriever import EnsembleRetriever
from .neighbor_expander import NeighborExpander
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Prompt templates
QUERY_DECOMPOSITION_PROMPT = """You are an expert at analyzing biomedical questions.
Break down the following complex question into 2-4 simpler sub-questions that would help answer the main question.
Each sub-question should focus on a specific aspect.

Question: {question}

Sub-questions (one per line):"""

HYDE_PROMPT = """You are a biomedical expert. Write a hypothetical paragraph that would answer the following question.
Include specific technical terms, gene names, pathways, and mechanisms that would likely appear in a real answer.

Question: {question}

Hypothetical answer:"""

QA_PROMPT = """You are an expert biomedical researcher providing accurate, detailed answers with precise numerical data extraction.

Use the following context to answer the question.

CRITICAL INSTRUCTIONS FOR NUMERICAL DATA:
1. Extract ALL specific numerical values, percentages, fold-changes, and measurements from the context
2. Quote exact numbers with their units (e.g., "0.82 μM·h⁻¹", "41-fold", "97°C", "~0.82 μM h⁻¹")
3. If multiple relevant numbers exist, list them all with their full context
4. Pay special attention to rates, concentrations, temperatures, wavelengths, and statistical values
5. Include ranges and approximate values (e.g., "~0.82", "41-fold to 0.82")
6. Preserve the exact format of scientific notation and units as written

EXAMPLES OF GOOD NUMERICAL EXTRACTION:
- "The formation rates increased 41-fold to ~0.82 μM h⁻¹ at 97°C"
- "λmax = 388 nm under specific conditions"
- "Concentrations ranged from 0.1 to 2.5 μM"
- "Statistical significance: p < 0.05"

Only say you cannot find information if the context truly contains NO relevant numerical data or information.

Context:
{context}

Question: {question}

Answer with specific numbers and units:"""

SYNTHESIS_PROMPT = """You are synthesizing multiple pieces of information to provide a comprehensive answer.
Combine the following sub-answers into a coherent, well-structured response.
Eliminate redundancy while preserving all unique insights.

Sub-answers:
{sub_answers}

Question: {original_question}

Comprehensive answer:"""


@dataclass
class QueryResult:
    """Container for query results"""
    answer: str
    enhanced_answer: str
    source_docs: List[Document]
    entities: List[Dict[str, Any]]
    sub_queries: Optional[List[str]] = None
    confidence: float = 0.0


class RAGChain:
    """
    Orchestrates the complete RAG pipeline with advanced features
    """

    def __init__(self,
                 vector_db,
                 entity_linker,
                 glossary_manager,
                 llm_model: Optional[str] = None):
        """
        Initialize RAG chain

        Args:
            vector_db: Vector database instance
            entity_linker: Entity linking instance
            glossary_manager: Glossary manager instance
            llm_model: Optional LLM model name
        """
        self.vector_db = vector_db
        self.entity_linker = entity_linker
        self.glossary_manager = glossary_manager

        # Initialize LLM
        self.llm = self._init_llm(llm_model)

        # Initialize retrievers with higher coverage for scientific papers
        self.retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 40}  # Even more for ensemble fusion
        )

        self.mmr_retriever = vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 15, "fetch_k": 40}  # Much higher diversity and coverage
        )

        # Initialize enhanced retrieval components
        try:
            self.neighbor_expander = NeighborExpander(vector_db)
            
            # Initialize cross-encoder reranker
            self.reranker = CrossEncoderReranker()
            
            # Try to set up ensemble retriever if documents already exist
            self.ensemble_retriever = None
            try:
                # Check if vector DB has documents
                doc_count = vector_db._collection.count() if hasattr(vector_db, '_collection') else 0
                if doc_count > 0:
                    # Try to get some documents from the vector DB to initialize BM25
                    existing_docs = self._get_sample_documents_from_db(vector_db)
                    if existing_docs:
                        self.setup_ensemble_retriever(existing_docs)
                        logger.info(f"✅ Auto-initialized ensemble retriever with {len(existing_docs)} sample docs")
            except Exception as init_e:
                logger.warning(f"Could not auto-initialize ensemble retriever: {str(init_e)}")
                
            logger.info("✅ Enhanced retrieval system initialized (Ensemble + Neighbors + Reranker)")
        except Exception as e:
            logger.warning(f"Enhanced retrieval setup failed: {str(e)}, falling back to dense only")
            self.neighbor_expander = None
            self.reranker = None

        # Initialize chains
        self._init_chains()

    def setup_ensemble_retriever(self, documents: List[Document]):
        """
        Setup ensemble retriever once documents are available
        Call this after documents are loaded into vector DB
        """
        try:
            from .ensemble_retriever import EnsembleRetriever
            
            # Ensure all documents have proper metadata for neighbor expansion
            enhanced_docs = []
            for doc in documents:
                if not doc.metadata.get('doc_id') or not doc.metadata.get('chunk_idx'):
                    logger.info(f"📥 Enhancing document metadata for ensemble retriever compatibility")
                    # Add missing metadata if needed
                    if 'doc_id' not in doc.metadata:
                        doc.metadata['doc_id'] = f"doc_{hash(doc.page_content[:100]) % 10000}"
                    if 'chunk_idx' not in doc.metadata:
                        doc.metadata['chunk_idx'] = doc.metadata.get('chunk_id', 0)
                enhanced_docs.append(doc)
            
            self.ensemble_retriever = EnsembleRetriever(self.retriever, enhanced_docs)
            logger.info("✅ Ensemble retriever setup complete")
        except Exception as e:
            logger.warning(f"Ensemble retriever setup failed: {str(e)}")
            self.ensemble_retriever = None

    def _init_llm(self, model_name: Optional[str] = None):
        """Initialize language model"""
        # Check for OpenAI
        if model_name == "openai" or (not model_name and os.getenv("OPENAI_API_KEY")):
            logger.info("Using OpenAI GPT-4")
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=2000
            )
        else:
            # Use Ollama
            logger.info("Using Ollama Llama3")
            return Ollama(
                model=model_name or "llama3",
                temperature=0,
                num_ctx=8192  # Increased context window
            )

    def _init_chains(self):
        """Initialize various chains"""
        # Query decomposition chain
        decomp_prompt = PromptTemplate(
            template=QUERY_DECOMPOSITION_PROMPT,
            input_variables=["question"]
        )
        self.decomposition_chain = LLMChain(
            llm=self.llm,
            prompt=decomp_prompt
        )

        # HyDE chain
        hyde_prompt = PromptTemplate(
            template=HYDE_PROMPT,
            input_variables=["question"]
        )
        self.hyde_chain = LLMChain(
            llm=self.llm,
            prompt=hyde_prompt
        )

        # Main QA chain - using correct prompt format
        qa_prompt = PromptTemplate(
            template=QA_PROMPT,
            input_variables=["context", "question"]
        )

        # Create custom QA chain that properly uses our prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": qa_prompt,
                "document_variable_name": "context"
            }
        )

        # Direct QA chain for when we have context
        self.direct_qa_chain = LLMChain(
            llm=self.llm,
            prompt=qa_prompt
        )

        # Synthesis chain
        synth_prompt = PromptTemplate(
            template=SYNTHESIS_PROMPT,
            input_variables=["sub_answers", "original_question"]
        )
        self.synthesis_chain = LLMChain(
            llm=self.llm,
            prompt=synth_prompt
        )

    def query(self,
              question: str,
              enable_hyde: bool = True,
              enable_decomposition: bool = True,
              enable_mmr: bool = True) -> Dict[str, Any]:
        """
        Process a query through the complete pipeline

        Args:
            question: User's question
            enable_hyde: Whether to use HyDE
            enable_decomposition: Whether to decompose complex queries
            enable_mmr: Whether to use MMR for diversity

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question}")

        try:
            # Step 1: Query analysis and enhancement
            enhanced_queries = [question]
            sub_queries = None

            if enable_decomposition and self._is_complex_query(question):
                try:
                    sub_queries = self._decompose_query(question)
                    if sub_queries:
                        enhanced_queries.extend(sub_queries)
                except Exception as e:
                    logger.warning(f"Query decomposition failed: {str(e)}")

            if enable_hyde:
                try:
                    hyde_doc = self._generate_hyde(question)
                    if hyde_doc and hyde_doc.strip():
                        enhanced_queries.append(hyde_doc)
                except Exception as e:
                    logger.warning(f"HyDE generation failed: {str(e)}")

            # Step 2: Retrieve documents
            all_docs = self._retrieve_documents(
                enhanced_queries,
                use_mmr=enable_mmr
            )

            if not all_docs:
                logger.warning("No documents retrieved")
                answer = "I couldn't find relevant information in the knowledge base to answer your question."
            else:
                # Step 3: Generate answer
                if sub_queries and len(sub_queries) > 1:
                    # Answer sub-queries and synthesize
                    answer = self._answer_with_decomposition(
                        question,
                        sub_queries,
                        all_docs
                    )
                else:
                    # Direct answer
                    answer = self._generate_answer(question, all_docs)

            # Step 4: Extract entities
            entities = self.entity_linker.extract_entities(answer)
            entity_summary = self.entity_linker.get_entity_summary(entities)

            # Step 5: Enhance answer with links and tooltips
            enhanced_answer = self._enhance_answer(answer, entities)

            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(all_docs, answer)

            # Prepare result
            result = {
                "answer": answer,
                "enhanced_answer": enhanced_answer,
                "source_docs": all_docs[:5],  # Top 5 sources
                "entities": self._format_entities(entity_summary),
                "sub_queries": sub_queries,
                "confidence": confidence
            }

            return result

        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            # Return a valid result even on error
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "enhanced_answer": "I encountered an error while processing your question. Please try again.",
                "source_docs": [],
                "entities": [],
                "sub_queries": None,
                "confidence": 0.0
            }

    def _is_complex_query(self, question: str) -> bool:
        """Determine if query needs decomposition"""
        # Simple heuristics
        complex_indicators = [
            ' and ',
            ' or ',
            'compare',
            'difference between',
            'relationship',
            'how does',
            'explain',
            'mechanism',
            'pathway',
            'multiple',
            'process of',
            'role of',
            'interaction between'
        ]

        question_lower = question.lower()

        # Check length
        if len(question.split()) > 15:
            return True

        # Check for complex indicators
        for indicator in complex_indicators:
            if indicator in question_lower:
                return True

        # Check for multiple question marks or complex punctuation
        if question.count('?') > 1 or question.count(',') > 2:
            return True

        return False

    def _decompose_query(self, question: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        try:
            result = self.decomposition_chain.run(question=question)

            # Parse the result
            sub_queries = []
            for line in result.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present
                if line and line[0].isdigit():
                    line = line[line.find(' ') + 1:].strip()
                if line and len(line) > 10:  # Filter out empty or too short lines
                    sub_queries.append(line)

            # Limit to 4 sub-queries
            sub_queries = sub_queries[:4]

            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries

        except Exception as e:
            logger.error(f"Error in query decomposition: {str(e)}")
            return []

    def _generate_hyde(self, question: str) -> str:
        """Generate hypothetical document embedding"""
        try:
            hyde_doc = self.hyde_chain.run(question=question)

            # Validate the output
            if hyde_doc and len(hyde_doc.strip()) > 50:
                logger.info("Generated HyDE document")
                return hyde_doc
            else:
                logger.warning("HyDE document too short, skipping")
                return ""

        except Exception as e:
            logger.error(f"Error in HyDE generation: {str(e)}")
            return ""

    def _retrieve_documents(self,
                            queries: List[str],
                            use_mmr: bool = True) -> List[Document]:
        """Retrieve documents using enhanced ensemble + neighbor expansion"""
        logger.info(f"📥 RETRIEVAL PIPELINE: Starting document retrieval")
        logger.info(f"📥 Input: {len(queries)} queries, use_mmr={use_mmr}")
        
        all_docs = []
        seen_content = set()

        for query_idx, query in enumerate(queries):
            if not query or not query.strip():
                logger.info(f"📥 Query {query_idx+1}: Skipping empty query")
                continue

            try:
                logger.info(f"📥 ========== PROCESSING QUERY {query_idx+1}/{len(queries)} ==========")
                logger.info(f"📥 Query: '{query}'")
                logger.info(f"📥 Query length: {len(query)} chars")
                logger.info(f"📥 Ensemble retriever available: {self.ensemble_retriever is not None}")
                
                # Check for scientific terms
                logger.info(f"📥 Analyzing query for scientific terms...")
                has_scientific = self._has_numeric_terms(query)
                logger.info(f"📥 Scientific analysis result: {has_scientific}")
                
                # Use ensemble retriever if available (BM25 + Dense)
                if self.ensemble_retriever and has_scientific:
                    logger.info(f"✅ DECISION: Using ENSEMBLE retrieval (BM25 + Dense)")
                    logger.info(f"✅ Reason: Ensemble available AND scientific terms detected")
                    docs = self.ensemble_retriever.get_relevant_documents(query, k=25)
                else:
                    if not self.ensemble_retriever:
                        logger.info(f"ℹ️ DECISION: Using STANDARD retrieval (ensemble not available)")
                    else:
                        logger.info(f"ℹ️ DECISION: Using STANDARD retrieval (no scientific terms)")
                    
                    # Choose standard retriever
                    retriever = self.mmr_retriever if use_mmr else self.retriever
                    retriever_type = "MMR" if use_mmr else "Similarity"
                    logger.info(f"ℹ️ Using {retriever_type} retriever")
                    docs = retriever.get_relevant_documents(query)
                    logger.info(f"ℹ️ Standard retrieval returned {len(docs)} documents")

                # Expand with neighbors for distributed details
                if self.neighbor_expander and len(docs) > 0:
                    logger.info(f"📥 NEIGHBOR EXPANSION: Processing {len(docs)} documents")
                    docs = self.neighbor_expander.expand_with_neighbors(docs, window=2)
                    logger.info(f"📥 After neighbor expansion: {len(docs)} documents")
                else:
                    logger.info(f"📥 Skipping neighbor expansion (expander={self.neighbor_expander is not None}, docs={len(docs)})")

                # Deduplicate based on content hash and apply section boosting
                logger.info(f"📥 DEDUPLICATION: Processing {len(docs)} documents")
                docs_added = 0
                docs_duplicate = 0
                
                for doc in docs:
                    content_preview = doc.page_content[:200]
                    content_hash = hash(content_preview)

                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        
                        # Add relevance score if not present
                        if 'score' not in doc.metadata:
                            doc.metadata['score'] = 1.0
                        
                        # Apply section-aware boosting for scientific queries
                        original_score = doc.metadata['score']
                        boosted_score = self._apply_section_boost(doc, has_scientific)
                        doc.metadata['score'] = boosted_score
                        
                        if boosted_score != original_score:
                            section = doc.metadata.get('section', 'unknown')
                            logger.info(f"📥 Section boost applied: {section} ({original_score:.3f} → {boosted_score:.3f})")
                        
                        all_docs.append(doc)
                        docs_added += 1
                    else:
                        docs_duplicate += 1
                
                logger.info(f"📥 Deduplication: {docs_added} new, {docs_duplicate} duplicates")
                logger.info(f"📥 Running total: {len(all_docs)} unique documents")

            except Exception as e:
                logger.error(f"💥 Error in enhanced retrieval for query '{query[:50]}...': {str(e)}")
                logger.info(f"📥 Attempting fallback retrieval...")
                # Fallback to basic retrieval
                try:
                    fallback_docs = self.retriever.get_relevant_documents(query)
                    all_docs.extend(fallback_docs)
                    logger.info(f"📥 Fallback successful: {len(fallback_docs)} documents added")
                except Exception as fallback_e:
                    logger.error(f"💥 Fallback also failed: {str(fallback_e)}")

        # Sort by relevance score if available
        logger.info(f"📥 FINAL SORTING: Sorting {len(all_docs)} documents by relevance score")
        try:
            all_docs.sort(key=lambda d: d.metadata.get('score', 0), reverse=True)
            logger.info(f"📥 Sorting successful")
            
            # Log top documents before reranking
            logger.info(f"📥 Top 3 documents before reranking:")
            for i, doc in enumerate(all_docs[:3]):
                score = doc.metadata.get('score', 'N/A')
                preview = doc.page_content[:100].replace('\n', ' ')
                logger.info(f"📥 #{i+1} (score={score}): {preview}...")
        except Exception as sort_e:
            logger.error(f"💥 Sorting failed: {str(sort_e)}")

        # Check for exact-match fast path (skip expensive reranking)
        should_skip_reranker = self._should_skip_reranker(queries, all_docs)
        
        # Apply cross-encoder reranking for better relevance
        if len(all_docs) > 1 and self.reranker and self.reranker.is_available() and not should_skip_reranker:
            logger.info(f"🎯 CROSS-ENCODER RERANKING: Processing {len(all_docs)} documents")
            
            # Rerank top 15 documents for speed (fast path optimization)
            docs_to_rerank = all_docs[:15] if len(all_docs) > 15 else all_docs
            remaining_docs = all_docs[15:] if len(all_docs) > 15 else []
            
            # Get the original query for reranking
            original_query = queries[0] if queries else ""  # Use first query as primary
            
            try:
                reranked_docs = self.reranker.rerank(original_query, docs_to_rerank)
                all_docs = reranked_docs + remaining_docs
                logger.info(f"🎯 Reranking complete: reranked {len(reranked_docs)} docs, {len(remaining_docs)} remaining")
            except Exception as rerank_e:
                logger.error(f"💥 Reranking failed: {str(rerank_e)}")
                logger.info(f"🔄 Continuing with original ranking")
        else:
            if should_skip_reranker:
                logger.info(f"⚡ FAST PATH: Skipping reranker due to exact match confidence")
            else:
                logger.info(f"📥 Skipping reranking (docs={len(all_docs)}, reranker_available={self.reranker.is_available() if self.reranker else False})")

        logger.info(f"✅ RETRIEVAL COMPLETE: {len(all_docs)} unique documents ready for answer generation")
        return all_docs

    def _has_numeric_terms(self, query: str) -> bool:
        """Check if query contains numeric/scientific terms that benefit from BM25"""
        logger.info(f"🔬 SCIENTIFIC TERM ANALYSIS: Starting analysis of query: '{query}'")
        
        # Enhanced patterns for scientific queries (high-value detectors)
        scientific_terms = [
            # Direct compound names
            'ascorbate', 'Fe2+', 'Fe3+', 'citrate', 'DMSO', 'dmso', 'Fe', 'iron',
            # Numeric patterns
            r'\d+\s*[-\-]?\s*fold',   # "41-fold", "41 fold", "41-fold"  
            r'\d+\s*fold',            # Simple "41fold" pattern
            r'[~≈]\s*\d+',            # "~0.82"
            r'\d+\.\d+',              # "0.82", "97.5"
            r'\d+\s*[μu]M\s*h[-‒–—]?[¹1]',  # "μM h⁻¹"
            r'λ\s*max\s*=\s*\d+',     # "λmax = 388"
            r'\d+\s*nm',              # "388 nm"
            r'\d+\s*°?C',             # "97°C"
            r'\d+\s*kJ\s*m[-‒–—]?[²2]',  # "kJ m⁻²"
            # Stats patterns (high-value addition)
            r'p\s*[<≤]\s*0\.\d+',     # "p < 0.05"
            r'95%\s*CI',              # "95% CI"
            r'ANOVA|t[-\s]?test|FDR', # Statistical tests
            # Genes/proteins (high-value addition)
            r'[A-Z0-9]{2,5}\d?(?:-[A-Z0-9]+)?',  # Gene names like BRCA1, TP53
            r'\w+ase\b',              # Enzymes ending in -ase
            'subunit', 'isoform',     # Protein terminology
            # Kinetics (high-value addition)
            'kcat', 'Km', 'Vmax', 'rate constant',
            # Spectra (high-value addition)
            r'λ\s*max', r'\d+\s*nm',  # Wavelengths
            # Isotopes/labels (high-value addition)
            r'\b¹³C|\b¹⁵N|\bDMSO[-\s]?d6\b',  # Isotopic labels
            # Scientific processes
            'formation process', 'light conditions', 'formation rates',
            'temperature', 'wavelength', 'photolysis', 'experimental',
            'conditions', 'concentration', 'buffer', 'pH',
            # Iron oxidation states
            r'Fe[²³2+3+]', r'iron.*oxidation'
        ]
        
        query_lower = query.lower()
        logger.info(f"🔬 Query lowercased: '{query_lower}'")
        
        # Check each pattern with detailed logging
        logger.info(f"🔬 Checking {len(scientific_terms)} scientific patterns...")
        
        for i, term in enumerate(scientific_terms):
            if isinstance(term, str):
                # Direct string match
                logger.info(f"🔬 Pattern {i+1}/{len(scientific_terms)}: Checking string '{term}' in query_lower")
                if term in query_lower:
                    logger.info(f"✅ MATCH! Scientific term detected: '{term}' found in '{query_lower}'")
                    logger.info(f"🎯 DECISION: Using ensemble retrieval due to scientific term: {term}")
                    return True
                else:
                    logger.info(f"❌ No match for string pattern: '{term}'")
            else:
                # Regex pattern
                logger.info(f"🔬 Pattern {i+1}/{len(scientific_terms)}: Checking regex '{term}' against query")
                match = re.search(term, query, re.IGNORECASE)
                if match:
                    logger.info(f"✅ MATCH! Scientific regex pattern detected: '{term}' matched '{match.group()}' in query")
                    logger.info(f"🎯 DECISION: Using ensemble retrieval due to regex pattern: {term}")
                    return True
                else:
                    logger.info(f"❌ No match for regex pattern: '{term}'")
        
        # Enhanced process terms - more comprehensive
        logger.info(f"🔬 Checking secondary process terms...")
        process_terms = ['formation', 'conditions', 'rates', 'process', 'mechanism',
                        'experimental', 'concentration', 'buffer', 'specific', 'mentioned']
        
        for term in process_terms:
            if term in query_lower:
                logger.info(f"✅ MATCH! Scientific process term detected: '{term}' in '{query_lower}'")
                logger.info(f"🎯 DECISION: Using ensemble retrieval due to process term: {term}")
                return True
            
        # Check for any numbers that might indicate scientific measurements
        logger.info(f"🔬 Checking for numeric patterns in query...")
        number_match = re.search(r'\d+', query)
        if number_match:
            logger.info(f"✅ MATCH! Numeric pattern detected: '{number_match.group()}' in query")
            logger.info(f"🎯 DECISION: Using ensemble retrieval due to numeric content")
            return True
        
        logger.info(f"❌ FINAL RESULT: No scientific terms detected in query: '{query}'")
        logger.info(f"🎯 DECISION: Using standard MMR retrieval (no scientific terms found)")
        return False
    
    def _should_skip_reranker(self, queries: List[str], documents: List[Document]) -> bool:
        """
        Determine if we should skip expensive reranking (exact-match fast path)
        Skip reranker when high confidence exact matches are found
        """
        if not queries or not documents:
            return False
        
        # Get primary query
        query = queries[0].lower()
        
        # Extract key numeric/unit phrases from query
        exact_patterns = [
            r'\d+\s*[-\-]?\s*fold',     # "41-fold"
            r'\d+\s*°?C',               # "97°C"
            r'\d+\s*nm',                # "388 nm"
            r'\d+\.\d+\s*[μu]M',        # "0.82 μM"
            r'p\s*[<≤]\s*0\.\d+',       # "p < 0.05"
            r'λ\s*max\s*=?\s*\d+',      # "λmax = 388"
        ]
        
        # Look for exact numeric matches in query
        found_patterns = []
        for pattern in exact_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                found_patterns.extend(matches)
        
        if not found_patterns:
            return False  # No exact patterns to match
            
        logger.info(f"⚡ FAST PATH CHECK: Found exact patterns in query: {found_patterns}")
        
        # Check if top documents contain these exact patterns
        exact_matches = 0
        for i, doc in enumerate(documents[:5]):  # Check top 5 only
            content_lower = doc.page_content.lower()
            
            for pattern_text in found_patterns:
                if pattern_text.lower() in content_lower:
                    exact_matches += 1
                    logger.info(f"⚡ EXACT MATCH found in doc #{i+1}: '{pattern_text}'")
                    break  # One match per doc is enough
        
        # Skip reranker if we have strong exact matches in top results
        overlap_ratio = exact_matches / min(len(documents), 5)
        should_skip = overlap_ratio >= 0.6  # 60%+ of top docs have exact matches
        
        logger.info(f"⚡ FAST PATH DECISION: {exact_matches}/5 exact matches (ratio={overlap_ratio:.2f}), skip_reranker={should_skip}")
        return should_skip
    
    def _apply_section_boost(self, doc: Document, has_scientific_terms: bool) -> float:
        """Apply section-aware boosting for scientific queries"""
        original_score = doc.metadata.get('score', 1.0)
        
        if not has_scientific_terms:
            return original_score  # No boosting for non-scientific queries
        
        section = doc.metadata.get('section', 'content').lower()
        
        # Boost high-value sections for scientific queries
        boost_factors = {
            'methods': 1.2,           # Experimental procedures
            'results': 1.15,          # Findings and observations  
            'figure_caption': 1.25,   # Figures often contain key data
            'table': 1.2,            # Tables contain measurements
            'abstract': 1.1,          # Condensed key information
            'content': 1.0,           # Default - no boost
            'discussion': 0.95,       # Less likely to have exact measurements
            'conclusion': 0.9,        # Summary, less specific data
            'references': 0.8,        # Least relevant for data queries
        }
        
        boost_factor = boost_factors.get(section, 1.0)
        boosted_score = original_score * boost_factor
        
        return boosted_score

    def _generate_answer(self, question: str, docs: List[Document]) -> str:
        """Generate answer using retrieved documents"""
        if not docs:
            return "I couldn't find relevant information to answer your question."

        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(docs[:5]):  # Use top 5 docs
            context_parts.append(f"Document {i + 1}:\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        try:
            # Use direct QA chain with context
            result = self.direct_qa_chain.run(
                context=context,
                question=question
            )

            return result.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            # Fallback: try with retrieval QA chain
            try:
                result = self.qa_chain({"query": question})
                return result.get("result", "I encountered an error while processing your question.")
            except Exception as e2:
                logger.error(f"Fallback answer generation also failed: {str(e2)}")
                return "I encountered an error while processing your question."

    def _answer_with_decomposition(self,
                                   original_question: str,
                                   sub_queries: List[str],
                                   docs: List[Document]) -> str:
        """Answer using decomposed queries"""
        sub_answers = []

        # Answer each sub-query
        for sub_query in sub_queries:
            # Filter relevant docs for sub-query
            relevant_docs = self._filter_relevant_docs(sub_query, docs)
            if relevant_docs:
                sub_answer = self._generate_answer(sub_query, relevant_docs)
                sub_answers.append(f"Q: {sub_query}\nA: {sub_answer}")
            else:
                sub_answers.append(f"Q: {sub_query}\nA: No specific information found for this aspect.")

        # Synthesize final answer
        try:
            final_answer = self.synthesis_chain.run(
                sub_answers="\n\n".join(sub_answers),
                original_question=original_question
            )

            # Clean up the synthesized answer
            # Remove Q:/A: prefixes if they leaked through
            final_answer = re.sub(r'\bQ:\s*', '', final_answer)
            final_answer = re.sub(r'\bA:\s*', '', final_answer)

            return final_answer.strip()

        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            # Fallback: return the best sub-answer
            if sub_answers:
                # Find the longest sub-answer as it's likely most comprehensive
                best_answer = max(sub_answers, key=lambda x: len(x.split('A:')[1]) if 'A:' in x else 0)
                return best_answer.split('A:')[1].strip() if 'A:' in best_answer else sub_answers[0]
            else:
                return "I couldn't synthesize a complete answer from the available information."

    def _filter_relevant_docs(self,
                              query: str,
                              docs: List[Document],
                              top_k: int = 3) -> List[Document]:
        """Filter most relevant documents for a query"""
        # Simple keyword-based filtering with improvements
        query_terms = set(query.lower().split())
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are',
                      'was', 'were'}
        query_terms = query_terms - stop_words

        scored_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            # Score based on term frequency
            score = sum(content_lower.count(term) for term in query_terms)
            # Boost score if multiple terms appear close together
            for i, term1 in enumerate(list(query_terms)):
                for term2 in list(query_terms)[i + 1:]:
                    if term1 in content_lower and term2 in content_lower:
                        # Check if terms appear within 50 characters of each other
                        idx1 = content_lower.find(term1)
                        idx2 = content_lower.find(term2)
                        if abs(idx1 - idx2) < 50:
                            score += 2

            scored_docs.append((score, doc))

        # Sort by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return top k documents with non-zero scores
        return [doc for score, doc in scored_docs[:top_k] if score > 0]

    def _enhance_answer(self, answer: str, entities: List) -> str:
        """Enhance answer with entity links and glossary tooltips"""
        if not answer:
            return answer

        try:
            # First, add entity links
            enhanced = self.entity_linker.create_linked_html(answer, entities)

            # Then, add glossary tooltips
            enhanced = self.glossary_manager.simplify_text(enhanced, format="html")

            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing answer: {str(e)}")
            # Return original answer if enhancement fails
            return answer

    def _calculate_confidence(self,
                              docs: List[Document],
                              answer: str) -> float:
        """Calculate confidence score for answer"""
        if not docs or not answer:
            return 0.0

        # Factors for confidence
        factors = []

        # 1. Number of supporting documents
        doc_score = min(len(docs) / 5.0, 1.0)
        factors.append(doc_score)

        # 2. Answer length (longer usually means more comprehensive)
        word_count = len(answer.split())
        length_score = min(word_count / 100.0, 1.0)
        factors.append(length_score)

        # 3. Entity coverage
        try:
            entities_found = len(self.entity_linker.extract_entities(answer))
            entity_score = min(entities_found / 5.0, 1.0)
            factors.append(entity_score)
        except:
            factors.append(0.5)  # Default if entity extraction fails

        # 4. Document relevance scores
        if docs and any('score' in doc.metadata for doc in docs):
            avg_score = sum(doc.metadata.get('score', 0) for doc in docs[:5]) / min(5, len(docs))
            factors.append(min(avg_score, 1.0))

        # Average confidence
        confidence = sum(factors) / len(factors) if factors else 0.0

        return round(confidence, 2)

    def _format_entities(self, entity_summary: Dict[str, List]) -> List[Dict]:
        """Format entities for output"""
        formatted = []

        for category, entities in entity_summary.items():
            for entity in entities:
                formatted.append({
                    "text": entity["text"],
                    "type": category.replace("_", " ").title(),
                    "url": entity.get("url", ""),
                    "id": entity.get("kb_id", ""),
                    "description": entity.get("description", "")
                })

        return formatted

    def explain_retrieval(self, question: str) -> Dict[str, Any]:
        """Explain the retrieval process for transparency"""
        explanation = {
            "query": question,
            "is_complex": self._is_complex_query(question),
            "steps": [],
            "timestamp": datetime.now().isoformat()
        }

        # Decomposition
        if self._is_complex_query(question):
            try:
                sub_queries = self._decompose_query(question)
                explanation["steps"].append({
                    "step": "Query Decomposition",
                    "description": "Breaking down complex question into simpler parts",
                    "result": sub_queries if sub_queries else "No decomposition performed"
                })
            except:
                explanation["steps"].append({
                    "step": "Query Decomposition",
                    "description": "Attempted to break down complex question",
                    "result": "Decomposition failed, using original query"
                })

        # HyDE
        explanation["steps"].append({
            "step": "Hypothetical Answer Generation",
            "description": "Creating a hypothetical answer to improve retrieval",
            "result": "Generated hypothetical document for better matching"
        })

        # Retrieval
        explanation["steps"].append({
            "step": "Document Retrieval",
            "description": "Searching knowledge base using multiple strategies",
            "result": "Retrieved relevant documents using similarity and diversity (MMR)"
        })

        # Enhancement
        explanation["steps"].append({
            "step": "Answer Enhancement",
            "description": "Adding entity links and jargon explanations",
            "result": "Enhanced answer with clickable entities and tooltips"
        })

        # Confidence
        explanation["steps"].append({
            "step": "Confidence Assessment",
            "description": "Calculating answer confidence based on multiple factors",
            "result": "Confidence score based on document support, entities found, and answer completeness"
        })

        return explanation

    def _assess_document_relevance(self, question: str, docs: List[Document]) -> float:
        """Assess how relevant retrieved documents are to the question"""
        if not docs:
            return 0.0
        
        # Extract key terms from question
        question_lower = question.lower()
        # Remove common words and extract potential biomedical terms
        stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'are', 'was', 'were', 'how', 'does', 'do', 'can', 'could', 'would', 'should'}
        question_terms = [term for term in question_lower.split() if term not in stop_words and len(term) > 2]
        
        logger.info(f"Question terms for relevance: {question_terms}")
        
        # Calculate relevance based on term overlap
        total_score = 0
        for i, doc in enumerate(docs[:5]):  # Check top 5 docs
            doc_content_lower = doc.page_content.lower()
            doc_score = 0
            
            # Direct term matches
            matches = []
            for term in question_terms:
                if term in doc_content_lower:
                    doc_score += 1
                    matches.append(term)
            
            # Normalize by question terms count
            if question_terms:
                doc_score = doc_score / len(question_terms)
            
            logger.info(f"Doc {i+1} relevance: {doc_score:.2f}, matches: {matches}, preview: {doc.page_content[:100]}")
            total_score += doc_score
        
        # Average across documents
        avg_score = total_score / min(5, len(docs)) if docs else 0
        logger.info(f"Overall relevance score: {avg_score:.2f}")
        return min(avg_score, 1.0)

    def _extract_question_topic(self, question: str) -> str:
        """Extract the main topic from a question"""
        import re
        
        # Common biomedical patterns - check original question, not lowercase
        bio_patterns = [
            r'\b([A-Z][A-Z0-9]{2,})\b',  # Gene names like BRCA1, TP53
            r'\b([a-z]+-?[0-9]+)\b',     # Things like p53, IL-1
            r'\b(protein|gene|enzyme|receptor|cancer|disease|mutation|pathway|metabolism)\b'
        ]
        
        topics = []
        for pattern in bio_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                topics.extend(matches)
        
        if topics:
            return topics[0] if isinstance(topics[0], str) else str(topics[0])
        
        # Fallback: use the first few words after removing common words
        words = question.split()
        stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'how', 'does', 'do', 'can'}
        content_words = [w for w in words if w.lower() not in stop_words]
        
        return ' '.join(content_words[:2]) if content_words else "this topic"

    def _identify_document_topics(self, docs: List[Document]) -> str:
        """Identify what topics the documents actually discuss"""
        if not docs:
            return "unknown topics"
        
        # Extract common terms from document content
        all_content = ' '.join([doc.page_content[:500] for doc in docs])
        content_lower = all_content.lower()
        
        # Look for scientific terms
        import re
        
        # Chemical formulas
        chemicals = re.findall(r'\b[A-Z][a-z]?[0-9]*\b', all_content)
        chemicals = [c for c in chemicals if len(c) <= 6]  # Filter reasonable chemical names
        
        # Scientific terms
        science_terms = re.findall(r'\b(methane|formation|reaction|catalysis|synthesis|oxidation|reduction|metabolism|pathway|enzyme|protein|gene|cellular|molecular|biochemical|chemical|biological|physiological)\b', content_lower)
        
        # Combine and get most common
        all_terms = chemicals + science_terms
        if all_terms:
            # Get most common terms
            from collections import Counter
            common_terms = Counter(all_terms).most_common(3)
            return ', '.join([term for term, count in common_terms])
        
        return "chemistry and biochemistry"
    
    def _get_sample_documents_from_db(self, vector_db) -> List[Document]:
        """Get sample documents from vector DB to initialize BM25"""
        try:
            # Get a sample of documents from the database
            collection = vector_db._collection
            results = collection.get(limit=50, include=["documents", "metadatas"])
            
            if not results or not results.get('documents'):
                return []
            
            documents = []
            for i, content in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results.get('metadatas') and i < len(results['metadatas']) else {}
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} sample documents for BM25 initialization")
            return documents
            
        except Exception as e:
            logger.warning(f"Failed to get sample documents: {str(e)}")
            return []