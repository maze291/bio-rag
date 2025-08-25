# BioRAG - Biomedical Research Assistant

🚨 **CRITICAL ISSUE - SEEKING HELP**

Enhanced PDF extraction works perfectly, but semantic retrieval is completely broken.

## Problem:
- ✅ Data extracted: '41-fold', '0.02', '0.82 μM·h⁻¹' exists in chunks
- ❌ Retrieval broken: Semantic search cannot find chunks with the data
- ❌ Bot claims: 'cannot find relevant information' despite data being present

## Key Missing Data:
From methane formation paper:
- 'rates increased 41-fold to ~0.82 μM h⁻¹ at 97°C'
- 'marginal CH₄ formation rates at 30°C (~0.02 μM h⁻¹)'

## Technical Details:
- Vector DB: ChromaDB + SciBERT embeddings
- Issue: Chunk 40 contains '41-fold' but NEVER retrieved by semantic search
- Enhanced extraction working (Unicode symbols preserved)

## Test Case:
Query: 'What are exact production rates?'
Expected: Should find '0.02 → 0.82' data
Actual: Claims 'cannot find information'

HELP NEEDED: Why is semantic search failing to retrieve chunks containing search terms?

