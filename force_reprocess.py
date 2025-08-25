#!/usr/bin/env python3
"""
Force complete reprocessing of PDFs with enhanced extraction
"""

import sys
import os
import shutil
from pathlib import Path
sys.path.append('biorag')

print("Force Reprocessing with Enhanced PDF Extraction")
print("=" * 50)

# Clear ALL cached data
print("\n1. Clearing all cached data...")

# Remove vector database
vector_db_path = Path("./vector_db")
if vector_db_path.exists():
    shutil.rmtree(vector_db_path)
    print("✓ Vector database cleared")

# Clear Chrome data if exists
chroma_path = Path("./chroma.sqlite3")
if chroma_path.exists():
    chroma_path.unlink()
    print("✓ Chroma cache cleared")

print("\n2. Enhanced extraction settings applied:")
print("✓ Chunk size: 1000 → 1500 characters")
print("✓ Chunk overlap: 200 → 300 characters") 
print("✓ Table/figure extraction enabled")
print("✓ Section-aware chunking")
print("✓ Retrieval coverage: 5 → 8 documents")
print("✓ MMR diversity: 15 → 25 candidates")

print("\n3. Improvements for scientific papers:")
print("✓ [TABLE] markers preserved")
print("✓ [FIGURE CAPTION] extraction")
print("✓ [SECTION] headers maintained")
print("✓ Results/Methods sections prioritized")

print("\n🚀 Ready for re-processing!")
print("\nNext steps:")
print("1. Restart the API server (python api_server.py)")
print("2. Re-upload your PDF")
print("3. The enhanced extraction should now capture:")
print("   • Temperature data (97°C, 30°C conditions)")
print("   • Production rates (0.02 → 0.82 μM·h⁻¹)")
print("   • Fold changes (41×, >11× citrate boost)")
print("   • CH4:C2H6 ratios (~110, 190-1100)")
print("   • Table/figure quantitative data")

print("\n✨ Test queries after upload:")
print("   - 'What are the exact production rates?'")
print("   - 'Show me the 41-fold increase data'")
print("   - 'What specific numbers are in the tables?'")