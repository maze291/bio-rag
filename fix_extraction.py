#!/usr/bin/env python3
"""
Fix PDF extraction by clearing corrupted data and re-processing
"""

import sys
import os
import shutil
from pathlib import Path
sys.path.append('biorag')

from biorag.core.vectordb import VectorDBManager
from biorag.core.ingest import IngestPipeline

def fix_extraction():
    print("Fixing PDF Text Extraction Issues")
    print("=" * 40)
    
    # 1. Clear corrupted vector database
    vector_db_path = Path("./vector_db")
    if vector_db_path.exists():
        print(f"Clearing corrupted vector database...")
        shutil.rmtree(vector_db_path)
        print("Vector database cleared")
    
    # 2. Clear ingestion cache
    ingester = IngestPipeline()
    ingester.doc_cache.clear()
    print("Ingestion cache cleared")
    
    # 3. Test text cleaning on sample corrupted text
    sample_corrupted = """
    Thus, the two factors heat and light synergistically combine for a stable and enhanced ROS and CHa formation. 
    ROS-generated CHsz is derived from biomass with Fe** to Fe*** and releasing organic radicals and thus enhance 
    ROS-driven CHa formation at 37°C.
    """
    
    print("\n--- Testing Text Cleaning ---")
    print("Before:", repr(sample_corrupted[:100]))
    
    cleaned = ingester._clean_text(sample_corrupted)
    print("After:", repr(cleaned[:100]))
    
    # 4. Show improvements
    improvements = [
        ("CHa", "CH4"),
        ("CHsz", "CH4"), 
        ("C2Hs", "C2H6"),
        ("Fe**", "Fe2+"),
        ("Fe***", "Fe3+")
    ]
    
    print("\n--- Text Corrections Applied ---")
    for old, new in improvements:
        if old in sample_corrupted and new in cleaned:
            print(f"✓ {old} -> {new}")
    
    print("\n✅ Text extraction fixes applied!")
    print("\nNext steps:")
    print("1. Restart the API server")
    print("2. Re-upload your PDF")
    print("3. Test queries again")

if __name__ == "__main__":
    fix_extraction()