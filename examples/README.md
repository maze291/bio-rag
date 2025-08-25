# 📊 BioRAG Examples

This directory contains sample data and configuration files to help you get started with BioRAG quickly.

## 📁 Files Overview

| File | Description | Usage |
|------|-------------|--------|
| `sample_biomedical_text.txt` | Comprehensive biomedical research paper sample | Document ingestion testing |
| `rss_feeds.txt` | Curated list of biomedical RSS feeds | RSS ingestion and real-time updates |

## 🚀 Quick Start Examples

### 1. **Basic Document Processing**

```bash
# Process the sample document
python cli.py --ingest examples/sample_biomedical_text.txt

# Ask questions about it
python cli.py --query "What is the function of BRCA1?"
python cli.py --query "How do PARP inhibitors work?"
python cli.py --query "What are the treatment options for BRCA-associated cancers?"
```

### 2. **RSS Feed Integration**

```bash
# Use a curated RSS feed
python cli.py --rss "https://pubmed.ncbi.nlm.nih.gov/rss/search/cancer/" --query "What are the latest cancer research findings?"

# Multiple feeds for comprehensive analysis
python cli.py \
  --rss "https://www.nature.com/nm.rss" \
  --rss "https://www.cell.com/cell/current.rss" \
  --query "Summarize recent advances in precision medicine"
```

### 3. **Web Interface Demo**

```bash
# Start the Streamlit interface
streamlit run main.py

# Then:
# 1. Upload examples/sample_biomedical_text.txt
# 2. Try these questions:
#    - "Explain BRCA1's role in DNA repair"
#    - "What are HRD biomarkers?"
#    - "Compare different PARP inhibitors"
```

## 🎯 Interactive Demo Scenarios

### **Scenario 1: Cancer Researcher**
**Goal**: Understanding BRCA genetics and treatment options

```bash
# Step 1: Load research data
python cli.py --ingest examples/sample_biomedical_text.txt

# Step 2: Research questions
python cli.py --query "What's the difference between BRCA1 and BRCA2 mutations?"
python cli.py --query "Which patients benefit most from PARP inhibitors?"
python cli.py --query "How does homologous recombination deficiency affect treatment?"
```

**Expected Results**: 
- Entity links to UniProt (BRCA1, BRCA2)
- Jargon tooltips for "homologous recombination", "synthetic lethality"
- Source citations from the sample document

### **Scenario 2: Clinical Scientist**
**Goal**: Staying current with latest research

```bash
# Step 1: Fetch latest papers
python cli.py --rss "https://pubmed.ncbi.nlm.nih.gov/rss/search/immunotherapy/" --query "What are the newest immunotherapy approaches?"

# Step 2: Compare with existing knowledge
python cli.py --ingest examples/sample_biomedical_text.txt --query "How does this compare to current PARP inhibitor strategies?"
```

### **Scenario 3: Medical Student**
**Goal**: Learning complex biomedical concepts

```bash
# Start with web interface for visual learning
streamlit run main.py

# Upload sample document and ask learning questions:
# - "Explain apoptosis in simple terms"
# - "What causes cancer cells to become resistant to treatment?"
# - "How do DNA repair mechanisms work?"
```

**Expected Results**:
- Simplified explanations via glossary tooltips
- Visual entity highlighting
- Links to external databases for deeper learning

## 📚 Sample Questions by Topic

### **DNA Repair & Genetics**
```bash
python cli.py --query "How does homologous recombination repair DNA damage?"
python cli.py --query "What happens when BRCA1 is mutated?"
python cli.py --query "Explain the concept of synthetic lethality"
```

### **Cancer Treatment**
```bash
python cli.py --query "How do PARP inhibitors selectively target cancer cells?"
python cli.py --query "What are the side effects of olaparib treatment?"
python cli.py --query "When is combination therapy most effective?"
```

### **Clinical Applications**
```bash
python cli.py --query "How do you test for BRCA mutations?"
python cli.py --query "What are the prevention options for BRCA carriers?"
python cli.py --query "How do you monitor treatment response?"
```

## 🔧 Advanced Usage Examples

### **Batch Processing**
```bash
# Process multiple documents and export results
for topic in cancer genetics immunotherapy; do
    python cli.py \
        --rss "https://pubmed.ncbi.nlm.nih.gov/rss/search/${topic}/" \
        --query "Summarize key findings in ${topic}" \
        --export "results_${topic}.json"
done
```

### **Custom Entity Recognition**
```bash
# Focus on specific entity types
python cli.py --ingest examples/sample_biomedical_text.txt --query "List all genes mentioned in this paper"
python cli.py --query "What chemicals and drugs are discussed?"
python cli.py --query "What diseases are associated with these findings?"
```

### **Comparative Analysis**
```bash
# Compare different sources
python cli.py \
    --ingest examples/sample_biomedical_text.txt \
    --rss "https://www.nature.com/ng.rss" \
    --query "How do these findings compare to recent genetics research?"
```

## 📝 Creating Your Own Examples

### **1. Document Examples**
Create files with rich biomedical content:
```
examples/
├── cardiology_paper.pdf
├── neuroscience_abstract.txt
├── clinical_trial_results.html
└── genomics_review.txt
```

### **2. Domain-Specific RSS Lists**
Create specialized feed collections:
```
examples/feeds/
├── cardiology_feeds.txt
├── neuroscience_feeds.txt
├── oncology_feeds.txt
└── genetics_feeds.txt
```

### **3. Question Templates**
Prepare domain-specific question sets:
```python
# examples/question_templates.py
GENETICS_QUESTIONS = [
    "What genes are associated with {disease}?",
    "How does {gene} mutation affect {process}?",
    "What are the therapeutic targets in {pathway}?"
]
```

## 🎨 Customization Tips

### **Tailoring for Your Research Area**
1. **Replace sample document** with papers from your field
2. **Update RSS feeds** to focus on your research interests  
3. **Customize entity knowledge base** with domain-specific terms
4. **Add field-specific glossary terms** for your jargon

### **Creating Demo Scenarios**
1. **Prepare realistic questions** that showcase BioRAG's strengths
2. **Use documents with rich entity content** (genes, drugs, diseases)
3. **Include complex concepts** that benefit from jargon simplification
4. **Test entity linking** to ensure major terms have database links

## 🔍 Troubleshooting Examples

### **No Results Found**
```bash
# Check if documents were processed
python cli.py --query "How many documents are in the database?"

# Verify entity recognition
python cli.py --query "What entities were found in the uploaded documents?"
```

### **Poor Quality Results**
```bash
# Try more specific questions
python cli.py --query "What is the molecular mechanism of BRCA1 in DNA repair?"
# Instead of: "Tell me about BRCA1"

# Use multiple documents
python cli.py --ingest doc1.pdf doc2.pdf --query "your question"
```

### **RSS Feed Issues**
```bash
# Test individual feeds
python cli.py --rss "https://www.nature.com/nm.rss" --query "test"

# Check feed validity
curl -I "https://www.nature.com/nm.rss"
```

## 💡 Pro Tips

1. **Start Simple**: Begin with the sample document before adding your own
2. **Use Specific Questions**: "How does X affect Y?" works better than "Tell me about X"
3. **Combine Sources**: Mix documents and RSS feeds for comprehensive answers
4. **Leverage Entity Links**: Click through to external databases for deeper research
5. **Export Results**: Save interesting Q&A sessions for future reference
6. **Test Regularly**: Use the self-test to verify everything works after changes

---

**Need help?** Check the main [README.md](../README.md) or run `python cli.py --help`