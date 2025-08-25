#!/usr/bin/env python3
"""
BioRAG Setup Script
Alternative to Poetry for pip-based installations
"""

from setuptools import setup, find_packages
import pathlib

# Get the long description from the README file
HERE = pathlib.Path(__file__).parent
long_description = (HERE / "README.md").read_text(encoding='utf-8')


# Read requirements
def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name="biorag",
    version="1.0.0",
    description="Biomedical RAG with Entity Linking and Jargon Simplification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/biorag",
    author="Your Name",
    author_email="your.email@domain.com",
    license="MIT",

    # Classifiers help users find your project by categorizing it
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],

    keywords="biomedical rag nlp entity-linking scientific-literature",

    # Package discovery
    packages=find_packages(),
    python_requires=">=3.8",

    # Dependencies
    install_requires=read_requirements(),

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch[cuda]",
        ],
        "full": [
            "en_ner_bionlp13cg_md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bionlp13cg_md-0.5.3.tar.gz",
            "en_ner_bc5cdr_md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz",
        ]
    },

    # Include additional files
    package_data={
        "biorag": ["*.yaml", "*.yml", "*.txt"],
        "core": ["*.yaml", "*.yml"],
        "examples": ["*.txt", "*.pdf", "*.html"],
    },

    # Console scripts
    entry_points={
        "console_scripts": [
            "biorag=cli:main",
            "biorag-cli=cli:main",
            "biorag-ui=main:main",
        ],
    },

    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/biorag/issues",
        "Source": "https://github.com/yourusername/biorag",
        "Documentation": "https://github.com/yourusername/biorag#readme",
    },
)