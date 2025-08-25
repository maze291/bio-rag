#!/usr/bin/env python3
"""
BioRAG CLI Interface
Command-line interface for the BioRAG system with all fixes applied
"""

import sys
import os
from pathlib import Path
import argparse
import json
import tempfile
import hashlib
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from typing import List, Optional, Dict, Any
import pickle

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

from core.ingest import IngestPipeline
from core.vectordb import VectorDBManager
from core.linker import EntityLinker
from core.glossary import GlossaryManager
from core.rag_chain import RAGChain

console = Console()


class BioRAGCLI:
    """Command-line interface for BioRAG system"""

    def __init__(self):
        self.ingester = IngestPipeline()
        self.db_manager = VectorDBManager()
        self.entity_linker = EntityLinker()
        self.glossary_mgr = GlossaryManager()
        self.vector_db = None
        self.rag_chain = None
        self.conversation_history = []

    def ingest_files(self, file_paths: List[str], show_progress: bool = True):
        """Ingest multiple files with detailed progress"""
        all_docs = []
        failed_files = []

        if show_progress:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
            ) as progress:
                # Main task for overall progress
                main_task = progress.add_task("Ingesting files...", total=len(file_paths))

                for file_path in file_paths:
                    file_name = Path(file_path).name

                    # Sub-task for individual file
                    file_task = progress.add_task(f"Processing {file_name}", total=100)

                    try:
                        # Update progress stages
                        progress.update(file_task, advance=30, description=f"Reading {file_name}...")
                        docs = self.ingester.ingest_file(file_path)

                        progress.update(file_task, advance=40, description=f"Chunking {file_name}...")
                        all_docs.extend(docs)

                        progress.update(file_task, advance=30, description=f"Completed {file_name}")
                        console.print(f"✅ Ingested {file_name}: {len(docs)} chunks")

                    except Exception as e:
                        console.print(f"❌ Error with {file_path}: {str(e)}", style="red")
                        failed_files.append((file_path, str(e)))

                    progress.advance(main_task)
                    progress.remove_task(file_task)
        else:
            # Simple progress for non-interactive mode
            for file_path in file_paths:
                try:
                    docs = self.ingester.ingest_file(file_path)
                    all_docs.extend(docs)
                    console.print(f"✅ Ingested {Path(file_path).name}: {len(docs)} chunks")
                except Exception as e:
                    console.print(f"❌ Error with {file_path}: {str(e)}", style="red")
                    failed_files.append((file_path, str(e)))

        if failed_files:
            console.print(f"\n⚠️  {len(failed_files)} files failed to process", style="yellow")

        return all_docs

    def build_database(self, docs: List, show_existing_info: bool = True):
        """Build or update vector database with metadata tracking"""
        with console.status("[bold green]Building vector database..."):
            # Check for existing DB and embedding model compatibility
            if self.vector_db is None:
                self.vector_db = self.db_manager.create_db(docs)
                db_info = self._save_db_metadata()
            else:
                # Verify embedding model compatibility
                if not self._check_embedding_compatibility():
                    console.print("⚠️  Warning: Embedding model mismatch detected", style="yellow")
                    if not console.input("Continue anyway? [y/N]: ").lower().startswith('y'):
                        return

                self.db_manager.add_documents(self.vector_db, docs)
                db_info = self._update_db_metadata()

            self.rag_chain = RAGChain(
                self.vector_db,
                self.entity_linker,
                self.glossary_mgr
            )

            if show_existing_info and db_info:
                console.print(f"✅ Vector database ready! Total documents: {db_info['total_docs']}", style="green")

    def query(self, question: str, show_sources: bool = False, export_result: bool = False):
        """Query the knowledge base with optional export"""
        if self.rag_chain is None:
            console.print("❌ No knowledge base loaded. Please ingest documents first.", style="red")
            return None

        with console.status("[bold cyan]Thinking..."):
            try:
                response = self.rag_chain.query(question)
            except Exception as e:
                console.print(f"❌ Error: {str(e)}", style="red")
                return None

        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": response["answer"],
            "entities": response["entities"],
            "sources": [doc.metadata for doc in response.get("source_docs", [])]
        })

        # Display answer in a nice panel
        console.print("\n")
        console.print(Panel(
            Markdown(response["answer"]),
            title="🤖 Answer",
            border_style="cyan"
        ))

        # Show entities in a table
        if response["entities"]:
            console.print("\n")
            entity_table = Table(title="🔬 Detected Entities")
            entity_table.add_column("Type", style="cyan")
            entity_table.add_column("Entity", style="green")
            entity_table.add_column("ID", style="yellow")
            entity_table.add_column("URL", style="blue", overflow="fold")

            for entity in response["entities"][:10]:  # Show top 10
                entity_table.add_row(
                    entity["type"],
                    entity["text"],
                    entity.get("id", "N/A"),
                    entity.get("url", "N/A")
                )

            console.print(entity_table)

            if len(response["entities"]) > 10:
                console.print(f"[dim]...and {len(response['entities']) - 10} more entities[/dim]")

        # Show sources if requested
        if show_sources and response.get("source_docs"):
            console.print("\n")
            console.print(Panel("📚 Sources", style="yellow"))
            for i, doc in enumerate(response["source_docs"][:3]):
                console.print(f"\n[yellow]Source {i + 1}:[/yellow] {doc.metadata.get('source', 'Unknown')}")
                console.print(f"[dim]{doc.page_content[:200]}...[/dim]")

        return response

    def export_conversation(self, output_path: str, format: str = "json"):
        """Export conversation history to file"""
        if not self.conversation_history:
            console.print("No conversation history to export", style="yellow")
            return

        try:
            output_path = Path(output_path)

            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)

            elif format == "markdown":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("# BioRAG Conversation History\n\n")
                    for entry in self.conversation_history:
                        f.write(f"## {entry['timestamp']}\n\n")
                        f.write(f"**Question:** {entry['question']}\n\n")
                        f.write(f"**Answer:** {entry['answer']}\n\n")
                        if entry['entities']:
                            f.write("**Entities:**\n")
                            for ent in entry['entities']:
                                f.write(f"- {ent['text']} ({ent['type']})\n")
                            f.write("\n")
                        f.write("---\n\n")

            console.print(f"✅ Exported conversation to {output_path}", style="green")

        except Exception as e:
            console.print(f"❌ Export failed: {str(e)}", style="red")

    def interactive_mode(self):
        """Run interactive chat mode"""
        console.print(Panel(
            "[bold cyan]BioRAG Interactive Mode[/bold cyan]\n"
            "Type your questions, or use commands:\n"
            "  /help - Show commands\n"
            "  /load <file> - Load a document\n"
            "  /stats - Show KB statistics\n"
            "  /export <file> - Export conversation\n"
            "  /clear - Clear console\n"
            "  /history - Show conversation history\n"
            "  /exit - Exit",
            title="🧬 Welcome to BioRAG",
            border_style="cyan"
        ))

        while True:
            try:
                question = console.input("\n[bold cyan]You:[/bold cyan] ")

                if question.startswith("/"):
                    self._handle_command(question)
                elif question.strip():
                    self.query(question)

            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="yellow")
                break
            except Exception as e:
                console.print(f"❌ Error: {str(e)}", style="red")

    def _handle_command(self, command: str):
        """Handle special commands"""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/exit":
            console.print("👋 Goodbye!", style="yellow")
            sys.exit(0)

        elif cmd == "/help":
            help_text = """
[bold]Available Commands:[/bold]
  /help          - Show this help
  /load <file>   - Load a document
  /stats         - Show knowledge base statistics
  /clear         - Clear the console
  /export <file> - Export conversation (add .json or .md extension)
  /history       - Show conversation history summary
  /exit          - Exit the program
            """
            console.print(Panel(help_text, title="Help", border_style="green"))

        elif cmd == "/load" and len(parts) > 1:
            file_path = parts[1]
            if Path(file_path).exists():
                docs = self.ingest_files([file_path])
                if docs:
                    self.build_database(docs)
            else:
                console.print(f"File not found: {file_path}", style="red")

        elif cmd == "/stats":
            if self.vector_db:
                stats_table = Table(title="📊 Knowledge Base Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")

                # Get real statistics
                db_stats = self.db_manager.get_collection_stats(self.vector_db)

                stats_table.add_row("Total Documents", str(db_stats["total_documents"]))
                stats_table.add_row("Embedding Model", db_stats["embedding_model"])
                stats_table.add_row("Collection Name", db_stats["collection_name"])

                # Get embedding dimensions dynamically
                if hasattr(self.db_manager.embeddings, 'client'):
                    sample_embedding = self.db_manager.embeddings.embed_query("test")
                    stats_table.add_row("Vector Dimensions", str(len(sample_embedding)))

                if db_stats.get("metadata_fields"):
                    stats_table.add_row("Metadata Fields", ", ".join(db_stats["metadata_fields"]))

                stats_table.add_row("Conversation History", str(len(self.conversation_history)))

                console.print(stats_table)
            else:
                console.print("No knowledge base loaded.", style="yellow")

        elif cmd == "/clear":
            console.clear()

        elif cmd == "/export" and len(parts) > 1:
            file_path = parts[1]
            format = "markdown" if file_path.endswith('.md') else "json"
            self.export_conversation(file_path, format)

        elif cmd == "/history":
            if self.conversation_history:
                history_table = Table(title="📜 Conversation History")
                history_table.add_column("Time", style="cyan")
                history_table.add_column("Question", style="green", overflow="fold")
                history_table.add_column("Entities", style="yellow")

                for entry in self.conversation_history[-10:]:  # Last 10
                    timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
                    question = entry['question'][:50] + "..." if len(entry['question']) > 50 else entry['question']
                    entities = len(entry['entities'])

                    history_table.add_row(timestamp, question, str(entities))

                console.print(history_table)
                if len(self.conversation_history) > 10:
                    console.print(f"[dim]Showing last 10 of {len(self.conversation_history)} entries[/dim]")
            else:
                console.print("No conversation history yet.", style="yellow")

        else:
            console.print(f"Unknown command: {cmd}", style="red")

    def _save_db_metadata(self) -> Dict[str, Any]:
        """Save database metadata for compatibility checking"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "embedding_model": self.db_manager.embedding_model_name,
            "total_docs": self.db_manager.get_collection_stats(self.vector_db)["total_documents"]
        }

        metadata_path = Path(self.db_manager.persist_directory) / ".biorag_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except:
            pass

        return metadata

    def _update_db_metadata(self) -> Dict[str, Any]:
        """Update database metadata"""
        metadata_path = Path(self.db_manager.persist_directory) / ".biorag_metadata.json"

        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            metadata.update({
                "last_updated": datetime.now().isoformat(),
                "total_docs": self.db_manager.get_collection_stats(self.vector_db)["total_documents"]
            })

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return metadata
        except:
            return {}

    def _check_embedding_compatibility(self) -> bool:
        """Check if current embedding model matches database"""
        metadata_path = Path(self.db_manager.persist_directory) / ".biorag_metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get("embedding_model") == self.db_manager.embedding_model_name
            except:
                pass

        return True  # Assume compatible if no metadata


def main():
    parser = argparse.ArgumentParser(
        description="BioRAG - Biomedical RAG with Entity Linking and Jargon Simplification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli.py

  # Ingest files and query
  python cli.py --ingest paper1.pdf paper2.pdf --query "What genes are involved?"

  # Ingest from RSS
  python cli.py --rss https://pubmed.ncbi.nlm.nih.gov/rss/search/cancer --query "Latest findings"

  # Export results
  python cli.py --query "explain apoptosis" --export results.json

  # Self-test
  python cli.py --selftest
        """
    )

    parser.add_argument(
        "--ingest", "-i",
        nargs="+",
        help="Files to ingest (PDFs, HTML, TXT)"
    )

    parser.add_argument(
        "--rss", "-r",
        help="RSS feed URL to ingest"
    )

    parser.add_argument(
        "--url", "-u",
        help="Web page URL to ingest"
    )

    parser.add_argument(
        "--query", "-q",
        help="Question to ask"
    )

    parser.add_argument(
        "--show-sources", "-s",
        action="store_true",
        help="Show source documents"
    )

    parser.add_argument(
        "--export", "-e",
        help="Export results to file (.json or .md)"
    )

    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run self-test with sample data"
    )

    parser.add_argument(
        "--db-path",
        default="./vector_db",
        help="Path to vector database (default: ./vector_db)"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = BioRAGCLI()
    cli.db_manager.persist_directory = args.db_path

    # Handle self-test
    if args.selftest:
        console.print(Panel("🧪 Running Self-Test", style="bold magenta"))

        # Test ingestion
        test_text = """
        Recent studies have identified BRCA1 and TP53 mutations in breast cancer patients.
        The protein p53, encoded by the TP53 gene, acts as a tumor suppressor.
        Treatment with tamoxifen showed promising results in ER-positive cases.
        Apoptosis was observed in cells treated with doxorubicin.
        """

        # Create temp file using proper temp directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_text)
            test_file = f.name

        try:
            # Test ingestion
            console.print("\n1️⃣ Testing document ingestion...")
            docs = cli.ingest_files([test_file], show_progress=not args.no_progress)

            # Build DB
            console.print("\n2️⃣ Building vector database...")
            cli.build_database(docs)

            # Test queries
            console.print("\n3️⃣ Testing queries...")
            test_queries = [
                "What genes are mentioned?",
                "Explain apoptosis in simple terms"
            ]

            for q in test_queries:
                console.print(f"\n[cyan]Query:[/cyan] {q}")
                result = cli.query(q, show_sources=True)

                # Test export
                if result:
                    export_file = tempfile.mktemp(suffix='.json')
                    cli.export_conversation(export_file, format='json')
                    Path(export_file).unlink()

            console.print("\n✅ Self-test completed successfully!", style="green")

        finally:
            # Cleanup
            Path(test_file).unlink()

        return

    # Load existing database if available
    db_path = Path(args.db_path)
    if db_path.exists():
        console.print(f"Loading existing database from {db_path}...", style="cyan")
        try:
            cli.vector_db = cli.db_manager.load_db(str(db_path))
            cli.rag_chain = RAGChain(cli.vector_db, cli.entity_linker, cli.glossary_mgr)

            # Show stats
            stats = cli.db_manager.get_collection_stats(cli.vector_db)
            console.print(f"Loaded {stats['total_documents']} documents", style="green")
        except Exception as e:
            console.print(f"Warning: Could not load existing database: {str(e)}", style="yellow")

    # Handle ingestion
    all_docs = []

    if args.ingest:
        docs = cli.ingest_files(args.ingest, show_progress=not args.no_progress)
        all_docs.extend(docs)

    if args.rss:
        with console.status(f"[bold green]Fetching RSS feed..."):
            try:
                docs = cli.ingester.ingest_rss(args.rss)
                all_docs.extend(docs)
                console.print(f"✅ Fetched {len(docs)} articles from RSS")
            except Exception as e:
                console.print(f"❌ RSS Error: {str(e)}", style="red")

    if args.url:
        with console.status(f"[bold green]Fetching web page..."):
            try:
                docs = cli.ingester.ingest_url(args.url)
                all_docs.extend(docs)
                console.print(f"✅ Fetched web page")
            except Exception as e:
                console.print(f"❌ URL Error: {str(e)}", style="red")

    # Build database if we have new documents
    if all_docs:
        cli.build_database(all_docs)

    # Handle query
    if args.query:
        result = cli.query(args.query, args.show_sources)

        # Export if requested
        if args.export and result:
            format = "markdown" if args.export.endswith('.md') else "json"
            cli.export_conversation(args.export, format)

    # Enter interactive mode if no specific action
    elif not any([args.ingest, args.rss, args.url, args.query]):
        cli.interactive_mode()


if __name__ == "__main__":
    main()