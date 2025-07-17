#!/usr/bin/env python3
"""
Chronicler: A tool that analyzes GitHub repositories and generates documentation.
"""

import os
import sys
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

# Local imports
from repository import Repository
from documentation import DocumentationGenerator
from llm_service import LLMService
from rag_system import RAGSystem

from dotenv import load_dotenv
load_dotenv()

# Initialize Typer app
app = typer.Typer(help="Chronicler: GitHub repository documentation generator")
console = Console()

@app.command()
def analyze(
    repo: str = typer.Argument(..., help="GitHub repository URL or local path"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for documentation"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="OpenAI API key"
    ),
    model_provider: str = typer.Option(
        "openai", "--provider", "-p", help="Model provider: 'openai', 'ollama', or 'local'"
    ),
    llm_model: str = typer.Option(
        "gpt-4o", "--llm-model", "-l", help="LLM model name (e.g., 'gpt-4o' for OpenAI, 'llama2' for Ollama)"
    ),
    embedding_model: str = typer.Option(
        "text-embedding-ada-002", "--embedding-model", "-e", 
        help="Embedding model name (e.g., 'text-embedding-ada-002' for OpenAI, 'llama2' for Ollama, 'all-MiniLM-L6-v2' for local)"
    ),
    temperature: float = typer.Option(
        0.2, "--temperature", "-t", help="Temperature for LLM generation (0.0-1.0)"
    ),
    ollama_host: Optional[str] = typer.Option(
        None, "--ollama-host", help="Ollama host URL (default: http://localhost:11434)"
    ),
):
    """Analyze a repository and generate documentation."""
    # Display welcome message
    console.print(
        Panel.fit(
            "🔍 [bold blue]Chronicler[/bold blue] - Repository Documentation Generator",
            border_style="blue",
        )
    )
    
    # Set environment variables if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    if ollama_host:
        os.environ["OLLAMA_HOST"] = ollama_host
    
    try:
        # Initialize components
        repository = Repository(repo)
        
        # Initialize LLM service with specified provider and models
        console.print(f"[bold]Using {model_provider} models:[/bold] LLM={llm_model}, Embeddings={embedding_model}")
        llm_service = LLMService(
            model_provider=model_provider,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            temperature=temperature
        )
        
        rag_system = RAGSystem(repository, llm_service)
        doc_generator = DocumentationGenerator(repository, rag_system, llm_service)
        
        # Process repository
        console.print("[bold]Analyzing repository...[/bold]")
        repository.clone_or_load()
        
        console.print("[bold]Building knowledge base...[/bold]")
        rag_system.build_knowledge_base()
        
        console.print("[bold]Generating documentation...[/bold]")
        documentation = doc_generator.generate()
        
        console.print("[bold]Suggesting improvements...[/bold]")
        suggestions = doc_generator.suggest_improvements()
        
        # Output results
        if output_dir:
            output_path = output_dir
        else:
            output_path = Path(os.path.join(repository.repo_path, "docs"))
            
        output_path.mkdir(exist_ok=True, parents=True)
        
        with open(os.path.join(output_path, "README.md"), "w") as f:
            f.write(documentation)
            
        with open(os.path.join(output_path, "suggestions.md"), "w") as f:
            f.write(suggestions)
            
        console.print(f"[bold green]Documentation generated successfully at {output_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return 1
        
    return 0

if __name__ == "__main__":
    app()
