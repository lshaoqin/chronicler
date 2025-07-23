#!/usr/bin/env python3
"""
Chronicler: A tool that analyzes GitHub repositories and generates documentation.

Commands:
- create: Generate documentation from scratch by analyzing repository files
- update: Update existing documentation based on git commit changes
"""

import os
import typer
from pathlib import Path
from typing import Optional, List
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

# Common parameters for commands
common_params = {
    "repo": typer.Argument(..., help="GitHub repository URL or local path"),
    "output_dir": typer.Option(
        None, "--output", "-o", help="Output directory for documentation"
    ),
    "api_key": typer.Option(
        None, "--api-key", "-k", help="API key for the selected provider (OpenAI or Google)"
    ),
    "llm_provider": typer.Option(
        None, "--llm-provider", "--provider", "-p", help="LLM provider: 'openai', 'gemini', 'ollama', or 'local' (defaults to LLM_PROVIDER env var or MODEL_PROVIDER env var or 'openai')"
    ),
    "embedding_provider": typer.Option(
        None, "--embedding-provider", "-ep", help="Embedding provider: 'openai', 'gemini', 'ollama', or 'local' (defaults to EMBEDDING_PROVIDER env var or MODEL_PROVIDER env var or 'openai')"
    ),
    "llm_model": typer.Option(
        None, "--llm-model", "-l", help="LLM model name (e.g., 'gpt-4o' for OpenAI, 'llama2' for Ollama) (defaults to LLM_MODEL env var)"
    ),
    "embedding_model": typer.Option(
        None, "--embedding-model", "-e", 
        help="Embedding model name (e.g., 'text-embedding-ada-002' for OpenAI, 'llama2' for Ollama) (defaults to EMBEDDING_MODEL env var)"
    ),
    "temperature": typer.Option(
        0.2, "--temperature", "-t", help="Temperature for LLM generation (0.0-1.0)"
    ),
    "ollama_host": typer.Option(
        None, "--ollama-host", help="Ollama host URL (default: http://localhost:11434)"
    ),
}

@app.command()
def create(
    repo: str = common_params["repo"],
    output_dir: Optional[Path] = common_params["output_dir"],
    api_key: Optional[str] = common_params["api_key"],
    llm_provider: Optional[str] = common_params["llm_provider"],
    embedding_provider: Optional[str] = common_params["embedding_provider"],
    llm_model: Optional[str] = common_params["llm_model"],
    embedding_model: Optional[str] = common_params["embedding_model"],
    temperature: float = common_params["temperature"],
    ollama_host: Optional[str] = common_params["ollama_host"],
    file_extensions: Optional[List[str]] = typer.Option(
        None, "--extensions", "-x", help="File extensions to process (e.g., py,js,md)"
    ),
):
    """Create documentation from scratch by analyzing repository files."""
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
    
    # Set model configuration if specified via command line
    if llm_provider:
        os.environ["LLM_PROVIDER"] = llm_provider
        
    if embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = embedding_provider
    
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
    
    if embedding_model:
        os.environ["EMBEDDING_MODEL"] = embedding_model
    
    # Get the effective providers and models
    effective_llm_provider = os.environ.get("LLM_PROVIDER", os.environ.get("MODEL_PROVIDER", "openai"))
    effective_embedding_provider = os.environ.get("EMBEDDING_PROVIDER", os.environ.get("MODEL_PROVIDER", "openai"))
    effective_llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
    effective_embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Parse file extensions if provided
    extensions_list = None
    if file_extensions:
        extensions_list = [f".{ext.strip()}" for ext in file_extensions.split(",")]
    
    try:
        # Initialize components
        repository = Repository(repo)
        
        # Initialize LLM service with specified providers and models
        console.print(f"[bold]Using providers:[/bold] LLM={effective_llm_provider} ({effective_llm_model}), Embeddings={effective_embedding_provider} ({effective_embedding_model})")
        llm_service = LLMService(
            llm_provider=effective_llm_provider,
            embedding_provider=effective_embedding_provider,
            temperature=temperature
        )
        
        # Process repository
        console.print("[bold]Cloning/loading repository...[/bold]")
        repository.clone_or_load()
        
        # Initialize RAG system and documentation generator
        rag_system = RAGSystem(repository, llm_service)
        doc_generator = DocumentationGenerator(repository, rag_system, llm_service, output_dir)
        
        console.print("[bold]Building knowledge base...[/bold]")
        rag_system.build_knowledge_base()
        
        console.print("[bold]Creating documentation...[/bold]")
        documentation = doc_generator.create_documentation(extensions_list)
        
        console.print(f"[bold green]Documentation created successfully at {doc_generator.output_dir}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return 1
        
    return 0


@app.command()
def update(
    repo: str = common_params["repo"],
    output_dir: Optional[Path] = common_params["output_dir"],
    api_key: Optional[str] = common_params["api_key"],
    llm_provider: Optional[str] = common_params["llm_provider"],
    embedding_provider: Optional[str] = common_params["embedding_provider"],
    llm_model: Optional[str] = common_params["llm_model"],
    embedding_model: Optional[str] = common_params["embedding_model"],
    temperature: float = common_params["temperature"],
    ollama_host: Optional[str] = common_params["ollama_host"],
    commit: Optional[str] = typer.Option(
        None, "--commit", "-c", help="Git commit hash to analyze (default: latest commit)"
    ),
):
    """Update documentation based on git commit changes."""
    
    # Display welcome message
    console.print(
        Panel.fit(
            "🔄 [bold blue]Chronicler[/bold blue] - Documentation Update",
            border_style="blue",
        )
    )
    
    # Set environment variables if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    if ollama_host:
        os.environ["OLLAMA_HOST"] = ollama_host
    
    # Set model configuration if specified via command line
    if llm_provider:
        os.environ["LLM_PROVIDER"] = llm_provider
        
    if embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = embedding_provider
    
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
    
    if embedding_model:
        os.environ["EMBEDDING_MODEL"] = embedding_model
    
    # Get the effective providers and models
    effective_llm_provider = os.environ.get("LLM_PROVIDER", os.environ.get("MODEL_PROVIDER", "openai"))
    effective_embedding_provider = os.environ.get("EMBEDDING_PROVIDER", os.environ.get("MODEL_PROVIDER", "openai"))
    effective_llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
    effective_embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    try:
        # Initialize components
        repository = Repository(repo)
        
        # Initialize LLM service with specified providers and models
        console.print(f"[bold]Using providers:[/bold] LLM={effective_llm_provider} ({effective_llm_model}), Embeddings={effective_embedding_provider} ({effective_embedding_model})")
        llm_service = LLMService(
            llm_provider=effective_llm_provider,
            embedding_provider=effective_embedding_provider,
            temperature=temperature
        )
        
        # Process repository
        console.print("[bold]Cloning/loading repository...[/bold]")
        repository.clone_or_load()
        
        # Initialize RAG system and documentation generator
        rag_system = RAGSystem(repository, llm_service)
        doc_generator = DocumentationGenerator(repository, rag_system, llm_service, output_dir)
        
        console.print("[bold]Building knowledge base...[/bold]")
        rag_system.build_knowledge_base()
        
        console.print("[bold]Updating documentation based on commit changes...[/bold]")
        documentation = doc_generator.update_documentation(commit)
        
        console.print(f"[bold green]Documentation updated successfully at {doc_generator.output_dir}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return 1
        
    return 0


if __name__ == "__main__":
    app()
