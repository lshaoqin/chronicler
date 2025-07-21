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
        None, "--api-key", "-k", help="OpenAI API key"
    ),
    "model_provider": typer.Option(
        "openai", "--provider", "-p", help="Model provider: 'openai', 'ollama', or 'local'"
    ),
    "llm_model": typer.Option(
        "gpt-4o", "--llm-model", "-l", help="LLM model name (e.g., 'gpt-4o' for OpenAI, 'llama2' for Ollama)"
    ),
    "embedding_model": typer.Option(
        "text-embedding-ada-002", "--embedding-model", "-e", 
        help="Embedding model name (e.g., 'text-embedding-ada-002' for OpenAI, 'llama2' for Ollama, 'all-MiniLM-L6-v2' for local)"
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
    model_provider: str = common_params["model_provider"],
    llm_model: str = common_params["llm_model"],
    embedding_model: str = common_params["embedding_model"],
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
    
    # Parse file extensions if provided
    extensions_list = None
    if file_extensions:
        extensions_list = [f".{ext.strip()}" for ext in file_extensions.split(",")]
    
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
    model_provider: str = common_params["model_provider"],
    llm_model: str = common_params["llm_model"],
    embedding_model: str = common_params["embedding_model"],
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


@app.command()
def analyze(
    repo: str = common_params["repo"],
    output_dir: Optional[Path] = common_params["output_dir"],
    api_key: Optional[str] = common_params["api_key"],
    model_provider: str = common_params["model_provider"],
    llm_model: str = common_params["llm_model"],
    embedding_model: str = common_params["embedding_model"],
    temperature: float = common_params["temperature"],
    ollama_host: Optional[str] = common_params["ollama_host"],
):
    """Legacy command: Analyze a repository and generate documentation."""
    
    console.print(
        Panel.fit(
            "⚠️ [bold yellow]Warning: 'analyze' is a legacy command[/bold yellow]\n"
            "Consider using 'create' or 'update' commands instead.",
            border_style="yellow",
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
        doc_generator = DocumentationGenerator(repository, rag_system, llm_service, output_dir)
        
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
        with open(os.path.join(doc_generator.output_dir, "README.md"), "w") as f:
            f.write(documentation)
            
        with open(os.path.join(doc_generator.output_dir, "suggestions.md"), "w") as f:
            f.write(suggestions)
            
        console.print(f"[bold green]Documentation generated successfully at {doc_generator.output_dir}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return 1
        
    return 0

if __name__ == "__main__":
    app()
