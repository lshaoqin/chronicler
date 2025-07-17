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
):
    """Analyze a repository and generate documentation."""
    # Display welcome message
    console.print(
        Panel.fit(
            "🔍 [bold blue]Chronicler[/bold blue] - Repository Documentation Generator",
            border_style="blue",
        )
    )
    
    # Set API key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        # Initialize components
        repository = Repository(repo)
        llm_service = LLMService()
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
