"""
Documentation generator module.
"""

import os
import difflib
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from repository import Repository
from rag_system import RAGSystem
from llm_service import LLMService


class DocumentationGenerator:
    """Class for generating documentation from repository analysis."""
    
    def __init__(self, repository: Repository, rag_system: RAGSystem, llm_service: LLMService):
        """
        Initialize the documentation generator.
        
        Args:
            repository: Repository object
            rag_system: RAG system for retrieving context
            llm_service: LLM service for generating content
        """
        self.repository = repository
        self.rag_system = rag_system
        self.llm_service = llm_service
        self.console = Console()
        
    def generate(self) -> str:
        """
        Generate documentation for the repository.
        
        Returns:
            Generated documentation as a string
        """
        # Start with repository structure analysis
        structure = self.repository.get_directory_structure()
        structure_analysis = self.llm_service.analyze_repository_structure(structure)
        
        # Get relevant files for documentation
        python_files = self.repository.get_files_by_extension('.py')
        
        # Generate documentation sections
        sections = [
            "# Repository Documentation\n\n",
            "## Overview\n\n",
            structure_analysis,
            "\n\n## Key Components\n\n"
        ]
        
        # Process key files
        for file_path in python_files[:10]:  # Limit to 10 files to avoid API overuse
            self.console.print(f"Processing file: {file_path}")
            
            try:
                # Get context for the file
                context = self.rag_system.get_file_context(file_path)
                
                # Generate documentation for the file
                file_doc = self.llm_service.generate_documentation(context, file_path)
                
                # Add to sections
                sections.append(f"### {file_path}\n\n")
                sections.append(file_doc)
                sections.append("\n\n")
            except Exception as e:
                self.console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")
        
        # Combine all sections
        return "".join(sections)
    
    def suggest_improvements(self) -> str:
        """
        Suggest improvements to the existing documentation.
        
        Returns:
            Suggestions as a string
        """
        # Get current README content
        current_docs = self.repository.readme_content
        
        if not current_docs:
            return "No existing README.md found to suggest improvements for."
        
        # Get context from repository
        context = ""
        
        # Get context from key files
        python_files = self.repository.get_files_by_extension('.py')
        for file_path in python_files[:5]:  # Limit to 5 files
            try:
                file_context = self.rag_system.get_file_context(file_path)
                context += f"\n\n## {file_path}\n\n{file_context}"
            except Exception:
                pass
        
        # Generate suggestions
        suggestions = self.llm_service.suggest_documentation_improvements(current_docs, context)
        
        # Generate an improved version
        improved_docs = self._generate_improved_docs(current_docs, suggestions)
        
        # Generate diff
        diff = self.llm_service.generate_diff(current_docs, improved_docs)
        
        return f"""# Documentation Improvement Suggestions

## Current Documentation Analysis

{suggestions}

## Suggested Improvements

{diff}

## Improved Documentation

```markdown
{improved_docs}
```
"""
    
    def _generate_improved_docs(self, current_docs: str, suggestions: str) -> str:
        """
        Generate an improved version of the documentation based on suggestions.
        
        Args:
            current_docs: Current documentation
            suggestions: Improvement suggestions
            
        Returns:
            Improved documentation
        """
        prompt = f"""
        Here is the current documentation:
        
        ```
        {current_docs}
        ```
        
        Here are suggestions for improvement:
        
        ```
        {suggestions}
        ```
        
        Please create an improved version of the documentation that incorporates these suggestions.
        Only output the improved documentation content in markdown format, nothing else.
        """
        
        # Use the LLM to generate the improved version
        improved = self.llm_service.llm.invoke([
            {"role": "system", "content": "You are a documentation expert. Your task is to improve existing documentation based on suggestions."},
            {"role": "user", "content": prompt}
        ])
        
        return self.llm_service._extract_response_content(improved)
    
    def display_diff(self, original: str, improved: str) -> None:
        """
        Display a diff between original and improved documentation in the console.
        
        Args:
            original: Original documentation
            improved: Improved documentation
        """
        diff = difflib.unified_diff(
            original.splitlines(),
            improved.splitlines(),
            lineterm='',
            n=3
        )
        
        diff_text = '\n'.join(diff)
        
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        
        self.console.print(Panel(
            syntax,
            title="Documentation Diff",
            border_style="green"
        ))
    
    def save_documentation(self, content: str, output_path: str) -> None:
        """
        Save documentation to a file.
        
        Args:
            content: Documentation content
            output_path: Path to save the documentation
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.console.print(f"[green]Documentation saved to {output_path}[/green]")
