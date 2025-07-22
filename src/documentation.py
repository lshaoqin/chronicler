"""
Documentation generator module.
"""

import os
import difflib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from repository import Repository
from rag_system import RAGSystem
from llm_service import LLMService
from doc_storage import DocumentationStorage, DocSection


class DocumentationGenerator:
    """Class for generating documentation from repository analysis."""
    
    def __init__(self, repository: Repository, rag_system: RAGSystem, llm_service: LLMService, output_dir: Optional[Path] = None):
        """
        Initialize the documentation generator.
        
        Args:
            repository: Repository object
            rag_system: RAG system for retrieving context
            llm_service: LLM service for generating content
            output_dir: Output directory for documentation (default: repo_path/docs)
        """
        self.repository = repository
        self.rag_system = rag_system
        self.llm_service = llm_service
        self.console = Console()
        
        # Create timestamp for folder naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Extract repository name from URL or path
        repo_name = self._extract_repo_name(self.repository.repo_url_or_path)
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            # Use a fixed output directory in the current working directory with repo name and date
            # This ensures documentation is saved to the same location regardless of where the repository is cloned
            current_dir = os.getcwd()
            docs_dir = Path(os.path.join(current_dir, "docs"))
            docs_dir.mkdir(exist_ok=True, parents=True)
            
            # Create folder with repository name and date
            self.output_dir = Path(os.path.join(docs_dir, f"{repo_name}_{self.timestamp}"))
            self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize documentation storage directly in docs folder
        self.doc_storage = DocumentationStorage(str(self.output_dir))
        
    def create_documentation(self, file_extensions: Optional[List[str]] = None) -> str:
        """
        Create documentation for the repository by processing files incrementally.
        
        Args:
            file_extensions: List of file extensions to process (default: ['.py', '.js', '.ts', '.md'])
            
        Returns:
            Generated documentation as a string
        """
        if not file_extensions:
            file_extensions = ['.py', '.js', '.ts', '.md']
            
        # Start with repository structure analysis
        structure = self.repository.get_directory_structure()
        structure_analysis = self.llm_service.analyze_repository_structure(structure)
        
        # Generate a high-level project overview
        project_overview = self._generate_project_overview(structure_analysis)
        
        # Save high-level project overview
        overview_section = DocSection(
            file_path="__project_overview__",
            content=project_overview,
            metadata={"type": "project_overview", "priority": 1}
        )
        self.doc_storage.save_section(overview_section)
        
        # Save repository structure overview
        structure_section = DocSection(
            file_path="__structure_overview__",
            content=f"# Repository Structure\n\n{structure_analysis}",
            metadata={"type": "structure_overview", "priority": 2}
        )
        self.doc_storage.save_section(structure_section)
        
        # Get all relevant files for documentation
        files_to_process = []
        for ext in file_extensions:
            files_to_process.extend(self.repository.get_files_by_extension(ext))
            
        # Filter out less useful files like __init__.py
        files_to_process = [f for f in files_to_process if not self._should_skip_file(f)]
        
        # Process files incrementally with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Processing files...", total=len(files_to_process))
            
            for file_path in files_to_process:
                progress.update(task, description=f"Processing {file_path}")
                
                try:
                    # Get content and embed the file
                    content = self.repository.get_file_content(file_path)
                    
                    # Generate documentation for the file
                    doc_content = self._generate_file_documentation(file_path, content)
                    
                    # Save the documentation section
                    section = DocSection(
                        file_path=file_path,
                        content=doc_content,
                        metadata={"type": "file"}
                    )
                    self.doc_storage.save_section(section)
                    
                except Exception as e:
                    self.console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")
                    
                progress.update(task, advance=1)
        
        # Generate the full documentation
        full_doc = self.doc_storage.generate_full_documentation()
        
        # Save the full documentation
        self.save_documentation(full_doc, os.path.join(self.output_dir, "README.md"))
        
        return full_doc
    
    def update_documentation(self, commit_hash: Optional[str] = None) -> str:
        """
        Update documentation based on changes in a git commit.
        
        Args:
            commit_hash: Hash of the commit to analyze, or None for the latest commit
            
        Returns:
            Updated documentation as a string
        """
        # Get commit summary
        commit_summary = self.repository.get_commit_summary(commit_hash)
        self.console.print(f"Analyzing commit: {commit_summary['short_hash']} - {commit_summary['message']}")
        
        # Get changed files
        changed_files = commit_summary['changed_files']
        
        if not changed_files:
            self.console.print("[yellow]No relevant files changed in this commit.[/yellow]")
            return self.doc_storage.generate_full_documentation()
        
        # Process changed files with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Processing changed files...", total=len(changed_files))
            
            for file_path in changed_files:
                progress.update(task, description=f"Processing {file_path}")
                
                try:
                    # Get old and new content
                    old_content, new_content = self.repository.get_file_diff(file_path, commit_hash)
                    
                    # If file was deleted
                    if not new_content:
                        self.doc_storage.delete_section(file_path)
                        progress.update(task, advance=1)
                        continue
                    
                    # Update documentation for the file
                    self._update_file_documentation(file_path, old_content, new_content)
                    
                except Exception as e:
                    self.console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")
                    
                progress.update(task, advance=1)
        
        # Generate the full documentation
        full_doc = self.doc_storage.generate_full_documentation()
        
        # Save the full documentation
        self.save_documentation(full_doc, os.path.join(self.output_dir, "README.md"))
        
        return full_doc
    
    def _generate_file_documentation(self, file_path: str, content: str) -> str:
        """
        Generate documentation for a single file.
        
        Args:
            file_path: Path to the file
            content: Content of the file
            
        Returns:
            Generated documentation as a string
        """
        # Get context for the file
        context = self.rag_system.get_file_context(file_path)
        
        # Generate documentation for the file
        file_doc = self.llm_service.generate_documentation(context, file_path)
        
        return file_doc
    
    def _update_file_documentation(self, file_path: str, old_content: str, new_content: str) -> None:
        """
        Update documentation for a changed file.
        
        Args:
            file_path: Path to the file
            old_content: Previous content of the file
            new_content: New content of the file
        """
        # Get existing documentation section or create a new one
        section, is_new = self.doc_storage.get_section_for_update(file_path, "")
        
        # Get context for the file
        context = self.rag_system.get_file_context(file_path)
        
        # If it's a new file, generate documentation from scratch
        if is_new:
            doc_content = self._generate_file_documentation(file_path, new_content)
            section.content = doc_content
            self.doc_storage.save_section(section)
            return
        
        # For existing files, update the documentation based on changes
        prompt = f"""
        I need to update documentation for a file that has changed. Here is the relevant information:
        
        File path: {file_path}
        
        Current documentation:
        ```
        {section.content}
        ```
        
        File context:
        ```
        {context}
        ```
        
        Please update the documentation to reflect the current state of the file.
        Focus on accuracy and clarity. Preserve any useful information from the existing
        documentation that is still relevant. Add new information about any new features
        or changes. Remove information about features that no longer exist.
        
        Return ONLY the updated documentation content in markdown format.
        """
        
        # Generate updated documentation
        updated_doc = self.llm_service.llm.invoke([
            {"role": "system", "content": "You are a documentation expert. Your task is to update existing documentation based on changes to a file."},
            {"role": "user", "content": prompt}
        ])
        
        # Extract content from response
        updated_content = self.llm_service._extract_response_content(updated_doc)
        
        # Update the section
        section.content = updated_content
        section.last_updated = time.time()
        self.doc_storage.save_section(section)
        
    def generate(self) -> str:
        """
        Generate documentation for the repository (legacy method).
        
        Returns:
            Generated documentation as a string
        """
        self.console.print("[yellow]Warning: Using legacy documentation generation method.[/yellow]")
        self.console.print("[yellow]Consider using create_documentation() instead for better results.[/yellow]")
        
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
        
        # Regenerate documentation with improvements
        improved_docs = self.llm_service.regenerate_documentation(current_docs, context)
        
        # Generate diff using cydifflib
        diff_html = self._generate_diff_html(current_docs, improved_docs)
        
        return f"""# Documentation Improvement Suggestions

## Current Documentation Analysis

{suggestions}

## Diff Between Original and Improved Documentation

{diff_html}

## Improved Documentation

```markdown
{improved_docs}
```
"""
    
    def _generate_diff_html(self, original_docs: str, improved_docs: str) -> str:
        """
        Generate HTML diff between original and improved documentation using cydifflib.
        
        Args:
            original_docs: Original documentation content
            improved_docs: Improved documentation content
            
        Returns:
            HTML diff as a string
        """
        import cydifflib
        from pygments import highlight
        from pygments.lexers import DiffLexer
        from pygments.formatters import HtmlFormatter
        import html
        
        # Split the documents into lines
        original_lines = original_docs.splitlines()
        improved_lines = improved_docs.splitlines()
        
        # Generate unified diff
        diff = cydifflib.unified_diff(
            original_lines,
            improved_lines,
            lineterm='',
            fromfile='Original Documentation',
            tofile='Improved Documentation',
            n=3  # Context lines
        )
        
        # Convert diff to string
        diff_text = '\n'.join(list(diff))
        
        # Highlight the diff using Pygments
        formatter = HtmlFormatter(style='colorful')
        highlighted_diff = highlight(diff_text, DiffLexer(), formatter)
        
        # Add CSS for the diff
        css = formatter.get_style_defs('.highlight')
        
        # Create the HTML output
        html_diff = f"""
        <style>
        {css}
        .diff-container {{
            font-family: monospace;
            white-space: pre-wrap;
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f8f8f8;
        }}
        </style>
        <div class="diff-container">
        {highlighted_diff}
        </div>
        """
        
        return html_diff
    
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
    
    def _extract_repo_name(self, repo_url_or_path: str) -> str:
        """Extract repository name from URL or path."""
        if repo_url_or_path.endswith('/'):
            repo_url_or_path = repo_url_or_path[:-1]
            
        if '/' in repo_url_or_path:
            return repo_url_or_path.split('/')[-1]
        else:
            return os.path.basename(os.path.abspath(repo_url_or_path))
            
    def _should_skip_file(self, file_path: str) -> bool:
        """
        Determine if a file should be skipped in documentation generation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be skipped, False otherwise
        """
        # Skip __init__.py files
        if os.path.basename(file_path) == '__init__.py':
            return True
            
        # Skip test files
        if '/tests/' in file_path or file_path.endswith('_test.py') or file_path.endswith('test_.py'):
            return True
            
        # Skip configuration files
        if os.path.basename(file_path) in ['.gitignore', '.env', '.env.example', 'pyproject.toml', 'setup.cfg']:
            return True
            
        # Skip cache files and directories
        if '__pycache__' in file_path or file_path.endswith('.pyc'):
            return True
            
        return False
            
    def _generate_project_overview(self, structure_analysis: str) -> str:
        """
        Generate a high-level project overview.
        
        Args:
            structure_analysis: Analysis of the repository structure
            
        Returns:
            High-level project overview as a string
        """
        # Get key files for context
        key_files = []
        
        # Try to find important files in the repository
        for file_path in self.repository.get_files_by_extension('.py'):
            if file_path.endswith('main.py') or \
               'app.py' in file_path or \
               'core.py' in file_path or \
               file_path.endswith('__main__.py'):
                key_files.append(file_path)
        
        # If we didn't find any key files, get the first few Python files
        if not key_files:
            key_files = self.repository.get_files_by_extension('.py')[:5]
        
        # Get README if it exists
        readme_files = [f for f in self.repository.get_files_by_extension('.md') 
                       if os.path.basename(f).lower() == 'readme.md']
        if readme_files:
            key_files.append(readme_files[0])
                
        # Build context from key files
        context = ""
        for file_path in key_files:
            try:
                file_content = self.repository.get_file_content(file_path)
                # Limit content length to avoid token limits
                content_preview = file_content[:500] + "..." if len(file_content) > 500 else file_content
                context += f"\n\n## {file_path}\n```\n{content_preview}\n```\n"
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not read {file_path}: {str(e)}[/yellow]")
                
        # Generate the overview using LLM
        prompt = f"""
        Create a comprehensive high-level overview of this project based on the following information.
        Focus on explaining the project's purpose, architecture, and how components work together.
        
        Repository structure analysis:
        {structure_analysis}
        
        Key files content:
        {context}
        
        Your overview should include:
        1. Project Purpose and Main Functionality
        2. Architecture Overview (with component relationships)
        3. Key Workflows and Processes
        4. Technology Stack
        
        Format the output as a well-structured markdown document with appropriate headings.
        Start with a main heading '# Project Overview' followed by relevant subheadings.
        Make sure the overview is comprehensive but concise, focusing on the most important aspects of the project.
        """
        
        # Generate overview using LLM
        self.console.print("[bold]Generating project overview...[/bold]")
        overview_response = self.llm_service.llm.invoke([
            {"role": "system", "content": "You are a technical documentation expert. Your task is to create a high-level project overview that explains the architecture and functionality of a software project."},
            {"role": "user", "content": prompt}
        ])
        
        # Extract content from response
        overview_content = self.llm_service._extract_response_content(overview_response)
        
        return overview_content
    
    def save_documentation(self, content: str, output_path: str) -> None:
        """Save documentation to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.console.print(f"[green]Documentation saved to {output_path}[/green]")
