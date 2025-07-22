"""
Documentation generator module.
"""

import os
import time
from typing import Optional, List
from pathlib import Path
from rich.console import Console
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
        
        prompt = f"""
        Generate documentation for the following file: {file_path}
        
        File context:
        ```
        {context}
        ```
        
        Your documentation should include ONLY these sections:
        1. A clear description of the file's purpose and functionality
        2. Key components, classes, and functions. Do not quote code unless necessary.
        3. How this file interacts with other parts of the system
        
        Format the output as markdown with appropriate headings and code examples where relevant.
        
        CRITICAL INSTRUCTION: This is PURE DOCUMENTATION, not a code review. 
        - DO NOT include ANY sections titled "Improvement Suggestions", "Recommendations", or similar
        - DO NOT suggest any changes or improvements to the codebase
        - DO NOT critique the code quality, organization, or structure
        - DO NOT mention what "could be" or "should be" done differently
        - ONLY describe what exists in the codebase as it is currently implemented
        - FOCUS EXCLUSIVELY on factual description and explanation
        """
        
        # Generate documentation using the custom prompt
        response = self.llm_service.llm.invoke([
            {"role": "system", "content": "You are a technical documentation expert. Your task is to create clear, factual documentation for a code file."},
            {"role": "user", "content": prompt}
        ])
        
        # Extract content from response
        file_doc = self.llm_service._extract_response_content(response)
        
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
            
        # Skip README files
        if os.path.basename(file_path).lower() == 'readme.md':
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
        Create a comprehensive high-level overview documentation of this project based on the following information.
        Focus on explaining the project's purpose, architecture, and how components work together.
        
        Repository structure analysis:
        {structure_analysis}
        
        Key files content:
        {context}
        
        Your documentation should include ONLY these sections:
        1. Project Purpose and Main Functionality
        2. Architecture Overview (with component relationships)
        3. Key Workflows and Processes
        4. Technology Stack
        
        Format the output as a well-structured markdown document with appropriate headings.
        Start with a main heading '# Project Overview' followed by relevant subheadings.
        Make sure the overview is comprehensive but concise, focusing on the most important aspects of the project.
        
        CRITICAL INSTRUCTION: This is PURE DOCUMENTATION, not a code review. 
        - DO NOT include ANY sections titled "Improvement Suggestions", "Recommendations", or similar
        - DO NOT suggest any changes or improvements to the codebase
        - DO NOT critique the code quality, organization, or structure
        - DO NOT mention what "could be" or "should be" done differently
        - ONLY describe what exists in the codebase as it is currently implemented
        - FOCUS EXCLUSIVELY on factual description and explanation
        
        The documentation will be rejected if it contains any improvement suggestions or critiques.
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
