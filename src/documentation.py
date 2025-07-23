"""
Documentation generator module.
"""

import os
import time
from typing import Optional, List, Dict
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
        
    def _analyze_codebase_sections(self, files: List[str]) -> Dict[str, List[str]]:
        """
        Analyze the codebase and group files into logical documentation sections.
        
        Args:
            files: List of file paths to analyze
            
        Returns:
            Dictionary mapping section names to lists of file paths
        """
        # Get overall project structure and purpose
        structure_analysis = self.repository.get_directory_structure()
        
        # Create context for section planning
        files_info = []
        for file_path in files[:10]:  # Limit to avoid token limits
            try:
                content = self.repository.get_file_content(file_path)
                if content and len(content.strip()) > 0:
                    # Get first few lines and key information
                    preview = content[:500] + "..." if len(content) > 500 else content
                    files_info.append(f"File: {file_path}\nPreview: {preview}\n")
            except Exception:
                files_info.append(f"File: {file_path}\n(Could not read content)\n")
        
        files_context = "\n".join(files_info)
        
        prompt = f"""
        I need to create coherent documentation sections for a software project. Instead of documenting each file separately, I want to group related files into logical sections.
        
        Project structure analysis:
        ```
        {structure_analysis}
        ```
        
        Files to document:
        ```
        {files_context}
        ```
        
        Please analyze these files and group them into logical documentation sections. Consider:
        1. Functional groupings (e.g., "Authentication", "Data Processing", "API Endpoints")
        2. Architectural layers (e.g., "Core Components", "Utilities", "Configuration")
        3. Module relationships and dependencies
        4. User-facing vs internal components
        
        Respond with a JSON object where keys are section names and values are arrays of file paths:
        {{
            "Section Name 1": ["file1.py", "file2.py"],
            "Section Name 2": ["file3.py"],
            ...
        }}
        
        Make section names descriptive and user-friendly. Ensure every file is assigned to exactly one section.
        """
        
        # Generate section plan using LLM
        response = self.llm_service.llm.invoke([
            {"role": "system", "content": "You are an expert technical writer who specializes in creating well-organized documentation. You analyze codebases and group related files into coherent documentation sections."},
            {"role": "user", "content": prompt}
        ])
        
        # Extract and parse JSON response
        try:
            content = self.llm_service._extract_response_content(response)
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
            if json_match:
                sections = json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire content as JSON
                sections = json.loads(content)
                
            # Validate that all files in the section plan actually exist
            valid_files = set(files)
            validated_sections = {}
            
            for section_name, file_list in sections.items():
                # Only include files that actually exist in our file list
                existing_files = [f for f in file_list if f in valid_files]
                if existing_files:  # Only create section if it has valid files
                    validated_sections[section_name] = existing_files
            
            # Validate that all original files are assigned
            assigned_files = set()
            for file_list in validated_sections.values():
                assigned_files.update(file_list)
            
            # Add any unassigned files to a "Miscellaneous" section
            unassigned = valid_files - assigned_files
            if unassigned:
                validated_sections["Miscellaneous"] = list(unassigned)
                
            return validated_sections
            
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not parse section plan ({str(e)}). Using default grouping.[/yellow]")
            # Fallback: group by directory structure
            return self._group_files_by_directory(files)
    
    def _group_files_by_directory(self, files: List[str]) -> Dict[str, List[str]]:
        """
        Fallback method to group files by directory structure.
        
        Args:
            files: List of file paths
            
        Returns:
            Dictionary mapping section names to file lists
        """
        sections = {}
        for file_path in files:
            # Get directory name or use root for files in project root
            parts = file_path.split('/')
            if len(parts) > 1:
                section_name = f"{parts[-2].title()} Module"  # Use parent directory
            else:
                section_name = "Core Files"
                
            if section_name not in sections:
                sections[section_name] = []
            sections[section_name].append(file_path)
            
        return sections
    
    def create_documentation(self, file_extensions: Optional[List[str]] = None) -> str:
        """
        Create documentation for the repository by processing files in logical sections.
        
        Args:
            file_extensions: List of file extensions to process (default: ['.py', '.js', '.ts', '.md'])
            
        Returns:
            Generated documentation as a string
        """
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.md']
        
        # Get all files to process
        files_to_process = []
        for root, dirs, files in os.walk(self.repository.repo_path):
            # Skip hidden directories and common build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'build', 'dist']]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repository.repo_path)
                
                if (any(file.endswith(ext) for ext in file_extensions) and 
                    not self._should_skip_file(rel_path)):
                    files_to_process.append(rel_path)
        
        if not files_to_process:
            return "No files found to document."
        
        # Generate overall project overview first
        structure_analysis = self.repository.get_directory_structure()
        overview = self._generate_project_overview(structure_analysis)
        
        # Store the overview as a special section
        overview_section = DocSection(
            file_path="_overview",
            content=overview
        )
        self.doc_storage.save_section(overview_section)
        
        # Analyze codebase and create section plan
        self.console.print("[blue]Planning documentation sections...[/blue]")
        section_plan = self._analyze_codebase_sections(files_to_process)
        
        # Display section plan to user
        self.console.print("\n[green]Documentation sections planned:[/green]")
        for section_name, files in section_plan.items():
            self.console.print(f"  • {section_name}: {len(files)} files")
        self.console.print()
        
        # Process sections with progress bar
        generated_sections = {}  # Track generated sections for RAG integration
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Generating section documentation...", total=len(section_plan))
            
            for section_name, files_in_section in section_plan.items():
                progress.update(task, description=f"Processing {section_name}")
                
                try:
                    # Generate documentation for the entire section with awareness of other sections
                    section_doc = self._generate_section_documentation(
                        section_name, files_in_section, generated_sections, section_plan
                    )
                    
                    # Store documentation section
                    section = DocSection(
                        file_path=f"_section_{section_name.lower().replace(' ', '_')}",
                        content=section_doc,
                        metadata={"section_name": section_name, "files": files_in_section}
                    )
                    self.doc_storage.save_section(section)
                    
                    # Add this section to our generated sections for future reference
                    generated_sections[section_name] = {
                        "content": section_doc[:1500] + "..." if len(section_doc) > 1500 else section_doc,  # Truncated for context
                        "files": files_in_section,
                        "summary": self._extract_section_summary(section_doc)
                    }
                    
                    # Note: RAGSystem doesn't support dynamic document addition during generation
                    # Cross-section awareness is achieved through the generated_sections tracking
                    # and context sharing between sections
                    
                except Exception as e:
                    self.console.print(f"[red]Error processing section {section_name}: {str(e)}[/red]")
                    
                progress.update(task, advance=1)
        
        # Generate the full documentation
        full_doc = self.doc_storage.generate_full_documentation()
        
        # Save the full documentation
        self.save_documentation(full_doc, os.path.join(self.output_dir, "README.md"))
        
        return full_doc
    
    def _generate_section_documentation(self, section_name: str, files_in_section: List[str], 
                                      generated_sections: Dict[str, Dict] = None, 
                                      all_sections: Dict[str, List[str]] = None) -> str:
        """
        Generate documentation for a logical section containing multiple related files,
        with awareness of other sections in the documentation.
        
        Args:
            section_name: Name of the documentation section
            files_in_section: List of file paths in this section
            generated_sections: Dictionary of previously generated sections for cross-reference
            all_sections: Complete section plan for context awareness
            
        Returns:
            Generated documentation for the section
        """
        # Collect content and context for all files in the section
        files_content = []
        files_context = []
        
        for file_path in files_in_section:
            try:
                content = self.repository.get_file_content(file_path)
                if content and len(content.strip()) > 0:
                    files_content.append({
                        "path": file_path,
                        "content": content[:2000] + "..." if len(content) > 2000 else content  # Truncate to avoid token limits
                    })
                    
                    # Get RAG context for each file
                    context = self.rag_system.get_file_context(file_path)
                    files_context.append({
                        "path": file_path,
                        "context": context[:1000] + "..." if len(context) > 1000 else context
                    })
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not read {file_path}: {str(e)}[/yellow]")
                continue
        
        if not files_content:
            return f"# {section_name}\n\nNo readable files found in this section."
        
        # Query RAG system for related documentation content from existing codebase
        related_docs_context = ""
        try:
            # Query for related content using section name and file paths
            query_text = f"{section_name} {' '.join(files_in_section)}"
            related_context = self.rag_system.get_relevant_context(query_text, k=3)
            
            if related_context and len(related_context.strip()) > 0:
                related_docs_context = f"\n\nRelated context from codebase:\n{related_context[:800]}..."
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not query RAG for section context: {str(e)}[/yellow]")
        
        # Build context about other sections in the documentation
        other_sections_context = ""
        if generated_sections:
            other_sections_info = []
            for other_name, other_info in generated_sections.items():
                if other_name != section_name:
                    other_sections_info.append(f"- {other_name}: {other_info['summary']} (covers {len(other_info['files'])} files)")
            
            if other_sections_info:
                other_sections_context = f"\n\nOther sections in this documentation:\n" + "\n".join(other_sections_info)
        
        # Build section plan context
        section_plan_context = ""
        if all_sections:
            remaining_sections = [name for name in all_sections.keys() if name not in (generated_sections.keys() if generated_sections else [])]
            if remaining_sections:
                section_plan_context = f"\n\nUpcoming sections to be documented: {', '.join(remaining_sections)}"
        
        # Build comprehensive prompt for section documentation
        files_info = "\n\n".join([
            f"File: {fc['path']}\n```\n{fc['content']}\n```" 
            for fc in files_content
        ])
        
        context_info = "\n\n".join([
            f"Context for {ctx['path']}:\n{ctx['context']}"
            for ctx in files_context
        ])
        
        prompt = f"""
        Generate comprehensive documentation for the "{section_name}" section of this codebase.
        This section contains {len(files_in_section)} related files that work together.
        
        IMPORTANT: This section is part of a larger documentation. Be aware of other sections and reference them when appropriate.
        
        Files in this section:
        {files_info}
        
        Additional context from codebase analysis:
        {context_info}
        
        Related documentation sections:
        {related_docs_context}
        {other_sections_context}
        {section_plan_context}
        
        Create coherent documentation that:
        1. Starts with a clear overview of what this section does and its role in the project
        2. Explains how the files work together and their relationships
        3. Describes the key functionality, classes, and methods across all files
        4. Highlights important patterns, interfaces, or architectural decisions
        5. Explains the data flow and interactions between components
        6. References other sections when there are dependencies or relationships (use format: "See [Section Name] for details")
        7. Avoids duplicating information already covered in other sections
        9. Notes any configuration, dependencies, or setup requirements
        
        Write the documentation as a cohesive narrative that treats these files as parts of a unified system,
        rather than documenting each file separately. Focus on the bigger picture and how everything fits together.
        When referencing other sections, make it clear how they relate to this section.
        
        Use markdown formatting with appropriate headers, code blocks, and bullet points.
        Do not include a table of contents - start directly with the section content.
        """
        
        # Generate section documentation using LLM
        response = self.llm_service.llm.invoke([
            {"role": "system", "content": "You are an expert technical writer who specializes in creating comprehensive, coherent documentation that explains how multiple code files work together as a unified system."},
            {"role": "user", "content": prompt}
        ])
        
        # Extract content from response
        section_doc = self.llm_service._extract_response_content(response)
        
        return section_doc
    
    def _extract_section_summary(self, section_doc: str) -> str:
        """
        Extract a concise summary from section documentation for cross-section references.
        
        Args:
            section_doc: Full documentation content for a section
            
        Returns:
            Brief summary of the section's purpose and key components
        """
        # Try to extract the first paragraph or overview from the section
        lines = section_doc.split('\n')
        summary_lines = []
        
        # Skip headers and find the first substantial paragraph
        in_overview = False
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                in_overview = True
                continue
            if in_overview and not line.startswith('-') and not line.startswith('*') and not line.startswith('```'):
                summary_lines.append(line)
                if len(' '.join(summary_lines)) > 200:  # Keep summary reasonable length
                    break
        
        if summary_lines:
            summary = ' '.join(summary_lines)[:250] + ('...' if len(' '.join(summary_lines)) > 250 else '')
            return summary
        else:
            # Fallback: use section name as summary
            return f"Section containing files and functionality related to {section_doc.split()[0].lower() if section_doc else 'this module'}"
    
    def _load_existing_documentation(self) -> None:
        """
        Load existing documentation directly from the repository's README.md file.
        This allows update operations to preserve existing documentation while
        only updating sections for changed files.
        """
        try:
            # Look for README.md in the repository root
            readme_path = os.path.join(self.repository.repo_path, "README.md")
            
            if not os.path.exists(readme_path):
                self.console.print("[yellow]No existing README.md found. Starting fresh.[/yellow]")
                return
            
            # Read the existing README content
            with open(readme_path, 'r', encoding='utf-8') as f:
                existing_readme = f.read()
            
            if not existing_readme.strip():
                self.console.print("[yellow]README.md is empty. Starting fresh.[/yellow]")
                return
            
            self.console.print("[blue]Loading existing documentation from README.md[/blue]")
            
            # Store the existing README as a special section that will be used as the base
            # We'll create a special section to hold the existing content
            from doc_storage import DocSection
            existing_section = DocSection(
                file_path="__existing_readme__",
                content=existing_readme,
                metadata={"type": "existing_readme", "source": "README.md"}
            )
            self.doc_storage.save_section(existing_section)
            
            self.console.print("[green]Loaded existing README.md content[/green]")
            
        except Exception as e:
            self.console.print(f"[yellow]Could not load existing README.md: {str(e)}. Starting fresh.[/yellow]")
    
    def update_documentation(self, commit_hash: Optional[str] = None) -> str:
        """
        Update documentation based on changes in a git commit.
        
        Args:
            commit_hash: Hash of the commit to analyze, or None for the latest commit
            
        Returns:
            Updated documentation as a string
        """
        # First, try to load existing documentation from the most recent run
        self._load_existing_documentation()
        
        # Get commit summary
        commit_summary = self.repository.get_commit_summary(commit_hash)
        self.console.print(f"Analyzing commit: {commit_summary['short_hash']} - {commit_summary['message']}")
        
        # Get changed files
        changed_files = commit_summary['changed_files']
        
        if not changed_files:
            self.console.print("[yellow]No relevant files changed in this commit.[/yellow]")
            return self.doc_storage.generate_full_documentation()
        
        # Check if we have existing README content to work with
        existing_sections = self.doc_storage.get_all_sections()
        has_existing_readme = "__existing_readme__" in existing_sections
        
        if has_existing_readme:
            # Use smart update approach: preserve existing README and update only changed sections
            existing_readme = existing_sections["__existing_readme__"].content
            updated_readme = self._update_readme_selectively(existing_readme, changed_files, commit_hash)
            
            # Save the updated README
            self.save_documentation(updated_readme, os.path.join(self.output_dir, "README.md"))
            
            return updated_readme
        else:
            # No existing README, throw error
            raise ValueError("No existing README found. Please run the `create` command to create a README first.")
    
    def _update_readme_selectively(self, existing_readme: str, changed_files: List[str], commit_hash: Optional[str] = None) -> str:
        """
        Update the existing README by using RAG to find relevant sections and updating only those sections.
        
        Args:
            existing_readme: The current README content
            changed_files: List of files that have changed
            commit_hash: The commit hash being processed
            
        Returns:
            Updated README content with only relevant sections modified
        """
        self.console.print(f"[blue]Using RAG to selectively update README for {len(changed_files)} changed files[/blue]")
        
        # Filter significant files
        significant_files = [f for f in changed_files if not self._should_skip_file(f)]
        
        if not significant_files:
            self.console.print("[yellow]No significant files changed. Preserving existing README.[/yellow]")
            return existing_readme
        
        self.console.print(f"[blue]Found {len(significant_files)} significant files to analyze[/blue]")
        
        # Step 1: Parse the existing README into sections
        readme_sections = self._parse_readme_sections(existing_readme)
        self.console.print(f"[blue]Parsed {len(readme_sections)} sections from existing README[/blue]")
        
        # Step 2: For each changed file, use RAG to find relevant sections
        sections_to_update = set()
        file_changes_context = {}
        
        for file_path in significant_files:
            try:
                # Get the changes for this file
                old_content, new_content = self.repository.get_file_diff(file_path, commit_hash)
                
                if new_content:  # File exists (not deleted)
                    # Create context about what changed
                    change_context = f"File: {file_path}\nChanges: {self._summarize_file_changes(old_content, new_content)}"
                    file_changes_context[file_path] = {
                        'old_content': old_content,
                        'new_content': new_content,
                        'change_context': change_context
                    }
                    
                    # Use RAG to find relevant sections
                    relevant_sections = self._find_relevant_sections_with_rag(change_context, readme_sections)
                    sections_to_update.update(relevant_sections)
                    
                    self.console.print(f"[blue]File {file_path} affects {len(relevant_sections)} sections[/blue]")
                    
            except Exception as e:
                self.console.print(f"[red]Error analyzing {file_path}: {str(e)}[/red]")
        
        if not sections_to_update:
            self.console.print("[yellow]No relevant sections found to update. Preserving existing README.[/yellow]")
            return existing_readme
        
        self.console.print(f"[blue]Updating {len(sections_to_update)} sections based on file changes[/blue]")
        
        # Step 3: Update only the relevant sections
        updated_sections = {}
        for section_key in sections_to_update:
            try:
                updated_content = self._update_section_with_context(
                    readme_sections[section_key], 
                    file_changes_context, 
                    significant_files
                )
                updated_sections[section_key] = updated_content
                self.console.print(f"[green]Updated section: {section_key}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error updating section {section_key}: {str(e)}[/red]")
        
        # Step 4: Reconstruct the README with updated sections
        updated_readme = self._reconstruct_readme(readme_sections, updated_sections)
        
        self.console.print(f"[green]Successfully updated {len(updated_sections)} sections in README[/green]")
        return updated_readme
    
    def _parse_readme_sections(self, readme_content: str) -> Dict[str, str]:
        """
        Parse the README content into sections based on headers.
        
        Returns:
            Dictionary mapping section keys to section content
        """
        sections = {}
        lines = readme_content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if this is a header line
            if line.strip().startswith('#'):
                # Save the previous section if it exists
                if current_section is not None:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start a new section
                current_section = line.strip()
                current_content = [line]
            else:
                # Add to current section
                if current_section is not None:
                    current_content.append(line)
                else:
                    # Content before any headers - treat as intro
                    if 'intro' not in sections:
                        sections['intro'] = line
                    else:
                        sections['intro'] += '\n' + line
        
        # Don't forget the last section
        if current_section is not None:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _summarize_file_changes(self, old_content: Optional[str], new_content: str) -> str:
        """
        Create a summary of what changed in a file.
        
        Args:
            old_content: Previous file content (None if new file)
            new_content: Current file content
            
        Returns:
            Summary of changes
        """
        if old_content is None:
            return f"New file created with {len(new_content.splitlines())} lines"
        
        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines()
        
        # Simple change summary
        lines_added = len(new_lines) - len(old_lines)
        if lines_added > 0:
            return f"File modified: {lines_added} lines added, {len(new_lines)} total lines"
        elif lines_added < 0:
            return f"File modified: {abs(lines_added)} lines removed, {len(new_lines)} total lines"
        else:
            return f"File modified: content changed, {len(new_lines)} lines"
    
    def _find_relevant_sections_with_rag(self, change_context: str, readme_sections: Dict[str, str]) -> List[str]:
        """
        Use RAG with existing vector database embeddings to find which README sections are relevant to the file changes.
        
        Args:
            change_context: Description of what changed in the file
            readme_sections: Dictionary of README sections
            
        Returns:
            List of section keys that are relevant to the changes
        """
        if not readme_sections or not hasattr(self, 'rag_system') or not self.rag_system:
            # Fallback to keyword matching if RAG system not available
            return self._fallback_keyword_matching(change_context, readme_sections)
        
        try:
            # Use the existing RAG system to search for relevant content using embeddings
            # This leverages the already-built vector database with embeddings
            relevant_docs = self.rag_system.search(change_context, k=5)  # Reduced from 10 to 5 for more precision
            
            # Extract the sources (file paths) from relevant documents with relevance scores
            relevant_sources = set()
            high_relevance_sources = set()
            
            for i, doc in enumerate(relevant_docs):
                source = doc.metadata.get('source', '')
                if source:
                    relevant_sources.add(source)
                    # Consider top 2 results as high relevance
                    if i < 2:
                        high_relevance_sources.add(source)
            
            # Now find which README sections are most similar to the change context
            relevant_sections = []
            
            # For each README section, use more precise matching
            for section_key, section_content in readme_sections.items():
                section_lower = section_content.lower()
                is_relevant = False
                
                # Method 1: Check for high-relevance sources with stricter matching
                for source in high_relevance_sources:
                    filename = os.path.basename(source).lower()
                    
                    # More precise matching - look for the filename in context, not just anywhere
                    if (f"src/{filename}" in section_lower or 
                        f"`{filename}`" in section_lower or 
                        f"file: {filename}" in section_lower or
                        f"module: {filename}" in section_lower):
                        relevant_sections.append(section_key)
                        is_relevant = True
                        break
                
                # Method 2: Semantic similarity check with much higher threshold
                if not is_relevant:
                    try:
                        # Use only the change context for search, not section content
                        section_results = self.rag_system.search(change_context, k=3)
                        
                        # Check if this section's content appears in the top relevant results
                        for result in section_results:
                            result_content = result.page_content.lower()
                            section_words = set(section_content.lower().split())
                            result_words = set(result_content.split())
                            
                            # Much higher threshold for relevance - 40% overlap
                            if section_words and result_words:
                                overlap = len(section_words.intersection(result_words))
                                overlap_ratio = overlap / min(len(section_words), len(result_words))
                                
                                # Only consider highly overlapping content as relevant
                                if overlap_ratio > 0.4:  # Increased from 10% to 40%
                                    relevant_sections.append(section_key)
                                    is_relevant = True
                                    break
                    except Exception:
                        # If semantic search fails, skip this method
                        pass
                
                # Method 3: Check if the section specifically discusses the changed file's functionality
                if not is_relevant:
                    # Extract key terms from the change context
                    change_terms = [term.lower() for term in change_context.split() if len(term) > 4]
                    
                    # Count how many change-specific terms appear in the section
                    term_matches = sum(1 for term in change_terms if term in section_lower)
                    
                    # Only consider relevant if multiple specific terms match
                    if term_matches >= 3 and len(change_terms) > 0:
                        match_ratio = term_matches / len(change_terms)
                        if match_ratio > 0.5:  # At least 50% of change terms must be present
                            relevant_sections.append(section_key)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_relevant = []
            for section in relevant_sections:
                if section not in seen:
                    seen.add(section)
                    unique_relevant.append(section)
            
            self.console.print(f"[blue]RAG found {len(unique_relevant)} relevant sections using embeddings[/blue]")
            return unique_relevant
            
        except Exception as e:
            self.console.print(f"[yellow]Error in RAG embeddings search: {str(e)}. Using fallback method.[/yellow]")
            return self._fallback_keyword_matching(change_context, readme_sections)
    
    def _fallback_keyword_matching(self, change_context: str, readme_sections: Dict[str, str]) -> List[str]:
        """
        Fallback method using simple keyword matching when RAG is not available.
        
        Args:
            change_context: Description of what changed
            readme_sections: Dictionary of README sections
            
        Returns:
            List of section keys that match keywords
        """
        relevant_sections = []
        change_keywords = change_context.lower().split()
        
        for section_key, section_content in readme_sections.items():
            section_text = section_content.lower()
            # Check if any significant keywords from the change appear in the section
            if any(keyword in section_text for keyword in change_keywords if len(keyword) > 3):
                relevant_sections.append(section_key)
        
        self.console.print(f"[yellow]Fallback keyword matching found {len(relevant_sections)} relevant sections[/yellow]")
        return relevant_sections
    
    def _update_section_with_context(self, section_content: str, file_changes_context: Dict, changed_files: List[str]) -> str:
        """
        Update a specific README section based on the context of file changes.
        
        Args:
            section_content: Current content of the section
            file_changes_context: Dictionary of file changes with context
            changed_files: List of files that changed
            
        Returns:
            Updated section content
        """
        try:
            # Create context about all the changes
            changes_summary = "\n".join([
                f"- {file_path}: {context['change_context']}"
                for file_path, context in file_changes_context.items()
            ])
            
            # Use LLM to update the section intelligently
            update_prompt = f"""
            You are updating a section of technical documentation based on code changes.
            
            Current Section Content:
            {section_content}
            
            File Changes:
            {changes_summary}
            
            Please update this documentation section to reflect the changes while:
            1. Preserving the existing structure and style
            2. Only updating parts that are actually affected by the changes
            3. Maintaining accuracy and clarity
            4. Keeping the same markdown formatting
            
            If no updates are needed, return the original content unchanged.
            
            Updated Section:
            """
            
            # Use the correct LLM invocation method
            response = self.llm_service.llm.invoke([
                {"role": "system", "content": "You are a technical documentation expert. Update documentation sections based on code changes while preserving structure and style."},
                {"role": "user", "content": update_prompt}
            ])
            
            # Extract content from response
            updated_content = self.llm_service._extract_response_content(response)
            return updated_content.strip()
            
        except Exception as e:
            self.console.print(f"[yellow]Error updating section with LLM: {str(e)}. Preserving original.[/yellow]")
            return section_content
    
    def _reconstruct_readme(self, original_sections: Dict[str, str], updated_sections: Dict[str, str]) -> str:
        """
        Reconstruct the README by combining original and updated sections.
        
        Args:
            original_sections: Original README sections
            updated_sections: Updated sections to replace
            
        Returns:
            Complete reconstructed README
        """
        reconstructed_parts = []
        
        # Maintain the original order of sections
        for section_key, original_content in original_sections.items():
            if section_key in updated_sections:
                # Use the updated version
                reconstructed_parts.append(updated_sections[section_key])
            else:
                # Use the original version
                reconstructed_parts.append(original_content)
        
        return '\n\n'.join(reconstructed_parts)
    
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
        
        Format the output as markdown with appropriate headings.
        
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
                
        # Get installation and usage information
        installation_info = self._extract_installation_info()
        usage_info = self._extract_usage_info()
        
        # Generate the overview using LLM
        prompt = f"""
        Create a comprehensive high-level overview documentation of this project based on the following information.
        Focus on explaining the project's purpose, architecture, and how components work together.
        
        Repository structure analysis:
        {structure_analysis}
        
        Key files content:
        {context}
        
        Installation information:
        {installation_info}
        
        Usage information:
        {usage_info}
        
        Your documentation should include ONLY these sections:
        1. Project Purpose and Main Functionality
        2. Installation
        3. Usage
        4. Architecture Overview (with component relationships)
        5. Key Workflows and Processes
        6. Technology Stack
        
        Format the output as a well-structured markdown document with appropriate headings.
        Start with a main heading '# Project Overview' followed by relevant subheadings.
        Make sure the overview is comprehensive but concise, focusing on the most important aspects of the project.
        
        For the Installation section, include:
        - Prerequisites (Python version, etc.)
        - Step-by-step installation instructions
        - Environment setup requirements
        
        For the Usage section, include:
        - Basic command syntax
        - Available commands and their purposes
        - Common usage examples
        - Important command-line options
        
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
    
    def _extract_installation_info(self) -> str:
        """
        Extract installation information from the repository.
        
        Returns:
            Installation information as a formatted string
        """
        info = []
        
        # Check for requirements.txt
        requirements_path = os.path.join(self.repository.repo_path, 'requirements.txt')
        if os.path.exists(requirements_path):
            try:
                with open(requirements_path, 'r', encoding='utf-8') as f:
                    requirements_content = f.read()
                info.append(f"Requirements file found:\n```\n{requirements_content[:1000]}\n```")
            except Exception as e:
                info.append(f"Requirements file exists but could not be read: {str(e)}")
        
        # Check for setup.py
        setup_path = os.path.join(self.repository.repo_path, 'setup.py')
        if os.path.exists(setup_path):
            info.append("Setup.py file found for package installation")
        
        # Check for pyproject.toml
        pyproject_path = os.path.join(self.repository.repo_path, 'pyproject.toml')
        if os.path.exists(pyproject_path):
            info.append("pyproject.toml file found for modern Python packaging")
        
        # Check for package.json (for Node.js projects)
        package_json_path = os.path.join(self.repository.repo_path, 'package.json')
        if os.path.exists(package_json_path):
            info.append("package.json file found for Node.js dependencies")
        
        # Check for .env.example or similar
        env_files = ['.env.example', '.env.template', '.env.sample']
        for env_file in env_files:
            env_path = os.path.join(self.repository.repo_path, env_file)
            if os.path.exists(env_path):
                info.append(f"Environment template file found: {env_file}")
                break
        
        return "\n".join(info) if info else "No specific installation files found."
    
    def _extract_usage_info(self) -> str:
        """
        Extract usage information from the repository, particularly from main.py and CLI definitions.
        
        Returns:
            Usage information as a formatted string
        """
        info = []
        
        # Look for main.py or app.py
        main_files = ['main.py', 'app.py', '__main__.py']
        main_content = None
        main_file_found = None
        
        for main_file in main_files:
            main_path = os.path.join(self.repository.repo_path, main_file)
            if os.path.exists(main_path):
                try:
                    with open(main_path, 'r', encoding='utf-8') as f:
                        main_content = f.read()
                    main_file_found = main_file
                    break
                except Exception:
                    continue
        
        # Also check in src/ directory
        if not main_content:
            src_path = os.path.join(self.repository.repo_path, 'src')
            if os.path.exists(src_path):
                for main_file in main_files:
                    main_path = os.path.join(src_path, main_file)
                    if os.path.exists(main_path):
                        try:
                            with open(main_path, 'r', encoding='utf-8') as f:
                                main_content = f.read()
                            main_file_found = f"src/{main_file}"
                            break
                        except Exception:
                            continue
        
        if main_content:
            info.append(f"Main entry point found: {main_file_found}")
            
            # Extract CLI information if using typer, click, or argparse
            if 'typer' in main_content.lower():
                info.append("Uses Typer for CLI interface")
                # Extract command information
                import re
                commands = re.findall(r'@app\.command\(\)\s*\ndef\s+(\w+)', main_content)
                if commands:
                    info.append(f"Available commands: {', '.join(commands)}")
            elif 'click' in main_content.lower():
                info.append("Uses Click for CLI interface")
            elif 'argparse' in main_content.lower():
                info.append("Uses argparse for CLI interface")
            
            # Look for usage examples in docstrings or comments
            docstring_match = re.search(r'"""([^"]*?)"""', main_content, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1)
                if any(keyword in docstring.lower() for keyword in ['usage', 'example', 'command']):
                    info.append(f"Usage information in docstring:\n{docstring[:500]}")
        
        # Look for README files for additional usage info
        readme_files = ['README.md', 'README.rst', 'README.txt']
        for readme_file in readme_files:
            readme_path = os.path.join(self.repository.repo_path, readme_file)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()[:2000]  # First 2000 chars
                    if any(keyword in readme_content.lower() for keyword in ['usage', 'how to run', 'getting started']):
                        info.append(f"Usage information found in {readme_file}")
                except Exception:
                    pass
        
        return "\n".join(info) if info else "No specific usage information found in main files."
    
    def save_documentation(self, content: str, output_path: str) -> None:
        """Save documentation to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.console.print(f"[green]Documentation saved to {output_path}[/green]")
