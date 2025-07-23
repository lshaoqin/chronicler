"""
Documentation storage and versioning system.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class DocSection:
    """Class representing a documentation section."""
    
    def __init__(
        self,
        file_path: str,
        content: str,
        last_updated: float = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a documentation section.
        
        Args:
            file_path: Path to the file this section documents
            content: Markdown content of the section
            last_updated: Timestamp when this section was last updated
            metadata: Additional metadata for the section
        """
        self.file_path = file_path
        self.content = content
        self.last_updated = last_updated or time.time()
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the section to a dictionary."""
        return {
            "file_path": self.file_path,
            "content": self.content,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocSection':
        """Create a section from a dictionary."""
        return cls(
            file_path=data["file_path"],
            content=data["content"],
            last_updated=data["last_updated"],
            metadata=data["metadata"]
        )


class DocumentationStorage:
    """Class for storing and managing documentation sections."""
    
    def __init__(self, base_path: str):
        """
        Initialize the documentation storage.
        
        Args:
            base_path: Base path for storing documentation
        """
        self.base_path = Path(base_path)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.sections_dir = self.base_path / "sections"
        self.sections_dir.mkdir(exist_ok=True, parents=True)
        self.index_file = self.base_path / f"index_{self.timestamp}.json"
        self.index = self._load_index()
        
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the documentation index."""
        if not self.index_file.exists():
            return {}
            
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
            
    def _save_index(self) -> None:
        """Save the documentation index."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)
            
    def get_section(self, file_path: str) -> Optional[DocSection]:
        """
        Get a documentation section for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocSection object or None if not found
        """
        if file_path not in self.index:
            return None
            
        section_path = self.sections_dir / f"{self.index[file_path]['id']}.md"
        if not section_path.exists():
            return None
            
        try:
            with open(section_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return DocSection(
                file_path=file_path,
                content=content,
                last_updated=self.index[file_path]["last_updated"],
                metadata=self.index[file_path]["metadata"]
            )
        except Exception:
            return None
            
    def save_section(self, section: DocSection) -> None:
        """
        Save a documentation section.
        
        Args:
            section: DocSection object to save
        """
        # Generate a section ID if it doesn't exist
        if section.file_path not in self.index:
            section_id = str(len(self.index) + 1).zfill(6)
        else:
            section_id = self.index[section.file_path]["id"]
            
        # Update the index
        self.index[section.file_path] = {
            "id": section_id,
            "last_updated": section.last_updated,
            "metadata": section.metadata
        }
        
        # Save the section content
        section_path = self.sections_dir / f"{section_id}.md"
        with open(section_path, 'w', encoding='utf-8') as f:
            f.write(section.content)
            
        # Save the updated index
        self._save_index()
        
    def delete_section(self, file_path: str) -> bool:
        """
        Delete a documentation section.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if deleted, False if not found
        """
        if file_path not in self.index:
            return False
            
        section_id = self.index[file_path]["id"]
        section_path = self.sections_dir / f"{section_id}.md"
        
        if section_path.exists():
            section_path.unlink()
            
        del self.index[file_path]
        self._save_index()
        
        return True
        
    def list_sections(self) -> List[str]:
        """
        List all documented file paths.
        
        Returns:
            List of file paths
        """
        return list(self.index.keys())
        
    def get_all_sections(self) -> Dict[str, DocSection]:
        """
        Get all documentation sections.
        
        Returns:
            Dictionary mapping file paths to DocSection objects
        """
        sections = {}
        for file_path in self.index:
            section = self.get_section(file_path)
            if section:
                sections[file_path] = section
                
        return sections
        
    def generate_full_documentation(self) -> str:
        """
        Generate full documentation by combining all sections.
        
        Returns:
            Combined documentation as a string
        """
        sections = self.get_all_sections()
        if not sections:
            return f"# Repository Documentation (Generated: {self.timestamp})\n\nNo documentation sections found."
            
        # Get current date and time for documentation header
        generation_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Build the documentation starting with the header
        doc_parts = [f"# Repository Documentation\n\n**Generated: {generation_time}**\n\n"]
        
        # First, add the project overview section if it exists
        if "__project_overview__" in sections:
            doc_parts.append(sections["__project_overview__"].content)
            doc_parts.append("\n\n")
            # Remove from sections so it doesn't get added again
            del sections["__project_overview__"]
            
        # Next, add the structure overview if it exists
        if "__structure_overview__" in sections:
            doc_parts.append(sections["__structure_overview__"].content)
            doc_parts.append("\n\n")
            # Remove from sections so it doesn't get added again
            del sections["__structure_overview__"]
        
        # Separate different types of sections
        regular_file_sections = {}  # Actual file documentation
        generated_sections = {}     # Generated sections like _overview, _section_*
        special_sections = {}       # Other special sections starting with __
        
        for file_path, section in sections.items():
            if file_path.startswith("__"):
                # Skip already processed special sections
                if file_path not in ["__project_overview__", "__structure_overview__"]:
                    special_sections[file_path] = section
            elif file_path.startswith("_section_") or file_path == "_overview":
                # These are generated documentation sections with proper names in metadata
                generated_sections[file_path] = section
            else:
                # These are actual file documentation sections
                regular_file_sections[file_path] = section
        
        # Add generated sections (like _overview and _section_*) with proper headers
        if generated_sections:
            # Sort generated sections: _overview first, then _section_* alphabetically
            sorted_generated = sorted(generated_sections.items(), key=lambda x: (x[0] != "_overview", x[0]))
            
            for file_path, section in sorted_generated:
                if file_path == "_overview":
                    # Overview section - just add the content directly (it should have its own # header)
                    doc_parts.append(section.content)
                    doc_parts.append("\n\n")
                elif file_path.startswith("_section_"):
                    # Section documentation - use the section_name from metadata if available
                    section_name = section.metadata.get("section_name", "Documentation Section")
                    doc_parts.append(f"## {section_name}\n\n")
                    doc_parts.append(section.content)
                    doc_parts.append("\n\n")
        
        # Add regular file sections grouped by directory (if any exist)
        if regular_file_sections:
            # Sort regular sections by file path
            sorted_sections = sorted(regular_file_sections.items(), key=lambda x: x[0])
            
            # Group sections by directory
            grouped_sections = {}
            for file_path, section in sorted_sections:
                dir_path = os.path.dirname(file_path)
                if not dir_path:
                    dir_path = "/"
                    
                if dir_path not in grouped_sections:
                    grouped_sections[dir_path] = []
                    
                grouped_sections[dir_path].append((file_path, section))
            
            # Add a table of contents for regular file sections
            doc_parts.append("## File Documentation\n\n")
            doc_parts.append("### Table of Contents\n\n")
            for dir_path in sorted(grouped_sections.keys()):
                dir_name = os.path.basename(dir_path) or "Root"
                doc_parts.append(f"- [{dir_name}](#{{''.join(dir_name.lower().split())}})\n")
                for file_path, _ in grouped_sections[dir_path]:
                    file_name = os.path.basename(file_path)
                    doc_parts.append(f"  - [{file_name}](#{file_name.lower().replace('.', '-')})\n")
            
            doc_parts.append("\n")
            
            # Add each regular section grouped by directory
            for dir_path in sorted(grouped_sections.keys()):
                dir_name = os.path.basename(dir_path) or "Root"
                doc_parts.append(f"### {dir_name}\n\n")
                
                for file_path, section in grouped_sections[dir_path]:
                    file_name = os.path.basename(file_path)
                    doc_parts.append(f"#### {file_name}\n\n")
                    doc_parts.append(section.content)
                    doc_parts.append("\n\n")
        
        # Add any remaining special sections (those starting with __ but not already processed)
        for file_path, section in sorted(special_sections.items(), key=lambda x: x[0]):
            # Use a clean section name without the __ prefix/suffix
            section_name = file_path.strip("_").replace("_", " ").title()
            doc_parts.append(f"## {section_name}\n\n")
            doc_parts.append(section.content)
            doc_parts.append("\n\n")
                
        return "".join(doc_parts)
        
    def get_last_updated(self) -> Optional[float]:
        """
        Get the timestamp of the most recently updated section.
        
        Returns:
            Timestamp or None if no sections exist
        """
        if not self.index:
            return None
            
        return max(info["last_updated"] for info in self.index.values())
        
    def get_section_for_update(self, file_path: str, context: str) -> Tuple[DocSection, bool]:
        """
        Get a section for updating, creating a new one if it doesn't exist.
        
        Args:
            file_path: Path to the file
            context: Context for the new section if created
            
        Returns:
            Tuple of (DocSection, is_new)
        """
        section = self.get_section(file_path)
        if section:
            return section, False
            
        # Create a new section with placeholder content
        new_section = DocSection(
            file_path=file_path,
            content=f"Documentation for `{file_path}`\n\n{context}",
            metadata={"created_at": time.time()}
        )
        
        return new_section, True
