"""
Repository module for handling GitHub repositories.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import git
from rich.progress import Progress
from rich.console import Console


class Repository:
    """Class for handling repository operations."""
    
    def __init__(self, repo_url_or_path: str):
        """
        Initialize a Repository object.
        
        Args:
            repo_url_or_path: GitHub URL or local path to the repository
        """
        self.repo_url_or_path = repo_url_or_path
        self.repo_path = None
        self.is_temp = False
        self.repo = None
        self.file_index = {}
        self.readme_content = ""
        self.console = Console()
        
    def clone_or_load(self) -> None:
        """Clone a repository from URL or load from local path."""
        if self.repo_url_or_path.startswith(('http://', 'https://', 'git@')):
            # It's a remote repository URL
            temp_dir = tempfile.mkdtemp()
            try:
                self.repo = git.Repo.clone_from(self.repo_url_or_path, temp_dir)
                self.repo_path = temp_dir
                self.is_temp = True
            except git.GitCommandError as e:
                if self.is_temp and os.path.exists(self.repo_path):
                    shutil.rmtree(self.repo_path)
                raise ValueError(f"Failed to clone repository: {str(e)}")
        else:
            # It's a local path
            repo_path = os.path.abspath(self.repo_url_or_path)
            if not os.path.exists(repo_path):
                raise ValueError(f"Repository path does not exist: {repo_path}")
                
            try:
                self.repo = git.Repo(repo_path)
                self.repo_path = repo_path
            except git.InvalidGitRepositoryError:
                raise ValueError(f"Not a valid git repository: {repo_path}")
        
        # Load README if it exists
        self._load_readme()
        
        # Index all files
        self._index_files()
        
    def _load_readme(self) -> None:
        """Load the README file content if it exists."""
        readme_paths = [
            os.path.join(self.repo_path, "README.md"),
            os.path.join(self.repo_path, "README"),
            os.path.join(self.repo_path, "readme.md"),
            os.path.join(self.repo_path, "Readme.md")
        ]
        
        for path in readme_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.readme_content = f.read()
                    return
                except Exception:
                    pass
                    
        self.readme_content = ""  # No README found or couldn't read it
        
    def _index_files(self) -> None:
        """Index all files in the repository."""
        with Progress() as progress:
            task = progress.add_task("[cyan]Indexing files...", total=None)
            
            for root, dirs, files in os.walk(self.repo_path):
                # Skip .git directory
                if '.git' in dirs:
                    dirs.remove('.git')
                
                # Skip other common directories to ignore
                for ignore_dir in ['.github', 'node_modules', '__pycache__', '.venv', 'venv', 'env']:
                    if ignore_dir in dirs:
                        dirs.remove(ignore_dir)
                
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.repo_path)
                    
                    # Skip binary files and very large files
                    try:
                        if os.path.getsize(full_path) > 1_000_000:  # Skip files larger than 1MB
                            continue
                            
                        # Try to read the first few bytes to check if it's binary
                        with open(full_path, 'r', encoding='utf-8') as f:
                            f.read(1024)
                            
                        self.file_index[rel_path] = {
                            'path': full_path,
                            'rel_path': rel_path,
                            'size': os.path.getsize(full_path),
                            'extension': os.path.splitext(file)[1].lower(),
                        }
                    except (UnicodeDecodeError, IOError):
                        # Skip binary files or files that can't be read
                        pass
            
            progress.update(task, completed=100)
    
    def get_file_content(self, file_path: str) -> str:
        """
        Get the content of a file.
        
        Args:
            file_path: Relative path to the file in the repository
            
        Returns:
            Content of the file as a string
        """
        if file_path not in self.file_index:
            raise ValueError(f"File not found: {file_path}")
            
        try:
            with open(self.file_index[file_path]['path'], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {str(e)}")
    
    def get_files_by_extension(self, extension: str) -> List[str]:
        """
        Get all files with a specific extension.
        
        Args:
            extension: File extension (e.g., '.py', '.js')
            
        Returns:
            List of relative file paths
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
            
        return [
            rel_path for rel_path, info in self.file_index.items()
            if info['extension'] == extension
        ]
    
    def get_directory_structure(self) -> Dict:
        """
        Get the directory structure of the repository.
        
        Returns:
            Dictionary representing the directory structure
        """
        structure = {}
        
        for rel_path in self.file_index.keys():
            parts = rel_path.split(os.sep)
            current = structure
            
            # Build the tree structure
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # It's a file
                    if '__files__' not in current:
                        current['__files__'] = []
                    current['__files__'].append(part)
                else:  # It's a directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        return structure
    
    def get_changed_files(self, commit_hash: Optional[str] = None) -> List[str]:
        """
        Get files changed in a specific commit or between HEAD and the previous commit.
        
        Args:
            commit_hash: Hash of the commit to analyze, or None for the latest commit
            
        Returns:
            List of relative paths to changed files
        """
        if not self.repo:
            raise ValueError("Repository not loaded. Call clone_or_load() first.")
            
        if commit_hash:
            # Get changes for a specific commit
            commit = self.repo.commit(commit_hash)
            prev_commit = commit.parents[0] if commit.parents else None
            if not prev_commit:
                # If it's the first commit, get all files in the commit
                return [item.a_path for item in commit.tree.traverse()]
                
            diffs = prev_commit.diff(commit)
        else:
            # Get changes between HEAD and its parent
            head = self.repo.head.commit
            prev_commit = head.parents[0] if head.parents else None
            if not prev_commit:
                # If it's the first commit, get all files in the commit
                return [item.a_path for item in head.tree.traverse()]
                
            diffs = prev_commit.diff(head)
            
        changed_files = []
        for diff in diffs:
            if diff.a_path and diff.a_path not in changed_files:
                changed_files.append(diff.a_path)
            if diff.b_path and diff.b_path != diff.a_path and diff.b_path not in changed_files:
                changed_files.append(diff.b_path)
                
        # Filter out files that aren't in the file index (e.g., binary files)
        return [f for f in changed_files if f in self.file_index]
    
    def get_file_diff(self, file_path: str, commit_hash: Optional[str] = None) -> Tuple[str, str]:
        """
        Get the diff for a specific file between commits.
        
        Args:
            file_path: Relative path to the file
            commit_hash: Hash of the commit to analyze, or None for the latest commit
            
        Returns:
            Tuple of (old_content, new_content)
        """
        if not self.repo:
            raise ValueError("Repository not loaded. Call clone_or_load() first.")
            
        if commit_hash:
            commit = self.repo.commit(commit_hash)
            prev_commit = commit.parents[0] if commit.parents else None
        else:
            commit = self.repo.head.commit
            prev_commit = commit.parents[0] if commit.parents else None
            
        # Get old content
        old_content = ""
        if prev_commit:
            try:
                old_blob = prev_commit.tree / file_path
                old_content = old_blob.data_stream.read().decode('utf-8')
            except (KeyError, UnicodeDecodeError):
                # File didn't exist or is binary
                pass
                
        # Get new content
        new_content = ""
        try:
            new_blob = commit.tree / file_path
            new_content = new_blob.data_stream.read().decode('utf-8')
        except (KeyError, UnicodeDecodeError):
            # File was deleted or is binary
            pass
            
        return old_content, new_content
    
    def get_commit_message(self, commit_hash: Optional[str] = None) -> str:
        """
        Get the message for a specific commit.
        
        Args:
            commit_hash: Hash of the commit to analyze, or None for the latest commit
            
        Returns:
            Commit message
        """
        if not self.repo:
            raise ValueError("Repository not loaded. Call clone_or_load() first.")
            
        if commit_hash:
            commit = self.repo.commit(commit_hash)
        else:
            commit = self.repo.head.commit
            
        return commit.message
    
    def get_commit_summary(self, commit_hash: Optional[str] = None) -> Dict[str, any]:
        """
        Get a summary of a commit including changed files and commit message.
        
        Args:
            commit_hash: Hash of the commit to analyze, or None for the latest commit
            
        Returns:
            Dictionary with commit information
        """
        if not self.repo:
            raise ValueError("Repository not loaded. Call clone_or_load() first.")
            
        if commit_hash:
            commit = self.repo.commit(commit_hash)
        else:
            commit = self.repo.head.commit
            
        changed_files = self.get_changed_files(commit_hash)
        
        return {
            "hash": commit.hexsha,
            "short_hash": commit.hexsha[:7],
            "author": f"{commit.author.name} <{commit.author.email}>",
            "date": commit.committed_datetime.isoformat(),
            "message": commit.message,
            "changed_files": changed_files,
            "stats": {
                "insertions": commit.stats.total["insertions"],
                "deletions": commit.stats.total["deletions"],
                "lines": commit.stats.total["lines"],
                "files": len(commit.stats.files)
            }
        }
    
    def cleanup(self) -> None:
        """Clean up temporary files if necessary."""
        if self.is_temp and self.repo_path and os.path.exists(self.repo_path):
            shutil.rmtree(self.repo_path)
            self.repo_path = None
            self.repo = None
