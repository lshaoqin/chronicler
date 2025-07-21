"""
RAG (Retrieval-Augmented Generation) system for enhancing documentation generation.
"""

import os
import re
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from repository import Repository
from llm_service import LLMService
from vector_db_config import VectorDBConfig


class RAGSystem:
    """RAG system for repository documentation."""
    
    def __init__(self, repository: Repository, llm_service: LLMService):
        """
        Initialize the RAG system.
        
        Args:
            repository: Repository object
            llm_service: LLM service for embeddings and completions
        """
        self.repository = repository
        self.llm_service = llm_service
        self.vector_store = None
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def build_knowledge_base(self) -> None:
        """Build the knowledge base from repository files."""
        # Process README first if it exists
        if self.repository.readme_content:
            self._process_document("README.md", self.repository.readme_content)
        
        # Process all files in the repository
        for file_path in self.repository.file_index.keys():
            # Skip README.md as it's already processed
            if file_path == "README.md":
                continue
                
            try:
                content = self.repository.get_file_content(file_path)
                self._process_document(file_path, content)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Create vector store
        self._create_vector_store()
    
    def _process_document(self, file_path: str, content: str) -> None:
        """
        Process a document and add it to the documents list.
        
        Args:
            file_path: Path to the document
            content: Content of the document
        """
        # Get file extension
        _, extension = os.path.splitext(file_path.lower())
        
        # Skip processing binary or very large files
        if len(content) > 1_000_000:  # Skip content larger than 1MB
            return
        
        try:
            # Process based on file type
            if extension == '.py':
                self._process_python_file(file_path, content)
            elif extension in ['.md', '.rst', '.txt']:
                self._process_documentation_file(file_path, content)
            elif extension in ['.js', '.ts', '.jsx', '.tsx']:
                self._process_javascript_file(file_path, content)
            elif extension in ['.java', '.kt', '.scala']:
                self._process_java_like_file(file_path, content)
            elif extension in ['.c', '.cpp', '.h', '.hpp', '.cc']:
                self._process_c_like_file(file_path, content)
            elif extension in ['.go']:
                self._process_go_file(file_path, content)
            elif extension in ['.rb']:
                self._process_ruby_file(file_path, content)
            elif extension in ['.php']:
                self._process_php_file(file_path, content)
            elif extension in ['.html', '.htm', '.xml', '.css', '.scss', '.sass', '.less']:
                self._process_markup_file(file_path, content)
            elif extension in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
                self._process_config_file(file_path, content)
            else:
                # For other file types, just process as plain text
                self._process_generic_file(file_path, content)
        except Exception as e:
            print(f"Error in processing {file_path}: {str(e)}")
    
    def _process_python_file(self, file_path: str, content: str) -> None:
        """Process Python files."""
        # Extract docstrings
        docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL)
        
        # Extract comments (single line)
        comments = re.findall(r'#\s*(.*?)$', content, re.MULTILINE)
        
        # Combine docstrings and comments
        doc_content = "\n\n".join(docstrings) + "\n\n" + "\n".join(comments)
        
        # Add the full content as well
        chunks = self.text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i,
                        "type": "code"
                    }
                )
            )
        
        # Add the extracted documentation
        if doc_content.strip():
            doc_chunks = self.text_splitter.split_text(doc_content)
            for i, chunk in enumerate(doc_chunks):
                self.documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk": i,
                            "type": "documentation"
                        }
                    )
                )
    
    def _process_documentation_file(self, file_path: str, content: str) -> None:
        """Process documentation files (md, rst, txt)."""
        chunks = self.text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i,
                        "type": "documentation"
                    }
                )
            )
    
    def _process_javascript_file(self, file_path: str, content: str) -> None:
        """Process JavaScript/TypeScript files."""
        # Extract JSDoc comments
        jsdocs = re.findall(r'/\*\*(.*?)\*/', content, re.DOTALL)
        
        # Extract single line comments
        comments = re.findall(r'//\s*(.*?)$', content, re.MULTILINE)
        
        # Combine comments
        doc_content = "\n\n".join(jsdocs) + "\n\n" + "\n".join(comments)
        
        # Add the full content
        chunks = self.text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i,
                        "type": "code"
                    }
                )
            )
        
        # Add extracted documentation
        if doc_content.strip():
            doc_chunks = self.text_splitter.split_text(doc_content)
            for i, chunk in enumerate(doc_chunks):
                self.documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk": i,
                            "type": "documentation"
                        }
                    )
                )
    
    def _process_java_like_file(self, file_path: str, content: str) -> None:
        """Process Java-like files."""
        # Extract JavaDoc comments
        javadocs = re.findall(r'/\*\*(.*?)\*/', content, re.DOTALL)
        
        # Extract single line comments
        comments = re.findall(r'//\s*(.*?)$', content, re.MULTILINE)
        
        # Combine comments
        doc_content = "\n\n".join(javadocs) + "\n\n" + "\n".join(comments)
        
        # Add the full content and documentation similar to Python files
        self._add_code_and_docs(file_path, content, doc_content)
    
    def _process_c_like_file(self, file_path: str, content: str) -> None:
        """Process C-like files."""
        # Extract multi-line comments
        multiline_comments = re.findall(r'/\*(.*?)\*/', content, re.DOTALL)
        
        # Extract single line comments
        comments = re.findall(r'//\s*(.*?)$', content, re.MULTILINE)
        
        # Combine comments
        doc_content = "\n\n".join(multiline_comments) + "\n\n" + "\n".join(comments)
        
        # Add the full content and documentation
        self._add_code_and_docs(file_path, content, doc_content)
    
    def _process_go_file(self, file_path: str, content: str) -> None:
        """Process Go files."""
        # Extract single line comments
        comments = re.findall(r'//\s*(.*?)$', content, re.MULTILINE)
        
        # Combine comments
        doc_content = "\n".join(comments)
        
        # Add the full content and documentation
        self._add_code_and_docs(file_path, content, doc_content)
    
    def _process_ruby_file(self, file_path: str, content: str) -> None:
        """Process Ruby files."""
        # Extract multi-line comments
        multiline_comments = re.findall(r'=begin(.*?)=end', content, re.DOTALL)
        
        # Extract single line comments
        comments = re.findall(r'#\s*(.*?)$', content, re.MULTILINE)
        
        # Combine comments
        doc_content = "\n\n".join(multiline_comments) + "\n\n" + "\n".join(comments)
        
        # Add the full content and documentation
        self._add_code_and_docs(file_path, content, doc_content)
    
    def _process_php_file(self, file_path: str, content: str) -> None:
        """Process PHP files."""
        # Extract doc comments
        doc_comments = re.findall(r'/\*\*(.*?)\*/', content, re.DOTALL)
        
        # Extract single line comments
        comments = re.findall(r'//\s*(.*?)$', content, re.MULTILINE)
        
        # Extract hash comments
        hash_comments = re.findall(r'#\s*(.*?)$', content, re.MULTILINE)
        
        # Combine comments
        doc_content = "\n\n".join(doc_comments) + "\n\n" + "\n".join(comments) + "\n\n" + "\n".join(hash_comments)
        
        # Add the full content and documentation
        self._add_code_and_docs(file_path, content, doc_content)
    
    def _process_markup_file(self, file_path: str, content: str) -> None:
        """Process markup files (HTML, XML, CSS)."""
        # Extract HTML/XML comments
        if file_path.endswith(('.html', '.htm', '.xml')):
            comments = re.findall(r'<!--(.*?)-->', content, re.DOTALL)
            doc_content = "\n\n".join(comments)
        # Extract CSS comments
        elif file_path.endswith(('.css', '.scss', '.sass', '.less')):
            comments = re.findall(r'/\*(.*?)\*/', content, re.DOTALL)
            doc_content = "\n\n".join(comments)
        else:
            doc_content = ""
        
        # Add the full content and documentation
        self._add_code_and_docs(file_path, content, doc_content)
    
    def _process_config_file(self, file_path: str, content: str) -> None:
        """Process configuration files."""
        # For config files, just add the full content
        chunks = self.text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i,
                        "type": "configuration"
                    }
                )
            )
    
    def _process_generic_file(self, file_path: str, content: str) -> None:
        """Process any other text file."""
        chunks = self.text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i,
                        "type": "other"
                    }
                )
            )
    
    def _add_code_and_docs(self, file_path: str, content: str, doc_content: str) -> None:
        """Helper method to add both code content and documentation."""
        # Add the full content
        chunks = self.text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i,
                        "type": "code"
                    }
                )
            )
        
        # Add extracted documentation if it exists
        if doc_content.strip():
            doc_chunks = self.text_splitter.split_text(doc_content)
            for i, chunk in enumerate(doc_chunks):
                self.documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk": i,
                            "type": "documentation"
                        }
                    )
                )
    
    def _create_vector_store(self) -> None:
        """Create a vector store from the documents."""
        if not self.documents:
            raise ValueError("No documents to create vector store from")
        
        # Use VectorDBConfig to create the appropriate vector store
        vector_db_config = VectorDBConfig(self.llm_service.get_embedding_model())
        self.vector_store = vector_db_config.create_vector_store(self.documents)
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_knowledge_base() first.")
            
        return self.vector_store.similarity_search(query, k=k)
    
    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Get relevant context for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            Concatenated context string
        """
        documents = self.search(query, k=k)
        context = "\n\n".join([
            f"Source: {doc.metadata['source']}\n{doc.page_content}"
            for doc in documents
        ])
        return context
    
    def get_file_context(self, file_path: str) -> str:
        """
        Get context for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Context string for the file
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_knowledge_base() first.")
            
        # Filter documents by source
        file_docs = [doc for doc in self.documents if doc.metadata['source'] == file_path]
        
        if not file_docs:
            return ""
            
        # Sort by chunk number to maintain order
        file_docs.sort(key=lambda x: x.metadata['chunk'])
        
        return "\n\n".join([doc.page_content for doc in file_docs])
    
    def find_relevant_files(self, query: str, k: int = 5) -> List[str]:
        """
        Find relevant files for a query.
        
        Args:
            query: Query string
            k: Number of files to return
            
        Returns:
            List of file paths
        """
        documents = self.search(query, k=k*2)  # Get more docs than needed to ensure we get k unique files
        
        # Extract unique file paths
        file_paths = []
        for doc in documents:
            file_path = doc.metadata['source']
            if file_path not in file_paths:
                file_paths.append(file_path)
                
            if len(file_paths) >= k:
                break
                
        return file_paths
