"""
LLM service for interacting with language models.
"""

import os
import json
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel


class LLMService:
    """Service for interacting with language models."""
    
    def __init__(self, 
                 model_provider: str = None,
                 llm_model_name: str = None,
                 embedding_model_name: str = None,
                 temperature: float = 0.2):
        """
        Initialize the LLM service.
        
        Args:
            model_provider: Provider of the model ("openai", "ollama", or "local"). If None, reads from MODEL_PROVIDER env var.
            llm_model_name: Name of the LLM model to use. If None, reads from LLM_MODEL env var.
            embedding_model_name: Name of the embedding model to use. If None, reads from EMBEDDING_MODEL env var.
            temperature: Temperature for LLM generation (higher = more creative)
        """
        # Get model provider from environment variable or use the provided value (with fallback to "openai")
        env_provider = os.environ.get("MODEL_PROVIDER", "openai").lower()
        self.model_provider = model_provider.lower() if model_provider else env_provider
        
        # Get model names from environment variables or use the provided values (with appropriate defaults)
        self.llm_model_name = llm_model_name or os.environ.get("LLM_MODEL", "gpt-4o")
        self.embedding_model_name = embedding_model_name or os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.temperature = temperature
        
        # Initialize LLM based on provider
        self.llm = self._initialize_llm()
        
        # Initialize embedding model based on provider
        self.embedding_model = self._initialize_embedding_model()
    
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the appropriate LLM based on the provider."""
        if self.model_provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
                )
            return ChatOpenAI(
                model_name=self.llm_model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )
        elif self.model_provider == "ollama":
            # Default Ollama host is localhost:11434
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            return ChatOllama(
                model=self.llm_model_name,
                temperature=self.temperature,
                base_url=ollama_host
            )
        elif self.model_provider == "local":
            # For local models, we use Ollama as the backend
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            return ChatOllama(
                model=self.llm_model_name,
                temperature=self.temperature,
                base_url=ollama_host
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def _initialize_embedding_model(self) -> Embeddings:
        """Initialize the appropriate embedding model based on the provider."""
        if self.model_provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
                )
            return OpenAIEmbeddings(
                model=self.embedding_model_name,
                openai_api_key=api_key
            )
        elif self.model_provider == "ollama":
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            return OllamaEmbeddings(
                model=self.embedding_model_name,
                base_url=ollama_host
            )
        elif self.model_provider == "local":
            # Use HuggingFace Sentence Transformers for local embeddings
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model_name or "all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def get_embedding_model(self) -> Embeddings:
        """Get the embedding model."""
        return self.embedding_model
        
    def _extract_response_content(self, response) -> str:
        """
        Extract content from different response formats based on the model provider.
        
        Args:
            response: Response from the language model
            
        Returns:
            Extracted content as a string
        """
        if hasattr(response, 'content'):
            # OpenAI format
            return response.content
        elif isinstance(response, dict):
            # Various dictionary formats
            if 'content' in response:
                return response['content']
            elif 'text' in response:
                return response['text']
            elif 'message' in response and isinstance(response['message'], dict):
                if 'content' in response['message']:
                    return response['message']['content']
            # Try to convert the whole dict to string as last resort
            return str(response)
        elif isinstance(response, str):
            # Direct string response
            return response
        else:
            # Fallback: convert to string
            return str(response)
    
    def generate_documentation(self, context: str, file_path: str) -> str:
        """
        Generate documentation for a file.
        
        Args:
            context: Context about the file
            file_path: Path to the file
            
        Returns:
            Generated documentation
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a technical documentation expert. Your task is to generate "
                "comprehensive documentation for a code file. Focus on explaining the "
                "purpose, functionality, and usage of the code. Use markdown formatting."
            ),
            HumanMessagePromptTemplate.from_template(
                "Please generate documentation for the following file: {file_path}\n\n"
                "Here is the content and context of the file:\n\n{context}\n\n"
                "Generate a comprehensive documentation section that explains what this "
                "file does, its main components, and how to use it. Format your response "
                "in markdown."
            )
        ])
        
        formatted_prompt = prompt.format_prompt(
            file_path=file_path,
            context=context
        )
        
        response = self.llm.invoke(formatted_prompt.to_messages())
        return self._extract_response_content(response)
        
    def analyze_repository_structure(self, structure: Dict) -> str:
        """
        Analyze the repository structure.
        
        Args:
            structure: Repository structure dictionary
            
        Returns:
            Analysis of the repository structure
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a software architecture expert. Your task is to analyze "
                "the structure of a repository and provide insights about its organization."
            ),
            HumanMessagePromptTemplate.from_template(
                "Please analyze the following repository structure:\n\n"
                "{structure}\n\n"
                "Provide a summary of the repository organization and identify the main "
                "components. Your response will be used for documentation, so do not sound uncertain or "
                "include any additional information or instructions. Format your response in markdown."
            )
        ])
        
        formatted_prompt = prompt.format_prompt(
            structure=json.dumps(structure, indent=2)
        )
        
        response = self.llm.invoke(formatted_prompt.to_messages())
        return self._extract_response_content(response)
        
    def find_relevant_files(self, query: str, file_list: List[str]) -> List[str]:
        """
        Find relevant files based on a query.
        
        Args:
            query: Search query
            file_list: List of files to search
            
        Returns:
            List of relevant files
        """
        if not file_list:
            return []
            
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a code search assistant. Your task is to identify the most relevant "
                "files for a given query from a list of files."
            ),
            HumanMessagePromptTemplate.from_template(
                "Query: {query}\n\n"
                "Files:\n{files}\n\n"
                "Return a JSON array of the most relevant file paths for this query. "
                "Only include files that are likely to be relevant. Return at most 5 files."
            )
        ])
        
        formatted_prompt = prompt.format_prompt(
            query=query,
            files="\n".join(file_list)
        )
        
        response = self.llm.invoke(formatted_prompt.to_messages())
        content = self._extract_response_content(response)
        
        try:
            # Extract JSON array from response
            result = json.loads(content)
            if isinstance(result, list):
                return result
            return []
        except:
            # Fallback: try to extract file paths from text response
            lines = content.split("\n")
            files = []
            for line in lines:
                for file in file_list:
                    if file in line:
                        files.append(file)
                        break
            return files[:5]  # Return at most 5 files
            