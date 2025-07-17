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
                 model_provider: str = "openai",
                 llm_model_name: str = "gpt-4o",
                 embedding_model_name: str = "text-embedding-ada-002",
                 temperature: float = 0.2):
        """
        Initialize the LLM service.
        
        Args:
            model_provider: Provider of the model ("openai", "ollama", or "local")
            llm_model_name: Name of the LLM model to use
            embedding_model_name: Name of the embedding model to use
            temperature: Temperature for LLM generation (higher = more creative)
        """
        self.model_provider = model_provider.lower()
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
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
                "Provide a summary of the repository organization, identify the main "
                "components, and suggest any improvements to the structure. Format your "
                "response in markdown."
            )
        ])
        
        formatted_prompt = prompt.format_prompt(
            structure=json.dumps(structure, indent=2)
        )
        
        response = self.llm.invoke(formatted_prompt.to_messages())
        return self._extract_response_content(response)
        
    def suggest_documentation_improvements(self, current_docs: str, context: str) -> str:
        """
        Suggest improvements to existing documentation.
        
        Args:
            current_docs: Current documentation content
            context: Additional context about the repository
            
        Returns:
            Suggested improvements
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a technical documentation expert. Your task is to suggest "
                "improvements to existing documentation for a software project."
            ),
            HumanMessagePromptTemplate.from_template(
                "Here is the current documentation:\n\n{current_docs}\n\n"
                "Here is additional context about the repository:\n\n{context}\n\n"
                "Please suggest specific improvements to the documentation. Consider:\n"
                "1. Missing information that should be added\n"
                "2. Unclear sections that should be clarified\n"
                "3. Better organization or structure\n"
                "4. Examples or tutorials that would be helpful\n"
                "5. Any other improvements\n\n"
                "Format your response in markdown and provide specific suggestions "
                "with examples where appropriate."
            )
        ])
        
        formatted_prompt = prompt.format_prompt(
            current_docs=current_docs,
            context=context
        )
        
        response = self.llm.invoke(formatted_prompt.to_messages())
        return self._extract_response_content(response)
        
    def generate_diff(self, original: str, improved: str) -> str:
        """
        Generate a diff between original and improved documentation.
        
        Args:
            original: Original documentation
            improved: Improved documentation
            
        Returns:
            Diff in markdown format
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a documentation expert. Your task is to create a clear diff "
                "between an original document and an improved version."
            ),
            HumanMessagePromptTemplate.from_template(
                "Original document:\n```\n{original}\n```\n\n"
                "Improved document:\n```\n{improved}\n```\n\n"
                "Please create a clear diff that shows what has been added, removed, or changed. "
                "Format your response in markdown, using color coding (green for additions, "
                "red for removals) and clear section headers."
            )
        ])
        
        formatted_prompt = prompt.format_prompt(
            original=original,
            improved=improved
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
        improved = self.llm.invoke([
            {"role": "system", "content": "You are a documentation expert. Your task is to improve existing documentation based on suggestions."},
            {"role": "user", "content": prompt}
        ])
        
        return self._extract_response_content(improved)
