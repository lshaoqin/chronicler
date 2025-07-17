"""
LLM service for interacting with language models.
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


class LLMService:
    """Service for interacting with language models."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.2,
            openai_api_key=self.api_key
        )
        
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.api_key
        )
        
    def get_embedding_model(self):
        """Get the embedding model."""
        return self.embedding_model
        
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
        
        response = self.llm(formatted_prompt.to_messages())
        return response.content
        
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
        
        response = self.llm(formatted_prompt.to_messages())
        return response.content
        
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
        
        response = self.llm(formatted_prompt.to_messages())
        return response.content
        
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
        
        response = self.llm(formatted_prompt.to_messages())
        return response.content
        
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
        
        response = self.llm(formatted_prompt.to_messages())
        
        try:
            # Extract JSON array from response
            result = json.loads(response.content)
            if isinstance(result, list):
                return result
            return []
        except:
            # Fallback: try to extract file paths from text response
            lines = response.content.split("\n")
            files = []
            for line in lines:
                for file in file_list:
                    if file in line:
                        files.append(file)
                        break
            return files[:5]  # Return at most 5 files
