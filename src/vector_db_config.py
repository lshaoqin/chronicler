"""
Vector database configuration and connection management.
"""

import os
from typing import Dict, Any, Optional, Union
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore


class VectorDBConfig:
    """Configuration and connection management for vector databases."""
    
    # Supported vector database types
    SUPPORTED_DBS = ["faiss", "pinecone", "weaviate", "chroma", "qdrant", "milvus"]
    
    def __init__(self, embedding_model: Embeddings):
        """
        Initialize vector database configuration.
        
        Args:
            embedding_model: Embedding model to use for vector database
        """
        self.embedding_model = embedding_model
        self.db_type = os.environ.get("VECTOR_DB_TYPE", "faiss").lower()
        
        if self.db_type not in self.SUPPORTED_DBS:
            raise ValueError(f"Unsupported vector database type: {self.db_type}. "
                            f"Supported types are: {', '.join(self.SUPPORTED_DBS)}")
        
        # Load configuration based on database type
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration for the selected vector database from environment variables.
        
        Returns:
            Dictionary of configuration parameters
        """
        config = {}
        
        if self.db_type == "pinecone":
            # Import here to avoid unnecessary dependencies if not used
            from langchain_community.vectorstores import Pinecone
            
            config["api_key"] = os.environ.get("PINECONE_API_KEY")
            config["environment"] = os.environ.get("PINECONE_ENVIRONMENT")
            config["index_name"] = os.environ.get("PINECONE_INDEX_NAME")
            config["namespace"] = os.environ.get("PINECONE_NAMESPACE", "default")
            
            if not config["api_key"] or not config["environment"] or not config["index_name"]:
                raise ValueError(
                    "Pinecone configuration incomplete. Please set PINECONE_API_KEY, "
                    "PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME environment variables."
                )
                
        elif self.db_type == "weaviate":
            # Import here to avoid unnecessary dependencies if not used
            from langchain_community.vectorstores import Weaviate
            
            config["url"] = os.environ.get("WEAVIATE_URL")
            config["api_key"] = os.environ.get("WEAVIATE_API_KEY")
            config["index_name"] = os.environ.get("WEAVIATE_INDEX_NAME", "Chronicler")
            config["text_key"] = os.environ.get("WEAVIATE_TEXT_KEY", "content")
            
            if not config["url"]:
                raise ValueError(
                    "Weaviate configuration incomplete. Please set WEAVIATE_URL environment variable."
                )
                
        elif self.db_type == "chroma":
            # Import here to avoid unnecessary dependencies if not used
            from langchain.vectorstores import Chroma
            
            config["persist_directory"] = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            config["collection_name"] = os.environ.get("CHROMA_COLLECTION_NAME", "chronicler")
            
        elif self.db_type == "qdrant":
            # Import here to avoid unnecessary dependencies if not used
            from langchain_community.vectorstores import Qdrant
            
            config["url"] = os.environ.get("QDRANT_URL")
            config["api_key"] = os.environ.get("QDRANT_API_KEY")
            config["collection_name"] = os.environ.get("QDRANT_COLLECTION_NAME", "chronicler")
            
            if not config["url"]:
                raise ValueError(
                    "Qdrant configuration incomplete. Please set QDRANT_URL environment variable."
                )
                
        elif self.db_type == "milvus":
            # Import here to avoid unnecessary dependencies if not used
            from langchain_community.vectorstores import Milvus
            
            config["uri"] = os.environ.get("MILVUS_URI")
            config["collection_name"] = os.environ.get("MILVUS_COLLECTION_NAME", "chronicler")
            config["connection_args"] = {}
            
            # Add username/password if provided
            if os.environ.get("MILVUS_USERNAME") and os.environ.get("MILVUS_PASSWORD"):
                config["connection_args"]["user"] = os.environ.get("MILVUS_USERNAME")
                config["connection_args"]["password"] = os.environ.get("MILVUS_PASSWORD")
            
            if not config["uri"]:
                raise ValueError(
                    "Milvus configuration incomplete. Please set MILVUS_URI environment variable."
                )
        
        return config
    
    def create_vector_store(self, documents: list) -> VectorStore:
        """
        Create a vector store from documents using the configured database.
        
        Args:
            documents: List of documents to add to the vector store
            
        Returns:
            Vector store instance
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
            
        try:
            if self.db_type == "faiss":
                return FAISS.from_documents(documents, self.embedding_model)
                
            elif self.db_type == "pinecone":
                from langchain_community.vectorstores import Pinecone
                import pinecone
                
                # Initialize Pinecone
                pinecone.init(
                    api_key=self.config["api_key"],
                    environment=self.config["environment"]
                )
                
                return Pinecone.from_documents(
                    documents,
                    self.embedding_model,
                    index_name=self.config["index_name"],
                    namespace=self.config["namespace"]
                )
                
            elif self.db_type == "weaviate":
                from langchain.vectorstores import Weaviate
                import weaviate
                from weaviate.auth import AuthApiKey
                
                # Initialize Weaviate client
                auth_config = None
                if self.config.get("api_key"):
                    auth_config = AuthApiKey(api_key=self.config["api_key"])
                    
                client = weaviate.Client(
                    url=self.config["url"],
                    auth_client_secret=auth_config
                )
                
                return Weaviate.from_documents(
                    documents,
                    self.embedding_model,
                    client=client,
                    index_name=self.config["index_name"],
                    text_key=self.config["text_key"]
                )
                
            elif self.db_type == "chroma":
                from langchain_community.vectorstores import Chroma
                
                return Chroma.from_documents(
                    documents,
                    self.embedding_model,
                    persist_directory=self.config["persist_directory"],
                    collection_name=self.config["collection_name"]
                )
                
            elif self.db_type == "qdrant":
                from langchain_community.vectorstores import Qdrant
                import qdrant_client
                
                # Initialize Qdrant client
                client = qdrant_client.QdrantClient(
                    url=self.config["url"],
                    api_key=self.config.get("api_key")
                )
                
                return Qdrant.from_documents(
                    documents,
                    self.embedding_model,
                    url=self.config["url"],
                    prefer_grpc=True,
                    collection_name=self.config["collection_name"],
                    client=client
                )
                
            elif self.db_type == "milvus":
                from langchain_community.vectorstores import Milvus
                
                return Milvus.from_documents(
                    documents,
                    self.embedding_model,
                    connection_args=self.config["connection_args"],
                    collection_name=self.config["collection_name"]
                )
                
        except Exception as e:
            print(f"Error creating {self.db_type} vector store: {str(e)}")
            print("Falling back to FAISS vector store")
            return FAISS.from_documents(documents, self.embedding_model)
            
        # Default fallback to FAISS
        return FAISS.from_documents(documents, self.embedding_model)
