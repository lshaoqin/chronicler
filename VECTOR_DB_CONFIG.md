# Vector Database Configuration

Chronicler supports connecting to various external or self-hosted vector databases through environment variables configured in a `.env` file.

## Supported Vector Databases

Chronicler currently supports the following vector databases:

- **FAISS** (default, local): No external service required
- **Pinecone**: Cloud-based vector database
- **Weaviate**: Self-hosted or cloud-based vector database
- **Chroma**: Self-hosted or cloud-based vector database
- **Qdrant**: Self-hosted or cloud-based vector database
- **Milvus**: Self-hosted or cloud-based vector database

## Configuration via Environment Variables

To configure a vector database connection, create a `.env` file in the root directory of the project and set the appropriate environment variables.

### General Configuration

```
# Select the vector database type (default: faiss)
VECTOR_DB_TYPE=faiss  # Options: faiss, pinecone, weaviate, chroma, qdrant, milvus
```

### Pinecone Configuration

```
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment  # e.g., us-west1-gcp
PINECONE_INDEX_NAME=your_index_name
PINECONE_NAMESPACE=your_namespace  # Optional, defaults to "default"
```

### Weaviate Configuration

```
VECTOR_DB_TYPE=weaviate
WEAVIATE_URL=your_weaviate_url  # e.g., https://your-instance.weaviate.network
WEAVIATE_API_KEY=your_api_key  # Optional if authentication is not required
WEAVIATE_INDEX_NAME=your_index_name  # Optional, defaults to "Chronicler"
WEAVIATE_TEXT_KEY=your_text_key  # Optional, defaults to "content"
```

### Chroma Configuration

```
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=your_persist_directory  # Optional, defaults to "./chroma_db"
CHROMA_COLLECTION_NAME=your_collection_name  # Optional, defaults to "chronicler"
```

### Qdrant Configuration

```
VECTOR_DB_TYPE=qdrant
QDRANT_URL=your_qdrant_url  # e.g., http://localhost:6333
QDRANT_API_KEY=your_api_key  # Optional if authentication is not required
QDRANT_COLLECTION_NAME=your_collection_name  # Optional, defaults to "chronicler"
```

### Milvus Configuration

```
VECTOR_DB_TYPE=milvus
MILVUS_URI=your_milvus_uri  # e.g., http://localhost:19530
MILVUS_COLLECTION_NAME=your_collection_name  # Optional, defaults to "chronicler"
MILVUS_USERNAME=your_username  # Optional
MILVUS_PASSWORD=your_password  # Optional
```

## Fallback Mechanism

If the connection to the configured vector database fails, Chronicler will automatically fall back to using a local FAISS vector store. This ensures that the application can continue to function even if there are issues with the external vector database connection.

## Installation

To use a specific vector database, make sure to install the required dependencies. The necessary packages are listed in the `requirements.txt` file with comments indicating which packages are needed for each vector database.

You can install only the dependencies you need. For example:

```bash
# For Pinecone
pip install pinecone-client

# For Weaviate
pip install weaviate-client

# For Chroma
pip install chromadb

# For Qdrant
pip install qdrant-client

# For Milvus
pip install pymilvus
```

## Example .env File

```
# OpenAI API Key (required for OpenAI provider)
OPENAI_API_KEY=your_openai_api_key

# Ollama configuration (required for Ollama provider)
OLLAMA_HOST=http://localhost:11434

# Vector Database Configuration
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=chronicler
```
