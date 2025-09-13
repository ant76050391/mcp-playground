"""ChromaDB-based vector store implementation for RAG MCP Server."""

import logging
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with vector embedding and metadata."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChromaDBVectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(self, 
                 db_path: str = ".vectordb",
                 collection_name: str = "rag_documents",
                 embedding_function=None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of the collection to use
            embedding_function: Function to generate embeddings (optional)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Please install: pip install chromadb")
        
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = None
        self.collection = None
        
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create database directory if it doesn't exist
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception as collection_error:
                # Collection doesn't exist, create new one
                logger.info(f"Collection not found ({collection_error}), creating new collection: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def create_document_metadata(self, 
                               file_path: str,
                               file_type: str,
                               extractor_type: str = "unknown",
                               chunk_index: int = 0) -> Dict[str, Any]:
        """Create standardized metadata for a document."""
        file_path_obj = Path(file_path)
        
        return {
            "file_path": str(file_path_obj),
            "file_name": file_path_obj.name,
            "file_type": file_type.lower(),
            "extractor_type": extractor_type,
            "extraction_time": int(time.time()),
            "file_size": file_path_obj.stat().st_size if file_path_obj.exists() else 0,
            "chunk_index": chunk_index,
            "content_hash": ""  # Will be set during upsert
        }
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _generate_document_id(self, file_path: str, chunk_index: int = 0) -> str:
        """Generate unique document ID."""
        path_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()[:8]
        return f"doc_{path_hash}_{chunk_index}"
    
    def upsert_document(self, 
                       content: str,
                       file_path: str,
                       embedding: List[float],
                       metadata: Optional[Dict[str, Any]] = None,
                       chunk_index: int = 0) -> str:
        """
        Insert or update a single document.
        
        Returns:
            Document ID
        """
        doc_id = self._generate_document_id(file_path, chunk_index)
        content_hash = self._generate_content_hash(content)
        
        # Prepare metadata
        if metadata is None:
            # Try to infer file type from path
            file_type = Path(file_path).suffix[1:] if Path(file_path).suffix else "unknown"
            metadata = self.create_document_metadata(file_path, file_type, chunk_index=chunk_index)
        
        metadata["content_hash"] = content_hash
        
        try:
            # Check if document already exists and hasn't changed
            existing = self.get_document(doc_id)
            if existing and existing.metadata.get("content_hash") == content_hash:
                logger.debug(f"Document {doc_id} unchanged, skipping upsert")
                return doc_id
            
            # Upsert document
            self.collection.upsert(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            logger.debug(f"Upserted document {doc_id} from {file_path}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to upsert document {doc_id}: {e}")
            raise
    
    def upsert_documents(self, 
                        documents: List[VectorDocument]) -> List[str]:
        """
        Insert or update multiple documents in batch.
        
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        ids = []
        contents = []
        embeddings = []
        metadatas = []
        
        for doc in documents:
            # Generate ID if not provided
            if not doc.id:
                doc.id = self._generate_document_id(
                    doc.metadata.get("file_path", "unknown"), 
                    doc.metadata.get("chunk_index", 0)
                )
            
            # Generate content hash
            if doc.metadata:
                doc.metadata["content_hash"] = self._generate_content_hash(doc.content)
            
            ids.append(doc.id)
            contents.append(doc.content)
            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata or {})
        
        try:
            self.collection.upsert(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Upserted {len(documents)} documents to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to upsert {len(documents)} documents: {e}")
            raise
    
    def search(self, 
               query_embeddings: List[List[float]],
               n_results: int = 10,
               include: List[str] = None,
               where: Dict[str, Any] = None) -> Dict[str, List]:
        """
        Search for similar documents.
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            include: What to include in results ['documents', 'metadatas', 'distances', 'embeddings']
            where: Filter conditions
            
        Returns:
            ChromaDB search results
        """
        if include is None:
            include = ['documents', 'metadatas', 'distances']
        
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=include,
                where=where
            )
            
            logger.debug(f"ChromaDB search returned {len(results.get('ids', []))} results")
            return results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            # Safe checking for results
            ids = results.get('ids', [])
            documents = results.get('documents', [])
            embeddings = results.get('embeddings', [])
            metadatas = results.get('metadatas', [])
            
            # Check if any results were found
            try:
                if len(ids) == 0:
                    return None
            except (TypeError, AttributeError):
                # Handle case where ids might not be a list/array
                if not ids:
                    return None
            
            return VectorDocument(
                id=ids[0] if len(ids) > 0 else "",
                content=documents[0] if len(documents) > 0 else "",
                embedding=embeddings[0] if len(embeddings) > 0 else None,
                metadata=metadatas[0] if len(metadatas) > 0 else None
            )
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def delete_documents_by_path(self, file_path: str) -> int:
        """Delete all documents for a specific file path."""
        try:
            # Find all documents for this file path
            results = self.collection.get(
                where={"file_path": file_path}
            )
            
            if results and 'ids' in results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents for {file_path}")
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete documents for {file_path}: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "db_path": str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "error": str(e)
            }
    
    def reset_collection(self):
        """Reset (clear) the entire collection. Use with caution!"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning(f"Reset ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise