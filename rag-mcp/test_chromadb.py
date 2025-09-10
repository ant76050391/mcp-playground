#!/usr/bin/env python3
"""
ChromaDB Vector Store Validation Test
====================================
This script validates that documents are properly stored in ChromaDB
and that vector search functionality works correctly.
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from vector_store.chroma_store import ChromaDBVectorStore
    from server import BGEEmbeddingModel  # Import from server.py
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_chromadb_integration():
    """Test ChromaDB integration with actual vector search."""
    print("🔍 ChromaDB Vector Store Validation Test")
    print("=" * 50)
    
    # Initialize BGE embedding model
    print("📊 Loading BGE-M3 Korean embedding model...")
    model_path = project_root / ".model" / "bge-m3-korean-q4_k_m-2.gguf"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        embedding_model = BGEEmbeddingModel(str(model_path))
        embedding_model.load()
        print("✅ BGE-M3 Korean model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return False
    
    # Initialize ChromaDB vector store
    print("🗂️  Connecting to ChromaDB vector store...")
    try:
        vector_store = ChromaDBVectorStore(
            db_path=".vectordb",
            collection_name="rag_documents"
        )
        print("✅ ChromaDB vector store connected")
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        return False
    
    # Get collection statistics
    print("\n📊 Collection Statistics:")
    stats = vector_store.get_collection_stats()
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Database path: {stats['db_path']}")
    
    if stats['total_documents'] == 0:
        print("⚠️  No documents found in ChromaDB. Run the server first to populate data.")
        return False
    
    # Test vector search with different queries
    test_queries = [
        "ChromaDB vector search",
        "텍스트 추출 문서",
        "Korean language processing",
        "RAG 시스템",
        "embedding similarity"
    ]
    
    print(f"\n🔍 Testing vector search with {len(test_queries)} queries:")
    print("-" * 50)
    
    all_tests_passed = True
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔎 Test {i}: '{query}'")
        
        try:
            # Generate query embedding
            start_time = time.time()
            query_embedding = embedding_model.encode(query)
            embedding_time = time.time() - start_time
            
            print(f"   📊 Query embedding: {len(query_embedding)} dimensions ({embedding_time:.3f}s)")
            
            # Perform vector search
            search_start = time.time()
            results = vector_store.search(
                query_embeddings=[query_embedding],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            search_time = time.time() - search_start
            
            print(f"   🔍 Search completed in {search_time:.3f}s")
            
            # Display results
            if results.get('ids') and len(results['ids'][0]) > 0:
                print(f"   ✅ Found {len(results['ids'][0])} results:")
                
                for j, (doc_id, distance) in enumerate(zip(results['ids'][0][:3], results['distances'][0][:3])):
                    similarity = 1 - distance
                    doc_content = results['documents'][0][j][:100] + "..." if len(results['documents'][0][j]) > 100 else results['documents'][0][j]
                    metadata = results['metadatas'][0][j]
                    
                    print(f"      {j+1}. ID: {doc_id}")
                    print(f"         Similarity: {similarity:.4f}")
                    print(f"         File: {metadata.get('file_name', 'Unknown')}")
                    print(f"         Content: {doc_content}")
                    print()
            else:
                print("   ⚠️  No search results found")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            all_tests_passed = False
    
    # Test document retrieval by ID
    print("\n🔎 Testing document retrieval:")
    try:
        # Get a few document IDs
        all_results = vector_store.search(
            query_embeddings=[embedding_model.encode("test")],
            n_results=3,
            include=['documents', 'metadatas']
        )
        
        if all_results.get('ids') and len(all_results['ids'][0]) > 0:
            test_doc_id = all_results['ids'][0][0]
            print(f"   🔎 Retrieving document: {test_doc_id}")
            
            doc = vector_store.get_document(test_doc_id)
            if doc:
                print(f"   ✅ Document retrieved successfully")
                print(f"      Content length: {len(doc.content) if doc.content else 0} characters")
                
                # Safe embedding dimension check
                try:
                    embed_dim = len(doc.embedding) if doc.embedding is not None else 'None'
                    print(f"      Embedding dimension: {embed_dim}")
                except (TypeError, AttributeError):
                    print(f"      Embedding dimension: Error getting dimension")
                
                # Safe metadata check
                try:
                    metadata_keys = list(doc.metadata.keys()) if doc.metadata else 'None'
                    print(f"      Metadata keys: {metadata_keys}")
                except (TypeError, AttributeError):
                    print(f"      Metadata keys: Error getting keys")
            else:
                print(f"   ❌ Failed to retrieve document {test_doc_id}")
                all_tests_passed = False
        else:
            print("   ⚠️  No documents available for retrieval test")
            
    except Exception as e:
        print(f"   ❌ Document retrieval test failed: {e}")
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All ChromaDB validation tests PASSED!")
        print("✅ Vector storage and search functionality is working correctly")
        return True
    else:
        print("❌ Some tests FAILED!")
        print("⚠️  Check the ChromaDB configuration and data")
        return False

if __name__ == "__main__":
    success = test_chromadb_integration()
    sys.exit(0 if success else 1)