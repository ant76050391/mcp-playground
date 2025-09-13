#!/usr/bin/env python3
"""
Test script for granular file change handling performance.

This script simulates file changes and measures performance improvement
between full reload vs granular processing.
"""

import asyncio
import time
import tempfile
import os
import shutil
from pathlib import Path
import json
import hashlib

# Mock the server components for testing
class MockTextExtractor:
    def extract_file(self, file_path):
        """Mock single file extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return MockExtractionResult(
                success=True,
                content=content,
                file_path=file_path,
                metadata=MockMetadata(
                    file_type='txt',
                    extractor_name='MockExtractor'
                )
            )
        except Exception as e:
            return MockExtractionResult(success=False, content="", file_path=file_path)

class MockExtractionResult:
    def __init__(self, success, content, file_path, metadata=None):
        self.success = success
        self.content = content
        self.file_path = file_path
        self.metadata = metadata
    
    def get_clean_content(self):
        return self.content.strip()

class MockMetadata:
    def __init__(self, file_type, extractor_name):
        self.file_type = file_type
        self.extractor_name = extractor_name

class MockVectorStore:
    def __init__(self):
        self.documents = {}
        
    def delete_documents_by_path(self, file_path):
        """Delete documents by file path."""
        deleted = 0
        keys_to_delete = []
        for key, doc in self.documents.items():
            if doc.get('file_path') == file_path:
                keys_to_delete.append(key)
                deleted += 1
        
        for key in keys_to_delete:
            del self.documents[key]
            
        return deleted
    
    def upsert_documents(self, vector_documents):
        """Mock upsert operation."""
        upserted_ids = []
        for doc in vector_documents:
            doc_id = doc.id if doc.id else f"doc_{len(self.documents)}"
            self.documents[doc_id] = {
                'content': doc.content,
                'file_path': doc.metadata.get('file_path', ''),
                'embedding': getattr(doc, 'embedding', None)
            }
            upserted_ids.append(doc_id)
        
        # Simulate processing time
        time.sleep(0.01 * len(vector_documents))  # 10ms per document
        return upserted_ids

class MockVectorDocument:
    def __init__(self, id, content, embedding, metadata):
        self.id = id
        self.content = content
        self.embedding = embedding
        self.metadata = metadata

class MockEmbeddingModel:
    def encode(self, text):
        """Mock embedding generation."""
        # Simulate embedding processing time
        time.sleep(0.005)  # 5ms per embedding
        return [0.1] * 384  # Mock 384-dim embedding

class GranularFileProcessor:
    """Mock implementation of granular file processing."""
    
    def __init__(self):
        self.text_extractor = MockTextExtractor()
        self.vector_store = MockVectorStore()
        self.embedding_model = MockEmbeddingModel()
        self.documents = []
    
    async def _process_added_files(self, file_paths):
        """Process newly added files."""
        processed = 0
        vector_docs = []
        
        for file_path in file_paths:
            try:
                extraction_result = self.text_extractor.extract_file(file_path)
                
                if extraction_result.success and extraction_result.content.strip():
                    clean_content = extraction_result.get_clean_content()
                    if clean_content:
                        self.documents.append(clean_content)
                        
                        embedding = self.embedding_model.encode(clean_content[:1000])
                        
                        metadata = {
                            'file_path': file_path,
                            'file_type': 'txt',
                            'extractor_type': 'MockExtractor',
                            'chunk_index': 0
                        }
                        
                        vector_docs.append({
                            'content': clean_content,
                            'file_path': file_path,
                            'embedding': embedding,
                            'metadata': metadata
                        })
                    
                    processed += 1
                    
            except Exception as e:
                print(f"Failed to process added file {file_path}: {e}")
        
        if vector_docs:
            await self._upsert_vector_documents(vector_docs)
            
        return processed
    
    async def _process_modified_files(self, file_paths):
        """Process modified files."""
        processed = 0
        
        for file_path in file_paths:
            try:
                # Remove existing
                deleted_count = self.vector_store.delete_documents_by_path(file_path)
                
                # Re-extract
                extraction_result = self.text_extractor.extract_file(file_path)
                
                if extraction_result.success and extraction_result.content.strip():
                    clean_content = extraction_result.get_clean_content()
                    if clean_content:
                        self.documents.append(clean_content)
                        
                        embedding = self.embedding_model.encode(clean_content[:1000])
                        
                        metadata = {
                            'file_path': file_path,
                            'file_type': 'txt',
                            'extractor_type': 'MockExtractor',
                            'chunk_index': 0
                        }
                        
                        vector_docs = [{
                            'content': clean_content,
                            'file_path': file_path,
                            'embedding': embedding,
                            'metadata': metadata
                        }]
                        
                        await self._upsert_vector_documents(vector_docs)
                    
                    processed += 1
                    
            except Exception as e:
                print(f"Failed to process modified file {file_path}: {e}")
        
        return processed
    
    async def _process_deleted_files(self, file_paths):
        """Process deleted files."""
        processed = 0
        
        for file_path in file_paths:
            try:
                deleted_count = self.vector_store.delete_documents_by_path(file_path)
                if deleted_count > 0:
                    processed += 1
                    
            except Exception as e:
                print(f"Failed to process deleted file {file_path}: {e}")
        
        return processed
    
    async def _upsert_vector_documents(self, vector_docs):
        """Helper method to upsert vector documents."""
        if not vector_docs:
            return
            
        vector_documents = []
        
        for doc_data in vector_docs:
            vector_doc = MockVectorDocument(
                id=self._generate_document_id(doc_data['file_path']),
                content=doc_data['content'],
                embedding=doc_data['embedding'],
                metadata=doc_data['metadata']
            )
            vector_documents.append(vector_doc)
        
        self.vector_store.upsert_documents(vector_documents)
    
    def _generate_document_id(self, file_path, chunk_index=0):
        """Generate consistent document ID based on file path and chunk."""
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"doc_{path_hash}_{chunk_index}"
    
    async def process_changes_granular(self, changes):
        """Process file changes granularly."""
        start_time = time.time()
        total_processed = 0
        
        for change_type, file_paths in changes.items():
            if not file_paths:
                continue
                
            if change_type == 'added':
                processed = await self._process_added_files(file_paths)
                total_processed += processed
            elif change_type == 'modified':
                processed = await self._process_modified_files(file_paths)
                total_processed += processed
            elif change_type == 'deleted':
                processed = await self._process_deleted_files(file_paths)
                total_processed += processed
        
        elapsed = time.time() - start_time
        return {
            'processed': total_processed,
            'elapsed': elapsed,
            'method': 'granular'
        }
    
    async def process_changes_full_reload(self, all_files):
        """Simulate full reload processing."""
        start_time = time.time()
        
        # Clear everything
        self.documents = []
        self.vector_store.documents = {}
        
        # Re-process all files
        vector_docs = []
        processed = 0
        
        for file_path in all_files:
            if os.path.exists(file_path):
                try:
                    extraction_result = self.text_extractor.extract_file(file_path)
                    
                    if extraction_result.success and extraction_result.content.strip():
                        clean_content = extraction_result.get_clean_content()
                        if clean_content:
                            self.documents.append(clean_content)
                            
                            embedding = self.embedding_model.encode(clean_content[:1000])
                            
                            metadata = {
                                'file_path': file_path,
                                'file_type': 'txt',
                                'extractor_type': 'MockExtractor',
                                'chunk_index': 0
                            }
                            
                            vector_docs.append({
                                'content': clean_content,
                                'file_path': file_path,
                                'embedding': embedding,
                                'metadata': metadata
                            })
                        
                        processed += 1
                        
                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")
        
        # Batch upsert all documents
        if vector_docs:
            await self._upsert_vector_documents(vector_docs)
        
        elapsed = time.time() - start_time
        return {
            'processed': processed,
            'elapsed': elapsed,
            'method': 'full_reload'
        }

async def test_performance():
    """Test performance comparison between granular vs full reload."""
    
    print("🚀 Starting granular file change performance test...")
    
    # Create test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = []
        
        # Create test files
        print("📝 Creating test files...")
        for i in range(20):  # 20 files for realistic test
            file_path = os.path.join(temp_dir, f"test_file_{i:02d}.txt")
            with open(file_path, 'w') as f:
                f.write(f"This is test document {i}.\n")
                f.write(f"Content line 1 for document {i}.\n")
                f.write(f"Content line 2 for document {i}.\n")
                f.write(f"This document has some unique content: {i * 123}.\n")
            test_files.append(file_path)
        
        processor = GranularFileProcessor()
        
        # Initial setup - process all files
        print("🏗️  Initial processing of all files...")
        initial_result = await processor.process_changes_full_reload(test_files)
        print(f"   Initial processing: {initial_result['processed']} files in {initial_result['elapsed']:.3f}s")
        
        # Test scenarios
        test_scenarios = [
            {
                'name': '1 file modified',
                'changes': {
                    'modified': [test_files[0]]
                }
            },
            {
                'name': '3 files modified',
                'changes': {
                    'modified': test_files[0:3]
                }
            },
            {
                'name': '1 added, 2 modified, 1 deleted',
                'changes': {
                    'added': [os.path.join(temp_dir, 'new_file.txt')],
                    'modified': test_files[0:2],
                    'deleted': [test_files[-1]]
                }
            }
        ]
        
        # Create additional file for testing
        with open(os.path.join(temp_dir, 'new_file.txt'), 'w') as f:
            f.write("This is a new test file.\n")
        
        # Modify some test files
        for i in range(3):
            with open(test_files[i], 'a') as f:
                f.write(f"Modified content added at {time.time()}\n")
        
        print("\n📊 Performance Comparison Results:")
        print("=" * 60)
        
        for scenario in test_scenarios:
            print(f"\n🧪 Test Scenario: {scenario['name']}")
            
            # Test granular processing
            processor_granular = GranularFileProcessor()
            await processor_granular.process_changes_full_reload(test_files)  # Setup
            
            granular_result = await processor_granular.process_changes_granular(scenario['changes'])
            
            # Test full reload processing
            processor_full = GranularFileProcessor()
            await processor_full.process_changes_full_reload(test_files)  # Setup
            
            full_reload_result = await processor_full.process_changes_full_reload(test_files)
            
            # Calculate improvement
            improvement = ((full_reload_result['elapsed'] - granular_result['elapsed']) / full_reload_result['elapsed']) * 100
            
            print(f"   📈 Granular Processing: {granular_result['elapsed']:.3f}s")
            print(f"   📉 Full Reload:        {full_reload_result['elapsed']:.3f}s")
            print(f"   ⚡ Performance Gain:   {improvement:.1f}% faster")
            print(f"   📄 Files Processed:    {granular_result['processed']} vs {full_reload_result['processed']}")
        
        print("\n" + "=" * 60)
        print("✅ Performance test completed!")

if __name__ == "__main__":
    asyncio.run(test_performance())