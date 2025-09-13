#!/usr/bin/env python3
"""RAG MCP Server - Simple implementation."""

import asyncio
import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from llama_cpp import Llama
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.models import ServerCapabilities
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
)

# Text extraction module
from text_extraction import TextExtractionEngine

# Vector store module
from vector_store import ChromaDBVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGEEmbeddingModel:
    """BGE-M3 Korean embedding model wrapper using llama-cpp-python."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
    
    def load(self):
        """Load the GGUF model with optimized settings."""
        try:
            # Log system information
            cpu_cores = os.cpu_count()
            logger.info(f"System: {platform.system()} {platform.release()}")
            logger.info(f"CPU cores available: {cpu_cores}")
            logger.info(f"Loading BGE-M3 Korean model from: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                embedding=True,      # Enable embedding mode
                n_ctx=8192,          # Context window (match model training size)
                verbose=False,       # Minimize logs
                n_threads=cpu_cores, # Use all CPU cores
                n_batch=512,         # Batch size optimization
                use_mmap=True,       # Memory mapping for faster loading
                use_mlock=False,     # Disable memory locking (can cause issues on macOS)
                n_gpu_layers=24,     # Use GPU acceleration (BERT has 24 layers)
            )
            logger.info("BGE-M3 Korean model loaded successfully!")
            logger.info(f"Model using {cpu_cores} CPU threads with batch size 512")
            logger.info("GPU acceleration enabled with 24 layers on Metal")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise
    
    def encode(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            embedding = self.model.create_embedding(text)
            # Extract the embedding vector from the response
            if isinstance(embedding, dict) and 'data' in embedding:
                return embedding['data'][0]['embedding']
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def benchmark_embedding(self):
        """Benchmark embedding performance with different text lengths."""
        logger.info("Starting embedding benchmark...")
        test_texts = [
            "짧은 텍스트",
            "중간 길이의 텍스트입니다. 성능 테스트를 위한 문장으로 적당한 길이를 가지고 있습니다.",
            "긴 텍스트입니다. " * 20 + "이 텍스트는 임베딩 성능을 측정하기 위해 만들어진 긴 문장입니다."
        ]
        
        for i, text in enumerate(test_texts, 1):
            start = time.time()
            embedding = self.encode(text)
            elapsed = time.time() - start
            logger.info(f"[BENCH] Text {i} ({len(text)} chars, dim={len(embedding)}): {elapsed:.3f}s")
        
        logger.info("Embedding benchmark completed")

class KoreanBM25Retriever:
    """한국어 최적화 BM25 검색기"""
    
    def __init__(self, tokenizer_type: str = "okt"):
        """
        tokenizer_type: 'okt', 'kkma' 중 선택
        벤치마크 결과 'okt'가 가장 우수한 성능
        """
        self.tokenizer_type = tokenizer_type
        self.tokenizer = None
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """형태소 분석기 초기화"""
        try:
            if self.tokenizer_type == "okt":
                from konlpy.tag import Okt
                # JVM 초기화 중 오류가 발생할 수 있으므로 즉시 테스트
                self.tokenizer = Okt()
                # 간단한 테스트로 정상 작동 확인
                test_result = self.tokenizer.morphs("테스트")
                logger.info("Korean BM25: Using Okt tokenizer")
            elif self.tokenizer_type == "kkma":
                from konlpy.tag import Kkma
                self.tokenizer = Kkma()
                # 간단한 테스트로 정상 작동 확인
                test_result = self.tokenizer.morphs("테스트") 
                logger.info("Korean BM25: Using Kkma tokenizer")
            else:
                raise ValueError(f"Unsupported tokenizer: {self.tokenizer_type}")
        except Exception as e:
            logger.error(f"Failed to initialize Korean tokenizer: {e}")
            logger.warning("This may be due to Java version compatibility (KoNLPy requires Java 8-11, current Java 24)")
            logger.info("Falling back to enhanced regex tokenizer")
            self.tokenizer = None
            self.tokenizer_type = "regex"
        
    def _tokenize_korean(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화"""
        if not text or not text.strip():
            return []
        
        try:
            if self.tokenizer and self.tokenizer_type != "regex":
                morphs = self.tokenizer.morphs(text)
                tokens = [word for word in morphs if len(word) > 1 and word.replace(' ', '').isalnum()]
                return tokens
            else:
                # Regex fallback tokenizer
                import re
                # 한글, 영문, 숫자 추출
                tokens = re.findall(r'[가-힣]{2,}|[a-zA-Z]{2,}|[0-9]+', text)
                return [token for token in tokens if len(token) > 1]
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            # Last resort - simple split
            return [word for word in text.split() if len(word) > 1]
    
    def build_index(self, documents: List[str]):
        """BM25 인덱스 구축"""
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return
            
        start_time = time.time()
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        try:
            from rank_bm25 import BM25Okapi
            
            self.documents = documents
            self.tokenized_docs = []
            
            for i, doc in enumerate(documents):
                tokenized = self._tokenize_korean(doc)
                self.tokenized_docs.append(tokenized)
                if (i + 1) % 100 == 0:
                    logger.info(f"Tokenized {i + 1}/{len(documents)} documents")
            
            self.bm25 = BM25Okapi(self.tokenized_docs)
            
            elapsed = time.time() - start_time
            logger.info(f"[TIMER] BM25 index building took {elapsed:.3f}s")
            
        except ImportError as e:
            logger.error(f"Failed to import rank-bm25: {e}")
            logger.error("Please install rank-bm25: pip install rank-bm25")
            raise
        except Exception as e:
            logger.error(f"BM25 index building failed: {e}")
            raise
            
    def search(self, query: str, top_k: int = 50) -> List[tuple]:
        """BM25 검색 수행"""
        if not self.bm25:
            logger.error("BM25 index not built. Call build_index() first.")
            return []
        
        if not query or not query.strip():
            return []
        
        try:
            tokenized_query = self._tokenize_korean(query)
            if not tokenized_query:
                return []
                
            scores = self.bm25.get_scores(tokenized_query)
            
            doc_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

class BGERerankerModel:
    """BGE-Reranker-v2-M3-Ko Korean reranker model wrapper using llama-cpp-python."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
    
    def load(self):
        """Load the GGUF reranker model with optimized settings."""
        try:
            cpu_cores = os.cpu_count()
            logger.info(f"Loading BGE-Reranker-v2-M3-Ko from: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                embedding=False,     # Reranker mode (not embedding)
                n_ctx=8192,          # Context window
                verbose=False,       # Minimize logs
                n_threads=cpu_cores, # Use all CPU cores
                n_batch=512,         # Batch size optimization
                use_mmap=True,       # Memory mapping
                use_mlock=False,     # Disable memory locking
                n_gpu_layers=24,     # GPU acceleration
            )
            logger.info("BGE-Reranker-v2-M3-Ko model loaded successfully!")
            logger.info(f"Reranker using {cpu_cores} CPU threads with GPU acceleration")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[tuple]:
        """한국어 최적화 재순위 처리"""
        if not self.model:
            raise RuntimeError("Reranker model not loaded. Call load() first.")
        
        if not documents:
            return []
        
        try:
            scores = []
            for i, document in enumerate(documents):
                input_text = self._prepare_korean_input(query, document)
                
                response = self.model(
                    input_text,
                    max_tokens=1,
                    temperature=0.0,
                    logits_all=True
                )
                
                # 유사도 점수 계산 (logits에서 추출)
                if 'logits' in response:
                    logits = response['logits']
                    # 간단한 점수 계산 (실제로는 더 복잡한 로직 필요)
                    score = sum(logits) / len(logits) if logits else 0.0
                else:
                    score = 0.0
                
                scores.append((i, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # 폴백: 원래 순서 반환
            return [(i, 1.0) for i in range(min(top_k, len(documents)))]
    
    def _prepare_korean_input(self, query: str, document: str) -> str:
        """한국어 특화 입력 전처리"""
        max_query_tokens = 512
        max_doc_tokens = 7168
        
        # 간단한 토큰 길이 추정 (한글 1글자 ≈ 1토큰)
        query_truncated = query[:max_query_tokens] if len(query) > max_query_tokens else query
        doc_truncated = document[:max_doc_tokens] if len(document) > max_doc_tokens else document
        
        # Cross-encoder 입력 형식
        return f"[CLS] {query_truncated} [SEP] {doc_truncated} [SEP]"

class HybridFusion:
    """Dense + Sparse 결과 융합"""
    
    @staticmethod
    def reciprocal_rank_fusion(dense_results: List[tuple], 
                             sparse_results: List[tuple], 
                             k: int = 60) -> List[tuple]:
        """Reciprocal Rank Fusion (RRF) 알고리즘"""
        fusion_scores = {}
        
        # Dense retrieval 점수 정규화 (doc_id, score 형태 가정)
        for rank, (doc_id, score) in enumerate(dense_results, 1):
            fusion_scores[doc_id] = 1.0 / (k + rank)
            
        # Sparse retrieval 점수 추가
        for rank, (doc_id, score) in enumerate(sparse_results, 1):
            if doc_id in fusion_scores:
                fusion_scores[doc_id] += 1.0 / (k + rank)
            else:
                fusion_scores[doc_id] = 1.0 / (k + rank)
        
        # 점수순 정렬
        fused = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

class WatchFilesMonitor:
    """Real-time file monitoring using watchfiles library."""
    
    def __init__(self, documents_path: str, config: Dict[str, Any], text_extractor=None):
        self.documents_path = Path(documents_path)
        self.config = config.get('watchfiles', {})
        self.text_extractor = text_extractor
        self.ignore_paths = set(self.config.get('ignore_paths', ['.git', '__pycache__', '.DS_Store']))
        self.debounce_ms = self.config.get('debounce_ms', 100)
        self.batch_delay_ms = self.config.get('batch_delay_ms', 500)
        self.is_monitoring = False
        self.monitor_task = None
        self.change_callback = None
        logger.info(f"WatchFilesMonitor initialized for: {self.documents_path}")
    
    def set_change_callback(self, callback):
        """Set callback function for file change events."""
        self.change_callback = callback
        logger.info("File change callback registered")
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on extension and ignore rules."""
        # Check extension using text extractor's supported formats
        if self.text_extractor:
            supported_extensions = self.text_extractor.get_supported_extensions()
            if file_path.suffix.lower() not in supported_extensions:
                return False
        else:
            # Fallback to basic text files if no text extractor available
            if file_path.suffix.lower() not in {'.txt', '.md'}:
                return False
        
        # Check ignore paths
        for ignore in self.ignore_paths:
            if ignore in str(file_path):
                return False
                
        return True
    
    async def start_monitoring(self):
        """Start the file monitoring task."""
        if self.is_monitoring:
            logger.warning("File monitoring already started")
            return
            
        if not self.config.get('enabled', True):
            logger.info("WatchFiles monitoring disabled in config")
            return
            
        try:
            from watchfiles import awatch
            logger.info(f"Starting watchfiles monitoring on: {self.documents_path}")
            self.is_monitoring = True
            
            # Start monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("✅ WatchFiles monitoring started successfully")
            
        except ImportError:
            logger.error("watchfiles library not installed. Install with: pip install watchfiles")
        except Exception as e:
            logger.error(f"Failed to start file monitoring: {e}")
            self.is_monitoring = False
    
    async def stop_monitoring(self):
        """Stop the file monitoring task."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("WatchFiles monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop using watchfiles."""
        try:
            from watchfiles import awatch, Change
            
            logger.info("WatchFiles monitoring loop started")
            batch_changes = []
            last_change_time = 0
            
            async for changes in awatch(self.documents_path, recursive=self.config.get('watch_recursive', True)):
                current_time = time.time() * 1000  # milliseconds
                
                # Filter and categorize changes
                for change_type, file_path in changes:
                    file_path = Path(file_path)
                    
                    if not self.should_process_file(file_path):
                        continue
                    
                    change_info = {
                        'type': self._get_change_type(change_type),
                        'path': file_path,
                        'timestamp': current_time
                    }
                    batch_changes.append(change_info)
                    last_change_time = current_time
                
                # Debounce: wait for batch_delay_ms after last change
                if batch_changes:
                    await asyncio.sleep(self.batch_delay_ms / 1000)
                    
                    # Check if more changes arrived during delay
                    if (time.time() * 1000) - last_change_time >= self.batch_delay_ms:
                        await self._process_changes(batch_changes)
                        batch_changes.clear()
                        
        except asyncio.CancelledError:
            logger.info("WatchFiles monitoring cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in watchfiles monitoring loop: {e}")
            self.is_monitoring = False
    
    def _get_change_type(self, watchfiles_change):
        """Convert watchfiles change type to our format."""
        try:
            from watchfiles import Change
            if watchfiles_change == Change.added:
                return 'added'
            elif watchfiles_change == Change.modified:
                return 'modified'
            elif watchfiles_change == Change.deleted:
                return 'deleted'
            else:
                return 'unknown'
        except ImportError:
            return 'modified'  # fallback
    
    async def _is_file_stable(self, file_path: Path, wait_time: float = 3.0) -> bool:
        """파일이 안정적으로 복사 완료되었는지 확인 (드래그 앤 드랍 대응)"""
        try:
            if not file_path.exists():
                return False
            
            # 초기 파일 크기 체크
            initial_size = file_path.stat().st_size
            if initial_size == 0:
                return False  # 빈 파일은 불안정
            
            # 대용량 파일의 경우 더 긴 대기 시간
            if initial_size > 100 * 1024 * 1024:  # 100MB 초과
                wait_time = 5.0  # 5초 대기
                
            logger.info(f"🕒 Checking file stability for {file_path.name} ({initial_size:,} bytes)...")
            
            # 비동기 대기
            await asyncio.sleep(wait_time)
            
            # 최종 크기와 비교
            try:
                final_size = file_path.stat().st_size
                stable = initial_size == final_size and final_size > 0
                
                if stable:
                    logger.info(f"✅ File stable: {file_path.name}")
                else:
                    logger.warning(f"⚠️  File still copying: {file_path.name} ({initial_size} → {final_size})")
                
                return stable
            except FileNotFoundError:
                return False  # 파일이 사라짐
                
        except Exception as e:
            logger.error(f"Error checking file stability: {e}")
            return False
    
    async def _process_changes(self, changes):
        """Process batched file changes with stability check."""
        if not changes:
            return
            
        logger.info(f"Processing {len(changes)} file changes")
        
        # Group changes by type
        changes_by_type = {'added': [], 'modified': [], 'deleted': []}
        for change in changes:
            change_type = change['type']
            file_path = Path(change['path'])
            
            # 추가/수정된 파일의 경우 안정성 체크
            if change_type in ['added', 'modified'] and file_path.suffix.lower() in ['.pdf']:
                if not await self._is_file_stable(file_path):
                    logger.warning(f"⏸️  Skipping unstable file: {file_path.name}")
                    continue
            
            if change_type in changes_by_type:
                changes_by_type[change_type].append(change['path'])
        
        # Call callback if registered
        if self.change_callback:
            try:
                await self.change_callback(changes_by_type)
            except Exception as e:
                logger.error(f"Error in file change callback: {e}")
        
        # Log changes
        for change_type, paths in changes_by_type.items():
            if paths:
                logger.info(f"📁 {change_type.title()}: {len(paths)} files")
                for path in paths[:3]:  # Show first 3 files
                    logger.info(f"   - {path}")
                if len(paths) > 3:
                    logger.info(f"   ... and {len(paths) - 3} more files")


class RAGServer:
    """RAG MCP Server with embedding model."""
    
    def __init__(self):
        self.embedding_model = None
        self.reranker_model = None
        self.bm25_retriever = None
        self.config = {}
        self.file_monitor = None
        self.text_extractor = None  # Multi-format text extraction engine
        self.vector_store = None  # ChromaDB vector storage
        self.documents = []  # Store documents for BM25 and reranker
        self.document_file_mapping = {}  # Map document index to file path
        self.model_path = Path(__file__).parent / ".model" / "bge-m3-korean-q4_k_m-2.gguf"
        self.reranker_path = Path(__file__).parent / ".model" / "bge-reranker-v2-m3-ko-q4_k_m.gguf"
        self.config_path = Path(__file__).parent / "config.json"
        self.documents_path = Path(__file__).parent / "documents"
    
    def load_config(self):
        """Load configuration from config.json."""
        start_time = time.time()
        try:
            if self.config_path.exists():
                logger.info(f"Loading configuration from: {self.config_path}")
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded: {list(self.config.keys())}")
            else:
                logger.info("No config.json found, using default settings")
                self.config = {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {}
        finally:
            elapsed = time.time() - start_time
            logger.info(f"[TIMER] Configuration loading took {elapsed:.3f}s")

    async def initialize_model(self):
        """Initialize the embedding model."""
        total_start = time.time()
        try:
            # Load configuration first
            config_start = time.time()
            self.load_config()
            
            # Initialize text extraction engine
            extractor_start = time.time()
            logger.info("Initializing text extraction engine...")
            
            # Get file processing config with 1GB default
            file_processing_config = self.config.get('file_processing', {})
            max_file_size = file_processing_config.get('max_file_size', 1073741824)  # 1GB default
            
            extraction_config = self.config.get('text_extraction', {
                'max_workers': min(32, os.cpu_count() + 4),
                'max_file_size': max_file_size,
                'use_multiprocessing': False
            })
            self.text_extractor = TextExtractionEngine(extraction_config)
            extractor_elapsed = time.time() - extractor_start
            logger.info(f"[TIMER] Text extraction engine initialization took {extractor_elapsed:.3f}s")
            
            # Initialize ChromaDB vector store
            vector_start = time.time()
            logger.info("Initializing ChromaDB vector store...")
            try:
                vector_config = self.config.get('vector_store', {})
                db_path = vector_config.get('db_path', '.vectordb')
                collection_name = vector_config.get('collection_name', 'rag_documents')
                
                self.vector_store = ChromaDBVectorStore(
                    db_path=db_path,
                    collection_name=collection_name
                )
                logger.info(f"ChromaDB vector store initialized at: {db_path}")
                
                # Get collection stats
                stats = self.vector_store.get_collection_stats()
                logger.info(f"Vector store stats: {stats}")
                
            except ImportError:
                logger.warning("ChromaDB not available, falling back to naive dense search")
                self.vector_store = None
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                logger.warning("Falling back to naive dense search")
                self.vector_store = None
                
            vector_elapsed = time.time() - vector_start
            logger.info(f"[TIMER] Vector store initialization took {vector_elapsed:.3f}s")
            
            # Initialize file monitor
            monitor_start = time.time()
            logger.info("Initializing watchfiles monitor...")
            self.file_monitor = WatchFilesMonitor(str(self.documents_path), self.config, self.text_extractor)
            self.file_monitor.set_change_callback(self._handle_file_changes)
            monitor_elapsed = time.time() - monitor_start
            logger.info(f"[TIMER] File monitor initialization took {monitor_elapsed:.3f}s")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load embedding model
            model_start = time.time()
            logger.info("Initializing BGE-M3 Korean embedding model...")
            self.embedding_model = BGEEmbeddingModel(str(self.model_path))
            self.embedding_model.load()
            model_elapsed = time.time() - model_start
            logger.info(f"[TIMER] Model loading took {model_elapsed:.3f}s")
            
            # Test model with a simple embedding
            test_start = time.time()
            logger.info("Testing embedding model...")
            try:
                test_text = "안녕하세요"
                test_embedding = self.embedding_model.encode(test_text)
                logger.info(f"Model test successful! Embedding dimension: {len(test_embedding)}")
                
                # Run benchmark after successful test
                self.embedding_model.benchmark_embedding()
                
            except Exception as e:
                logger.error(f"Model test failed: {e}")
                # Continue without failing - model is loaded, just test failed
                logger.info("Continuing without embedding test...")
            test_elapsed = time.time() - test_start
            logger.info(f"[TIMER] Model testing took {test_elapsed:.3f}s")
            
            # Load reranker model if available
            reranker_start = time.time()
            if self.reranker_path.exists():
                logger.info("Initializing BGE-Reranker-v2-M3-Ko model...")
                try:
                    self.reranker_model = BGERerankerModel(str(self.reranker_path))
                    self.reranker_model.load()
                    logger.info("Reranker model loaded successfully!")
                except Exception as e:
                    logger.error(f"Failed to load reranker model: {e}")
                    logger.info("Continuing without reranker model...")
                    self.reranker_model = None
            else:
                logger.info(f"Reranker model not found: {self.reranker_path}")
                logger.info("Continuing without reranker model...")
                self.reranker_model = None
            reranker_elapsed = time.time() - reranker_start
            logger.info(f"[TIMER] Reranker initialization took {reranker_elapsed:.3f}s")
            
            # Initialize BM25 retriever
            bm25_start = time.time()
            logger.info("Initializing Korean BM25 retriever...")
            try:
                tokenizer_type = self.config.get('bm25', {}).get('tokenizer', 'okt')
                self.bm25_retriever = KoreanBM25Retriever(tokenizer_type=tokenizer_type)
                logger.info("BM25 retriever initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize BM25 retriever: {e}")
                logger.info("Continuing without BM25 retriever...")
                self.bm25_retriever = None
            bm25_elapsed = time.time() - bm25_start
            logger.info(f"[TIMER] BM25 initialization took {bm25_elapsed:.3f}s")
            
            # Initial scan and extraction of documents directory
            scan_start = time.time()
            logger.info("Performing initial document extraction...")
            self.documents = []  # Reset document list
            self.document_file_mapping = {}  # Reset document mapping
            
            # Extract text from all supported documents using TextExtractionEngine
            try:
                def progress_callback(processed, total, result):
                    if processed % 10 == 0 or processed == total:
                        logger.info(f"Extracted {processed}/{total} documents...")
                
                extraction_result = self.text_extractor.extract_directory(
                    self.documents_path,
                    recursive=True,
                    include_hidden=False,
                    progress_callback=progress_callback
                )
                
                # Process successful extractions and populate vector store
                vector_docs = []
                for doc in extraction_result.documents:
                    if doc.success and doc.content.strip():
                        # Use cleaned content for RAG
                        clean_content = doc.get_clean_content()
                        if clean_content:
                            doc_index = len(self.documents)
                            self.documents.append(clean_content)
                            self.document_file_mapping[doc_index] = doc.file_path
                            
                            # Prepare document for vector store
                            if self.vector_store:
                                try:
                                    embedding = self.embedding_model.encode(clean_content[:1000])  # Limit context for embedding
                                    
                                    # Create metadata
                                    metadata = self.vector_store.create_document_metadata(
                                        file_path=doc.file_path,
                                        file_type=str(doc.metadata.file_type.value) if hasattr(doc.metadata.file_type, 'value') else str(doc.metadata.file_type),
                                        extractor_type=doc.metadata.extractor_name,
                                        chunk_index=0
                                    )
                                    
                                    vector_docs.append({
                                        'content': clean_content,
                                        'file_path': doc.file_path,
                                        'embedding': embedding,
                                        'metadata': metadata
                                    })
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to prepare vector document for {doc.file_path}: {e}")
                        else:
                            logger.warning(f"No clean content extracted from {doc.file_path}")
                    elif not doc.success:
                        logger.warning(f"Failed to extract {doc.file_path}: {doc.error_message}")
                
                logger.info(f"Extracted text from {len(self.documents)} documents")
                logger.info(f"Success rate: {extraction_result.success_rate:.1f}%")
                
                # Populate ChromaDB vector store with initial documents using efficient batch upsert
                if self.vector_store and vector_docs:
                    try:
                        vector_start = time.time()
                        
                        # Create VectorDocument objects for batch upsert
                        from vector_store.chroma_store import VectorDocument
                        vector_documents = []
                        
                        for doc_data in vector_docs:
                            vector_doc = VectorDocument(
                                id="",  # Let the vector store generate the ID
                                content=doc_data['content'],
                                embedding=doc_data['embedding'],
                                metadata=doc_data['metadata']
                            )
                            vector_documents.append(vector_doc)
                        
                        # Efficient batch upsert
                        upserted_ids = self.vector_store.upsert_documents(vector_documents)
                        
                        vector_elapsed = time.time() - vector_start
                        logger.info(f"🗂️  Populated ChromaDB with {len(upserted_ids)} documents (batch upsert)")
                        logger.info(f"[TIMER] Initial vector store population took {vector_elapsed:.3f}s")
                        
                        # Log vector store stats
                        stats = self.vector_store.get_collection_stats()
                        logger.info(f"📊 Vector store total documents: {stats['total_documents']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to populate vector store: {e}")
                
                # Log extraction statistics
                stats = self.text_extractor.get_extraction_stats()
                logger.info(f"Text extraction engine stats: {stats['total_extractors']} extractors, "
                           f"supports {len(stats['supported_extensions'])} file types")
                
            except Exception as e:
                logger.error(f"Failed to extract documents: {e}")
                self.documents = []
            
            scan_elapsed = time.time() - scan_start
            logger.info(f"[TIMER] Initial document extraction took {scan_elapsed:.3f}s")
            
            # Build BM25 index if documents available and BM25 retriever initialized
            if self.documents and self.bm25_retriever:
                index_start = time.time()
                logger.info("Building BM25 index from loaded documents...")
                try:
                    self.bm25_retriever.build_index(self.documents)
                    logger.info("BM25 index built successfully!")
                except Exception as e:
                    logger.error(f"BM25 index building failed: {e}")
                index_elapsed = time.time() - index_start
                logger.info(f"[TIMER] BM25 indexing took {index_elapsed:.3f}s")
            
            total_elapsed = time.time() - total_start
            logger.info(f"[TIMER] TOTAL initialization took {total_elapsed:.3f}s")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    async def _dense_search(self, query: str, top_k: int = 50) -> List[tuple]:
        """Dense retrieval using BGE-M3 embedding with ChromaDB or fallback to naive search"""
        if not self.embedding_model:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query)
            
            # Use ChromaDB if available
            if self.vector_store:
                results = self.vector_store.search(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, 100),  # Reasonable limit
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Convert ChromaDB results to document indices
                scores = []
                if results.get('ids') and results.get('distances'):
                    for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                        # Convert distance to similarity score (ChromaDB returns cosine distance)
                        similarity = 1 - distance
                        
                        # Find document index in self.documents by content matching
                        if results.get('documents') and i < len(results['documents'][0]):
                            doc_content = results['documents'][0][i]
                            # Find matching document in self.documents
                            for doc_idx, doc in enumerate(self.documents):
                                if doc.startswith(doc_content[:100]):  # Match by content prefix
                                    scores.append((doc_idx, similarity))
                                    break
                
                return scores[:top_k]
            
            # Fallback to naive search if ChromaDB not available
            if not self.documents:
                return []
                
            scores = []
            for i, doc in enumerate(self.documents):
                doc_embedding = self.embedding_model.encode(doc[:1000])  # Truncate for speed
                
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                norm_a = sum(a * a for a in query_embedding) ** 0.5
                norm_b = sum(b * b for b in doc_embedding) ** 0.5
                
                if norm_a > 0 and norm_b > 0:
                    similarity = dot_product / (norm_a * norm_b)
                    scores.append((i, similarity))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    async def _handle_file_changes(self, changes: Dict[str, List]):
        """Handle file changes with differential processing for optimal performance."""
        logger.info("🔄 File changes detected, processing differential updates...")
        
        total_start = time.time()
        changes_processed = 0
        
        try:
            # Process each change type separately for efficiency
            for change_type, file_paths in changes.items():
                if not file_paths:
                    continue
                    
                logger.info(f"📁 Processing {len(file_paths)} {change_type} files...")
                
                if change_type == 'deleted':
                    changes_processed += await self._process_deleted_files(file_paths)
                elif change_type == 'added':
                    changes_processed += await self._process_added_files(file_paths)
                elif change_type == 'modified':
                    changes_processed += await self._process_modified_files(file_paths)
            
            # Rebuild BM25 index only if documents were actually changed
            if changes_processed > 0 and self.bm25_retriever:
                await self._rebuild_bm25_index()
            
        except Exception as e:
            logger.error(f"Failed to handle file changes: {e}")
        
        total_elapsed = time.time() - total_start
        logger.info(f"[TIMER] Differential file processing took {total_elapsed:.3f}s ({changes_processed} files changed)")
    
    async def _process_deleted_files(self, file_paths: List[str]) -> int:
        """Process deleted files by removing them from vector store and documents list."""
        deleted_count = 0
        
        for file_path in file_paths:
            try:
                file_path_str = str(Path(file_path))
                
                # Remove from ChromaDB vector store
                if self.vector_store:
                    deleted_docs = self.vector_store.delete_documents_by_path(file_path_str)
                    if deleted_docs > 0:
                        logger.info(f"🗑️  Deleted {deleted_docs} vectors for {Path(file_path).name}")
                
                # Remove from documents list using mapping
                removed_indices = [idx for idx, mapped_path in self.document_file_mapping.items() 
                                 if mapped_path == file_path_str]
                
                if removed_indices:
                    # Remove from documents list in reverse order to maintain indices
                    for idx in sorted(removed_indices, reverse=True):
                        if idx < len(self.documents):
                            del self.documents[idx]
                            del self.document_file_mapping[idx]
                    
                    # Reindex mapping after deletion
                    self._reindex_document_mapping()
                    
                    logger.info(f"📚 Removed {len(removed_indices)} documents from memory for {Path(file_path).name}")
                    deleted_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process deleted file {file_path}: {e}")
        
        return deleted_count
    
    async def _process_added_files(self, file_paths: List[str]) -> int:
        """Process newly added files by extracting and indexing them."""
        added_count = 0
        
        for file_path in file_paths:
            try:
                added_count += await self._process_single_file(file_path, is_new=True)
            except Exception as e:
                logger.error(f"Failed to process added file {file_path}: {e}")
        
        return added_count
    
    async def _process_modified_files(self, file_paths: List[str]) -> int:
        """Process modified files by updating their vectors and content."""
        modified_count = 0
        
        for file_path in file_paths:
            try:
                # First remove old version
                file_path_str = str(Path(file_path))
                
                if self.vector_store:
                    self.vector_store.delete_documents_by_path(file_path_str)
                
                # Remove from documents list using mapping  
                removed_indices = [idx for idx, mapped_path in self.document_file_mapping.items() 
                                 if mapped_path == file_path_str]
                
                if removed_indices:
                    # Remove from documents list in reverse order to maintain indices
                    for idx in sorted(removed_indices, reverse=True):
                        if idx < len(self.documents):
                            del self.documents[idx]
                            del self.document_file_mapping[idx]
                    
                    # Reindex mapping after deletion
                    self._reindex_document_mapping()
                
                # Then add updated version
                modified_count += await self._process_single_file(file_path, is_new=False)
                
            except Exception as e:
                logger.error(f"Failed to process modified file {file_path}: {e}")
        
        return modified_count
    
    async def _process_single_file(self, file_path: str, is_new: bool = True) -> int:
        """Process a single file: extract, embed, and index."""
        try:
            file_path_obj = Path(file_path)
            str_file_path = str(file_path_obj)
            
            # Check if file is already processed (avoid duplicate processing)
            if str_file_path in self.document_file_mapping.values():
                logger.debug(f"File {file_path_obj.name} already processed, skipping")
                return 0
            
            # Extract text from the single file
            extraction_start = time.time()
            extraction_result = self.text_extractor.extract_single(Path(file_path))
            
            if not extraction_result.success:
                logger.warning(f"Failed to extract {file_path_obj.name}: {extraction_result.error_message}")
                return 0
            
            clean_content = extraction_result.get_clean_content()
            if not clean_content or not clean_content.strip():
                logger.warning(f"No clean content extracted from {file_path_obj.name}")
                return 0
            
            # Add to documents list
            doc_index = len(self.documents)
            self.documents.append(clean_content)
            self.document_file_mapping[doc_index] = str(file_path_obj)
            
            # Process for vector store
            if self.vector_store and self.embedding_model:
                try:
                    embed_start = time.time()
                    embedding = self.embedding_model.encode(clean_content[:1000])  # Limit context
                    embed_elapsed = time.time() - embed_start
                    
                    # Create metadata
                    metadata = self.vector_store.create_document_metadata(
                        file_path=str(file_path_obj),
                        file_type=str(extraction_result.metadata.file_type.value) if hasattr(extraction_result.metadata.file_type, 'value') else str(extraction_result.metadata.file_type),
                        extractor_type=extraction_result.metadata.extractor_name,
                        chunk_index=0
                    )
                    
                    # Upsert single document
                    vector_start = time.time()
                    self.vector_store.upsert_document(
                        content=clean_content,
                        file_path=str(file_path_obj),
                        embedding=embedding,
                        metadata=metadata
                    )
                    vector_elapsed = time.time() - vector_start
                    
                    action = "Added" if is_new else "Updated"
                    logger.info(f"✅ {action} {file_path_obj.name} (extract: {time.time() - extraction_start:.3f}s, embed: {embed_elapsed:.3f}s, vector: {vector_elapsed:.3f}s)")
                    
                except Exception as e:
                    logger.error(f"Failed to process vector for {file_path}: {e}")
            
            return 1
            
        except Exception as e:
            logger.error(f"Failed to process single file {file_path}: {e}")
            return 0
    
    def _reindex_document_mapping(self):
        """Reindex document mapping after deletions to maintain consistency."""
        new_mapping = {}
        for new_idx, old_idx in enumerate(sorted(self.document_file_mapping.keys())):
            if old_idx < len(self.documents):
                new_mapping[new_idx] = self.document_file_mapping[old_idx]
        self.document_file_mapping = new_mapping
    
    async def _rebuild_bm25_index(self):
        """Rebuild BM25 index from current documents."""
        if not self.documents or not self.bm25_retriever:
            return
            
        try:
            index_start = time.time()
            self.bm25_retriever.build_index(self.documents)
            index_elapsed = time.time() - index_start
            logger.info(f"🔍 BM25 index rebuilt with {len(self.documents)} documents ({index_elapsed:.3f}s)")
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index: {e}")
    
    async def start_monitoring(self):
        """Start file monitoring."""
        if self.file_monitor:
            await self.file_monitor.start_monitoring()
    
    async def stop_monitoring(self):
        """Stop file monitoring."""
        if self.file_monitor:
            await self.file_monitor.stop_monitoring()
    
    async def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """3단계 하이브리드 검색: Dense + Sparse + Reranking"""
        if not self.documents:
            return []
        
        search_start = time.time()
        logger.info(f"Starting hybrid search for query: '{query[:50]}...'")
        
        # Get configuration
        config = self.config.get('hybrid_search', {})
        dense_top_k = config.get('dense_top_k', 50)
        sparse_top_k = config.get('sparse_top_k', 50)
        rerank_candidates = config.get('rerank_candidates', 30)
        
        # 1단계: Dense Retrieval (BGE-M3)
        dense_start = time.time()
        logger.info("Stage 1: Dense retrieval...")
        dense_results = await self._dense_search(query, top_k=dense_top_k)
        dense_elapsed = time.time() - dense_start
        logger.info(f"[TIMER] Dense retrieval took {dense_elapsed:.3f}s, found {len(dense_results)} results")
        
        # 2단계: Sparse Retrieval (BM25)
        sparse_start = time.time()
        logger.info("Stage 2: Sparse retrieval...")
        sparse_results = []
        if self.bm25_retriever:
            sparse_results = self.bm25_retriever.search(query, top_k=sparse_top_k)
        sparse_elapsed = time.time() - sparse_start
        logger.info(f"[TIMER] Sparse retrieval took {sparse_elapsed:.3f}s, found {len(sparse_results)} results")
        
        # 3단계: Fusion & Reranking
        fusion_start = time.time()
        logger.info("Stage 3: Fusion and reranking...")
        
        if not dense_results and not sparse_results:
            logger.info("No results from dense or sparse retrieval")
            return []
        
        # 결과 융합 (RRF)
        if dense_results and sparse_results:
            fused_results = HybridFusion.reciprocal_rank_fusion(dense_results, sparse_results)
        elif dense_results:
            fused_results = dense_results
        else:
            fused_results = sparse_results
        
        # 상위 candidates를 리랭커에 전달
        top_candidates = fused_results[:rerank_candidates]
        
        # Reranking with Korean optimization
        final_results = []
        if self.reranker_model and len(top_candidates) > 1:
            try:
                candidate_docs = [self.documents[doc_id] for doc_id, _ in top_candidates]
                reranked = self.reranker_model.rerank(query, candidate_docs, top_k=top_k)
                
                # 결과 포맷팅
                for rank_idx, (orig_idx, score) in enumerate(reranked):
                    if rank_idx < len(top_candidates):
                        doc_id, fusion_score = top_candidates[orig_idx]
                        final_results.append({
                            'rank': rank_idx + 1,
                            'doc_id': doc_id,
                            'content': self.documents[doc_id][:500] + '...' if len(self.documents[doc_id]) > 500 else self.documents[doc_id],
                            'rerank_score': score,
                            'fusion_score': fusion_score,
                            'method': 'hybrid_reranked'
                        })
                        
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                # 폴백: fusion 결과만 사용
                for i, (doc_id, score) in enumerate(top_candidates[:top_k]):
                    final_results.append({
                        'rank': i + 1,
                        'doc_id': doc_id,
                        'content': self.documents[doc_id][:500] + '...' if len(self.documents[doc_id]) > 500 else self.documents[doc_id],
                        'fusion_score': score,
                        'method': 'hybrid_fusion_only'
                    })
        else:
            # Reranker 없이 fusion 결과만 사용
            for i, (doc_id, score) in enumerate(top_candidates[:top_k]):
                final_results.append({
                    'rank': i + 1,
                    'doc_id': doc_id,
                    'content': self.documents[doc_id][:500] + '...' if len(self.documents[doc_id]) > 500 else self.documents[doc_id],
                    'fusion_score': score,
                    'method': 'hybrid_fusion_only'
                })
        
        fusion_elapsed = time.time() - fusion_start
        logger.info(f"[TIMER] Fusion & reranking took {fusion_elapsed:.3f}s")
        
        total_elapsed = time.time() - search_start
        logger.info(f"[TIMER] Total hybrid search took {total_elapsed:.3f}s, returning {len(final_results)} results")
        
        return final_results

# Create server instances
rag_server = RAGServer()
app = Server("rag-mcp")

@app.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="rag://documents",
            name="Document Collection",
            description="Access to RAG document collection",
            mimeType="application/json",
        )
    ]

@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific resource."""
    if uri == "rag://documents":
        return "RAG document collection placeholder - implementation coming soon!"
    else:
        raise ValueError(f"Unknown resource: {uri}")

@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_documents",
            description="Search through document collection using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    """Handle tool calls."""
    if name == "search_documents":
        query = arguments.get("query") if arguments else ""
        limit = arguments.get("limit", 5) if arguments else 5
        
        if not query.strip():
            return [TextContent(type="text", text="❌ Query cannot be empty")]
        
        try:
            # Use hybrid search (Dense + Sparse + Reranking)
            search_results = await rag_server.hybrid_search(query, top_k=limit)
            
            if not search_results:
                return [TextContent(type="text", text=f"🔍 No results found for query: '{query}'")]
            
            # Format results for display
            results_text = f"🔍 Hybrid Search Results for: '{query}'\n"
            results_text += f"📊 3-Stage Pipeline: Dense (BGE-M3) + Sparse (BM25) + Reranking (BGE-Reranker-v2-M3-Ko)\n"
            results_text += f"🎯 Found {len(search_results)} results\n\n"
            
            for result in search_results:
                results_text += f"📄 Rank #{result['rank']}\n"
                results_text += f"   Method: {result['method']}\n"
                
                if 'rerank_score' in result:
                    results_text += f"   Rerank Score: {result['rerank_score']:.4f}\n"
                if 'fusion_score' in result:
                    results_text += f"   Fusion Score: {result['fusion_score']:.4f}\n"
                
                results_text += f"   Content: {result['content']}\n\n"
            
            # Add pipeline status
            pipeline_status = "\n🔧 Pipeline Status:\n"
            pipeline_status += f"   ✅ Dense Retrieval (BGE-M3): {'Available' if rag_server.embedding_model else 'Unavailable'}\n"
            pipeline_status += f"   ✅ Sparse Retrieval (BM25): {'Available' if rag_server.bm25_retriever else 'Unavailable'}\n"
            pipeline_status += f"   ✅ Reranking (BGE-Reranker): {'Available' if rag_server.reranker_model else 'Unavailable'}\n"
            pipeline_status += f"   📚 Document Count: {len(rag_server.documents)}"
            
            results_text += pipeline_status
            
            return [TextContent(type="text", text=results_text)]
            
        except Exception as e:
            error_msg = f"❌ Hybrid search failed: {str(e)}"
            logger.error(error_msg)
            return [TextContent(type="text", text=error_msg)]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main server entry point."""
    main_start = time.time()
    logger.info("Starting RAG MCP Server...")
    
    # Initialize embedding model before starting server
    try:
        init_start = time.time()
        await rag_server.initialize_model()
        init_elapsed = time.time() - init_start
        logger.info(f"[TIMER] Server initialization completed in {init_elapsed:.3f}s")
        
        # Start file monitoring
        monitor_start = time.time()
        await rag_server.start_monitoring()
        monitor_elapsed = time.time() - monitor_start
        logger.info(f"[TIMER] File monitoring started in {monitor_elapsed:.3f}s")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        return
    
    logger.info("Server running on stdio - ready to accept MCP requests")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-mcp",
                server_version="0.1.0",
                capabilities=ServerCapabilities(
                    resources={},
                    tools={},
                    prompts={}
                )
            )
        )
    
    total_elapsed = time.time() - main_start
    logger.info(f"[TIMER] TOTAL server runtime: {total_elapsed:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())