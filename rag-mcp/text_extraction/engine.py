"""Main TextExtractionEngine for processing multiple file formats."""

import logging
import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

from .core.models import ExtractedDocument, ExtractionResult, FileType
from .core.base import BaseExtractor
from .utils.file_utils import FileTypeDetector, FileScanner

# Import extractors
from .extractors.pdf_extractor import PDFExtractor, FallbackPDFExtractor
from .extractors.office_extractor import OfficeExtractor
from .extractors.web_extractor import WebExtractor, TextExtractor
from .extractors.data_extractor import DataExtractor

logger = logging.getLogger(__name__)

class TextExtractionEngine:
    """High-performance text extraction engine supporting multiple file formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.file_detector = FileTypeDetector()
        self.file_scanner = FileScanner(self.file_detector)
        
        # Performance settings
        self.max_workers = self.config.get('max_workers', min(32, (multiprocessing.cpu_count() or 1) + 4))
        self.use_multiprocessing = self.config.get('use_multiprocessing', False)
        self.batch_size = self.config.get('batch_size', 10)
        
        # File size limits (in bytes)
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB default
        
        # Initialize extractors
        self.extractors = self._initialize_extractors()
        
        logger.info(f"TextExtractionEngine initialized with {len(self.extractors)} extractors")
        logger.info(f"Max workers: {self.max_workers}, Multiprocessing: {self.use_multiprocessing}")
    
    def _initialize_extractors(self) -> List[BaseExtractor]:
        """Initialize all available extractors."""
        extractors = []
        extractor_config = self.config.get('extractors', {})
        
        # PDF extractors (primary + fallback)
        try:
            pdf_extractor = PDFExtractor(extractor_config.get('pdf', {}))
            extractors.append(pdf_extractor)
            if pdf_extractor.available:
                logger.info("PDF extractor (PyMuPDF4LLM) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PDF extractor: {e}")
        
        # Fallback PDF extractor
        try:
            fallback_pdf = FallbackPDFExtractor(extractor_config.get('pdf_fallback', {}))
            extractors.append(fallback_pdf)
            if fallback_pdf.available:
                logger.info("Fallback PDF extractor (basic PyMuPDF) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize fallback PDF extractor: {e}")
        
        # Office documents extractor
        try:
            office_extractor = OfficeExtractor(extractor_config.get('office', {}))
            extractors.append(office_extractor)
            logger.info("Office documents extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Office extractor: {e}")
        
        # Web documents extractor
        try:
            web_extractor = WebExtractor(extractor_config.get('web', {}))
            extractors.append(web_extractor)
            if web_extractor.available:
                logger.info("Web documents extractor (BeautifulSoup) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Web extractor: {e}")
        
        # Text extractor (always available)
        try:
            text_extractor = TextExtractor(extractor_config.get('text', {}))
            extractors.append(text_extractor)
            logger.info("Text extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Text extractor: {e}")
        
        # Data extractor
        try:
            data_extractor = DataExtractor(extractor_config.get('data', {}))
            extractors.append(data_extractor)
            logger.info("Data files extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Data extractor: {e}")
        
        return extractors
    
    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions."""
        return self.file_detector.get_supported_extensions()
    
    def can_extract(self, file_path: Path) -> bool:
        """Check if any extractor can handle this file."""
        return any(extractor.can_extract(file_path) for extractor in self.extractors)
    
    def extract_single(self, file_path: Path) -> ExtractedDocument:
        """Extract text from a single file."""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return ExtractedDocument(
                file_path=str(file_path),
                content="",
                metadata=None,
                success=False,
                error_message="File does not exist"
            )
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.warning(f"File too large ({file_size} bytes): {file_path}")
                return ExtractedDocument(
                    file_path=str(file_path),
                    content="",
                    metadata=None,
                    success=False,
                    error_message=f"File too large ({file_size} bytes)"
                )
        except OSError as e:
            logger.error(f"Could not get file stats for {file_path}: {e}")
            return ExtractedDocument(
                file_path=str(file_path),
                content="",
                metadata=None,
                success=False,
                error_message=f"Could not access file: {e}"
            )
        
        # Find suitable extractor
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                logger.debug(f"Using {extractor.name} for {file_path}")
                try:
                    return extractor.extract(file_path)
                except Exception as e:
                    logger.warning(f"{extractor.name} failed for {file_path}: {e}")
                    continue
        
        # No extractor found
        file_type = self.file_detector.detect_file_type(file_path)
        error_msg = f"No suitable extractor found for file type: {file_type.value}"
        logger.warning(f"{error_msg}: {file_path}")
        
        return ExtractedDocument(
            file_path=str(file_path),
            content="",
            metadata=None,
            success=False,
            error_message=error_msg
        )
    
    def extract_batch(self, file_paths: List[Path], 
                     progress_callback: Optional[callable] = None) -> ExtractionResult:
        """Extract text from multiple files with parallel processing."""
        start_time = time.time()
        total_files = len(file_paths)
        
        if total_files == 0:
            return ExtractionResult(
                documents=[],
                total_files=0,
                successful_extractions=0,
                failed_extractions=0,
                total_processing_time=0.0
            )
        
        logger.info(f"Starting batch extraction of {total_files} files")
        
        # Choose executor based on configuration
        if self.use_multiprocessing and total_files > 4:
            # Use ProcessPoolExecutor for CPU-intensive tasks
            executor_class = ProcessPoolExecutor
            logger.info("Using ProcessPoolExecutor for batch extraction")
        else:
            # Use ThreadPoolExecutor for I/O-bound tasks
            executor_class = ThreadPoolExecutor
            logger.info("Using ThreadPoolExecutor for batch extraction")
        
        documents = []
        successful = 0
        failed = 0
        
        # Process files in batches to manage memory
        batch_size = min(self.batch_size, total_files)
        
        with executor_class(max_workers=self.max_workers) as executor:
            for i in range(0, total_files, batch_size):
                batch = file_paths[i:i + batch_size]
                
                # Submit batch for processing
                future_to_path = {
                    executor.submit(self.extract_single, path): path 
                    for path in batch
                }
                
                # Collect results
                for future in future_to_path:
                    path = future_to_path[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per file
                        documents.append(result)
                        
                        if result.success:
                            successful += 1
                        else:
                            failed += 1
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(successful + failed, total_files, result)
                        
                    except Exception as e:
                        logger.error(f"Extraction failed for {path}: {e}")
                        failed_result = ExtractedDocument(
                            file_path=str(path),
                            content="",
                            metadata=None,
                            success=False,
                            error_message=str(e)
                        )
                        documents.append(failed_result)
                        failed += 1
                        
                        if progress_callback:
                            progress_callback(successful + failed, total_files, failed_result)
        
        total_time = time.time() - start_time
        
        logger.info(f"Batch extraction completed in {total_time:.2f}s")
        logger.info(f"Success: {successful}, Failed: {failed}, Success rate: {(successful/total_files)*100:.1f}%")
        
        return ExtractionResult(
            documents=documents,
            total_files=total_files,
            successful_extractions=successful,
            failed_extractions=failed,
            total_processing_time=total_time
        )
    
    def extract_directory(self, directory: Path, 
                         recursive: bool = True,
                         include_hidden: bool = False,
                         file_types: Optional[List[FileType]] = None,
                         progress_callback: Optional[callable] = None) -> ExtractionResult:
        """Extract text from all supported files in a directory."""
        logger.info(f"Scanning directory: {directory}")
        
        # Scan for files
        file_paths = self.file_scanner.scan_directory(
            directory, 
            recursive=recursive,
            include_hidden=include_hidden,
            max_file_size=self.max_file_size
        )
        
        # Filter by specific file types if requested
        if file_types:
            file_paths = self.file_scanner.filter_by_type(file_paths, file_types)
        
        logger.info(f"Found {len(file_paths)} files to extract")
        
        if not file_paths:
            return ExtractionResult(
                documents=[],
                total_files=0,
                successful_extractions=0,
                failed_extractions=0,
                total_processing_time=0.0
            )
        
        return self.extract_batch(file_paths, progress_callback)
    
    async def extract_batch_async(self, file_paths: List[Path],
                                 progress_callback: Optional[callable] = None) -> ExtractionResult:
        """Asynchronous batch extraction using thread pool."""
        loop = asyncio.get_event_loop()
        
        # Run batch extraction in thread pool
        result = await loop.run_in_executor(
            None, 
            self.extract_batch, 
            file_paths, 
            progress_callback
        )
        
        return result
    
    async def extract_directory_async(self, directory: Path,
                                    recursive: bool = True,
                                    include_hidden: bool = False,
                                    file_types: Optional[List[FileType]] = None,
                                    progress_callback: Optional[callable] = None) -> ExtractionResult:
        """Asynchronous directory extraction."""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            self.extract_directory,
            directory,
            recursive,
            include_hidden,
            file_types,
            progress_callback
        )
        
        return result
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about available extractors."""
        stats = {
            "total_extractors": len(self.extractors),
            "available_extractors": [],
            "supported_extensions": list(self.get_supported_extensions()),
            "max_workers": self.max_workers,
            "use_multiprocessing": self.use_multiprocessing,
            "max_file_size_mb": self.max_file_size / (1024 * 1024)
        }
        
        for extractor in self.extractors:
            extractor_info = {
                "name": extractor.name,
                "version": extractor.version,
                "available": getattr(extractor, 'available', True)
            }
            stats["available_extractors"].append(extractor_info)
        
        return stats