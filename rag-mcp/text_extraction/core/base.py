"""Base extractor abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional, Dict, Any
import logging
import time
from datetime import datetime

from .models import ExtractedDocument, ExtractionMetadata, FileType

logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """Abstract base class for document extractors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"
    
    @abstractmethod
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle the given file."""
        pass
    
    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from a single file."""
        pass
    
    def extract_batch(self, file_paths: List[Path]) -> Generator[ExtractedDocument, None, None]:
        """Extract text from multiple files (default implementation)."""
        for file_path in file_paths:
            try:
                result = self.extract(file_path)
                yield result
            except Exception as e:
                logger.error(f"Failed to extract from {file_path}: {e}")
                yield self._create_failed_result(file_path, str(e))
    
    def _create_metadata(self, file_path: Path, file_type: FileType, extraction_time: float, 
                        **kwargs) -> ExtractionMetadata:
        """Create extraction metadata."""
        stat = file_path.stat()
        
        return ExtractionMetadata(
            file_path=str(file_path),
            file_type=file_type,
            file_size=stat.st_size,
            modification_time=datetime.fromtimestamp(stat.st_mtime),
            extraction_time=extraction_time,
            extractor_name=self.name,
            extractor_version=self.version,
            **kwargs
        )
    
    def _create_failed_result(self, file_path: Path, error_message: str) -> ExtractedDocument:
        """Create a failed extraction result."""
        try:
            stat = file_path.stat()
            metadata = ExtractionMetadata(
                file_path=str(file_path),
                file_type=FileType.UNKNOWN,
                file_size=stat.st_size,
                modification_time=datetime.fromtimestamp(stat.st_mtime),
                extraction_time=0.0,
                extractor_name=self.name,
                extractor_version=self.version
            )
        except:
            # If we can't even get file stats, create minimal metadata
            metadata = ExtractionMetadata(
                file_path=str(file_path),
                file_type=FileType.UNKNOWN,
                file_size=0,
                modification_time=datetime.now(),
                extraction_time=0.0,
                extractor_name=self.name,
                extractor_version=self.version
            )
        
        return ExtractedDocument(
            file_path=str(file_path),
            content="",
            metadata=metadata,
            success=False,
            error_message=error_message
        )
    
    def _get_file_type(self, file_path: Path) -> FileType:
        """Detect file type from extension."""
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            '.txt': FileType.TEXT,
            '.log': FileType.TEXT,
            '.md': FileType.MARKDOWN,
            '.markdown': FileType.MARKDOWN,
            '.pdf': FileType.PDF,
            '.docx': FileType.DOCX,
            '.pptx': FileType.PPTX,
            '.xlsx': FileType.XLSX,
            '.html': FileType.HTML,
            '.htm': FileType.HTML,
            '.xml': FileType.XML,
            '.csv': FileType.CSV,
            '.json': FileType.JSON,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
        }
        
        return type_mapping.get(suffix, FileType.UNKNOWN)
    
    def _time_extraction(self, func, *args, **kwargs):
        """Time a function execution and return result with timing."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            extraction_time = time.time() - start_time
            return result, extraction_time
        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"Extraction failed after {extraction_time:.3f}s: {e}")
            raise