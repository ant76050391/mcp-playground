"""File utilities for text extraction."""

import mimetypes
import os
from pathlib import Path
from typing import List, Optional, Set
import logging

from ..core.models import FileType

logger = logging.getLogger(__name__)

# Try to import python-magic for more accurate MIME type detection
try:
    import magic
    MAGIC_AVAILABLE = True
    logger.info("python-magic available for enhanced file type detection")
except ImportError:
    MAGIC_AVAILABLE = False
    logger.info("python-magic not available, using mimetypes fallback")

class FileTypeDetector:
    """Detect file types for text extraction."""
    
    # Supported file extensions mapped to FileType
    EXTENSION_MAPPING = {
        '.txt': FileType.TEXT,
        '.log': FileType.TEXT,
        '.text': FileType.TEXT,
        '.md': FileType.MARKDOWN,
        '.markdown': FileType.MARKDOWN,
        '.pdf': FileType.PDF,
        '.docx': FileType.DOCX,
        '.doc': FileType.DOCX,  # Treat .doc as .docx for simplicity
        '.pptx': FileType.PPTX,
        '.ppt': FileType.PPTX,  # Treat .ppt as .pptx
        '.xlsx': FileType.XLSX,
        '.xls': FileType.XLSX,  # Treat .xls as .xlsx
        '.html': FileType.HTML,
        '.htm': FileType.HTML,
        '.xml': FileType.XML,
        '.csv': FileType.CSV,
        '.json': FileType.JSON,
        '.yaml': FileType.YAML,
        '.yml': FileType.YAML,
    }
    
    # MIME types for additional validation
    MIME_MAPPING = {
        'text/plain': FileType.TEXT,
        'text/markdown': FileType.MARKDOWN,
        'application/pdf': FileType.PDF,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCX,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileType.PPTX,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileType.XLSX,
        'text/html': FileType.HTML,
        'application/xml': FileType.XML,
        'text/xml': FileType.XML,
        'text/csv': FileType.CSV,
        'application/json': FileType.JSON,
        'application/x-yaml': FileType.YAML,
        'text/yaml': FileType.YAML,
    }
    
    def __init__(self):
        self.supported_extensions = set(self.EXTENSION_MAPPING.keys())
    
    def detect_file_type(self, file_path: Path) -> FileType:
        """Detect file type using extension and optionally MIME type."""
        # First try extension-based detection
        extension = file_path.suffix.lower()
        file_type = self.EXTENSION_MAPPING.get(extension, FileType.UNKNOWN)
        
        # If python-magic is available and we got UNKNOWN, try MIME detection
        if file_type == FileType.UNKNOWN and MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                file_type = self.MIME_MAPPING.get(mime_type, FileType.UNKNOWN)
                if file_type != FileType.UNKNOWN:
                    logger.debug(f"Detected {file_path} as {file_type.value} via MIME type: {mime_type}")
            except Exception as e:
                logger.warning(f"Failed to detect MIME type for {file_path}: {e}")
        
        # Fallback to mimetypes module
        if file_type == FileType.UNKNOWN:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                file_type = self.MIME_MAPPING.get(mime_type, FileType.UNKNOWN)
                if file_type != FileType.UNKNOWN:
                    logger.debug(f"Detected {file_path} as {file_type.value} via mimetypes: {mime_type}")
        
        return file_type
    
    def is_supported(self, file_path: Path) -> bool:
        """Check if file type is supported for extraction."""
        return self.detect_file_type(file_path) != FileType.UNKNOWN
    
    def get_supported_extensions(self) -> Set[str]:
        """Get set of supported file extensions."""
        return self.supported_extensions.copy()

class FileScanner:
    """Scan directories for extractable files."""
    
    def __init__(self, file_detector: Optional[FileTypeDetector] = None):
        self.detector = file_detector or FileTypeDetector()
    
    def scan_directory(self, directory: Path, 
                      recursive: bool = True,
                      include_hidden: bool = False,
                      max_file_size: Optional[int] = None) -> List[Path]:
        """
        Scan directory for extractable files.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            include_hidden: Whether to include hidden files
            max_file_size: Maximum file size in bytes (None for no limit)
        
        Returns:
            List of file paths that can be extracted
        """
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory does not exist or is not a directory: {directory}")
            return []
        
        files = []
        
        try:
            # Choose scanning method based on recursive flag
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in directory.glob(pattern):
                # Skip directories
                if not file_path.is_file():
                    continue
                
                # Skip hidden files if not included
                if not include_hidden and file_path.name.startswith('.'):
                    continue
                
                # Check file size limit
                if max_file_size is not None:
                    try:
                        if file_path.stat().st_size > max_file_size:
                            logger.debug(f"Skipping large file: {file_path}")
                            continue
                    except OSError:
                        logger.warning(f"Could not get file size for: {file_path}")
                        continue
                
                # Check if file type is supported
                if self.detector.is_supported(file_path):
                    files.append(file_path)
                    logger.debug(f"Found extractable file: {file_path}")
        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        logger.info(f"Found {len(files)} extractable files in {directory}")
        return files
    
    def filter_by_type(self, file_paths: List[Path], 
                      file_types: List[FileType]) -> List[Path]:
        """Filter file paths by specific file types."""
        filtered = []
        
        for file_path in file_paths:
            detected_type = self.detector.detect_file_type(file_path)
            if detected_type in file_types:
                filtered.append(file_path)
        
        return filtered