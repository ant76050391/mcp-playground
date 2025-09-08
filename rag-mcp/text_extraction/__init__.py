"""High-performance multi-format text extraction module for RAG systems."""

from .engine import TextExtractionEngine
from .core.models import (
    FileType, 
    ExtractedDocument, 
    ExtractionMetadata, 
    ExtractionResult,
    ChunkInfo
)
from .core.base import BaseExtractor
from .utils.file_utils import FileTypeDetector, FileScanner

# Extractors
from .extractors.pdf_extractor import PDFExtractor, FallbackPDFExtractor
from .extractors.office_extractor import OfficeExtractor
from .extractors.web_extractor import WebExtractor, TextExtractor
from .extractors.data_extractor import DataExtractor

__version__ = "1.0.0"
__author__ = "RAG MCP Server"

# Main exports
__all__ = [
    # Main engine
    "TextExtractionEngine",
    
    # Data models
    "FileType",
    "ExtractedDocument",
    "ExtractionMetadata", 
    "ExtractionResult",
    "ChunkInfo",
    
    # Base classes
    "BaseExtractor",
    
    # Utilities
    "FileTypeDetector",
    "FileScanner",
    
    # Extractors
    "PDFExtractor",
    "FallbackPDFExtractor",
    "OfficeExtractor",
    "WebExtractor",
    "TextExtractor",
    "DataExtractor",
]