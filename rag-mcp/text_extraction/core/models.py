"""Core data models for text extraction."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from enum import Enum

class FileType(Enum):
    """Supported file types for text extraction."""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    XML = "xml"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    UNKNOWN = "unknown"

@dataclass
class ChunkInfo:
    """Information about text chunks for better document understanding."""
    start_position: int
    end_position: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_type: Optional[str] = None  # paragraph, table, header, etc.

@dataclass
class ExtractionMetadata:
    """Metadata collected during text extraction."""
    file_path: str
    file_type: FileType
    file_size: int
    modification_time: datetime
    extraction_time: float
    extractor_name: str
    extractor_version: str
    encoding: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    language: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractedDocument:
    """Result of document text extraction."""
    file_path: str
    content: str
    metadata: ExtractionMetadata
    chunks: Optional[List[ChunkInfo]] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def get_clean_content(self) -> str:
        """Return cleaned text content optimized for RAG."""
        if not self.content:
            return ""
        
        # Basic text cleaning
        lines = self.content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 1:  # Remove empty and very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_text_stats(self) -> Dict[str, int]:
        """Return text statistics."""
        content = self.get_clean_content()
        return {
            "char_count": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.split('\n')),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()])
        }

@dataclass
class ExtractionResult:
    """Batch extraction result containing multiple documents."""
    documents: List[ExtractedDocument]
    total_files: int
    successful_extractions: int
    failed_extractions: int
    total_processing_time: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_extractions / self.total_files) * 100.0