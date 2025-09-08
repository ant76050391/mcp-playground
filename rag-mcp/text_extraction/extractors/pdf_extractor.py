"""PDF text extraction using PyMuPDF4LLM."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from ..core.base import BaseExtractor
from ..core.models import ExtractedDocument, FileType, ChunkInfo

logger = logging.getLogger(__name__)

class PDFExtractor(BaseExtractor):
    """High-performance PDF text extractor using PyMuPDF4LLM."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.version = "1.0.0"
        
        # Configuration options
        self.extract_images = config.get('extract_images', False) if config else False
        self.extract_tables = config.get('extract_tables', True) if config else True
        self.page_chunks = config.get('page_chunks', True) if config else True
        
        # Try to import PyMuPDF4LLM
        try:
            import pymupdf4llm
            self.pymupdf4llm = pymupdf4llm
            self.available = True
            logger.info("PyMuPDF4LLM initialized successfully")
        except ImportError as e:
            logger.error(f"PyMuPDF4LLM not available: {e}")
            self.available = False
    
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle PDF files."""
        return (
            self.available and 
            file_path.suffix.lower() == '.pdf' and 
            file_path.exists() and 
            file_path.is_file()
        )
    
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from PDF using PyMuPDF4LLM."""
        if not self.can_extract(file_path):
            return self._create_failed_result(
                file_path, 
                "PDF extractor not available or invalid file"
            )
        
        try:
            # Time the extraction
            content, extraction_time = self._time_extraction(
                self._extract_pdf_content, file_path
            )
            
            # Create metadata
            metadata = self._create_metadata(
                file_path, 
                FileType.PDF, 
                extraction_time,
                encoding="UTF-8"  # PyMuPDF4LLM outputs UTF-8
            )
            
            # Extract page information and create chunks if enabled
            chunks = None
            if self.page_chunks and content:
                chunks = self._create_page_chunks(content)
            
            # Update metadata with content statistics
            if content:
                words = content.split()
                metadata.word_count = len(words)
                metadata.char_count = len(content)
                
                # Try to detect page count from content structure
                page_markers = content.count('# Page ')  # PyMuPDF4LLM page markers
                if page_markers > 0:
                    metadata.page_count = page_markers
            
            return ExtractedDocument(
                file_path=str(file_path),
                content=content,
                metadata=metadata,
                chunks=chunks,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to extract PDF {file_path}: {e}")
            return self._create_failed_result(file_path, str(e))
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract PDF content using PyMuPDF4LLM."""
        try:
            # Use PyMuPDF4LLM to extract markdown-formatted text
            # This preserves document structure better than plain text
            md_text = self.pymupdf4llm.to_markdown(
                str(file_path),
                # Advanced options
                extract_images=self.extract_images,
                extract_tables=self.extract_tables,
                # Output formatting
                page_chunks=self.page_chunks,
                write_images=False,  # Don't write image files
                # Performance options
                margins=(0, 0, 0, 0),  # No margins for better text extraction
            )
            
            if not md_text or md_text.strip() == "":
                logger.warning(f"No text extracted from PDF: {file_path}")
                return ""
            
            return md_text
            
        except Exception as e:
            logger.error(f"PyMuPDF4LLM extraction failed for {file_path}: {e}")
            # Fallback: try basic extraction without advanced features
            try:
                md_text = self.pymupdf4llm.to_markdown(str(file_path))
                return md_text or ""
            except Exception as e2:
                logger.error(f"Fallback extraction also failed for {file_path}: {e2}")
                raise e2
    
    def _create_page_chunks(self, content: str) -> Optional[list[ChunkInfo]]:
        """Create chunk information based on page markers in PyMuPDF4LLM output."""
        if not content:
            return None
        
        chunks = []
        lines = content.split('\n')
        current_page = 1
        chunk_start = 0
        
        for i, line in enumerate(lines):
            # Look for PyMuPDF4LLM page markers
            if line.startswith('# Page ') or line.startswith('## Page '):
                # Create chunk for previous page
                if i > 0:
                    chunk_end = sum(len(l) + 1 for l in lines[:i])  # +1 for newline
                    chunks.append(ChunkInfo(
                        start_position=chunk_start,
                        end_position=chunk_end,
                        page_number=current_page,
                        section=f"Page {current_page}",
                        chunk_type="page"
                    ))
                    chunk_start = chunk_end
                
                # Extract page number if possible
                try:
                    page_str = line.split('Page ')[-1].split()[0]
                    current_page = int(page_str)
                except (ValueError, IndexError):
                    current_page += 1
        
        # Add final chunk
        if chunk_start < len(content):
            chunks.append(ChunkInfo(
                start_position=chunk_start,
                end_position=len(content),
                page_number=current_page,
                section=f"Page {current_page}",
                chunk_type="page"
            ))
        
        return chunks if chunks else None


# Fallback PDF extractor using basic PyMuPDF (fitz)
class FallbackPDFExtractor(BaseExtractor):
    """Fallback PDF extractor using basic PyMuPDF."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.version = "1.0.0-fallback"
        
        # Try to import basic fitz
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            self.available = True
            logger.info("Fallback PyMuPDF (fitz) initialized")
        except ImportError:
            self.available = False
            logger.warning("PyMuPDF (fitz) not available")
    
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle PDF files."""
        return (
            self.available and 
            file_path.suffix.lower() == '.pdf' and 
            file_path.exists()
        )
    
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from PDF using basic PyMuPDF."""
        if not self.can_extract(file_path):
            return self._create_failed_result(
                file_path, 
                "Fallback PDF extractor not available"
            )
        
        try:
            content, extraction_time = self._time_extraction(
                self._extract_pdf_basic, file_path
            )
            
            metadata = self._create_metadata(
                file_path, 
                FileType.PDF, 
                extraction_time,
                encoding="UTF-8"
            )
            
            if content:
                metadata.word_count = len(content.split())
                metadata.char_count = len(content)
            
            return ExtractedDocument(
                file_path=str(file_path),
                content=content,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Fallback PDF extraction failed for {file_path}: {e}")
            return self._create_failed_result(file_path, str(e))
    
    def _extract_pdf_basic(self, file_path: Path) -> str:
        """Basic PDF text extraction using PyMuPDF."""
        doc = self.fitz.open(str(file_path))
        text_content = []
        
        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append(f"# Page {page_num + 1}\n\n{text}")
            
            return "\n\n".join(text_content)
        
        finally:
            doc.close()