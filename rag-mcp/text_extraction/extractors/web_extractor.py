"""Web documents extractor (HTML, XML)."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import re

from ..core.base import BaseExtractor
from ..core.models import ExtractedDocument, FileType

logger = logging.getLogger(__name__)

class WebExtractor(BaseExtractor):
    """Extract text from HTML and XML documents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.version = "1.0.0"
        
        # Configuration options
        self.preserve_links = config.get('preserve_links', True) if config else True
        self.preserve_structure = config.get('preserve_structure', True) if config else True
        self.extract_meta = config.get('extract_meta', True) if config else True
        
        # Try to import BeautifulSoup
        try:
            from bs4 import BeautifulSoup, Comment
            self.BeautifulSoup = BeautifulSoup
            self.Comment = Comment
            self.available = True
            logger.debug("BeautifulSoup4 available")
        except ImportError:
            self.available = False
            logger.warning("BeautifulSoup4 not available")
    
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file."""
        if not self.available or not file_path.exists() or not file_path.is_file():
            return False
        
        suffix = file_path.suffix.lower()
        return suffix in ['.html', '.htm', '.xml']
    
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from HTML/XML document."""
        if not self.can_extract(file_path):
            return self._create_failed_result(
                file_path, 
                "Web extractor not available or invalid file"
            )
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in ['.html', '.htm']:
                return self._extract_html(file_path)
            elif suffix == '.xml':
                return self._extract_xml(file_path)
            else:
                return self._create_failed_result(file_path, f"Unsupported file type: {suffix}")
        
        except Exception as e:
            logger.error(f"Failed to extract from {file_path}: {e}")
            return self._create_failed_result(file_path, str(e))
    
    def _extract_html(self, file_path: Path) -> ExtractedDocument:
        """Extract text from HTML document."""
        content, extraction_time = self._time_extraction(
            self._extract_html_content, file_path
        )
        
        metadata = self._create_metadata(
            file_path, 
            FileType.HTML, 
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
    
    def _extract_html_content(self, file_path: Path) -> str:
        """Extract content from HTML document."""
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    html_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(file_path, 'rb') as f:
                html_content = f.read().decode('utf-8', errors='ignore')
        
        # Parse HTML
        soup = self.BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, self.Comment)):
            comment.extract()
        
        content_parts = []
        
        # Extract metadata if enabled
        if self.extract_meta:
            title = soup.find('title')
            if title and title.string:
                content_parts.append(f"# {title.string.strip()}")
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content_parts.append(f"## Description\n{meta_desc['content'].strip()}")
        
        # Extract main content
        main_content = self._extract_structured_html(soup)
        if main_content:
            content_parts.append(main_content)
        
        return "\n\n".join(content_parts)
    
    def _extract_structured_html(self, soup) -> str:
        """Extract HTML content while preserving structure."""
        content_parts = []
        
        # Process different HTML elements
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section', 'li', 'td', 'th']):
            text = element.get_text().strip()
            if not text:
                continue
            
            # Format based on element type
            tag = element.name
            
            if tag == 'h1':
                content_parts.append(f"# {text}")
            elif tag == 'h2':
                content_parts.append(f"## {text}")
            elif tag == 'h3':
                content_parts.append(f"### {text}")
            elif tag in ['h4', 'h5', 'h6']:
                content_parts.append(f"#### {text}")
            elif tag in ['p', 'div', 'article', 'section']:
                # Only add if it's not just a container for other elements
                if not element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section']):
                    content_parts.append(text)
            else:
                content_parts.append(text)
        
        # Handle links if preservation is enabled
        if self.preserve_links:
            links = soup.find_all('a', href=True)
            if links:
                link_parts = []
                for link in links:
                    link_text = link.get_text().strip()
                    href = link['href']
                    if link_text and href:
                        link_parts.append(f"[{link_text}]({href})")
                
                if link_parts:
                    content_parts.append("## Links\n" + "\n".join(link_parts))
        
        return "\n\n".join(content_parts)
    
    def _extract_xml(self, file_path: Path) -> ExtractedDocument:
        """Extract text from XML document."""
        content, extraction_time = self._time_extraction(
            self._extract_xml_content, file_path
        )
        
        metadata = self._create_metadata(
            file_path, 
            FileType.XML, 
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
    
    def _extract_xml_content(self, file_path: Path) -> str:
        """Extract content from XML document."""
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    xml_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            # Fallback to binary read
            with open(file_path, 'rb') as f:
                xml_content = f.read().decode('utf-8', errors='ignore')
        
        # Parse XML
        soup = self.BeautifulSoup(xml_content, 'xml')
        
        content_parts = []
        
        # Extract text from all elements, preserving structure
        for element in soup.find_all(string=True):
            text = element.strip()
            if text and not text.startswith('<?'):  # Skip XML declarations
                content_parts.append(text)
        
        # Clean up and join content
        cleaned_content = []
        for part in content_parts:
            # Remove excessive whitespace
            part = re.sub(r'\s+', ' ', part).strip()
            if part and len(part) > 1:
                cleaned_content.append(part)
        
        return "\n\n".join(cleaned_content)


# Simple text extractor for plain text and markdown files
class TextExtractor(BaseExtractor):
    """Extract text from plain text files (.txt, .md, etc.)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.version = "1.0.0"
    
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file."""
        if not file_path.exists() or not file_path.is_file():
            return False
        
        suffix = file_path.suffix.lower()
        return suffix in ['.txt', '.md', '.markdown', '.log', '.text', '.rst']
    
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from plain text file."""
        if not self.can_extract(file_path):
            return self._create_failed_result(file_path, "Not a supported text file")
        
        try:
            content, extraction_time = self._time_extraction(
                self._read_text_file, file_path
            )
            
            file_type = FileType.MARKDOWN if file_path.suffix.lower() in ['.md', '.markdown'] else FileType.TEXT
            
            metadata = self._create_metadata(
                file_path, 
                file_type, 
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
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return self._create_failed_result(file_path, str(e))
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read text file with encoding detection."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Final fallback
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')