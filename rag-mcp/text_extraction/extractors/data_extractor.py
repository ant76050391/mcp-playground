"""Data files extractor (CSV, JSON, YAML)."""

import logging
import json
import csv
from pathlib import Path
from typing import Optional, Dict, Any
import io

from ..core.base import BaseExtractor
from ..core.models import ExtractedDocument, FileType

logger = logging.getLogger(__name__)

class DataExtractor(BaseExtractor):
    """Extract text from structured data files."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.version = "1.0.0"
        
        # Configuration options
        self.max_rows = config.get('max_rows', 10000) if config else 10000
        self.sample_rows = config.get('sample_rows', False) if config else False
        
        # Try to import optional libraries
        self.polars_available = False
        self.yaml_available = False
        
        try:
            import polars as pl
            self.polars = pl
            self.polars_available = True
            logger.debug("Polars available for high-performance CSV processing")
        except ImportError:
            logger.debug("Polars not available, using standard csv module")
        
        try:
            import yaml
            self.yaml = yaml
            self.yaml_available = True
            logger.debug("PyYAML available")
        except ImportError:
            logger.debug("PyYAML not available")
    
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file."""
        if not file_path.exists() or not file_path.is_file():
            return False
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return True
        elif suffix == '.json':
            return True
        elif suffix in ['.yaml', '.yml']:
            return self.yaml_available
        
        return False
    
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from data file."""
        if not self.can_extract(file_path):
            return self._create_failed_result(
                file_path, 
                "Data extractor not available for this file type"
            )
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                return self._extract_csv(file_path)
            elif suffix == '.json':
                return self._extract_json(file_path)
            elif suffix in ['.yaml', '.yml']:
                return self._extract_yaml(file_path)
            else:
                return self._create_failed_result(file_path, f"Unsupported file type: {suffix}")
        
        except Exception as e:
            logger.error(f"Failed to extract from {file_path}: {e}")
            return self._create_failed_result(file_path, str(e))
    
    def _extract_csv(self, file_path: Path) -> ExtractedDocument:
        """Extract text from CSV file."""
        if self.polars_available:
            content, extraction_time = self._time_extraction(
                self._extract_csv_polars, file_path
            )
        else:
            content, extraction_time = self._time_extraction(
                self._extract_csv_standard, file_path
            )
        
        metadata = self._create_metadata(
            file_path, 
            FileType.CSV, 
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
    
    def _extract_csv_polars(self, file_path: Path) -> str:
        """Extract CSV content using Polars (high performance)."""
        try:
            # Read CSV with Polars
            df = self.polars.read_csv(
                str(file_path),
                n_rows=self.max_rows if not self.sample_rows else None
            )
            
            content_parts = [f"# CSV Data: {file_path.name}\n"]
            
            # Add basic info
            content_parts.append(f"**Rows:** {df.shape[0]}")
            content_parts.append(f"**Columns:** {df.shape[1]}")
            content_parts.append(f"**Column Names:** {', '.join(df.columns)}\n")
            
            # Convert to markdown table format
            content_parts.append("## Data Preview\n")
            
            # Header
            header = "| " + " | ".join(df.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            content_parts.append(header)
            content_parts.append(separator)
            
            # Data rows (limit for readability)
            preview_rows = min(50, df.shape[0])
            for i in range(preview_rows):
                row_data = []
                for col in df.columns:
                    value = str(df[col][i]) if df[col][i] is not None else ""
                    # Escape markdown characters and limit length
                    value = value.replace("|", "\\|")[:100]
                    row_data.append(value)
                
                row = "| " + " | ".join(row_data) + " |"
                content_parts.append(row)
            
            if df.shape[0] > preview_rows:
                content_parts.append(f"\n... and {df.shape[0] - preview_rows} more rows")
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.warning(f"Polars CSV extraction failed, falling back to standard: {e}")
            return self._extract_csv_standard(file_path)
    
    def _extract_csv_standard(self, file_path: Path) -> str:
        """Extract CSV content using standard csv module."""
        content_parts = [f"# CSV Data: {file_path.name}\n"]
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
                    # Detect delimiter
                    sample = csvfile.read(1024)
                    csvfile.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    
                    reader = csv.reader(csvfile, delimiter=delimiter)
                    rows = list(reader)
                
                break
            except (UnicodeDecodeError, csv.Error):
                continue
        else:
            raise ValueError("Could not read CSV file with any encoding")
        
        if not rows:
            return "Empty CSV file"
        
        # Process rows
        header = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
        
        content_parts.append(f"**Rows:** {len(data_rows)}")
        content_parts.append(f"**Columns:** {len(header)}")
        content_parts.append(f"**Column Names:** {', '.join(header)}\n")
        
        # Add data preview
        content_parts.append("## Data Preview\n")
        
        # Header
        if header:
            header_line = "| " + " | ".join(header) + " |"
            separator = "| " + " | ".join(["---"] * len(header)) + " |"
            content_parts.append(header_line)
            content_parts.append(separator)
        
        # Data rows (limit for readability)
        preview_rows = min(50, len(data_rows))
        for row in data_rows[:preview_rows]:
            # Pad row to match header length
            padded_row = row + [""] * (len(header) - len(row)) if len(row) < len(header) else row[:len(header)]
            
            escaped_row = []
            for cell in padded_row:
                cell_str = str(cell).replace("|", "\\|")[:100]
                escaped_row.append(cell_str)
            
            row_line = "| " + " | ".join(escaped_row) + " |"
            content_parts.append(row_line)
        
        if len(data_rows) > preview_rows:
            content_parts.append(f"\n... and {len(data_rows) - preview_rows} more rows")
        
        return "\n".join(content_parts)
    
    def _extract_json(self, file_path: Path) -> ExtractedDocument:
        """Extract text from JSON file."""
        content, extraction_time = self._time_extraction(
            self._extract_json_content, file_path
        )
        
        metadata = self._create_metadata(
            file_path, 
            FileType.JSON, 
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
    
    def _extract_json_content(self, file_path: Path) -> str:
        """Extract content from JSON file."""
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        else:
            raise ValueError("Could not read JSON file")
        
        content_parts = [f"# JSON Data: {file_path.name}\n"]
        
        # Add basic structure info
        content_parts.append(self._describe_json_structure(data))
        
        # Add formatted content
        content_parts.append("## Content\n")
        content_parts.append("```json")
        content_parts.append(json.dumps(data, indent=2, ensure_ascii=False)[:5000])  # Limit size
        content_parts.append("```")
        
        return "\n".join(content_parts)
    
    def _extract_yaml(self, file_path: Path) -> ExtractedDocument:
        """Extract text from YAML file."""
        content, extraction_time = self._time_extraction(
            self._extract_yaml_content, file_path
        )
        
        metadata = self._create_metadata(
            file_path, 
            FileType.YAML, 
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
    
    def _extract_yaml_content(self, file_path: Path) -> str:
        """Extract content from YAML file."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    data = self.yaml.safe_load(f)
                break
            except (UnicodeDecodeError, self.yaml.YAMLError):
                continue
        else:
            raise ValueError("Could not read YAML file")
        
        content_parts = [f"# YAML Data: {file_path.name}\n"]
        
        # Add structure description
        content_parts.append(self._describe_json_structure(data, "YAML"))
        
        # Add formatted content
        content_parts.append("## Content\n")
        content_parts.append("```yaml")
        yaml_str = self.yaml.dump(data, default_flow_style=False, allow_unicode=True)
        content_parts.append(yaml_str[:5000])  # Limit size
        content_parts.append("```")
        
        return "\n".join(content_parts)
    
    def _describe_json_structure(self, data, format_name="JSON") -> str:
        """Describe the structure of JSON/YAML data."""
        description = [f"## {format_name} Structure\n"]
        
        if isinstance(data, dict):
            description.append(f"**Type:** Object/Dictionary")
            description.append(f"**Keys:** {len(data)}")
            if data:
                description.append(f"**Top-level keys:** {', '.join(list(data.keys())[:10])}")
                if len(data) > 10:
                    description.append(f"... and {len(data) - 10} more keys")
        
        elif isinstance(data, list):
            description.append(f"**Type:** Array/List")
            description.append(f"**Items:** {len(data)}")
            if data:
                item_types = set(type(item).__name__ for item in data[:100])
                description.append(f"**Item types:** {', '.join(item_types)}")
        
        else:
            description.append(f"**Type:** {type(data).__name__}")
            description.append(f"**Value:** {str(data)[:200]}")
        
        return "\n".join(description) + "\n"