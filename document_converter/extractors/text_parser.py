"""
Text and CSV file parser
"""
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TextParser:
    """Parse TXT and CSV files"""
    
    def __init__(self):
        """Initialize text parser"""
        pass
    
    def parse_csv(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary with parsed data
        """
        try:
            # Try reading with pandas first (handles various encodings)
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            except Exception as e:
                logger.error(f"Failed to read CSV {file_path}: {e}")
                return {"headers": [], "rows": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to read CSV {file_path}: {e}")
            return {"headers": [], "rows": [], "error": str(e)}
        
        # Convert to list of dictionaries
        headers = df.columns.tolist()
        rows = df.values.tolist()
        
        return {
            "headers": headers,
            "rows": rows,
            "dataframe": df
        }
    
    def parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse TXT file with pattern matching
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Dictionary with parsed data
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            text = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                return {"headers": [], "rows": [], "error": "Could not decode file"}
        except Exception as e:
            logger.error(f"Failed to read TXT {file_path}: {e}")
            return {"headers": [], "rows": [], "error": str(e)}
        
        # Try to detect if it's tab-delimited or space-delimited
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {"headers": [], "rows": [], "error": "Empty file"}
        
        # Check if it looks like tab-delimited
        if '\t' in non_empty_lines[0]:
            return self._parse_delimited(text, delimiter='\t')
        
        # Check if it looks like CSV-like (comma separated)
        if ',' in non_empty_lines[0] and non_empty_lines[0].count(',') >= 2:
            return self._parse_delimited(text, delimiter=',')
        
        # Try space-delimited (multiple spaces)
        if re.search(r'\s{2,}', non_empty_lines[0]):
            return self._parse_space_delimited(text)
        
        # Otherwise, try to parse as structured text
        return self._parse_structured_text(text)
    
    def _parse_delimited(self, text: str, delimiter: str) -> Dict[str, Any]:
        """Parse delimited text"""
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {"headers": [], "rows": []}
        
        # First line as headers if it looks like headers
        headers = non_empty_lines[0].split(delimiter)
        headers = [h.strip() for h in headers]
        
        # Remaining lines as data
        rows = []
        for line in non_empty_lines[1:]:
            if line.strip():
                row = line.split(delimiter)
                row = [cell.strip() for cell in row]
                # Pad or trim to match header length
                while len(row) < len(headers):
                    row.append("")
                row = row[:len(headers)]
                rows.append(row)
        
        return {"headers": headers, "rows": rows}
    
    def _parse_space_delimited(self, text: str) -> Dict[str, Any]:
        """Parse space-delimited text (multiple spaces)"""
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {"headers": [], "rows": []}
        
        # Split by multiple spaces
        headers = re.split(r'\s{2,}', non_empty_lines[0])
        headers = [h.strip() for h in headers]
        
        rows = []
        for line in non_empty_lines[1:]:
            if line.strip():
                row = re.split(r'\s{2,}', line)
                row = [cell.strip() for cell in row]
                while len(row) < len(headers):
                    row.append("")
                row = row[:len(headers)]
                rows.append(row)
        
        return {"headers": headers, "rows": rows}
    
    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """
        Parse structured text with pattern matching
        
        Looks for common patterns like:
        - Item: value
        - Column headers followed by data rows
        """
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {"headers": [], "rows": []}
        
        # Try to find header line (contains common header keywords)
        header_keywords = ['item', 'product', 'name', 'qty', 'quantity', 'price', 
                          'amount', 'value', 'date', 'code', 'number']
        
        header_idx = None
        for i, line in enumerate(non_empty_lines[:10]):  # Check first 10 lines
            line_lower = line.lower()
            keyword_count = sum(1 for kw in header_keywords if kw in line_lower)
            if keyword_count >= 2:  # Found header if 2+ keywords
                header_idx = i
                break
        
        if header_idx is not None:
            # Extract headers
            header_line = non_empty_lines[header_idx]
            # Try splitting by common delimiters
            if '\t' in header_line:
                headers = [h.strip() for h in header_line.split('\t')]
            elif ',' in header_line:
                headers = [h.strip() for h in header_line.split(',')]
            else:
                headers = re.split(r'\s{2,}', header_line)
                headers = [h.strip() for h in headers]
            
            # Extract data rows
            rows = []
            for line in non_empty_lines[header_idx + 1:]:
                if line.strip() and not line.startswith('---'):  # Skip separators
                    if '\t' in line:
                        row = [cell.strip() for cell in line.split('\t')]
                    elif ',' in line and line.count(',') >= 2:
                        row = [cell.strip() for cell in line.split(',')]
                    else:
                        row = re.split(r'\s{2,}', line)
                        row = [cell.strip() for cell in row]
                    
                    # Pad or trim
                    while len(row) < len(headers):
                        row.append("")
                    row = row[:len(headers)]
                    
                    # Only add if row has some non-empty values
                    if any(cell.strip() for cell in row):
                        rows.append(row)
            
            return {"headers": headers, "rows": rows}
        
        # If no header found, return as single column
        return {"headers": ["Content"], "rows": [[line] for line in non_empty_lines]}


