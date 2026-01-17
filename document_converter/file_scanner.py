"""
File scanner to detect and categorize files
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FileScanner:
    """Scan and categorize files for processing"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.csv': 'csv',
        # '.xls': 'excel',  # Skip Excel files - user doesn't want them converted
        # '.xlsx': 'excel',  # Skip Excel files - user doesn't want them converted
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.tif': 'image',
        '.tiff': 'image'
    }
    
    def __init__(self, input_folder: Path):
        """
        Initialize file scanner
        
        Args:
            input_folder: Path to EmailAttachments or input folder
        """
        self.input_folder = Path(input_folder)
    
    def scan_files(self, recursive: bool = True) -> List[Dict[str, any]]:
        """
        Scan folder for supported files
        
        Args:
            recursive: Whether to scan subdirectories
            
        Returns:
            List of file info dictionaries
        """
        files = []
        
        if not self.input_folder.exists():
            logger.error(f"Input folder does not exist: {self.input_folder}")
            return files
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in self.input_folder.glob(pattern):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                if file_ext in self.SUPPORTED_EXTENSIONS:
                    file_info = self._analyze_file(file_path)
                    if file_info:
                        files.append(file_info)
        
        logger.info(f"Scanned {len(files)} supported files from {self.input_folder}")
        return files
    
    def _analyze_file(self, file_path: Path) -> Optional[Dict[str, any]]:
        """
        Analyze file and extract metadata
        
        Args:
            file_path: Path to file
            
        Returns:
            File info dictionary or None if file cannot be processed
        """
        try:
            file_ext = file_path.suffix.lower()
            file_type = self.SUPPORTED_EXTENSIONS.get(file_ext)
            
            if not file_type:
                return None
            
            # Check if file is scanned or digital
            is_scanned = self._is_scanned_file(file_path, file_type)
            
            file_info = {
                "path": file_path,
                "filename": file_path.name,
                "extension": file_ext,
                "file_type": file_type,
                "is_scanned": is_scanned,
                "size": file_path.stat().st_size,
                "modified_time": file_path.stat().st_mtime
            }
            
            return file_info
        
        except Exception as e:
            logger.warning(f"Error analyzing file {file_path}: {e}")
            return None
    
    def _is_scanned_file(self, file_path: Path, file_type: str) -> bool:
        """
        Determine if file is scanned (requires OCR) or digital
        
        Args:
            file_path: Path to file
            file_type: Detected file type
            
        Returns:
            True if file is scanned, False if digital
        """
        # Images are always considered scanned
        if file_type == 'image':
            return True
        
        # For PDFs, check if they contain text
        if file_type == 'pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    if len(reader.pages) == 0:
                        return True
                    
                    # Check first page for text
                    first_page = reader.pages[0]
                    text = first_page.extract_text()
                    
                    # If less than 100 characters, likely scanned
                    if text and len(text.strip()) > 100:
                        return False
                    return True
            except Exception:
                # If we can't read it, assume scanned
                return True
        
        # Text and CSV files are always digital
        if file_type in ('text', 'csv', 'excel'):
            return False
        
        return False
    
    def filter_by_type(self, files: List[Dict], file_type: str) -> List[Dict]:
        """
        Filter files by type
        
        Args:
            files: List of file info dictionaries
            file_type: Type to filter by
            
        Returns:
            Filtered list
        """
        return [f for f in files if f['file_type'] == file_type]
    
    def filter_scanned(self, files: List[Dict]) -> List[Dict]:
        """
        Filter only scanned files (require OCR)
        
        Args:
            files: List of file info dictionaries
            
        Returns:
            Filtered list of scanned files
        """
        return [f for f in files if f['is_scanned']]


