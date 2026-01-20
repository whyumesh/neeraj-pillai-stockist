"""
Logging utilities for processing tracking
"""
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class ProcessingLogger:
    """Logger for document processing operations"""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize logger
        
        Args:
            log_file: Path to JSON log file for structured logging
        """
        self.log_file = log_file
        self.logs: List[Dict[str, Any]] = []
        
        # Create CSV log file path (same directory as JSON log, with .csv extension)
        if log_file:
            self.csv_log_file = log_file.parent / f"{log_file.stem}_conversion_log.csv"
        else:
            self.csv_log_file = None
        
        # Initialize CSV log file with headers if it doesn't exist
        if self.csv_log_file:
            self._init_csv_log()
        
        # Setup Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_csv_log(self) -> None:
        """Initialize CSV log file with headers if it doesn't exist"""
        if self.csv_log_file and not self.csv_log_file.exists():
            self.csv_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'File Name',
                    'File Type',
                    'Classification',
                    'Conversion Possible',
                    'Items Extracted',
                    'Sections Found',
                    'Status',
                    'Confidence',
                    'Output File',
                    'Error/Notes'
                ])
    
    def log_processing(self, 
                      filename: str,
                      file_type: str,
                      classification: str,
                      confidence: float,
                      status: str,
                      error: Optional[str] = None,
                      output_file: Optional[str] = None,
                      items_count: Optional[int] = None,
                      sections_count: Optional[int] = None) -> None:
        """
        Log a processing event
        
        Args:
            filename: Source filename
            file_type: Detected file type
            classification: Classification result
            confidence: Classification confidence
            status: Processing status (success, error, skipped)
            error: Error message if any
            output_file: Output Excel file path if successful
            items_count: Number of items extracted (optional)
            sections_count: Number of sections found (optional)
        """
        timestamp = datetime.now().isoformat()
        
        # Determine if conversion is possible
        conversion_possible = "Yes" if status == "success" and (items_count is None or items_count > 0) else "No"
        if status == "error" or error:
            conversion_possible = "No"
        
        log_entry = {
            "timestamp": timestamp,
            "filename": filename,
            "file_type": file_type,
            "classification": classification,
            "confidence": confidence,
            "status": status,
            "error": error,
            "output_file": output_file,
            "items_count": items_count,
            "sections_count": sections_count,
            "conversion_possible": conversion_possible
        }
        
        self.logs.append(log_entry)
        
        # Write to JSON log file
        if self.log_file:
            self.save_logs()
        
        # Write to CSV log file
        if self.csv_log_file:
            self._write_csv_entry(
                timestamp, filename, file_type, classification, conversion_possible,
                items_count or 0, sections_count or 0, status, confidence,
                output_file, error or ""
            )
        
        # Also log to console
        if status == "success":
            items_info = f" ({items_count} items)" if items_count is not None else ""
            self.logger.info(f"Processed {filename} -> {classification} (confidence: {confidence:.2f}){items_info}")
        elif status == "error":
            self.logger.error(f"Error processing {filename}: {error}")
        else:
            self.logger.warning(f"Skipped {filename}: {error}")
    
    def _write_csv_entry(self, timestamp: str, filename: str, file_type: str,
                        classification: str, conversion_possible: str,
                        items_count: int, sections_count: int, status: str,
                        confidence: float, output_file: Optional[str],
                        error_notes: str) -> None:
        """Write a single entry to CSV log file"""
        if not self.csv_log_file:
            return
        
        self.csv_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Append mode - file already has headers
        with open(self.csv_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                filename,
                file_type,
                classification,
                conversion_possible,
                items_count,
                sections_count,
                status,
                f"{confidence:.2f}" if confidence else "",
                output_file or "",
                error_notes
            ])
    
    def save_logs(self) -> None:
        """Save logs to JSON file"""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics (Stock & Sales only)"""
        total = len(self.logs)
        successful = sum(1 for log in self.logs if log['status'] == 'success')
        errors = sum(1 for log in self.logs if log['status'] == 'error')
        skipped = sum(1 for log in self.logs if log['status'] == 'skipped')
        
        stock_count = sum(1 for log in self.logs if log.get('classification') == 'stock_sales_report')
        other_count = sum(1 for log in self.logs if log.get('classification') == 'other')
        
        return {
            "total_files": total,
            "successful": successful,
            "errors": errors,
            "skipped": skipped,
            "stock_sales_reports": stock_count,
            "other_documents": other_count,
            "success_rate": (successful / total * 100) if total > 0 else 0
        }


