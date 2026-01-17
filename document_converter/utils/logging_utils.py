"""
Logging utilities for processing tracking
"""
import json
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
        
        # Setup Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_processing(self, 
                      filename: str,
                      file_type: str,
                      classification: str,
                      confidence: float,
                      status: str,
                      error: Optional[str] = None,
                      output_file: Optional[str] = None) -> None:
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
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "file_type": file_type,
            "classification": classification,
            "confidence": confidence,
            "status": status,
            "error": error,
            "output_file": output_file
        }
        
        self.logs.append(log_entry)
        
        # Write to JSON log file
        if self.log_file:
            self.save_logs()
        
        # Also log to console
        if status == "success":
            self.logger.info(f"Processed {filename} -> {classification} (confidence: {confidence:.2f})")
        elif status == "error":
            self.logger.error(f"Error processing {filename}: {error}")
        else:
            self.logger.warning(f"Skipped {filename}: {error}")
    
    def save_logs(self) -> None:
        """Save logs to JSON file"""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = len(self.logs)
        successful = sum(1 for log in self.logs if log['status'] == 'success')
        errors = sum(1 for log in self.logs if log['status'] == 'error')
        skipped = sum(1 for log in self.logs if log['status'] == 'skipped')
        
        po_count = sum(1 for log in self.logs if log.get('classification') == 'purchase_order')
        stock_count = sum(1 for log in self.logs if log.get('classification') == 'stock_sales_report')
        
        return {
            "total_files": total,
            "successful": successful,
            "errors": errors,
            "skipped": skipped,
            "purchase_orders": po_count,
            "stock_sales_reports": stock_count,
            "success_rate": (successful / total * 100) if total > 0 else 0
        }


