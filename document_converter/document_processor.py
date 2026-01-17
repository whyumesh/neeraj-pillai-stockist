"""
Main document processor - Orchestrates classification and conversion
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import pickle

from .file_scanner import FileScanner
from .ocr_processor import OCRProcessor
from .document_classifier import DocumentClassifier
from .extractors.text_parser import TextParser
from .extractors.po_extractor import PurchaseOrderExtractor
from .extractors.stock_sales_extractor import StockSalesExtractor
from .excel_converter import ExcelConverter
from .utils.logging_utils import ProcessingLogger
from .config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing orchestrator"""
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize processor"""
        self.config = config or ConfigLoader()
        
        # Initialize components
        self.ocr_processor = OCRProcessor(self.config)
        self.text_parser = TextParser()
        self.po_extractor = PurchaseOrderExtractor()
        self.stock_extractor = StockSalesExtractor()
        self.excel_converter = ExcelConverter(self.config)
        
        # Load ML model if available
        self.ml_model = self._load_ml_model()
        self.classifier = DocumentClassifier(self.config, self.ml_model)
        
        # Initialize logger
        log_file = Path(self.config.get('paths.log_file', 'Output/ProcessingLog.json'))
        self.processing_logger = ProcessingLogger(log_file)
    
    def _load_ml_model(self):
        """Load trained ML model if available"""
        model_path = Path(__file__).parent / "models" / "classifier_model.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
        return None
    
    def process_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single file
        
        Args:
            file_info: File information from FileScanner
            
        Returns:
            Processing result dictionary
        """
        file_path = file_info['path']
        filename = file_info['filename']
        file_type = file_info['file_type']
        is_scanned = file_info['is_scanned']
        
        logger.info(f"Processing: {filename}")
        
        try:
            # Extract text content
            text = self._extract_text(file_path, file_type, is_scanned)
            
            if not text:
                error = "Could not extract text from file"
                self.processing_logger.log_processing(
                    filename, file_type, 'unknown', 0.0, 'error', error
                )
                return {"status": "error", "error": error}
            
            # Classify document
            classification, confidence = self.classifier.classify(file_path, text)
            
            # Extract data based on classification
            extracted_data = self._extract_data(file_path, file_type, text, classification)
            
            # Convert to Excel
            output_path = self._get_output_path(file_path, classification)
            success = self._convert_to_excel(extracted_data, output_path, filename, classification)
            
            if success:
                self.processing_logger.log_processing(
                    filename, file_type, classification, confidence, 'success',
                    output_file=str(output_path)
                )
                return {
                    "status": "success",
                    "classification": classification,
                    "confidence": confidence,
                    "output_file": str(output_path)
                }
            else:
                error = "Failed to create Excel file"
                self.processing_logger.log_processing(
                    filename, file_type, classification, confidence, 'error', error
                )
                return {"status": "error", "error": error}
        
        except Exception as e:
            error = str(e)
            logger.error(f"Error processing {filename}: {e}")
            self.processing_logger.log_processing(
                filename, file_type, 'unknown', 0.0, 'error', error
            )
            return {"status": "error", "error": error}
    
    def _extract_text(self, file_path: Path, file_type: str, is_scanned: bool) -> str:
        """Extract text from file"""
        if file_type == 'pdf':
            if is_scanned:
                text, _, _ = self.ocr_processor.extract_text_from_pdf(file_path)
            else:
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                except:
                    text, _, _ = self.ocr_processor.extract_text_from_pdf(file_path)
            return text
        
        elif file_type == 'text':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except:
                    return ""
        
        elif file_type == 'csv':
            # For CSV, return empty - will be handled separately
            return ""
        
        elif file_type == 'image':
            # For images, use OCR
            from PIL import Image
            img = Image.open(file_path)
            text, _, _ = self.ocr_processor.extract_text_from_image(img)
            return text
        
        return ""
    
    def _extract_data(self, file_path: Path, file_type: str, text: str, classification: str) -> Dict[str, Any]:
        """Extract structured data from file"""
        if file_type == 'csv':
            parsed = self.text_parser.parse_csv(file_path)
            return {
                "headers": parsed.get("headers", []),
                "rows": parsed.get("rows", []),
                "type": "table"
            }
        
        if classification == 'purchase_order':
            if file_type == 'pdf':
                return self.po_extractor.extract_from_pdf(str(file_path))
            else:
                return self.po_extractor.extract_from_text(text)
        
        elif classification == 'stock_sales_report':
            if file_type == 'pdf':
                return self.stock_extractor.extract_from_pdf(str(file_path))
            else:
                return self.stock_extractor.extract_from_text(text)
        
        else:
            # Generic text parsing
            if file_type == 'text':
                parsed = self.text_parser.parse_txt(file_path)
                return {
                    "headers": parsed.get("headers", []),
                    "rows": parsed.get("rows", []),
                    "type": "table"
                }
            else:
                # Return raw text
                return {"text": text, "type": "text"}
    
    def _convert_to_excel(self, data: Dict[str, Any], output_path: Path, 
                         source_filename: str, classification: str) -> bool:
        """Convert extracted data to Excel"""
        if classification == 'purchase_order':
            return self.excel_converter.convert_po_to_excel(data, output_path, source_filename)
        
        elif classification == 'stock_sales_report':
            return self.excel_converter.convert_stock_sales_to_excel(data, output_path, source_filename)
        
        else:
            # Generic table conversion
            if data.get("type") == "table":
                headers = data.get("headers", [])
                rows = data.get("rows", [])
                return self.excel_converter.convert_table_data_to_excel(
                    headers, rows, output_path, source_filename
                )
            return False
    
    def _get_output_path(self, file_path: Path, classification: str) -> Path:
        """Get output Excel file path"""
        base_name = file_path.stem
        
        if classification == 'purchase_order':
            output_dir = Path(self.config.get('paths.po_output', 'Output/PurchaseOrders'))
            output_path = output_dir / f"{base_name}_PO.xlsx"
        elif classification == 'stock_sales_report':
            output_dir = Path(self.config.get('paths.stock_output', 'Output/StockSalesReports'))
            output_path = output_dir / f"{base_name}_SS.xlsx"
        else:
            output_dir = Path(self.config.get('paths.output_folder', 'Output'))
            output_path = output_dir / f"{base_name}.xlsx"
        
        return output_path
    
    def process_folder(self, input_folder: Path) -> Dict[str, Any]:
        """
        Process all files in folder
        
        Args:
            input_folder: Path to input folder
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing folder: {input_folder}")
        
        scanner = FileScanner(input_folder)
        files = scanner.scan_files()
        
        results = []
        for file_info in files:
            result = self.process_file(file_info)
            results.append(result)
        
        # Save log
        self.processing_logger.save_logs()
        
        # Get statistics
        stats = self.processing_logger.get_statistics()
        
        return {
            "total_files": len(files),
            "results": results,
            "statistics": stats
        }

