"""
Main document processor - Orchestrates classification and conversion
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import pickle
import time

from .file_scanner import FileScanner
from .ocr_processor import OCRProcessor
from .document_classifier import DocumentClassifier
from .extractors.text_parser import TextParser
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
        
        # Initialize components (Stock & Sales only)
        self.ocr_processor = OCRProcessor(self.config)
        self.text_parser = TextParser()
        self.stock_extractor = StockSalesExtractor()
        self.excel_converter = ExcelConverter(self.config)
        
        # Load ML model if available
        self.ml_model = self._load_ml_model()
        self.classifier = DocumentClassifier(self.config, self.ml_model)
        
        # Initialize logger
        log_file = Path(self.config.get('paths.log_file', 'Output/ProcessingLog.json'))
        self.processing_logger = ProcessingLogger(log_file)
    
    def _load_ml_model(self):
        """Load trained ML model if available (prefers Ensemble, then Stock-only binary model)"""
        models_dir = Path(__file__).parent / "models"
        
        # Try ensemble model first (strongest model)
        model_path = models_dir / "classifier_model_ensemble.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    logger.info("Loaded Ensemble Voting Classifier model")
                    return model
            except Exception as e:
                logger.warning(f"Failed to load Ensemble model: {e}")
        
        # Try new Stock-only binary model
        model_path = models_dir / "classifier_model_stock_only.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    logger.info("Loaded Stock-only binary classifier model")
                    return model
            except Exception as e:
                logger.warning(f"Failed to load Stock-only model: {e}")
        
        # Fallback to old model (for backward compatibility)
        model_path = models_dir / "classifier_model.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    logger.info("Loaded legacy classifier model (consider retraining with Ensemble model)")
                    return model
            except Exception as e:
                logger.warning(f"Failed to load legacy ML model: {e}")
        
        logger.info("No ML model found - using rule-based classification only")
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
                error_msg, suggestions = self._generate_extraction_error(filename, file_type, is_scanned)
                self.processing_logger.log_processing(
                    filename, file_type, 'unknown', 0.0, 'error', error_msg,
                    items_count=0,
                    sections_count=0
                )
                return {
                    "status": "error",
                    "error": error_msg,
                    "suggestions": suggestions
                }
            
            # Classify document (binary: stock_sales_report or other)
            classification, confidence = self.classifier.classify(file_path, text)
            
            # Skip non-Stock files (only process Stock & Sales reports)
            if classification != 'stock_sales_report':
                self.processing_logger.log_processing(
                    filename, file_type, classification, confidence, 'skipped',
                    f"Not a Stock & Sales report (classification: {classification})",
                    items_count=0,
                    sections_count=0
                )
                return {
                    "status": "skipped",
                    "classification": classification,
                    "confidence": confidence,
                    "reason": "Not a Stock & Sales report"
                }
            
            # Extract data based on classification (only Stock & Sales) with retry logic
            extracted_data = self._extract_data_with_retry(file_path, file_type, text, classification)
            
            # Extract counts from data for logging
            items_count = None
            sections_count = None
            if extracted_data:
                items = extracted_data.get("items", [])
                items_count = len(items) if items else 0
                sections = extracted_data.get("sections", [])
                sections_count = len(sections) if sections else 0
            
            # Apply graceful degradation if extraction failed
            if classification == 'stock_sales_report' and items_count == 0:
                extracted_data = self._apply_graceful_degradation(
                    file_path, file_type, text, extracted_data
                )
                # Update counts after graceful degradation
                items = extracted_data.get("items", [])
                items_count = len(items) if items else 0
                sections = extracted_data.get("sections", [])
                sections_count = len(sections) if sections else 0
            
            # Convert to Excel
            output_path = self._get_output_path(file_path, classification)
            success = self._convert_to_excel(extracted_data, output_path, filename, classification)
            
            if success:
                self.processing_logger.log_processing(
                    filename, file_type, classification, confidence, 'success',
                    output_file=str(output_path),
                    items_count=items_count,
                    sections_count=sections_count
                )
                return {
                    "status": "success",
                    "classification": classification,
                    "confidence": confidence,
                    "output_file": str(output_path),
                    "items_count": items_count,
                    "sections_count": sections_count
                }
            else:
                error_msg, suggestions = self._generate_excel_error(filename, extracted_data)
                self.processing_logger.log_processing(
                    filename, file_type, classification, confidence, 'error', error_msg,
                    items_count=items_count,
                    sections_count=sections_count
                )
                return {
                    "status": "error",
                    "error": error_msg,
                    "suggestions": suggestions
                }
        
        except Exception as e:
            error_msg, suggestions = self._generate_user_friendly_error(e, filename, file_type)
            logger.error(f"Error processing {filename}: {e}")
            self.processing_logger.log_processing(
                filename, file_type, 'unknown', 0.0, 'error', error_msg,
                items_count=0,
                sections_count=0
            )
            return {
                "status": "error",
                "error": error_msg,
                "suggestions": suggestions,
                "technical_error": str(e)
            }
    
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
    
    def _extract_data_with_retry(self, file_path: Path, file_type: str, text: str, 
                                 classification: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Extract structured data with retry logic and exponential backoff
        
        Args:
            file_path: Path to file
            file_type: Type of file
            text: Extracted text content
            classification: Document classification
            max_retries: Maximum number of retry attempts
            
        Returns:
            Extracted data dictionary
        """
        if file_type == 'csv':
            parsed = self.text_parser.parse_csv(file_path)
            return {
                "headers": parsed.get("headers", []),
                "rows": parsed.get("rows", []),
                "type": "table"
            }
        
        # Only process Stock & Sales reports
        if classification == 'stock_sales_report':
            last_error = None
            last_result = None
            
            for attempt in range(max_retries):
                try:
                    if file_type == 'pdf':
                        result = self.stock_extractor.extract_from_pdf(str(file_path))
                    else:
                        result = self.stock_extractor.extract_from_text(text)
                    
                    # Check if extraction was successful (has items or at least metadata)
                    items = result.get("items", [])
                    if items or attempt == max_retries - 1:
                        # Success or last attempt - return result
                        if attempt > 0:
                            logger.info(f"Extraction succeeded on attempt {attempt + 1} for {file_path.name}")
                        return result
                    
                    # No items found, but not last attempt - retry
                    last_result = result
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Extraction returned 0 items for {file_path.name}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    last_error = e
                    wait_time = 2 ** attempt
                    logger.warning(f"Extraction failed for {file_path.name} on attempt {attempt + 1}/{max_retries}: {e}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} extraction attempts failed for {file_path.name}")
            
            # All retries exhausted - return last result or empty result
            if last_result:
                return last_result
            else:
                return {
                    "items": [],
                    "sections": [],
                    "period": None,
                    "diagnostics": {
                        "error": str(last_error) if last_error else "All retry attempts failed",
                        "retries": max_retries
                    }
                }
        
        # Should not reach here (non-Stock files are skipped earlier)
        return {"text": text, "type": "text"}
    
    def _extract_data(self, file_path: Path, file_type: str, text: str, classification: str) -> Dict[str, Any]:
        """Extract structured data from file (Stock & Sales only) - legacy method"""
        return self._extract_data_with_retry(file_path, file_type, text, classification, max_retries=1)
    
    def _apply_graceful_degradation(self, file_path: Path, file_type: str, text: str, 
                                    extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply graceful degradation when extraction fails
        
        If items extraction failed, try to extract at least:
        - Sections
        - Period
        - Metadata
        - Create Excel with headers and note about incomplete extraction
        
        Args:
            file_path: Path to file
            file_type: Type of file
            text: Extracted text content
            extracted_data: Previously extracted data (may be empty)
            
        Returns:
            Enhanced extracted data with at least metadata
        """
        logger.info(f"Applying graceful degradation for {file_path.name} - extraction returned 0 items")
        
        # Ensure we have at least basic structure
        if not extracted_data:
            extracted_data = {
                "items": [],
                "sections": [],
                "period": None,
                "diagnostics": {}
            }
        
        # Try to extract sections and period even if items failed
        if not extracted_data.get("sections"):
            sections = self.stock_extractor._extract_sections(text)
            extracted_data["sections"] = sections
            logger.info(f"Extracted {len(sections)} sections via graceful degradation")
        
        if not extracted_data.get("period"):
            period = self.stock_extractor._extract_period(text)
            extracted_data["period"] = period
            if period:
                logger.info(f"Extracted period via graceful degradation: {period}")
        
        # Add diagnostic note about incomplete extraction
        diagnostics = extracted_data.get("diagnostics", {})
        diagnostics["graceful_degradation"] = True
        diagnostics["degradation_reason"] = "No items extracted - providing metadata only"
        diagnostics["note"] = "This file may require manual review. Extraction was incomplete."
        extracted_data["diagnostics"] = diagnostics
        
        # Try alternative extraction methods
        # Try pattern-based extraction as fallback
        try:
            pattern_items = self.stock_extractor._extract_items_pattern_based(text)
            if pattern_items:
                logger.info(f"Pattern-based extraction found {len(pattern_items)} items")
                extracted_data["items"] = pattern_items
                diagnostics["fallback_method"] = "pattern_based"
                return extracted_data
        except Exception as e:
            logger.debug(f"Pattern-based fallback failed: {e}")
        
        # Try manual table detection as fallback
        try:
            manual_items = self.stock_extractor._extract_items_manual_detection(text)
            if manual_items:
                logger.info(f"Manual detection found {len(manual_items)} items")
                extracted_data["items"] = manual_items
                diagnostics["fallback_method"] = "manual_detection"
                return extracted_data
        except Exception as e:
            logger.debug(f"Manual detection fallback failed: {e}")
        
        # If still no items, at least we have metadata
        logger.warning(f"Graceful degradation complete for {file_path.name} - metadata extracted but no items found")
        return extracted_data
    
    def _generate_user_friendly_error(self, error: Exception, filename: str, file_type: str) -> tuple:
        """Generate user-friendly error message with suggestions"""
        error_str = str(error).lower()
        
        if "password" in error_str or "encrypted" in error_str:
            return (
                f"PDF file '{filename}' is password-protected",
                [
                    "Provide the password to decrypt the PDF",
                    "Remove password protection from the PDF before processing",
                    "Contact the file owner for the password"
                ]
            )
        elif "corrupted" in error_str or "invalid" in error_str or "cannot read" in error_str:
            return (
                f"PDF file '{filename}' appears to be corrupted or invalid",
                [
                    "Try opening the PDF in a PDF viewer to verify it's not corrupted",
                    "Re-save or re-create the PDF file",
                    "Check if the file was downloaded completely",
                    "Try converting the PDF to a different format"
                ]
            )
        elif "permission" in error_str or "access" in error_str:
            return (
                f"Cannot access file '{filename}' - permission denied",
                [
                    "Check file permissions",
                    "Ensure the file is not open in another application",
                    "Run the application with appropriate permissions"
                ]
            )
        elif "not found" in error_str or "does not exist" in error_str:
            return (
                f"File '{filename}' not found",
                [
                    "Verify the file path is correct",
                    "Check if the file was moved or deleted",
                    "Ensure the file exists in the specified location"
                ]
            )
        else:
            return (
                f"Error processing file '{filename}': {str(error)}",
                [
                    "Check the file format is supported",
                    "Verify the file is not corrupted",
                    "Try processing the file manually",
                    "Contact support if the issue persists"
                ]
            )
    
    def _generate_extraction_error(self, filename: str, file_type: str, is_scanned: bool) -> tuple:
        """Generate user-friendly error for text extraction failures"""
        if file_type == 'pdf':
            if is_scanned:
                return (
                    f"Could not extract text from scanned PDF '{filename}'",
                    [
                        "The PDF appears to be scanned (image-based)",
                        "Ensure OCR is properly configured",
                        "Check if OCR dependencies (PaddleOCR/Tesseract) are installed",
                        "Try preprocessing the PDF images for better OCR results"
                    ]
                )
            else:
                return (
                    f"Could not extract text from PDF '{filename}'",
                    [
                        "The PDF may be corrupted or have an unsupported format",
                        "Try opening the PDF in a PDF viewer to verify",
                        "Check if the PDF has text layers (not just images)",
                        "Consider using OCR if the PDF is image-based"
                    ]
                )
        else:
            return (
                f"Could not extract text from file '{filename}'",
                [
                    f"Verify the file format '{file_type}' is supported",
                    "Check file encoding (try UTF-8)",
                    "Ensure the file is not corrupted",
                    "Try opening the file in a text editor to verify"
                ]
            )
    
    def _generate_excel_error(self, filename: str, extracted_data: Dict[str, Any]) -> tuple:
        """Generate user-friendly error for Excel conversion failures"""
        items = extracted_data.get("items", [])
        diagnostics = extracted_data.get("diagnostics", {})
        
        if len(items) == 0:
            return (
                f"Failed to create Excel file for '{filename}' - no data extracted",
                [
                    "No items were extracted from the PDF",
                    "Check if the PDF format matches expected Stock & Sales report format",
                    "Review extraction diagnostics for details",
                    "Try manual extraction or contact support"
                ]
            )
        else:
            return (
                f"Failed to create Excel file for '{filename}'",
                [
                    "Check if output directory exists and is writable",
                    "Verify sufficient disk space is available",
                    "Ensure Excel file is not open in another application",
                    "Check file permissions for the output directory"
                ]
            )
    
    def _convert_to_excel(self, data: Dict[str, Any], output_path: Path, 
                         source_filename: str, classification: str) -> bool:
        """Convert extracted data to Excel (Stock & Sales only)"""
        if classification == 'stock_sales_report':
            return self.excel_converter.convert_stock_sales_to_excel(data, output_path, source_filename)
        
        # Generic table conversion fallback
        if data.get("type") == "table":
            headers = data.get("headers", [])
            rows = data.get("rows", [])
            return self.excel_converter.convert_table_data_to_excel(
                headers, rows, output_path, source_filename
            )
        return False
    
    def _get_output_path(self, file_path: Path, classification: str) -> Path:
        """Get output Excel file path (Stock & Sales only)"""
        base_name = file_path.stem
        
        if classification == 'stock_sales_report':
            output_dir = Path(self.config.get('paths.stock_output', 'Output/StockSalesReports'))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{base_name}_SS.xlsx"
        else:
            # Fallback (should not happen for Stock-only system)
            output_dir = Path(self.config.get('paths.output_folder', 'Output'))
            output_dir.mkdir(parents=True, exist_ok=True)
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

