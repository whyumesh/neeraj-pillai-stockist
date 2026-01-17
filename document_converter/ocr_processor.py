"""
OCR Processor with PaddleOCR (primary) and Tesseract (fallback)
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image
import numpy as np

from .utils.image_preprocessing import preprocess_image, pil_to_cv2
from .config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor with PaddleOCR and Tesseract fallback"""
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize OCR processor
        
        Args:
            config: Configuration loader instance
        """
        self.config = config or ConfigLoader()
        self.paddleocr = None
        self._initialize_ocr()
    
    def _initialize_ocr(self) -> None:
        """Initialize OCR engines"""
        primary = self.config.get('ocr.primary', 'paddleocr')
        
        # Initialize PaddleOCR if available
        if primary == 'paddleocr' and PADDLEOCR_AVAILABLE:
            try:
                # Use English model, enable table structure
                self.paddleocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=False,
                    show_log=False
                )
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
                self.paddleocr = None
        
        # Check Tesseract availability
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract OCR not available. Install pytesseract and Tesseract engine.")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, float, str]:
        """
        Extract text from PDF (handles both digital and scanned)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, confidence, method_used)
        """
        # First try to extract text directly (digital PDF)
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If we got substantial text, it's likely digital
            if len(text.strip()) > 100:
                return text.strip(), 0.9, "digital_pdf"
        except Exception as e:
            logger.debug(f"Digital PDF extraction failed: {e}")
        
        # If digital extraction failed or yielded little text, try OCR
        if PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.config.get('ocr.dpi', 300),
                    first_page=1,
                    last_page=10  # Process first 10 pages for speed
                )
                
                all_text = []
                total_confidence = 0.0
                method_used = None
                
                for img in images:
                    text, conf, method = self.extract_text_from_image(img)
                    if text:
                        all_text.append(text)
                        total_confidence += conf
                        if not method_used:
                            method_used = method
                
                if all_text:
                    avg_confidence = total_confidence / len(images) if images else 0.0
                    return "\n".join(all_text), avg_confidence, method_used or "ocr"
            except Exception as e:
                logger.error(f"OCR from PDF images failed: {e}")
        
        return "", 0.0, "failed"
    
    def extract_text_from_image(self, image: Image.Image) -> Tuple[str, float, str]:
        """
        Extract text from image using OCR
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (extracted_text, confidence, method_used)
        """
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Preprocess image
        preprocess_config = self.config.get('ocr.preprocessing', {})
        processed_img = preprocess_image(
            img_array,
            deskew=preprocess_config.get('deskew', True),
            binarize=preprocess_config.get('binarize', True),
            noise_removal=preprocess_config.get('noise_removal', True)
        )
        
        # Try PaddleOCR first
        if self.paddleocr is not None:
            try:
                result = self.paddleocr.ocr(processed_img, cls=True)
                
                if result and result[0]:
                    # Extract text and confidence
                    texts = []
                    confidences = []
                    
                    for line in result[0]:
                        if line and len(line) == 2:
                            text, (bbox, conf) = line
                            texts.append(text)
                            confidences.append(conf)
                    
                    if texts:
                        avg_confidence = np.mean(confidences) if confidences else 0.0
                        return "\n".join(texts), float(avg_confidence), "paddleocr"
            except Exception as e:
                logger.debug(f"PaddleOCR failed: {e}")
        
        # Fallback to Tesseract
        if TESSERACT_AVAILABLE:
            try:
                # Convert back to PIL for Tesseract
                if CV2_AVAILABLE and len(processed_img.shape) == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = Image.fromarray(processed_img)
                
                # Extract text with confidence
                data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                
                texts = []
                confidences = []
                
                for i, text in enumerate(data['text']):
                    if text.strip():
                        texts.append(text)
                        conf = int(data['conf'][i]) / 100.0 if data['conf'][i] != '-1' else 0.0
                        confidences.append(conf)
                
                if texts:
                    avg_confidence = np.mean(confidences) if confidences else 0.0
                    return " ".join(texts), float(avg_confidence), "tesseract"
                else:
                    # Try simple extraction without confidence
                    text = pytesseract.image_to_string(pil_img)
                    return text.strip(), 0.5, "tesseract"
            except Exception as e:
                logger.error(f"Tesseract OCR failed: {e}")
        
        return "", 0.0, "failed"
    
    def extract_tables_from_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Extract tables from image using PaddleOCR structure recognition
        
        Args:
            image: PIL Image object
            
        Returns:
            List of extracted tables
        """
        if self.paddleocr is None:
            return []
        
        try:
            img_array = np.array(image)
            if CV2_AVAILABLE and len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # PaddleOCR can detect table structure
            result = self.paddleocr.ocr(img_array, cls=True)
            
            # Parse table structure from OCR result
            # This is a simplified version - could be enhanced with table detection
            tables = []
            if result and result[0]:
                # Group text by rows based on Y coordinates
                lines = []
                for item in result[0]:
                    if item and len(item) == 2:
                        text, (bbox, conf) = item
                        y_center = (bbox[0][1] + bbox[2][1]) / 2
                        lines.append((y_center, text))
                
                # Sort by Y coordinate and extract as table
                lines.sort(key=lambda x: x[0])
                table = [line[1] for line in lines]
                if table:
                    tables.append({"rows": table, "type": "detected"})
            
            return tables
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []

