"""
Document classifier - Hybrid approach (rule-based + ML)
"""
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classify documents as Purchase Order or Stock & Sales Report"""
    
    def __init__(self, config=None, ml_model=None):
        """
        Initialize classifier
        
        Args:
            config: ConfigLoader instance
            ml_model: Trained ML model (optional)
        """
        from .config_loader import ConfigLoader
        self.config = config or ConfigLoader()
        self.ml_model = ml_model
        
        # Load classification keywords and patterns
        self.po_keywords = self.config.get('classification.po_keywords', [])
        self.stock_keywords = self.config.get('classification.stock_sales_keywords', [])
        
        # Compile filename patterns
        self.po_patterns = self._compile_patterns(
            self.config.get('classification.filename_patterns.po', [])
        )
        self.stock_patterns = self._compile_patterns(
            self.config.get('classification.filename_patterns.stock_sales', [])
        )
    
    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """Compile regex patterns for matching"""
        compiled = []
        for pattern in patterns:
            try:
                # Case-insensitive pattern
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
        return compiled
    
    def classify(self, file_path: Path, content: Optional[str] = None) -> Tuple[str, float]:
        """
        Classify document
        
        Args:
            file_path: Path to file
            content: Optional pre-extracted text content
            
        Returns:
            Tuple of (classification, confidence)
            classification: 'purchase_order', 'stock_sales_report', or 'unknown'
            confidence: 0.0 to 1.0
        """
        filename = file_path.name.lower()
        
        # Rule-based classification
        rule_score_po, rule_score_stock = self._rule_based_classify(filename, content)
        
        # ML classification if model available
        ml_score_po, ml_score_stock = self._ml_classify(filename, content)
        
        # Combine scores
        combined_po = (rule_score_po * 0.6) + (ml_score_po * 0.4)
        combined_stock = (rule_score_stock * 0.6) + (ml_score_stock * 0.4)
        
        # Determine classification
        confidence_threshold = self.config.get('classification.combined_confidence_threshold', 0.6)
        
        if combined_po > combined_stock and combined_po >= confidence_threshold:
            return 'purchase_order', combined_po
        elif combined_stock > combined_po and combined_stock >= confidence_threshold:
            return 'stock_sales_report', combined_stock
        else:
            # Check which is higher, but below threshold
            if combined_po > combined_stock:
                return 'purchase_order', combined_po
            elif combined_stock > combined_po:
                return 'stock_sales_report', combined_stock
            else:
                return 'unknown', max(combined_po, combined_stock)
    
    def _rule_based_classify(self, filename: str, content: Optional[str]) -> Tuple[float, float]:
        """
        Rule-based classification using filename and content patterns
        
        Returns:
            Tuple of (po_score, stock_score)
        """
        po_score = 0.0
        stock_score = 0.0
        
        # Filename pattern matching
        po_filename_matches = sum(1 for pattern in self.po_patterns if pattern.search(filename))
        stock_filename_matches = sum(1 for pattern in self.stock_patterns if pattern.search(filename))
        
        # Filename keyword matching
        po_filename_keywords = sum(1 for kw in self.po_keywords if kw.lower() in filename)
        stock_filename_keywords = sum(1 for kw in self.stock_keywords if kw.lower() in filename)
        
        # Combine filename signals (increased weights for better accuracy)
        if po_filename_matches > 0 or po_filename_keywords > 0:
            po_score += min((po_filename_matches * 0.4) + (po_filename_keywords * 0.3), 0.7)
        if stock_filename_matches > 0 or stock_filename_keywords > 0:
            stock_score += min((stock_filename_matches * 0.4) + (stock_filename_keywords * 0.3), 0.7)
        
        # Content analysis (if available)
        if content:
            content_lower = content[:2000].lower()  # Use first 2000 chars
            
            # Check for PO keywords in content
            po_content_matches = sum(1 for kw in self.po_keywords if kw.lower() in content_lower)
            stock_content_matches = sum(1 for kw in self.stock_keywords if kw.lower() in content_lower)
            
            # Normalize by keyword count (increased weight)
            if po_content_matches > 0:
                po_score += min(po_content_matches * 0.15, 0.4)
            if stock_content_matches > 0:
                stock_score += min(stock_content_matches * 0.15, 0.4)
            
            # Specific pattern matching
            # Purchase Orders often have PO number patterns
            po_number_patterns = [
                r'purchase\s+order\s+(?:no|number|#)?\s*[:\-]?\s*[\d\w\-]+',
                r'po\s+(?:no|number|#)?\s*[:\-]?\s*[\d\w\-]+',
                r'order\s+number\s*[:\-]?\s*[\d\w\-]+'
            ]
            for pattern in po_number_patterns:
                if re.search(pattern, content_lower):
                    po_score += 0.25
                    break
            
            # Stock reports often have quantity fields
            stock_patterns = [
                r'opening\s+qty',
                r'receipt\s+qty',
                r'issue\s+qty',
                r'closing\s+qty',
                r'opening\s+balance',
                r'closing\s+balance',
                r'stock\s+statement',
                r'item\s+description'
            ]
            for pattern in stock_patterns:
                if re.search(pattern, content_lower):
                    stock_score += 0.25
                    break
        
        # Normalize scores to 0-1 range
        po_score = min(po_score, 1.0)
        stock_score = min(stock_score, 1.0)
        
        return po_score, stock_score
    
    def _ml_classify(self, filename: str, content: Optional[str]) -> Tuple[float, float]:
        """
        ML-based classification
        
        Returns:
            Tuple of (po_score, stock_score)
        """
        if self.ml_model is None:
            return 0.0, 0.0
        
        try:
            # Prepare features for ML model
            features = self._extract_features(filename, content)
            
            # Predict with ML model
            # Assuming model.predict_proba() returns [po_prob, stock_prob]
            prediction = self.ml_model.predict_proba([features])[0]
            
            # Adjust based on model output format
            if len(prediction) == 2:
                return float(prediction[0]), float(prediction[1])
            elif len(prediction) == 3:  # [po, stock, unknown]
                return float(prediction[0]), float(prediction[1])
            else:
                return 0.0, 0.0
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return 0.0, 0.0
    
    def _extract_features(self, filename: str, content: Optional[str]) -> str:
        """
        Extract features for ML model (combines filename + content)
        
        Returns:
            Combined feature string
        """
        features = [filename]
        
        if content:
            # Use first 1000 characters
            features.append(content[:1000])
        
        return " ".join(features)


