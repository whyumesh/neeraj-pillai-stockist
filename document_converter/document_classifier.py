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
        
        # IMPROVED: Combine scores with better weighting
        # If ML model not available, use rule-based scores directly (scaled up)
        # If ML model available, combine but boost rule-based confidence
        
        if self.ml_model is None:
            # No ML model - use rule-based scores directly but ensure high confidence
            combined_po = rule_score_po
            combined_stock = rule_score_stock
            
            # Boost scores when rule-based is confident (>0.6)
            if rule_score_po > 0.6:
                combined_po = min(rule_score_po * 1.1, 0.98)  # Scale up to near-max
            if rule_score_stock > 0.6:
                combined_stock = min(rule_score_stock * 1.1, 0.98)  # Scale up to near-max
        else:
            # ML model available - weighted combination but favor rule-based
            combined_po = (rule_score_po * 0.7) + (ml_score_po * 0.3)
            combined_stock = (rule_score_stock * 0.7) + (ml_score_stock * 0.3)
            
            # If rule-based is very confident, boost the combined score
            if rule_score_po > 0.7 and combined_po < 0.85:
                combined_po = min(combined_po * 1.15, 0.98)
            if rule_score_stock > 0.7 and combined_stock < 0.85:
                combined_stock = min(combined_stock * 1.15, 0.98)
        
        # Ensure minimum confidence when clear winner
        score_diff = abs(combined_po - combined_stock)
        if score_diff > 0.2:  # Clear winner
            if combined_po > combined_stock and combined_po < 0.75:
                combined_po = 0.85  # Minimum high confidence for clear PO
            elif combined_stock > combined_po and combined_stock < 0.75:
                combined_stock = 0.85  # Minimum high confidence for clear stock
        
        # Normalize to ensure valid range
        combined_po = min(max(combined_po, 0.0), 0.99)
        combined_stock = min(max(combined_stock, 0.0), 0.99)
        
        # Determine classification
        confidence_threshold = self.config.get('classification.combined_confidence_threshold', 0.3)
        
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
        IMPROVED: Higher base scores to achieve maximum confidence
        
        Returns:
            Tuple of (po_score, stock_score)
        """
        po_score = 0.0
        stock_score = 0.0
        
        # Filename pattern matching - HIGHER WEIGHTS for better confidence
        po_filename_matches = sum(1 for pattern in self.po_patterns if pattern.search(filename))
        stock_filename_matches = sum(1 for pattern in self.stock_patterns if pattern.search(filename))
        
        # Filename keyword matching - EXPANDED matching
        po_filename_keywords = sum(1 for kw in self.po_keywords if kw.lower() in filename)
        stock_filename_keywords = sum(1 for kw in self.stock_keywords if kw.lower() in filename)
        
        # Additional common patterns in filenames
        # PO patterns: PO, PO_, _PO, purchase_order, etc.
        if re.search(r'\bpo[_\-\s]?\d+|purchase[\s_-]?order|order[\s_-]?\d+', filename, re.IGNORECASE):
            po_filename_keywords += 2  # Strong indicator
        
        # Stock patterns: STOCK, ST-, ST_, statement, report
        if re.search(r'\bstock|statement|st[\s_-]?[\d\-]|stockandsales', filename, re.IGNORECASE):
            stock_filename_keywords += 2  # Strong indicator
        
        # Combine filename signals - MUCH HIGHER BASE SCORES
        if po_filename_matches > 0 or po_filename_keywords > 0:
            # Base score from filename is now 0.8-0.95 for clear matches
            filename_base = min((po_filename_matches * 0.5) + (po_filename_keywords * 0.4), 0.95)
            po_score += filename_base
        
        if stock_filename_matches > 0 or stock_filename_keywords > 0:
            # Base score from filename is now 0.8-0.95 for clear matches
            filename_base = min((stock_filename_matches * 0.5) + (stock_filename_keywords * 0.4), 0.95)
            stock_score += filename_base
        
        # Content analysis (if available) - INCREASED WEIGHTS
        if content:
            content_lower = content[:3000].lower()  # Use more content for better matching
            
            # Check for PO keywords in content - INCREASED SCORING
            po_content_matches = sum(1 for kw in self.po_keywords if kw.lower() in content_lower)
            stock_content_matches = sum(1 for kw in self.stock_keywords if kw.lower() in content_lower)
            
            # Content scoring - higher weights
            if po_content_matches > 0:
                # Each keyword match adds more value
                po_score += min(po_content_matches * 0.25, 0.6)
            if stock_content_matches > 0:
                stock_score += min(stock_content_matches * 0.25, 0.6)
            
            # Specific pattern matching - STRONG INDICATORS
            # Purchase Orders often have PO number patterns
            po_number_patterns = [
                r'purchase\s+order\s+(?:no|number|#)?\s*[:\-]?\s*[\d\w\-]+',
                r'po\s+(?:no|number|#)?\s*[:\-]?\s*[\d\w\-]+',
                r'order\s+number\s*[:\-]?\s*[\d\w\-]+',
                r'po[\s_-]?\d{4,}',  # PO followed by numbers
                r'vendor|supplier.*purchase'  # Vendor/supplier mentions
            ]
            for pattern in po_number_patterns:
                if re.search(pattern, content_lower):
                    po_score += 0.4  # Strong indicator
                    break
            
            # Stock reports often have quantity fields - STRONG INDICATORS
            stock_patterns = [
                r'opening\s+qty',
                r'receipt\s+qty',
                r'issue\s+qty',
                r'closing\s+qty',
                r'opening\s+balance',
                r'closing\s+balance',
                r'stock\s+statement',
                r'item\s+description',
                r'opening\s+value.*receipt\s+value',  # Multiple value fields
                r'stock.*and.*sales',  # Stock and sales phrase
                r'abbott.*india',  # Common in stock reports
            ]
            pattern_matches = sum(1 for pattern in stock_patterns if re.search(pattern, content_lower))
            if pattern_matches > 0:
                stock_score += min(pattern_matches * 0.3, 0.7)  # Strong cumulative indicator
        
        # Boost confidence when clear indicators are present
        # If filename strongly indicates one type, boost that score
        if po_filename_keywords >= 2 and po_score < 0.8:
            po_score = 0.85  # Minimum confidence for clear PO indicators
        
        if stock_filename_keywords >= 2 and stock_score < 0.8:
            stock_score = 0.85  # Minimum confidence for clear stock indicators
        
        # Normalize scores to 0-1 range, but ensure high scores when clear
        po_score = min(po_score, 1.0)
        stock_score = min(stock_score, 1.0)
        
        # If one score is significantly higher, ensure it's at least 0.75
        score_diff = abs(po_score - stock_score)
        if score_diff > 0.3:
            if po_score > stock_score:
                po_score = max(po_score, 0.75)
            else:
                stock_score = max(stock_score, 0.75)
        
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


