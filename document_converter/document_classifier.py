"""
Document classifier - Binary classification (Stock & Sales vs Other)
Hybrid approach (rule-based + ML)
"""
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Binary classifier: Stock & Sales Report vs Other documents"""
    
    def __init__(self, config=None, ml_model=None):
        """
        Initialize classifier
        
        Args:
            config: ConfigLoader instance
            ml_model: Trained ML model (optional) - expects binary classification
        """
        from .config_loader import ConfigLoader
        self.config = config or ConfigLoader()
        self.ml_model = ml_model
        
        # Load classification keywords and patterns (Stock & Sales only)
        self.stock_keywords = self.config.get('classification.stock_sales_keywords', [])
        
        # Compile filename patterns (Stock & Sales only)
        self.stock_patterns = self._compile_patterns(
            self.config.get('classification.filename_patterns.stock_sales', [])
        )
        
        # Load PO detection keywords and patterns
        self.po_keywords = self.config.get('classification.po_keywords', [])
        self.po_filename_patterns = self._compile_patterns(
            self.config.get('classification.filename_patterns.po', [])
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
        Binary classification: Stock & Sales Report vs Other
        
        Args:
            file_path: Path to file
            content: Optional pre-extracted text content
            
        Returns:
            Tuple of (classification, confidence)
            classification: 'stock_sales_report' or 'other'
            confidence: 0.0 to 1.0 (probability of being Stock & Sales)
        """
        filename = file_path.name.lower()
        
        # Rule-based classification (returns stock score)
        rule_score_stock = self._rule_based_classify(filename, content)
        
        # ML classification if model available (binary: stock vs other)
        ml_score_stock = self._ml_classify(filename, content)
        
        # Combine scores
        if self.ml_model is None:
            # No ML model - use rule-based score directly
            combined_stock = rule_score_stock
            
            # Boost scores when rule-based is confident (>0.6)
            if rule_score_stock > 0.6:
                combined_stock = min(rule_score_stock * 1.1, 0.98)
        else:
            # ML model available - weighted combination favoring rule-based
            # ml_score_stock is already a probability from binary classifier
            combined_stock = (rule_score_stock * 0.7) + (ml_score_stock * 0.3)
            
            # If rule-based is very confident, boost the combined score
            if rule_score_stock > 0.7 and combined_stock < 0.85:
                combined_stock = min(combined_stock * 1.15, 0.98)
        
        # Normalize to ensure valid range
        combined_stock = min(max(combined_stock, 0.0), 0.99)
        
        # Adaptive threshold based on file characteristics
        base_threshold = self.config.get('classification.combined_confidence_threshold', 0.3)
        
        # Adjust threshold based on confidence levels
        # If we have very low confidence (<0.1), be more strict
        # If we have moderate confidence (0.1-0.4), use standard threshold
        # If we have high confidence (>0.7), be more lenient
        if combined_stock < 0.1:
            adaptive_threshold = base_threshold * 1.2  # More strict for very low scores
        elif combined_stock > 0.7:
            adaptive_threshold = base_threshold * 0.8  # More lenient for high scores
        else:
            adaptive_threshold = base_threshold
        
        # Flag uncertain classifications
        is_uncertain = False
        if 0.2 <= combined_stock < 0.5:
            is_uncertain = True
            logger.debug(f"Uncertain classification for file (score: {combined_stock:.2f})")
        
        # Determine classification
        if combined_stock >= adaptive_threshold:
            classification = 'stock_sales_report'
            confidence = combined_stock
            if is_uncertain:
                logger.warning(f"Uncertain classification: {classification} with confidence {confidence:.2f}")
        else:
            classification = 'other'
            confidence = 1.0 - combined_stock  # Confidence of being "other"
            if is_uncertain:
                logger.warning(f"Uncertain classification: {classification} with confidence {confidence:.2f}")
        
        return classification, confidence
    
    def _rule_based_classify(self, filename: str, content: Optional[str]) -> float:
        """
        Rule-based binary classification: Stock & Sales Report vs Other
        Returns score for Stock & Sales (0.0 to 1.0)
        
        Returns:
            stock_score: Probability of being Stock & Sales Report (0.0 to 1.0)
        """
        # FIRST: Check for PO indicators - if found, strongly reduce stock score
        # Enhanced PO patterns for better detection
        po_indicators = [
            r'\bpo\s*\d+',  # PO followed by number (e.g., PO1650, PO 1650)
            r'\bpurchase\s+order',
            r'\bpo\s+number',
            r'\bpo\s+no',
            r'\bpo\s+#',  # PO #123
            r'\border\s+number',  # Order number
            r'\border\s+no',  # Order no
            r'\border\s+#',  # Order #123
            r'^po\d+',  # Starts with PO and number
            r'^po\s*\d+',  # Starts with PO space number
            r'po_',  # PO_ pattern
            r'po-',  # PO- pattern
            r'p\.o\.\s*\d+',  # P.O. 123
            r'p/o\s*\d+',  # P/O 123
            r'\bpo\s+id',  # PO ID
            r'\bpurchase\s+order\s+number',  # Purchase order number
            r'\bpurchase\s+order\s+no',  # Purchase order no
            r'\bpo\s+ref',  # PO ref
            r'\bpo\s+reference',  # PO reference
        ]
        
        # Check filename for PO indicators with confidence scoring
        po_matches = sum(1 for pattern in po_indicators if re.search(pattern, filename, re.IGNORECASE))
        
        # Check configured PO patterns
        po_filename_matches = sum(1 for pattern in self.po_filename_patterns if pattern.search(filename))
        
        # Check filename for PO keywords
        po_keyword_matches = sum(1 for kw in self.po_keywords if kw.lower() in filename)
        
        # Calculate PO confidence from filename
        po_filename_confidence = (po_matches * 0.4) + (po_filename_matches * 0.3) + (po_keyword_matches * 0.3)
        
        # Check content for PO indicators with more thorough analysis
        po_content_confidence = 0.0
        if content:
            content_lower = content[:3000].lower()  # Check first 3000 chars for PO indicators
            content_po_matches = sum(1 for pattern in po_indicators if re.search(pattern, content_lower, re.IGNORECASE))
            content_po_keywords = sum(1 for kw in self.po_keywords if kw.lower() in content_lower)
            
            # Additional content-based PO indicators
            po_content_patterns = [
                r'purchase\s+order\s+details',
                r'po\s+details',
                r'order\s+details',
                r'purchase\s+order\s+date',
                r'po\s+date',
                r'order\s+date',
                r'vendor\s+name',  # Common in POs
                r'supplier\s+name',  # Common in POs
                r'delivery\s+address',  # Common in POs
                r'billing\s+address',  # Common in POs
            ]
            content_pattern_matches = sum(1 for pattern in po_content_patterns if re.search(pattern, content_lower, re.IGNORECASE))
            
            po_content_confidence = (content_po_matches * 0.3) + (content_po_keywords * 0.2) + (content_pattern_matches * 0.2)
        
        # Combined PO confidence
        total_po_confidence = max(po_filename_confidence, po_content_confidence)
        
        # Handle ambiguous cases (files with both PO and Stock indicators)
        has_stock_indicators = False
        if content:
            stock_indicators = [
                r'opening\s+qty',
                r'receipt\s+qty',
                r'issue\s+qty',
                r'closing\s+qty',
                r'stock\s+statement',
                r'item\s+description',
            ]
            has_stock_indicators = any(re.search(pattern, content[:2000].lower(), re.IGNORECASE) for pattern in stock_indicators)
        
        # If PO confidence is high, strongly reduce stock score
        if total_po_confidence > 0.5:
            if has_stock_indicators:
                # Ambiguous case - file has both PO and Stock indicators
                logger.warning(f"Ambiguous file: {filename} - has both PO and Stock indicators")
                return 0.15  # Very low but slightly higher than pure PO
            else:
                # Strong PO indicator - return very low stock score
                logger.debug(f"PO indicator found in filename/content: {filename} (confidence: {total_po_confidence:.2f})")
                return 0.01  # Very low confidence for stock
        elif total_po_confidence > 0.2:
            # Moderate PO indicator
            logger.debug(f"Moderate PO indicator found: {filename}")
            return 0.2  # Low confidence for stock
        
        stock_score = 0.0
        
        # Filename pattern matching - HIGHER WEIGHTS for better confidence
        stock_filename_matches = sum(1 for pattern in self.stock_patterns if pattern.search(filename))
        
        # Filename keyword matching - EXPANDED matching
        stock_filename_keywords = sum(1 for kw in self.stock_keywords if kw.lower() in filename)
        
        # Additional common patterns in filenames
        # Stock patterns: STOCK, ST-, ST_, statement, report
        if re.search(r'\bstock|statement|st[\s_-]?[\d\-]|stockandsales', filename, re.IGNORECASE):
            stock_filename_keywords += 2  # Strong indicator
        
        # Combine filename signals - MUCH HIGHER BASE SCORES
        if stock_filename_matches > 0 or stock_filename_keywords > 0:
            # Base score from filename is now 0.8-0.95 for clear matches
            filename_base = min((stock_filename_matches * 0.5) + (stock_filename_keywords * 0.4), 0.95)
            stock_score += filename_base
        
        # Content analysis (if available) - INCREASED WEIGHTS
        if content:
            content_lower = content[:3000].lower()  # Use more content for better matching
            
            # Check for Stock keywords in content - INCREASED SCORING
            stock_content_matches = sum(1 for kw in self.stock_keywords if kw.lower() in content_lower)
            
            # Content scoring - higher weights
            if stock_content_matches > 0:
                stock_score += min(stock_content_matches * 0.25, 0.6)
            
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
        if stock_filename_keywords >= 2 and stock_score < 0.8:
            stock_score = 0.85  # Minimum confidence for clear stock indicators
        
        # Normalize score to 0-1 range, but ensure high scores when clear
        stock_score = min(stock_score, 1.0)
        
        # If score is high, ensure it's at least 0.75
        if stock_score > 0.5:
            stock_score = max(stock_score, 0.75)
        
        return stock_score
    
    def _ml_classify(self, filename: str, content: Optional[str]) -> float:
        """
        ML-based binary classification: Stock & Sales vs Other
        
        Returns:
            stock_score: Probability of being Stock & Sales Report (0.0 to 1.0)
        """
        if self.ml_model is None:
            return 0.0
        
        try:
            # Prepare features for ML model
            features = self._extract_features(filename, content)
            
            # Predict with ML model (binary classifier)
            # Handle both Pipeline models and direct models
            prediction = self.ml_model.predict_proba([features])[0]
            
            # Get class names - handle Pipeline models
            if hasattr(self.ml_model, 'named_steps'):
                # Pipeline model - get classes from ensemble step
                if 'ensemble' in self.ml_model.named_steps:
                    classes = self.ml_model.named_steps['ensemble'].classes_
                else:
                    # Try to find classifier step
                    for step_name, step in self.ml_model.named_steps.items():
                        if hasattr(step, 'classes_'):
                            classes = step.classes_
                            break
                    else:
                        # Fallback: assume standard order
                        classes = np.array(['other', 'stock'])
            else:
                # Direct model
                classes = self.ml_model.classes_
            
            # Find index of 'stock' class
            if 'stock' in classes:
                stock_idx = list(classes).index('stock')
                return float(prediction[stock_idx])
            else:
                # Fallback: assume first class is 'other', second is 'stock'
                # Or if only one class, return that probability
                if len(prediction) == 2:
                    return float(prediction[1])  # Assume second is stock
                else:
                    return float(prediction[0])
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return 0.0
    
    def _extract_features(self, filename: str, content: Optional[str]) -> str:
        """
        Extract features for ML model (combines filename + content)
        
        Returns:
            Combined feature string
        """
        features = [filename]
        
        if content:
            # Use first 3000 characters (increased from 1000 for better feature representation)
            features.append(content[:3000])
        
        return " ".join(features)


