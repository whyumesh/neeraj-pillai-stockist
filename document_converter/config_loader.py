"""
Configuration loader for the document converter system
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage configuration settings"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config loader
        
        Args:
            config_path: Path to config.json file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # Use default configuration
            self.config = self._get_default_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'ocr.dpi')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "ocr": {
                "primary": "paddleocr",
                "fallback": "tesseract",
                "dpi": 300,
                "preprocessing": {
                    "deskew": True,
                    "binarize": True,
                    "noise_removal": True,
                    "confidence_threshold": 0.6
                }
            },
            "classification": {
                "po_keywords": [
                    "purchase order",
                    "po number",
                    "po no",
                    "order number",
                    "order no",
                    "po_",
                    "orderpurchase"
                ],
                "stock_sales_keywords": [
                    "stock statement",
                    "stock and sales",
                    "stockandsales",
                    "opening qty",
                    "receipt qty",
                    "issue qty",
                    "closing qty",
                    "st-",
                    "stok"
                ],
                "ml_confidence_threshold": 0.7,
                "combined_confidence_threshold": 0.6
            },
            "paths": {
                "input_folder": "EmailAttachments",
                "output_folder": "Output",
                "po_output": "Output/PurchaseOrders",
                "stock_output": "Output/StockSalesReports",
                "log_file": "Output/ProcessingLog.json"
            }
        }



