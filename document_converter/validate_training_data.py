"""
Training Data Validation Script
Scans training data folder, identifies PO vs Stock files, checks for mislabeled files,
and generates a data quality report.
"""
import sys
from pathlib import Path
import re
import logging
from collections import defaultdict
from typing import List, Dict, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_converter.file_scanner import FileScanner
from document_converter.ocr_processor import OCRProcessor
from document_converter.document_classifier import DocumentClassifier
from document_converter.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_stockist_code(filename: str) -> str:
    """Extract stockist code from filename"""
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return 'unknown'


def check_po_indicators(filename: str, content: str = "") -> Tuple[bool, List[str]]:
    """
    Check for PO indicators in filename and content
    
    Returns:
        Tuple of (is_po, indicators_found)
    """
    indicators_found = []
    filename_lower = filename.lower()
    content_lower = content[:1000].lower() if content else ""
    
    po_patterns = [
        (r'\bpo\s*\d+', 'PO followed by number'),
        (r'\bpurchase\s+order', 'Purchase order phrase'),
        (r'\bpo\s+number', 'PO number phrase'),
        (r'\bpo\s+no', 'PO no phrase'),
        (r'^po\d+', 'Starts with PO and number'),
        (r'po_', 'PO_ pattern'),
        (r'po-', 'PO- pattern'),
    ]
    
    for pattern, description in po_patterns:
        if re.search(pattern, filename_lower, re.IGNORECASE):
            indicators_found.append(f"Filename: {description}")
        if content and re.search(pattern, content_lower, re.IGNORECASE):
            indicators_found.append(f"Content: {description}")
    
    return len(indicators_found) > 0, indicators_found


def check_stock_indicators(filename: str, content: str = "") -> Tuple[bool, List[str]]:
    """
    Check for Stock & Sales indicators in filename and content
    
    Returns:
        Tuple of (is_stock, indicators_found)
    """
    indicators_found = []
    filename_lower = filename.lower()
    content_lower = content[:1000].lower() if content else ""
    
    stock_patterns = [
        (r'\bstock', 'Stock keyword'),
        (r'\bstatement', 'Statement keyword'),
        (r'\bst[\s_-]?[\d\-]', 'ST- pattern'),
        (r'\bstockandsales', 'Stockandsales keyword'),
        (r'opening\s+qty', 'Opening qty field'),
        (r'receipt\s+qty', 'Receipt qty field'),
        (r'issue\s+qty', 'Issue qty field'),
        (r'closing\s+qty', 'Closing qty field'),
        (r'stock\s+statement', 'Stock statement phrase'),
        (r'stock.*and.*sales', 'Stock and sales phrase'),
    ]
    
    for pattern, description in stock_patterns:
        if re.search(pattern, filename_lower, re.IGNORECASE):
            indicators_found.append(f"Filename: {description}")
        if content and re.search(pattern, content_lower, re.IGNORECASE):
            indicators_found.append(f"Content: {description}")
    
    return len(indicators_found) > 0, indicators_found


def validate_training_data(input_folder: Path) -> Dict:
    """
    Validate training data and identify mislabeled files
    
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating training data in: {input_folder}")
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return {}
    
    scanner = FileScanner(input_folder)
    files = scanner.scan_files()
    
    if not files:
        logger.error("No files found in input folder")
        return {}
    
    logger.info(f"Found {len(files)} total files")
    
    # Initialize components
    try:
        ocr_processor = OCRProcessor(ConfigLoader())
        classifier = DocumentClassifier(ConfigLoader(), ml_model=None)  # Rule-based only
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return {}
    
    results = {
        'total_files': len(files),
        'processed_files': 0,
        'error_files': 0,
        'po_files': [],
        'stock_files': [],
        'other_files': [],
        'mislabeled_files': [],
        'statistics': {
            'po_count': 0,
            'stock_count': 0,
            'other_count': 0,
            'mislabeled_count': 0
        }
    }
    
    processed_count = 0
    
    for file_info in files:
        file_path = file_info['path']
        filename = file_info['filename']
        processed_count += 1
        
        if processed_count % 50 == 0:
            logger.info(f"Processing {processed_count}/{len(files)} files...")
        
        try:
            # Extract text content
            text = ""
            if file_info['file_type'] == 'pdf':
                if file_info['is_scanned']:
                    try:
                        text, _, _ = ocr_processor.extract_text_from_pdf(file_path)
                    except Exception as e:
                        logger.debug(f"OCR failed for {file_path.name}: {e}")
                        results['error_files'] += 1
                        continue
                else:
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                    except Exception as e:
                        logger.debug(f"PDF extraction failed for {file_path.name}: {e}")
                        try:
                            text, _, _ = ocr_processor.extract_text_from_pdf(file_path)
                        except:
                            results['error_files'] += 1
                            continue
            elif file_info['file_type'] == 'text':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                    except Exception as e:
                        logger.debug(f"Text file reading failed for {file_path.name}: {e}")
                        results['error_files'] += 1
                        continue
            else:
                continue
            
            if not text or len(text.strip()) < 50:
                continue
            
            # Check indicators
            has_po, po_indicators = check_po_indicators(filename, text)
            has_stock, stock_indicators = check_stock_indicators(filename, text)
            
            # Classify using enhanced classifier
            try:
                classification, confidence = classifier.classify(file_path, text)
            except Exception as e:
                logger.debug(f"Classification failed for {file_path.name}: {e}")
                results['error_files'] += 1
                continue
            
            results['processed_files'] += 1
            
            file_result = {
                'filename': filename,
                'stockist_code': extract_stockist_code(filename),
                'classification': classification,
                'confidence': confidence,
                'has_po_indicators': has_po,
                'has_stock_indicators': has_stock,
                'po_indicators': po_indicators,
                'stock_indicators': stock_indicators,
            }
            
            # Check for mislabeling
            is_mislabeled = False
            if has_po and classification == 'stock_sales_report':
                is_mislabeled = True
                file_result['mislabeling_reason'] = 'PO file classified as stock'
                results['mislabeled_files'].append(file_result)
                results['statistics']['mislabeled_count'] += 1
            elif has_stock and classification == 'other' and not has_po:
                # Only flag as mislabeled if it has stock indicators but no PO indicators
                is_mislabeled = True
                file_result['mislabeling_reason'] = 'Stock file classified as other'
                results['mislabeled_files'].append(file_result)
                results['statistics']['mislabeled_count'] += 1
            
            # Categorize files
            if classification == 'stock_sales_report':
                results['stock_files'].append(file_result)
                results['statistics']['stock_count'] += 1
            elif has_po:
                results['po_files'].append(file_result)
                results['statistics']['po_count'] += 1
            else:
                results['other_files'].append(file_result)
                results['statistics']['other_count'] += 1
                
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")
            results['error_files'] += 1
            continue
    
    return results


def generate_report(results: Dict, output_path: Path = None):
    """Generate validation report"""
    if not results:
        logger.error("No results to report")
        return
    
    stats = results['statistics']
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING DATA VALIDATION REPORT")
    logger.info("="*80)
    logger.info(f"\nTotal files: {results['total_files']}")
    logger.info(f"Processed files: {results['processed_files']}")
    logger.info(f"Error files: {results['error_files']}")
    logger.info(f"\nClassification Statistics:")
    logger.info(f"  Stock & Sales Reports: {stats['stock_count']} ({stats['stock_count']/results['processed_files']*100:.1f}%)" if results['processed_files'] > 0 else "  Stock & Sales Reports: 0")
    logger.info(f"  PO Files: {stats['po_count']} ({stats['po_count']/results['processed_files']*100:.1f}%)" if results['processed_files'] > 0 else "  PO Files: 0")
    logger.info(f"  Other Files: {stats['other_count']} ({stats['other_count']/results['processed_files']*100:.1f}%)" if results['processed_files'] > 0 else "  Other Files: 0")
    logger.info(f"  Mislabeled Files: {stats['mislabeled_count']} ({stats['mislabeled_count']/results['processed_files']*100:.1f}%)" if results['processed_files'] > 0 else "  Mislabeled Files: 0")
    
    if stats['mislabeled_count'] > 0:
        logger.warning(f"\n⚠️  Found {stats['mislabeled_count']} potentially mislabeled files:")
        for item in results['mislabeled_files'][:20]:  # Show first 20
            logger.warning(f"  - {item['filename']}: {item.get('mislabeling_reason', 'Unknown')} (confidence: {item['confidence']:.2f})")
            if item.get('po_indicators'):
                logger.warning(f"    PO indicators: {', '.join(item['po_indicators'][:3])}")
            if item.get('stock_indicators'):
                logger.warning(f"    Stock indicators: {', '.join(item['stock_indicators'][:3])}")
        if len(results['mislabeled_files']) > 20:
            logger.warning(f"  ... and {len(results['mislabeled_files']) - 20} more")
    
    # Data quality checks
    logger.info(f"\nData Quality Checks:")
    if stats['po_count'] < 10:
        logger.warning(f"  ⚠️  Very few PO files ({stats['po_count']}). Model may struggle to distinguish POs.")
    if stats['stock_count'] < 50:
        logger.warning(f"  ⚠️  Few Stock files ({stats['stock_count']}). Consider adding more training data.")
    if stats['stock_count'] > 0 and stats['po_count'] > 0:
        ratio = stats['stock_count'] / stats['po_count']
        if ratio > 10:
            logger.warning(f"  ⚠️  High class imbalance: {ratio:.1f}:1 (Stock:PO). Consider balancing.")
        elif ratio < 0.1:
            logger.warning(f"  ⚠️  High class imbalance: {1/ratio:.1f}:1 (PO:Stock). Consider balancing.")
    
    # Save detailed report to JSON
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nDetailed report saved to: {output_path}")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Training Data Quality')
    parser.add_argument('--input', type=str, default='training data attachments',
                       help='Input folder path (default: training data attachments)')
    parser.add_argument('--output', type=str, 
                       default='document_converter/models/training_data_validation_report.json',
                       help='Output report path')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_path = Path(args.output)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Validate training data
    results = validate_training_data(input_folder)
    
    if not results:
        logger.error("Validation failed - no results generated")
        return
    
    # Generate report
    generate_report(results, output_path)
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

