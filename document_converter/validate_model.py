"""
Model Validation Script
Tests model on known PO and Stock files with comprehensive metrics.
"""
import sys
from pathlib import Path
import logging
import pickle
import json
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score
)

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


def identify_file_type(filename: str, content: str = "") -> str:
    """
    Identify file type based on filename and content patterns
    
    Returns:
        'po', 'stock', or 'unknown'
    """
    import re
    filename_lower = filename.lower()
    content_lower = content[:1000].lower() if content else ""
    
    # PO indicators
    po_patterns = [
        r'\bpo\s*\d+', r'\bpurchase\s+order', r'\bpo\s+number',
        r'\bpo\s+no', r'^po\d+', r'po_', r'po-'
    ]
    has_po = any(re.search(pattern, filename_lower, re.IGNORECASE) for pattern in po_patterns)
    if content:
        has_po = has_po or any(re.search(pattern, content_lower, re.IGNORECASE) for pattern in po_patterns)
    
    if has_po:
        return 'po'
    
    # Stock indicators
    stock_patterns = [
        r'\bstock', r'\bstatement', r'\bst[\s_-]?[\d\-]',
        r'\bstockandsales', r'opening\s+qty', r'receipt\s+qty',
        r'issue\s+qty', r'closing\s+qty', r'stock\s+statement'
    ]
    has_stock = any(re.search(pattern, filename_lower, re.IGNORECASE) for pattern in stock_patterns)
    if content:
        has_stock = has_stock or any(re.search(pattern, content_lower, re.IGNORECASE) for pattern in stock_patterns)
    
    if has_stock:
        return 'stock'
    
    return 'unknown'


def validate_model(
    model_path: Path,
    test_folder: Path,
    known_po_folder: Path = None,
    known_stock_folder: Path = None
) -> Dict:
    """
    Validate model performance on test files
    
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {}
    
    # Initialize components
    try:
        config = ConfigLoader()
        ocr_processor = OCRProcessor(config)
        classifier = DocumentClassifier(config, ml_model=model)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return {}
    
    results = {
        'model_path': str(model_path),
        'test_files': [],
        'predictions': [],
        'true_labels': [],
        'file_types': [],
        'statistics': {}
    }
    
    # Collect test files
    test_files = []
    
    # Add files from test folder
    if test_folder.exists():
        scanner = FileScanner(test_folder)
        files = scanner.scan_files()
        test_files.extend(files)
        logger.info(f"Found {len(files)} files in test folder")
    
    # Add known PO files
    if known_po_folder and known_po_folder.exists():
        scanner = FileScanner(known_po_folder)
        files = scanner.scan_files()
        for f in files:
            f['expected_type'] = 'po'
        test_files.extend(files)
        logger.info(f"Found {len(files)} known PO files")
    
    # Add known Stock files
    if known_stock_folder and known_stock_folder.exists():
        scanner = FileScanner(known_stock_folder)
        files = scanner.scan_files()
        for f in files:
            f['expected_type'] = 'stock'
        test_files.extend(files)
        logger.info(f"Found {len(files)} known Stock files")
    
    if not test_files:
        logger.error("No test files found")
        return {}
    
    logger.info(f"Total test files: {len(test_files)}")
    
    # Process files
    processed_count = 0
    for file_info in test_files:
        file_path = file_info['path']
        filename = file_info['filename']
        processed_count += 1
        
        if processed_count % 20 == 0:
            logger.info(f"Processing {processed_count}/{len(test_files)} files...")
        
        try:
            # Extract text
            text = ""
            if file_info['file_type'] == 'pdf':
                if file_info['is_scanned']:
                    try:
                        text, _, _ = ocr_processor.extract_text_from_pdf(file_path)
                    except:
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
                    except:
                        try:
                            text, _, _ = ocr_processor.extract_text_from_pdf(file_path)
                        except:
                            continue
            elif file_info['file_type'] == 'text':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                    except:
                        continue
            
            if not text or len(text.strip()) < 50:
                continue
            
            # Identify true file type
            true_type = file_info.get('expected_type')
            if not true_type:
                true_type = identify_file_type(filename, text)
            
            # Classify using model
            try:
                classification, confidence = classifier.classify(file_path, text)
            except Exception as e:
                logger.debug(f"Classification failed for {file_path.name}: {e}")
                continue
            
            # Map to binary labels
            predicted_label = 'stock' if classification == 'stock_sales_report' else 'other'
            true_label = 'stock' if true_type == 'stock' else 'other'
            
            results['test_files'].append({
                'filename': filename,
                'true_type': true_type,
                'predicted_classification': classification,
                'confidence': confidence,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'correct': predicted_label == true_label
            })
            
            results['predictions'].append(predicted_label)
            results['true_labels'].append(true_label)
            results['file_types'].append(true_type)
            
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")
            continue
    
    # Calculate metrics
    if len(results['predictions']) > 0:
        # Binary classification metrics
        y_true = [1 if label == 'stock' else 0 for label in results['true_labels']]
        y_pred = [1 if label == 'stock' else 0 for label in results['predictions']]
        
        # Overall accuracy
        accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
        results['statistics']['overall_accuracy'] = accuracy
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        
        results['statistics']['precision'] = {
            'other': float(precision[0]),
            'stock': float(precision[1])
        }
        results['statistics']['recall'] = {
            'other': float(recall[0]),
            'stock': float(recall[1])
        }
        results['statistics']['f1_score'] = {
            'other': float(f1[0]),
            'stock': float(f1[1])
        }
        results['statistics']['support'] = {
            'other': int(support[0]),
            'stock': int(support[1])
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        results['statistics']['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),  # Other predicted as Other
            'false_positive': int(cm[0, 1]),  # Other predicted as Stock
            'false_negative': int(cm[1, 0]),  # Stock predicted as Other
            'true_positive': int(cm[1, 1])  # Stock predicted as Stock
        }
        
        # PO-specific metrics (if we have PO files)
        po_files = [r for r in results['test_files'] if r['true_type'] == 'po']
        if po_files:
            po_correct = sum(1 for r in po_files if r['predicted_label'] == 'other')
            results['statistics']['po_detection'] = {
                'total_po_files': len(po_files),
                'correctly_classified_as_other': po_correct,
                'misclassified_as_stock': len(po_files) - po_correct,
                'precision': po_correct / len(po_files) if po_files else 0.0
            }
        
        # Stock-specific metrics
        stock_files = [r for r in results['test_files'] if r['true_type'] == 'stock']
        if stock_files:
            stock_correct = sum(1 for r in stock_files if r['predicted_label'] == 'stock')
            results['statistics']['stock_detection'] = {
                'total_stock_files': len(stock_files),
                'correctly_classified_as_stock': stock_correct,
                'misclassified_as_other': len(stock_files) - stock_correct,
                'precision': stock_correct / len(stock_files) if stock_files else 0.0
            }
    
    return results


def generate_report(results: Dict, output_path: Path = None):
    """Generate validation report"""
    if not results or not results.get('statistics'):
        logger.error("No results to report")
        return
    
    stats = results['statistics']
    
    logger.info("\n" + "="*80)
    logger.info("MODEL VALIDATION REPORT")
    logger.info("="*80)
    logger.info(f"\nModel: {results.get('model_path', 'Unknown')}")
    logger.info(f"Total test files: {len(results['test_files'])}")
    logger.info(f"\nOverall Accuracy: {stats['overall_accuracy']:.4f} ({stats['overall_accuracy']*100:.2f}%)")
    
    logger.info(f"\nPer-Class Metrics:")
    logger.info(f"  Other (Precision/Recall/F1): {stats['precision']['other']:.4f} / {stats['recall']['other']:.4f} / {stats['f1_score']['other']:.4f}")
    logger.info(f"  Stock (Precision/Recall/F1): {stats['precision']['stock']:.4f} / {stats['recall']['stock']:.4f} / {stats['f1_score']['stock']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    cm = stats['confusion_matrix']
    logger.info(f"  True Negative (Other->Other):  {cm['true_negative']}")
    logger.info(f"  False Positive (Other->Stock): {cm['false_positive']}")
    logger.info(f"  False Negative (Stock->Other): {cm['false_negative']}")
    logger.info(f"  True Positive (Stock->Stock):   {cm['true_positive']}")
    
    # PO detection metrics
    if 'po_detection' in stats:
        po_stats = stats['po_detection']
        logger.info(f"\nPO Detection Metrics:")
        logger.info(f"  Total PO files: {po_stats['total_po_files']}")
        logger.info(f"  Correctly classified as Other: {po_stats['correctly_classified_as_other']}")
        logger.info(f"  Misclassified as Stock: {po_stats['misclassified_as_stock']}")
        logger.info(f"  PO Detection Precision: {po_stats['precision']:.4f} ({po_stats['precision']*100:.2f}%)")
        
        if po_stats['misclassified_as_stock'] > 0:
            logger.warning(f"  ⚠️  {po_stats['misclassified_as_stock']} PO files were misclassified as Stock!")
    
    # Stock detection metrics
    if 'stock_detection' in stats:
        stock_stats = stats['stock_detection']
        logger.info(f"\nStock Detection Metrics:")
        logger.info(f"  Total Stock files: {stock_stats['total_stock_files']}")
        logger.info(f"  Correctly classified as Stock: {stock_stats['correctly_classified_as_stock']}")
        logger.info(f"  Misclassified as Other: {stock_stats['misclassified_as_other']}")
        logger.info(f"  Stock Detection Precision: {stock_stats['precision']:.4f} ({stock_stats['precision']*100:.2f}%)")
    
    # Show misclassified files
    misclassified = [r for r in results['test_files'] if not r['correct']]
    if misclassified:
        logger.warning(f"\n⚠️  Misclassified Files ({len(misclassified)}):")
        for item in misclassified[:20]:  # Show first 20
            logger.warning(f"  - {item['filename']}: {item['true_type']} -> {item['predicted_classification']} (confidence: {item['confidence']:.2f})")
        if len(misclassified) > 20:
            logger.warning(f"  ... and {len(misclassified) - 20} more")
    
    # Save detailed report to JSON
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nDetailed report saved to: {output_path}")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Model Performance')
    parser.add_argument('--model', type=str, 
                       default='document_converter/models/classifier_model_ensemble.pkl',
                       help='Model path')
    parser.add_argument('--test-folder', type=str, default='attachments test',
                       help='Test folder path')
    parser.add_argument('--known-po-folder', type=str, default=None,
                       help='Folder with known PO files (optional)')
    parser.add_argument('--known-stock-folder', type=str, default=None,
                       help='Folder with known Stock files (optional)')
    parser.add_argument('--output', type=str, 
                       default='document_converter/models/model_validation_report.json',
                       help='Output report path')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    test_folder = Path(args.test_folder) if args.test_folder else None
    known_po_folder = Path(args.known_po_folder) if args.known_po_folder else None
    known_stock_folder = Path(args.known_stock_folder) if args.known_stock_folder else None
    output_path = Path(args.output)
    
    if not model_path.exists():
        logger.error(f"Model file does not exist: {model_path}")
        return
    
    # Validate model
    results = validate_model(model_path, test_folder, known_po_folder, known_stock_folder)
    
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

