"""
Enhanced training script for Stock & Sales binary classification
Designed for 1500+ unique stockist files with ~10 files per stockist
"""
import sys
from pathlib import Path
import pickle
import logging
from typing import List, Tuple, Dict
from collections import defaultdict
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_converter.file_scanner import FileScanner
from document_converter.ocr_processor import OCRProcessor
from document_converter.document_classifier import DocumentClassifier
from document_converter.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_stockist_code(filename: str) -> str:
    """
    Extract stockist code from filename
    Format: <stockistcode>_<date>_<time>_<originalfilename>
    
    Args:
        filename: File name
        
    Returns:
        Stockist code or 'unknown'
    """
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return 'unknown'


def collect_training_data_stratified(
    input_folder: Path, 
    files_per_stockist: int = 10,
    max_total_files: int = None
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Collect training data with stratified sampling per stockist
    
    Args:
        input_folder: Path to EmailAttachments folder
        files_per_stockist: Number of files to sample per unique stockist
        max_total_files: Maximum total files to process (None = no limit)
        
    Returns:
        Tuple of (features, labels, stockist_stats)
        features: List of text feature strings
        labels: List of labels ('stock' or 'other')
        stockist_stats: Dictionary with stockist code -> count mapping
    """
    logger.info(f"Collecting training data from {input_folder}")
    logger.info(f"Target: {files_per_stockist} files per stockist")
    
    scanner = FileScanner(input_folder)
    files = scanner.scan_files()
    
    if not files:
        logger.error("No files found in input folder")
        return [], [], {}
    
    logger.info(f"Found {len(files)} total files")
    
    # Group files by stockist code
    stockist_files = defaultdict(list)
    for file_info in files:
        stockist_code = extract_stockist_code(file_info['filename'])
        stockist_files[stockist_code].append(file_info)
    
    logger.info(f"Found {len(stockist_files)} unique stockists")
    
    # Stratified sampling: sample files_per_stockist from each stockist
    sampled_files = []
    stockist_stats = {}
    
    for stockist_code, stockist_file_list in stockist_files.items():
        # Sample up to files_per_stockist files per stockist
        if len(stockist_file_list) > files_per_stockist:
            sampled = random.sample(stockist_file_list, files_per_stockist)
        else:
            sampled = stockist_file_list
        
        sampled_files.extend(sampled)
        stockist_stats[stockist_code] = len(sampled)
    
    # Limit total files if specified
    if max_total_files and len(sampled_files) > max_total_files:
        sampled_files = random.sample(sampled_files, max_total_files)
        logger.info(f"Limited to {max_total_files} total files")
    
    logger.info(f"Processing {len(sampled_files)} sampled files from {len(stockist_files)} stockists")
    
    # Initialize components
    ocr_processor = OCRProcessor(ConfigLoader())
    classifier = DocumentClassifier(ConfigLoader(), ml_model=None)  # Rule-based only for labeling
    
    features = []
    labels = []
    processed_count = 0
    
    for i, file_info in enumerate(sampled_files):
        file_path = file_info['path']
        processed_count += 1
        
        if processed_count % 50 == 0:
            logger.info(f"Processing {processed_count}/{len(sampled_files)} files...")
        
        try:
            # Extract text content
            text = ""
            if file_info['file_type'] == 'pdf':
                if file_info['is_scanned']:
                    text, _, _ = ocr_processor.extract_text_from_pdf(file_path)
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
                        text, _, _ = ocr_processor.extract_text_from_pdf(file_path)
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
            
            # Classify using rule-based classifier
            # For binary classification: stock_sales_report = 'stock', everything else = 'other'
            classification, confidence = classifier.classify(file_path, text)
            
            # Binary labeling: stock_sales_report -> 'stock', everything else -> 'other'
            if classification == 'stock_sales_report':
                label = 'stock'
            else:
                label = 'other'
            
            # Use filename + first 2000 chars of content as features
            feature_text = f"{file_path.name} {text[:2000]}"
            features.append(feature_text)
            labels.append(label)
            
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")
            continue
    
    # Log statistics
    stock_count = labels.count('stock')
    other_count = labels.count('other')
    
    logger.info(f"Collected {len(features)} training samples")
    logger.info(f"  Stock & Sales: {stock_count} ({stock_count/len(labels)*100:.1f}%)")
    logger.info(f"  Other: {other_count} ({other_count/len(labels)*100:.1f}%)")
    logger.info(f"  Unique stockists: {len(stockist_stats)}")
    
    return features, labels, stockist_stats


def train_binary_model(
    features: List[str], 
    labels: List[str],
    test_size: float = 0.2
) -> Tuple[Pipeline, Dict]:
    """
    Train binary classification model (Stock vs Other)
    
    Args:
        features: List of text features
        labels: List of labels ('stock' or 'other')
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info("Training binary classification model (Stock vs Other)...")
    
    if len(set(labels)) < 2:
        logger.error("Need at least 2 classes for binary classification")
        return None, {}
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=42, 
        stratify=labels
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Create pipeline with TF-IDF and Logistic Regression
    # Using more features and n-grams for better accuracy
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,  # Increased from 5000
            ngram_range=(1, 3),   # Include trigrams
            min_df=2,            # Minimum document frequency
            max_df=0.95,         # Maximum document frequency
            stop_words='english'  # Remove common English words
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=2000,       # Increased iterations
            C=1.0,                # Regularization strength
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Train
    logger.info("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    test_accuracy = pipeline.score(X_test, y_test)
    logger.info(f"Test accuracy: {test_accuracy:.2%}")
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
    
    # Detailed classification report
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    metrics = {
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return pipeline, metrics


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stock & Sales binary classifier')
    parser.add_argument('--input', type=str, default='attachments',
                       help='Input folder path (default: attachments)')
    parser.add_argument('--output', type=str, 
                       default='document_converter/models/classifier_model_stock_only.pkl',
                       help='Output model path')
    parser.add_argument('--files-per-stockist', type=int, default=10,
                       help='Number of files to sample per stockist (default: 10)')
    parser.add_argument('--max-total-files', type=int, default=None,
                       help='Maximum total files to process (default: None = no limit)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    input_folder = Path(args.input)
    output_path = Path(args.output)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Collect training data with stratified sampling
    features, labels, stockist_stats = collect_training_data_stratified(
        input_folder, 
        files_per_stockist=args.files_per_stockist,
        max_total_files=args.max_total_files
    )
    
    if len(features) < 20:
        logger.warning("Not enough training data. Need at least 20 samples.")
        logger.warning(f"Got {len(features)} samples. Please check your input folder.")
        return
    
    # Check class balance
    stock_count = labels.count('stock')
    other_count = labels.count('other')
    
    if stock_count < 5 or other_count < 5:
        logger.warning(f"Class imbalance detected: Stock={stock_count}, Other={other_count}")
        logger.warning("Model may not perform well. Consider collecting more diverse data.")
    
    # Train model
    model, metrics = train_binary_model(features, labels)
    
    if model is None:
        logger.error("Model training failed")
        return
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"\nModel saved to {output_path}")
    
    # Save metrics
    metrics_path = output_path.with_suffix('.metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save stockist stats
    stats_path = output_path.with_suffix('.stockist_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stockist_stats, f, indent=2)
    logger.info(f"Stockist statistics saved to {stats_path}")
    
    logger.info("\nTraining completed successfully!")


if __name__ == '__main__':
    main()


