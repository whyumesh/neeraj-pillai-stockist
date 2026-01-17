"""
Train ML classifier on existing EmailAttachments files
"""
import sys
from pathlib import Path
import pickle
import logging
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_converter.file_scanner import FileScanner
from document_converter.ocr_processor import OCRProcessor
from document_converter.document_classifier import DocumentClassifier
from document_converter.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_training_data(input_folder: Path, max_files: int = 100) -> Tuple[List[str], List[str]]:
    """
    Collect training data from EmailAttachments folder
    
    Args:
        input_folder: Path to EmailAttachments folder
        max_files: Maximum files to process for training
        
    Returns:
        Tuple of (features, labels) where features are text strings and labels are 'po' or 'stock'
    """
    logger.info(f"Collecting training data from {input_folder}")
    
    scanner = FileScanner(input_folder)
    files = scanner.scan_files()
    
    # Limit files for training
    files = files[:max_files]
    
    ocr_processor = OCRProcessor()
    features = []
    labels = []
    
    for i, file_info in enumerate(files):
        file_path = file_info['path']
        logger.info(f"Processing {i+1}/{len(files)}: {file_path.name}")
        
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
        
        if not text:
            continue
        
        # Classify using rule-based to get labels
        classifier = DocumentClassifier()
        classification, confidence = classifier.classify(file_path, text)
        
        # Only use high-confidence classifications for training
        if confidence >= 0.7:
            if classification == 'purchase_order':
                labels.append('po')
                features.append(file_path.name + " " + text[:1000])
            elif classification == 'stock_sales_report':
                labels.append('stock')
                features.append(file_path.name + " " + text[:1000])
    
    logger.info(f"Collected {len(features)} training samples")
    logger.info(f"  PO: {labels.count('po')}, Stock: {labels.count('stock')}")
    
    return features, labels


def train_model(features: List[str], labels: List[str]) -> Pipeline:
    """
    Train ML model
    
    Args:
        features: List of text features
        labels: List of labels ('po' or 'stock')
        
    Returns:
        Trained model pipeline
    """
    logger.info("Training ML model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create pipeline with TF-IDF and classifier
    # Using Logistic Regression as it often works better than Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.2%}")
    
    return pipeline


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train document classifier')
    parser.add_argument('--input', type=str, default='EmailAttachments',
                       help='Input folder path')
    parser.add_argument('--output', type=str, default='document_converter/models/classifier_model.pkl',
                       help='Output model path')
    parser.add_argument('--max-files', type=int, default=100,
                       help='Maximum files to process')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_path = Path(args.output)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Collect training data
    features, labels = collect_training_data(input_folder, args.max_files)
    
    if len(features) < 10:
        logger.warning("Not enough training data. Need at least 10 samples.")
        return
    
    # Train model
    model = train_model(features, labels)
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {output_path}")


if __name__ == '__main__':
    main()

