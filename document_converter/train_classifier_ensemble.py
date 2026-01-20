"""
Enhanced Ensemble Voting Classifier Training Script
Strong ML model combining 4 diverse algorithms for high accuracy
"""
import sys
from pathlib import Path
import pickle
import logging
from typing import List, Tuple, Dict
from collections import defaultdict
import random
import json
import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

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
    Reused from train_classifier_stock_only.py with error handling improvements
    
    Args:
        input_folder: Path to training data folder
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
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return [], [], {}
    
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
    try:
        ocr_processor = OCRProcessor(ConfigLoader())
        classifier = DocumentClassifier(ConfigLoader(), ml_model=None)  # Rule-based only for labeling
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return [], [], {}
    
    features = []
    labels = []
    processed_count = 0
    error_count = 0
    mislabeled_files = []  # Track potentially mislabeled files
    po_files_found = 0
    stock_files_found = 0
    
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
                    try:
                        text, _, _ = ocr_processor.extract_text_from_pdf(file_path)
                    except Exception as e:
                        logger.debug(f"OCR failed for {file_path.name}: {e}")
                        error_count += 1
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
                            error_count += 1
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
                        error_count += 1
                        continue
            else:
                # Skip unsupported file types
                continue
            
            if not text or len(text.strip()) < 50:
                continue
            
            # Check for PO indicators explicitly before classification
            filename_lower = file_path.name.lower()
            po_indicators = [
                r'\bpo\s*\d+', r'\bpurchase\s+order', r'\bpo\s+number', 
                r'\bpo\s+no', r'^po\d+', r'po_', r'po-'
            ]
            has_po_in_filename = any(re.search(pattern, filename_lower, re.IGNORECASE) for pattern in po_indicators)
            
            # Classify using enhanced rule-based classifier (now with PO detection)
            # For binary classification: stock_sales_report = 'stock', everything else = 'other'
            try:
                classification, confidence = classifier.classify(file_path, text)
            except Exception as e:
                logger.debug(f"Classification failed for {file_path.name}: {e}")
                error_count += 1
                continue
            
            # Binary labeling: stock_sales_report -> 'stock', everything else -> 'other'
            if classification == 'stock_sales_report':
                label = 'stock'
                stock_files_found += 1
                
                # Check for potential mislabeling: PO files labeled as stock
                if has_po_in_filename:
                    mislabeled_files.append({
                        'file': file_path.name,
                        'detected_type': 'PO (from filename)',
                        'labeled_as': 'stock',
                        'confidence': confidence
                    })
                    logger.warning(f"Potential mislabeling: PO file {file_path.name} classified as stock (confidence: {confidence:.2f})")
            else:
                label = 'other'
                if has_po_in_filename:
                    po_files_found += 1
            
            # Enhanced feature engineering: increase text length and add explicit indicators
            # Use filename + first 3000 chars of content (increased from 2000)
            feature_text = f"{file_path.name} {text[:3000]}"
            
            # Add explicit PO/Stock indicators as features to help model learn patterns
            if has_po_in_filename:
                feature_text = f"PO_INDICATOR {feature_text}"
            if 'stock' in filename_lower or 'statement' in filename_lower:
                feature_text = f"STOCK_INDICATOR {feature_text}"
            
            features.append(feature_text)
            labels.append(label)
            
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")
            error_count += 1
            continue
    
    # Log statistics
    stock_count = labels.count('stock')
    other_count = labels.count('other')
    
    logger.info(f"Collected {len(features)} training samples")
    logger.info(f"  Stock & Sales: {stock_count} ({stock_count/len(labels)*100:.1f}%)" if labels else "  Stock & Sales: 0")
    logger.info(f"  Other: {other_count} ({other_count/len(labels)*100:.1f}%)" if labels else "  Other: 0")
    logger.info(f"  PO files detected: {po_files_found}")
    logger.info(f"  Stock files detected: {stock_files_found}")
    logger.info(f"  Unique stockists: {len(stockist_stats)}")
    
    # Data quality checks
    if stock_count < 5 or other_count < 5:
        logger.warning(f"Class imbalance detected: Stock={stock_count}, Other={other_count}")
        logger.warning("Model may not perform well. Consider collecting more diverse data.")
    
    if po_files_found < 10:
        logger.warning(f"Very few PO files found ({po_files_found}). Model may struggle to distinguish POs from Stock reports.")
    
    if mislabeled_files:
        logger.warning(f"Found {len(mislabeled_files)} potentially mislabeled files:")
        for item in mislabeled_files[:10]:  # Show first 10
            logger.warning(f"  - {item['file']}: {item['detected_type']} labeled as {item['labeled_as']} (confidence: {item['confidence']:.2f})")
        if len(mislabeled_files) > 10:
            logger.warning(f"  ... and {len(mislabeled_files) - 10} more")
    
    if error_count > 0:
        logger.warning(f"  Errors encountered: {error_count} files skipped")
    
    return features, labels, stockist_stats


def train_strong_ensemble_model(
    features: List[str], 
    labels: List[str],
    validation_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[Pipeline, Dict]:
    """
    Train a strong ensemble model with detailed training parameters
    
    Model Architecture:
    - Ensemble Voting Classifier combining:
      1. Logistic Regression (L2 regularization)
      2. Random Forest (200 trees)
      3. Gradient Boosting (200 estimators)
      4. SVM (RBF kernel)
    
    Training Parameters:
    - Train/Validation/Test Split: 70%/15%/15%
    - Cross-Validation: 5-fold stratified
    - Feature Engineering: TF-IDF with 15K features, 1-3 n-grams
    
    Returns:
        Tuple of (trained_model, detailed_metrics_dict)
    """
    logger.info("="*80)
    logger.info("TRAINING STRONG ENSEMBLE MODEL")
    logger.info("="*80)
    
    if len(set(labels)) < 2:
        logger.error("Need at least 2 classes for binary classification")
        return None, {}
    
    # ========== DATA SPLITTING ==========
    logger.info("\n[1] DATA SPLITTING")
    logger.info("-" * 80)
    
    # First split: Train+Val vs Test (85% vs 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    # Second split: Train vs Validation (70% vs 15% of total)
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=y_temp
    )
    
    logger.info(f"Training set:   {len(X_train):5d} samples ({len(X_train)/len(features)*100:.1f}%)")
    logger.info(f"Validation set: {len(X_val):5d} samples ({len(X_val)/len(features)*100:.1f}%)")
    logger.info(f"Test set:       {len(X_test):5d} samples ({len(X_test)/len(features)*100:.1f}%)")
    logger.info(f"Total:          {len(features):5d} samples")
    
    # Class distribution
    train_stock = y_train.count('stock')
    train_other = y_train.count('other')
    logger.info(f"\nTraining class distribution:")
    logger.info(f"  Stock: {train_stock} ({train_stock/len(y_train)*100:.1f}%)")
    logger.info(f"  Other: {train_other} ({train_other/len(y_train)*100:.1f}%)")
    
    # ========== FEATURE ENGINEERING ==========
    logger.info("\n[2] FEATURE ENGINEERING")
    logger.info("-" * 80)
    
    tfidf = TfidfVectorizer(
        max_features=15000,      # Increased features for better representation
        ngram_range=(1, 3),       # Unigrams, bigrams, trigrams
        min_df=2,                 # Minimum document frequency
        max_df=0.95,              # Maximum document frequency (remove very common words)
        stop_words='english',      # Remove English stop words
        sublinear_tf=True,        # Apply sublinear TF scaling (1 + log(tf))
        norm='l2'                 # L2 normalization
    )
    
    logger.info("TF-IDF Parameters:")
    logger.info(f"  Max features: {15000}")
    logger.info(f"  N-gram range: (1, 3)")
    logger.info(f"  Min document frequency: 2")
    logger.info(f"  Max document frequency: 0.95")
    logger.info(f"  Stop words: English")
    logger.info(f"  Sublinear TF: True")
    logger.info(f"  Norm: L2")
    
    # Transform features
    logger.info("\nTransforming features...")
    start_time = time.time()
    try:
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)
        X_test_tfidf = tfidf.transform(X_test)
    except Exception as e:
        logger.error(f"Feature transformation failed: {e}")
        return None, {}
    
    transform_time = time.time() - start_time
    
    logger.info(f"Feature matrix shape: {X_train_tfidf.shape}")
    logger.info(f"Transformation time: {transform_time:.2f} seconds")
    
    # ========== LABEL ENCODING ==========
    logger.info("\n[3] LABEL ENCODING")
    logger.info("-" * 80)
    
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    y_val_num = le.transform(y_val)
    y_test_num = le.transform(y_test)
    
    logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # ========== MODEL ARCHITECTURE ==========
    logger.info("\n[4] MODEL ARCHITECTURE")
    logger.info("-" * 80)
    
    # Define individual models
    models = {}
    trained_models = {}
    model_scores = {}
    
    # 1. Logistic Regression (L2 regularization)
    logger.info("\n[4.1] Logistic Regression")
    logger.info("  Loss function: Logistic Loss (Cross-Entropy)")
    logger.info("  Regularization: L2 (Ridge)")
    logger.info("  Learning rate: Adaptive (LBFGS optimizer)")
    logger.info("  Max iterations: 2000")
    logger.info("  C (regularization strength): 1.0")
    logger.info("  Class weight: Balanced")
    
    try:
        models['lr'] = LogisticRegression(
            random_state=42,
            max_iter=2000,           # Maximum iterations
            C=1.0,                    # Inverse regularization strength (lower = stronger)
            penalty='l2',            # L2 regularization
            solver='lbfgs',          # Limited-memory BFGS (quasi-Newton method)
            class_weight='balanced'  # Handle class imbalance
            # Note: multi_class parameter not needed for binary classification
        )
    except Exception as e:
        logger.error(f"Failed to create Logistic Regression model: {e}")
        models['lr'] = None
    
    # 2. Random Forest
    logger.info("\n[4.2] Random Forest")
    logger.info("  Number of trees: 200")
    logger.info("  Max depth: 20")
    logger.info("  Min samples split: 5")
    logger.info("  Min samples leaf: 2")
    logger.info("  Max features: sqrt")
    logger.info("  Bootstrap: True")
    logger.info("  Class weight: Balanced")
    
    try:
        models['rf'] = RandomForestClassifier(
            n_estimators=200,        # Number of trees
            max_depth=20,            # Maximum depth
            min_samples_split=5,     # Minimum samples to split
            min_samples_leaf=2,      # Minimum samples in leaf
            max_features='sqrt',     # Features per split
            bootstrap=True,          # Bootstrap sampling
            random_state=42,
            class_weight='balanced',
            n_jobs=-1                # Use all CPU cores
        )
    except Exception as e:
        logger.error(f"Failed to create Random Forest model: {e}")
        models['rf'] = None
    
    # 3. Gradient Boosting
    logger.info("\n[4.3] Gradient Boosting")
    logger.info("  Number of estimators: 200")
    logger.info("  Learning rate: 0.1")
    logger.info("  Max depth: 5")
    logger.info("  Min samples split: 5")
    logger.info("  Loss function: Deviance (Log Loss)")
    logger.info("  Subsample: 0.8")
    
    try:
        models['gb'] = GradientBoostingClassifier(
            n_estimators=200,        # Number of boosting stages
            learning_rate=0.1,      # Learning rate (shrinkage)
            max_depth=5,             # Maximum depth of trees
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,           # Fraction of samples for each tree
            random_state=42,
            loss='log_loss'          # Log loss (logistic regression for classification)
        )
    except Exception as e:
        logger.error(f"Failed to create Gradient Boosting model: {e}")
        models['gb'] = None
    
    # 4. SVM (for ensemble diversity)
    logger.info("\n[4.4] Support Vector Machine")
    logger.info("  Kernel: RBF (Radial Basis Function)")
    logger.info("  C (regularization): 1.0")
    logger.info("  Gamma: scale (1 / n_features)")
    logger.info("  Class weight: Balanced")
    
    try:
        models['svm'] = SVC(
            kernel='rbf',            # Radial Basis Function kernel
            C=1.0,                   # Regularization parameter
            gamma='scale',           # Kernel coefficient
            probability=True,         # Enable probability estimates
            random_state=42,
            class_weight='balanced'
        )
    except Exception as e:
        logger.error(f"Failed to create SVM model: {e}")
        models['svm'] = None
    
    # ========== TRAINING INDIVIDUAL MODELS ==========
    logger.info("\n[5] TRAINING INDIVIDUAL MODELS")
    logger.info("-" * 80)
    
    for name, model in models.items():
        if model is None:
            logger.warning(f"Skipping {name.upper()} - model creation failed")
            continue
            
        logger.info(f"\nTraining {name.upper()}...")
        start_time = time.time()
        
        try:
            model.fit(X_train_tfidf, y_train_num)
            train_time = time.time() - start_time
            
            # Evaluate on validation set
            val_score = model.score(X_val_tfidf, y_val_num)
            model_scores[name] = val_score
            trained_models[name] = model
            
            logger.info(f"  Training time: {train_time:.2f} seconds")
            logger.info(f"  Validation accuracy: {val_score:.4f} ({val_score*100:.2f}%)")
        except Exception as e:
            logger.error(f"  Training failed for {name.upper()}: {e}")
            continue
    
    if not trained_models:
        logger.error("No models trained successfully!")
        return None, {}
    
    # ========== ENSEMBLE MODEL ==========
    logger.info("\n[6] CREATING ENSEMBLE MODEL")
    logger.info("-" * 80)
    
    # Create voting classifier with soft voting (uses probabilities)
    try:
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in trained_models.items()],
            voting='soft',           # Use probability predictions
            weights=None             # Equal weights (can be tuned)
        )
        
        logger.info("Ensemble configuration:")
        logger.info("  Voting: Soft (uses probability predictions)")
        logger.info("  Weights: Equal for all models")
        logger.info(f"  Models: {', '.join(trained_models.keys())}")
        
        # Train ensemble
        logger.info("\nTraining ensemble...")
        start_time = time.time()
        ensemble.fit(X_train_tfidf, y_train_num)
        ensemble_train_time = time.time() - start_time
        
        logger.info(f"Ensemble training time: {ensemble_train_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Ensemble creation/training failed: {e}")
        return None, {}
    
    # ========== VALIDATION ==========
    logger.info("\n[7] VALIDATION")
    logger.info("-" * 80)
    
    val_score = ensemble.score(X_val_tfidf, y_val_num)
    logger.info(f"Validation accuracy: {val_score:.4f} ({val_score*100:.2f}%)")
    
    # Cross-validation on training set
    logger.info("\nPerforming 5-fold cross-validation...")
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble, X_train_tfidf, y_train_num, cv=cv, scoring='accuracy')
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        logger.info(f"Std CV accuracy:  {cv_scores.std():.4f} (+/- {cv_scores.std()*2*100:.2f}%)")
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        cv_scores = np.array([])
    
    # ========== TEST EVALUATION ==========
    logger.info("\n[8] TEST SET EVALUATION")
    logger.info("-" * 80)
    
    test_accuracy = ensemble.score(X_test_tfidf, y_test_num)
    logger.info(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    y_pred = ensemble.predict(X_test_tfidf)
    y_pred_proba = ensemble.predict_proba(X_test_tfidf)[:, 1]
    
    # Classification report
    report = classification_report(y_test_num, y_pred, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test_num, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_num, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # ROC-AUC score
    try:
        roc_auc = roc_auc_score(y_test_num, y_pred_proba)
        logger.info(f"\nROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        roc_auc = None
    
    # ========== CREATE PIPELINE ==========
    logger.info("\n[9] CREATING PIPELINE")
    logger.info("-" * 80)
    
    # Create final pipeline with TF-IDF + Ensemble
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('ensemble', ensemble)
    ])
    
    # Verify compatibility
    try:
        # Test predict_proba
        test_features = ["test file name test content"]
        test_proba = pipeline.predict_proba(test_features)
        test_classes = pipeline.named_steps['ensemble'].classes_
        logger.info("✓ Model compatibility verified (predict_proba and classes_ work)")
    except Exception as e:
        logger.warning(f"Model compatibility check failed: {e}")
    
    # ========== METRICS SUMMARY ==========
    metrics = {
        'model_type': 'Ensemble Voting Classifier',
        'models': list(trained_models.keys()),
        'voting': 'soft',
        'training_parameters': {
            'train_size': len(X_train),
            'validation_size': len(X_val),
            'test_size': len(X_test),
            'train_percentage': len(X_train)/len(features)*100,
            'validation_percentage': len(X_val)/len(features)*100,
            'test_percentage': len(X_test)/len(features)*100
        },
        'feature_engineering': {
            'max_features': 15000,
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': 0.95,
            'stop_words': 'english',
            'sublinear_tf': True,
            'norm': 'l2'
        },
        'individual_model_scores': model_scores,
        'individual_model_configs': {
            'logistic_regression': {
                'max_iter': 2000,
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'loss_function': 'Logistic Loss (Cross-Entropy)',
                'learning_rate': 'Adaptive (LBFGS)',
                'class_weight': 'balanced'
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced'
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'loss_function': 'Deviance (Log Loss)',
                'subsample': 0.8
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'class_weight': 'balanced'
            }
        },
        'validation_accuracy': val_score,
        'cv_mean': float(cv_scores.mean()) if len(cv_scores) > 0 else None,
        'cv_std': float(cv_scores.std()) if len(cv_scores) > 0 else None,
        'test_accuracy': test_accuracy,
        'roc_auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'training_time_seconds': ensemble_train_time,
        'cross_validation': {
            'folds': 5,
            'method': 'StratifiedKFold',
            'scores': cv_scores.tolist() if len(cv_scores) > 0 else []
        },
        'label_encoder': {
            'classes': le.classes_.tolist(),
            'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
        }
    }
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    if len(cv_scores) > 0:
        logger.info(f"Cross-Validation Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    if roc_auc:
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return pipeline, metrics


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Strong Ensemble Stock & Sales Classifier')
    parser.add_argument('--input', type=str, default='training data attachments',
                       help='Input folder path (default: training data attachments)')
    parser.add_argument('--output', type=str, 
                       default='document_converter/models/classifier_model_ensemble.pkl',
                       help='Output model path')
    parser.add_argument('--files-per-stockist', type=int, default=10,
                       help='Number of files to sample per stockist')
    parser.add_argument('--max-total-files', type=int, default=None,
                       help='Maximum total files to process')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_folder = Path(args.input)
    output_path = Path(args.output)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Collect training data
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
    model, metrics = train_strong_ensemble_model(features, labels)
    
    if model is None:
        logger.error("Model training failed")
        return
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"\nModel saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return
    
    # Save detailed metrics
    metrics_path = output_path.with_suffix('.metrics.json')
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Detailed metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
    
    # Save stockist stats
    stats_path = output_path.with_suffix('.stockist_stats.json')
    try:
        with open(stats_path, 'w') as f:
            json.dump(stockist_stats, f, indent=2)
        logger.info(f"Stockist statistics saved to {stats_path}")
    except Exception as e:
        logger.warning(f"Failed to save stockist stats: {e}")
    
    logger.info("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

