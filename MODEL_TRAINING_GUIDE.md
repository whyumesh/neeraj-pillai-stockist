# Model Training Guide

This guide explains how to train, validate, and retrain the document classification model for highest accuracy.

## Overview

The system uses an Ensemble Voting Classifier that combines Random Forest and SVM models to classify documents as either "Stock & Sales Reports" or "Other" (including Purchase Orders). The model uses a hybrid approach combining rule-based classification with machine learning.

## System Architecture

### Classification Flow

1. **Rule-Based Classification**: First checks for PO indicators, then evaluates Stock indicators
2. **ML Classification**: Uses trained ensemble model to predict classification
3. **Score Combination**: Combines rule-based (70% weight) and ML (30% weight) scores
4. **Final Classification**: Returns 'stock_sales_report' or 'other' based on combined score

### Key Components

- **DocumentClassifier** (`document_converter/document_classifier.py`): Main classification logic with PO detection
- **StockSalesExtractor** (`document_converter/extractors/stock_sales_extractor.py`): Extracts data from Stock & Sales reports
- **Training Scripts**: 
  - `train_classifier_ensemble.py`: Main training script for ensemble model
  - `train_classifier_stock_only.py`: Alternative training script

## Training Process

### Step 1: Validate Training Data

Before training, validate your training data to ensure correct labeling:

```bash
python document_converter/validate_training_data.py --input "training data attachments" --output document_converter/models/training_data_validation_report.json
```

This script will:
- Scan all files in the training folder
- Identify PO files vs Stock files
- Check for mislabeled files
- Generate a data quality report

**What to look for:**
- PO files should be labeled as 'other', not 'stock'
- Stock files should be labeled as 'stock'
- Check the mislabeled files list and fix any issues
- Ensure you have at least 10 PO files and 50+ Stock files

### Step 2: Train the Model

Train the ensemble model with validated data:

```bash
python document_converter/train_classifier_ensemble.py --input "training data attachments" --output document_converter/models/classifier_model_ensemble.pkl --files-per-stockist 10
```

**Parameters:**
- `--input`: Path to training data folder (default: "training data attachments")
- `--output`: Output model path (default: `document_converter/models/classifier_model_ensemble.pkl`)
- `--files-per-stockist`: Number of files to sample per stockist (default: 10)
- `--max-total-files`: Maximum total files to process (optional)
- `--seed`: Random seed for reproducibility (default: 42)

**Training Process:**
1. Collects training data with stratified sampling per stockist
2. Uses enhanced classifier (with PO detection) for labeling
3. Extracts features: filename + first 3000 chars of content + PO/Stock indicators
4. Splits data: 70% train, 15% validation, 15% test
5. Trains ensemble model (Random Forest + SVM)
6. Evaluates on test set
7. Saves model and metrics

**Expected Output:**
- Model file: `classifier_model_ensemble.pkl`
- Metrics file: `classifier_model_ensemble.metrics.json`
- Stockist stats: `classifier_model_ensemble.stockist_stats.json`

### Step 3: Validate Model Performance

After training, validate the model on test files:

```bash
python document_converter/validate_model.py --model document_converter/models/classifier_model_ensemble.pkl --test-folder "attachments test" --output document_converter/models/model_validation_report.json
```

**What to check:**
- Overall accuracy should be >99%
- PO Detection precision should be >95%
- Stock Detection precision should be >98%
- Review misclassified files to identify patterns

### Step 4: Validate Extraction Quality

Validate that extraction is capturing all line items:

```bash
python document_converter/validate_extraction.py --input "attachments test" --output document_converter/models/extraction_validation_report.json
```

**What to check:**
- Average items per file should be reasonable (varies by document)
- Files with zero items should be <10% of total
- Review files with validation issues

## Retraining Process

### When to Retrain

Retrain the model when:
1. PO misclassification occurs (PO files classified as Stock)
2. New document types are introduced
3. Model accuracy drops below thresholds
4. Training data is updated or corrected

### Retraining Steps

1. **Backup existing model:**
   ```bash
   cp document_converter/models/classifier_model_ensemble.pkl document_converter/models/classifier_model_ensemble_backup.pkl
   ```

2. **Validate training data** (see Step 1 above)

3. **Fix any mislabeled files** in training data

4. **Retrain model** (see Step 2 above)

5. **Validate new model** (see Step 3 above)

6. **Compare with old model:**
   - Check if accuracy improved
   - Verify PO detection improved
   - Review misclassified files

7. **Deploy new model** if performance is better

## Configuration

### PO Detection Configuration

PO detection patterns are configured in `document_converter/config.json`:

```json
{
  "classification": {
    "po_keywords": [
      "purchase order",
      "po number",
      "po no",
      "order number",
      "po_",
      "po-"
    ],
    "filename_patterns": {
      "po": [
        "po\\d+",
        "purchase.*order",
        "^po"
      ]
    }
  }
}
```

### Stock Detection Configuration

Stock detection patterns are also in `config.json`:

```json
{
  "classification": {
    "stock_sales_keywords": [
      "stock statement",
      "stock and sales",
      "opening qty",
      "receipt qty",
      ...
    ],
    "filename_patterns": {
      "stock_sales": [
        "st-",
        "stock",
        "statement",
        ...
      ]
    }
  }
}
```

## Troubleshooting

### Issue: PO Files Still Misclassified

**Symptoms:** PO files (e.g., `PO1650_7293.pdf`) are classified as 'stock_sales_report'

**Solutions:**
1. Check if PO patterns are in config.json
2. Verify PO detection is working: Check logs for "PO indicator found"
3. Retrain model with more PO examples in training data
4. Increase PO file count in training set (aim for 100+ PO files)

### Issue: Missing Line Items in Extraction

**Symptoms:** Extracted Excel files have fewer items than original PDFs

**Solutions:**
1. Check extraction validation report
2. Review files with validation issues
3. Adjust extraction filters if needed (see `stock_sales_extractor.py`)
4. Check if PDFs are scanned (may need better OCR)

### Issue: Low Model Accuracy

**Symptoms:** Model accuracy <95% on test set

**Solutions:**
1. Validate training data quality
2. Ensure balanced classes (not too imbalanced)
3. Increase training data size
4. Check for mislabeled training files
5. Review feature engineering (may need adjustment)

### Issue: Model Not Loading

**Symptoms:** Error loading model file

**Solutions:**
1. Check if model file exists
2. Verify pickle file is not corrupted
3. Check Python version compatibility
4. Ensure all dependencies are installed

## Best Practices

1. **Regular Validation**: Run validation scripts regularly to catch issues early
2. **Version Control**: Keep backups of models before retraining
3. **Data Quality**: Always validate training data before training
4. **Incremental Updates**: Retrain with new data periodically
5. **Monitor Performance**: Track model performance over time
6. **Document Changes**: Keep notes on what changed between model versions

## Success Metrics

Target metrics for a well-trained model:

- **Overall Accuracy**: >99%
- **PO Detection Precision**: >95%
- **PO Detection Recall**: >95%
- **Stock Detection Precision**: >98%
- **Stock Detection Recall**: >98%
- **False Positives**: <1% (PO files misclassified as Stock)
- **Extraction Completeness**: >95% of line items extracted

## File Structure

```
document_converter/
├── models/
│   ├── classifier_model_ensemble.pkl          # Trained model
│   ├── classifier_model_ensemble.metrics.json # Training metrics
│   ├── classifier_model_ensemble.stockist_stats.json # Stockist statistics
│   ├── training_data_validation_report.json  # Training data validation
│   ├── model_validation_report.json           # Model validation
│   └── extraction_validation_report.json      # Extraction validation
├── train_classifier_ensemble.py               # Main training script
├── validate_training_data.py                  # Training data validation
├── validate_model.py                          # Model validation
└── validate_extraction.py                     # Extraction validation
```

## Additional Resources

- See `TECHNICAL_DOCUMENTATION.md` for technical details
- See `STOCK_ONLY_MIGRATION.md` for migration notes
- Check logs in `Output/ProcessingLog.json` for runtime classification results

