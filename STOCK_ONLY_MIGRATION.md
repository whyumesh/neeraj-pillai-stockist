# Stock & Sales Only System - Migration Summary

## Overview
The system has been migrated from a multi-class classification system (PO + Stock & Sales) to a **binary classification system focused exclusively on Stock & Sales reports**.

## Key Changes

### 1. Enhanced Training Script (`train_classifier_stock_only.py`)
- **Binary Classification**: Trains a model to distinguish Stock & Sales reports from all other documents
- **Stratified Sampling**: Samples ~10 files per unique stockist for balanced training
- **Improved Features**:
  - Increased TF-IDF features (10,000 vs 5,000)
  - Trigram support (1-3 grams)
  - Class balancing for imbalanced datasets
  - Cross-validation and detailed metrics

### 2. Binary Classifier (`document_classifier.py`)
- **Simplified Logic**: Returns `stock_sales_report` or `other` (no PO classification)
- **Rule-based Scoring**: Focused on Stock & Sales patterns only
- **ML Integration**: Supports binary ML models trained with `train_classifier_stock_only.py`
- **Higher Confidence**: Optimized scoring to achieve maximum confidence (>0.8) for clear Stock reports

### 3. Document Processor (`document_processor.py`)
- **Auto-skip Non-Stock Files**: Files classified as "other" are automatically skipped
- **Stock-only Processing**: Only Stock & Sales reports are extracted and converted
- **Model Loading**: Prefers `classifier_model_stock_only.pkl`, falls back to legacy model
- **Removed PO Extractor**: No longer imports or uses Purchase Order extractor

### 4. Configuration (`config.json`)
- **Removed PO Settings**: All Purchase Order keywords and patterns removed
- **Enhanced Stock Keywords**: Added more Stock & Sales patterns
- **Updated Paths**: Default input folder changed to `attachments`

### 5. Logging & Statistics (`logging_utils.py`)
- **Updated Statistics**: Removed PO counts, added "other_documents" count
- **Stock-focused Metrics**: Statistics now reflect Stock-only processing

### 6. Entry Points (`run_cli.py`, `main.py`)
- **Updated Defaults**: Input folder default changed to `attachments`
- **Updated Statistics Display**: Shows Stock & Sales and Other document counts

## Training the Model

To train the new binary classifier with your 1500+ stockist files:

```bash
python document_converter/train_classifier_stock_only.py \
    --input attachments \
    --files-per-stockist 10 \
    --max-total-files 15000 \
    --output document_converter/models/classifier_model_stock_only.pkl
```

### Training Parameters
- `--input`: Folder containing stockist files (default: `attachments`)
- `--files-per-stockist`: Number of files to sample per stockist (default: 10)
- `--max-total-files`: Maximum total files to process (default: None = no limit)
- `--output`: Output model path
- `--seed`: Random seed for reproducibility (default: 42)

### Training Output
The script generates:
1. **Model file**: `classifier_model_stock_only.pkl` - Trained binary classifier
2. **Metrics file**: `classifier_model_stock_only.metrics.json` - Training metrics and evaluation
3. **Stockist stats**: `classifier_model_stock_only.stockist_stats.json` - Per-stockist file counts

## Running the System

### CLI Mode
```bash
python run_cli.py --input attachments
```

### Processing Behavior
1. **Scans** all files in the input folder
2. **Classifies** each file as Stock & Sales or Other
3. **Skips** files classified as "other" (logs them as skipped)
4. **Extracts** data from Stock & Sales reports only
5. **Converts** to Excel format
6. **Logs** all operations to JSON and CSV files

## Benefits of Binary Classification

1. **Higher Accuracy**: Simpler binary problem vs multi-class classification
2. **Better Performance**: Faster classification with fewer features
3. **Easier Maintenance**: Single focus on Stock & Sales patterns
4. **Scalability**: Can handle 1500+ unique stockists with stratified sampling
5. **Confidence**: Optimized to achieve high confidence (>0.8) for clear Stock reports

## Migration Notes

- **Backward Compatibility**: System can still load legacy models but prefers new binary model
- **No Breaking Changes**: Existing Stock & Sales extraction logic unchanged
- **Log Format**: Logs now include "other" classification instead of "purchase_order"
- **Statistics**: Updated to reflect Stock-only focus

## Next Steps

1. **Train the Model**: Run the training script with your 1500+ stockist files
2. **Validate**: Test on a sample of files to verify accuracy
3. **Deploy**: Use the trained model for production processing
4. **Monitor**: Review logs to track classification accuracy and extraction quality


