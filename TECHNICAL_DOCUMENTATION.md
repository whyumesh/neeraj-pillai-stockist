# Document Classification & Conversion System - Technical Documentation

## 1. Project Overview

### Purpose
An offline, AI-powered system for automatically classifying and converting Purchase Order (PO) and Stock & Sales documents from various formats (PDF, TXT, CSV, scanned images) into structured Excel files.

### Business Context
- Processes documents from `EmailAttachments` folder
- Supports both digital and scanned documents
- Automatic classification and extraction
- Outputs formatted Excel files for downstream analysis

---

## 2. System Architecture

### Architecture Pattern
**Layered Pipeline Architecture** with clear separation of concerns:

```
File Detection → Text Extraction → Classification → Data Extraction → Excel Conversion
```

**Component Flow:**
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ File Scanner│ --> │ OCR/Text     │ --> │ Classifier  │ --> │ Extractors  │ --> │ Excel       │
│             │     │ Extractor    │     │             │     │             │     │ Converter   │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Design Principles

1. **Template-Free Extraction**: Does not rely on fixed templates; uses intelligent table detection
2. **Hybrid Classification**: Combines rule-based and ML approaches for flexibility
3. **Graceful Degradation**: Multiple fallback mechanisms at each stage
4. **Configuration-Driven**: Most parameters configurable without code changes
5. **Structured Logging**: Comprehensive logging for debugging and audit trails

---

## 3. Component Breakdown

### 3.1 File Scanner (`file_scanner.py`)

**Purpose**: Detects and categorizes supported files

**Key Functions**:
- `scan_files()`: Recursively scans for supported formats
- `_analyze_file()`: Extracts file metadata
- `_is_scanned_file()`: Detects scanned PDFs vs digital PDFs

**Algorithm**:
1. Scan directory with glob patterns (`**/*` for recursive)
2. Filter by supported extensions (`.pdf`, `.txt`, `.csv`, `.jpg`, `.png`)
3. For PDFs: Attempt text extraction; <100 chars → scanned
4. Return file info: path, type, is_scanned flag, size, timestamp

**Why this approach**:
- Pre-filters unsupported files (efficiency)
- Scanned detection optimizes OCR path
- Metadata aids logging and debugging

---

### 3.2 OCR Processor (`ocr_processor.py`)

**Purpose**: Extracts text from scanned documents and images

**Strategy**: Multi-tier fallback
```
Primary: PaddleOCR → Fallback: Tesseract → Final Fallback: PyPDF2
```

**Components**:
- **PaddleOCR**: Primary OCR engine (better accuracy for tables and mixed layouts)
- **Tesseract**: Fallback OCR engine (proven, widely available)
- **Image Preprocessing**: Deskew, binarize, noise removal (via OpenCV)

**Key Methods**:
- `extract_text_from_pdf()`: Handles both digital and scanned PDFs
- `extract_text_from_image()`: OCR on image files
- Uses `pdf2image` to convert PDF pages to images for OCR

**Why dual OCR**:
- PaddleOCR handles complex layouts better (tables, multi-column)
- Tesseract provides reliability if PaddleOCR fails
- Preprocessing improves accuracy on poor-quality scans

**Image Preprocessing** (`utils/image_preprocessing.py`):
- **Deskew**: Corrects document rotation
- **Binarize**: Converts to black/white (improves OCR accuracy)
- **Noise removal**: Cleans artifacts and speckles

---

### 3.3 Document Classifier (`document_classifier.py`)

**Purpose**: Classifies documents as Purchase Order or Stock & Sales Report

**Approach**: **Hybrid Rule-Based + Optional ML**

**Algorithm**:

#### Step 1: Rule-Based Scoring
- **Filename patterns**: Regex matching (`po_`, `stock`, `st-`)
- **Filename keywords**: Keyword presence counting
- **Content patterns**: PO numbers, quantity fields
- **Content keywords**: Domain keywords in content

**Scoring Formula**:
```python
# Filename scoring
po_score = (pattern_matches * 0.5) + (keyword_matches * 0.4)  # Max: 0.95

# Content scoring  
po_score += (content_keywords * 0.25) + (pattern_matches * 0.4)  # Max: 0.6 additional

# Final normalization: min(po_score, 1.0)
```

#### Step 2: ML Classification (if model available)
- Uses scikit-learn (trained via `train_classifier.py`)
- Features: filename + first 1000 chars of content
- Returns probability scores

#### Step 3: Score Combination
```python
# Without ML model (common case)
combined_score = rule_score * 1.1  # Boost to 0.98 max

# With ML model
combined_score = (rule_score * 0.7) + (ml_score * 0.3)
```

**Why Hybrid**:
- **Rule-based**: Fast, configurable, explainable
- **ML**: Learns from data, handles edge cases
- **Fallback**: Works even without trained model

**Confidence Boosting**:
- Clear indicators (2+ keyword matches) → Minimum 0.85 confidence
- Strong patterns found → Boosted to 0.75-0.98
- Clear winner (score diff > 0.2) → Minimum 0.85 confidence

---

### 3.4 Data Extractors (`extractors/`)

#### 3.4.1 Stock & Sales Extractor (`stock_sales_extractor.py`)

**Purpose**: Extracts structured data from Stock & Sales reports

**Extraction Strategies** (in priority order):

1. **Table Extraction** (pdfplumber) - Primary method
2. **Text-Based Parsing** - Fallback for space-delimited formats

**Table Extraction Flow**:
```python
1. Extract tables using pdfplumber (multiple strategies: lines_strict, text, default)
2. Detect header rows (multi-row headers common: "Opening Qty", "Receipt Qty")
3. Build normalized column names: "Opening Qty", "Opening Value", etc.
4. Parse data rows preserving exact structure
5. Filter out metadata rows (company names, totals, section headers)
```

**Header Detection**:
- Looks for rows with keywords: `qty`, `quantity`, `value`, `opening`, `receipt`
- Supports multi-row headers (category row + sub-header row)
- Normalizes to standard names

**Text Parsing (Fallback)**:
- Space-delimited format: `ITEM_NAME PCS QTY VALUE QTY VALUE...`
- Finds numeric sequence start (identifies unit words: PCS, CS, etc.)
- Parses 9 expected numeric values (opening qty/value, receipt qty/value, etc.)
- Validates line structure before parsing

**Why Two Strategies**:
- Tables capture exact structure (most accurate)
- Text parsing handles edge cases (some PDFs don't have proper tables)

**Section Detection**:
- Pattern: `COMPANY NAME (SECTION_NAME)`
- Regex: `^[A-Z][A-Z0-9 \-\.\&/]+?\(([A-Z0-9 \-\.\&/]+)\)\s*$`
- Each section gets separate sheet in Excel output

#### 3.4.2 Purchase Order Extractor (`po_extractor.py`)

**Purpose**: Extracts data from Purchase Order documents

**Challenges**:
- Variable column layouts across vendors
- Multi-word product names
- Blank cells must be preserved
- Header rows often confused with metadata

**Solution**: Template-free table extraction

**Algorithm**:
1. Extract all tables from PDF (pdfplumber with multiple strategies)
2. Identify item table (looks for PO-specific headers: "Sr.", "Product Name", "Qty")
3. Build headers from detected header rows (preserves original names)
4. Preserve ALL columns as-is (no normalization to predefined fields)
5. Map each row to header columns, preserving blanks

**Key Improvements**:
- **Header Detection**: Uses multiple indicators ("Name of Product" is strong signal)
- **Company Name Filtering**: Skips metadata rows (e.g., "OTT HEALTH", "CARE PVT LT")
- **Cell Cleaning**: Fixes fragmented text in cells
- **Blank Preservation**: Treats blanks as valid data (not errors)

**Header Detection Logic**:
- Strong match: Has item number AND product keywords → High confidence
- Strong match: Has "Name of Product" specifically → High confidence
- Moderate match: Has Qty AND (Product or Packing) → Medium confidence
- Filters out company name rows (short, all caps, no header keywords)

**Table Extraction Strategies**:
1. Text-based (primary): Better for preserving cell text integrity
2. Lines-strict (fallback): Better for PDFs with clear table borders
3. Default (fallback): Uses pdfplumber defaults

---

### 3.5 Excel Converter (`excel_converter.py`)

**Purpose**: Converts extracted data to formatted Excel files

**Features**:
- Multiple sheets per document (for sections)
- Header formatting (blue background, white text)
- Cell borders and alignment
- Auto-adjusted column widths
- Right-aligned numeric columns

**Methods**:
- `convert_stock_sales_to_excel()`: Creates multiple sheets (Summary + per-section)
- `convert_po_to_excel()`: Creates single sheet with all items
- `_add_items_to_sheet()`: Preserves column order from extraction

**Why openpyxl**:
- Full formatting control (not possible with pandas.to_excel)
- Multi-sheet support
- Handles large datasets efficiently

**Column Preservation**:
- Uses column order from first item (preserves PDF structure)
- No column reordering or filtering
- Preserves original header names

---

### 3.6 Document Processor (`document_processor.py`)

**Purpose**: Main orchestrator - coordinates all components

**Flow**:
```python
1. FileScanner.scan_files() → Get file list
2. For each file:
   a. OCRProcessor / PyPDF2 → Extract text
   b. DocumentClassifier.classify() → Determine type
   c. Extractor (PO or Stock) → Extract data
   d. ExcelConverter → Generate Excel
   e. ProcessingLogger → Log results
```

**Error Handling**:
- Try-except at each stage
- Returns status (success/error)
- Logs errors with details
- Continues processing remaining files

**Initialization**:
- Loads config
- Initializes OCR engines
- Loads ML model (if available)
- Sets up loggers

---

### 3.7 Configuration System (`config_loader.py`)

**Purpose**: Centralized configuration management

**Approach**:
- JSON file (`config.json`)
- Dot-notation access: `config.get('ocr.dpi')`
- Default values if key missing
- Hot-reloadable (read on each access)

**Why JSON**:
- No code changes for settings
- Human-readable
- Easy versioning

**Configuration Sections**:
- `ocr`: OCR settings (primary engine, DPI, preprocessing)
- `classification`: Keywords and patterns
- `paths`: Input/output directories
- `excel`: Formatting options

---

### 3.8 Logging System (`utils/logging_utils.py`)

**Purpose**: Tracks processing results and statistics

**Dual Format**:
1. **JSON Log**: Structured data for programmatic access
2. **CSV Log**: Human-readable with key metrics

**CSV Log Columns**:
- Timestamp
- File Name
- File Type
- Classification
- Conversion Possible (Yes/No)
- Items Extracted
- Sections Found
- Status
- Confidence
- Output File
- Error/Notes

**Why Both**:
- JSON: Machine-readable, supports automation/analysis
- CSV: Easy to open in Excel, quick review

**Automatic Generation**:
- Created in same directory as JSON log
- Appends on each file processing
- Headers written if file doesn't exist

---

## 4. Processing Flow (Detailed)

### Complete Pipeline:

```
1. USER ACTION
   ├─ GUI: Select folder → Process
   └─ CLI: python -m document_converter --input <folder>

2. INITIALIZATION
   ├─ ConfigLoader.load_config() → Read config.json
   ├─ DocumentProcessor.__init__() → Initialize components
   ├─ OCRProcessor.__init__() → Initialize PaddleOCR/Tesseract
   └─ ProcessingLogger.__init__() → Setup log files

3. FILE SCANNING
   ├─ FileScanner.scan_files() → Discover files
   ├─ Filter by extension → .pdf, .txt, .csv, .jpg, .png
   ├─ Analyze each file → Detect scanned vs digital
   └─ Return file list with metadata

4. FOR EACH FILE:
   
   4a. TEXT EXTRACTION
       ├─ Is scanned? → OCRProcessor.extract_text_from_pdf()
       │   ├─ Try PaddleOCR → Success?
       │   ├─ Try Tesseract → Success?
       │   └─ Preprocess image (deskew, binarize, denoise)
       └─ Is digital? → PyPDF2.extract_text() or direct read
   
   4b. CLASSIFICATION
       ├─ DocumentClassifier.classify()
       │   ├─ Rule-based scoring (filename + content)
       │   ├─ ML scoring (if model available)
       │   └─ Combine scores → Classification + Confidence
       └─ Return: ('purchase_order', 0.95) or ('stock_sales_report', 0.92)
   
   4c. DATA EXTRACTION
       ├─ PO? → PurchaseOrderExtractor.extract_from_pdf()
       │   ├─ Extract tables (pdfplumber)
       │   ├─ Identify item table (header detection)
       │   ├─ Parse rows → Preserve all columns
       │   └─ Return: {po_number, date, vendor, items: [...]}
       │
       └─ Stock? → StockSalesExtractor.extract_from_pdf()
           ├─ Extract tables (pdfplumber)
           ├─ Try text parsing (fallback)
           ├─ Detect sections (company names in parentheses)
           ├─ Parse items → Standardized column names
           └─ Return: {sections: [...], items: [...], period: ...}
   
   4d. EXCEL CONVERSION
       ├─ ExcelConverter.convert_xxx_to_excel()
       │   ├─ Create workbook (openpyxl)
       │   ├─ Add headers → Format (blue bg, white text)
       │   ├─ Add data rows → Apply borders, alignment
       │   ├─ Auto-adjust column widths
       │   └─ Save to Output/ folder
       └─ Return: True (success) or False (error)
   
   4e. LOGGING
       ├─ ProcessingLogger.log_processing()
       │   ├─ Write to JSON log
       │   ├─ Append to CSV log
       │   └─ Console output
       └─ Save logs on completion

5. COMPLETION
   ├─ ProcessingLogger.get_statistics() → Summary stats
   └─ Return results to user (GUI/CLI)
```

---

## 5. Why This Approach?

### Template-Free Extraction
- **Problem**: Vendor layouts vary widely
- **Solution**: Detect tables dynamically using pdfplumber
- **Benefit**: Works across different formats without code changes

### Hybrid Classification
- **Problem**: Pure rule-based can be rigid; pure ML needs training data
- **Solution**: Rules for speed/configurability; ML for learning
- **Benefit**: Works immediately and improves over time

### Multiple OCR Engines
- **Problem**: OCR accuracy varies by document quality
- **Solution**: PaddleOCR (best) → Tesseract (reliable) → PyPDF2 (fast)
- **Benefit**: Maximum reliability and accuracy

### Configuration-Driven
- **Problem**: Different vendors need different keywords
- **Solution**: JSON config for keywords, patterns, paths
- **Benefit**: No code changes for adjustments

### Dual Logging
- **Problem**: Need both programmatic access and human review
- **Solution**: JSON (structured) + CSV (readable)
- **Benefit**: Flexibility for different use cases

---

## 6. Key Technologies & Libraries

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **pdfplumber** | Table extraction from PDFs | Superior table detection vs PyPDF2 |
| **PaddleOCR** | Primary OCR engine | Better table/layout accuracy |
| **Tesseract** | Fallback OCR | Proven, widely available |
| **pandas** | Data manipulation | Handles large datasets |
| **openpyxl** | Excel generation | Full formatting control |
| **scikit-learn** | ML classification | Optional ML enhancement |
| **PyPDF2** | Digital PDF text extraction | Fast for native PDFs |
| **OpenCV** | Image preprocessing | OCR quality improvement |

---

## 7. Data Structures

### Extracted PO Data:
```python
{
    "po_number": "PO-1234",
    "date": "2025-01-15",
    "vendor": "Company Name",
    "items": [
        {
            "Sr.": 1,
            "Product Name": "Product ABC",
            "Pack": "10ML",
            "Qty": 100,
            "Free": 0,
            "Rate": 50.00,
            "Amount": 5000.00
        },
        ...
    ],
    "total_amount": 5000.00
}
```

### Extracted Stock & Sales Data:
```python
{
    "sections": ["ABBOTT INDIA", "SECTION_NAME"],
    "period": "Jan 2025",
    "items": [
        {
            "Item Description": "Product Name",
            "Opening Qty": 100.0,
            "Opening Value": 5000.0,
            "Receipt Qty": 50.0,
            "Receipt Value": 2500.0,
            "Issue Qty": 30.0,
            "Issue Value": 1500.0,
            "Closing Qty": 120.0,
            "Closing Value": 6000.0,
            "Dump Qty": 0.0,
            "section": "SECTION_NAME"
        },
        ...
    ]
}
```

---

## 8. Error Handling & Validation

**Validation Points**:
1. File existence and readability
2. Text extraction success (min length checks)
3. Classification confidence (thresholds)
4. Item count validation (warn if 0 items)
5. Excel write success (file creation)

**Error Recovery**:
- Logs errors and continues processing
- Returns partial results on failure
- Provides detailed error messages in logs

---

## 9. Performance Considerations

**Optimizations**:
- Lazy OCR initialization (only when needed)
- Batch file processing (sequential, single-threaded)
- Configurable DPI (default 300, can reduce for speed)
- Image preprocessing only for scanned files

**Memory Management**:
- Process one file at a time
- Release OCR engine resources
- Close file handles promptly

---

## 10. Extension Points

**To add new document types**:
1. Add extractor class in `extractors/`
2. Add classification keywords in `config.json`
3. Update `DocumentProcessor._extract_data()` to route to new extractor
4. Update `ExcelConverter` if output format differs

**To improve classification**:
1. Add keywords to `config.json`
2. Train ML model: `python train_classifier.py`
3. Model saved to `models/classifier_model.pkl`

**To customize output**:
1. Edit `excel_converter.py` formatting methods
2. Modify `config.json` for paths and sheet names

---

## 11. File Structure

```
document_converter/
├── __init__.py              # Package initialization
├── __main__.py              # Module entry point
├── main.py                  # CLI entry point
├── main_gui.py              # GUI interface (tkinter)
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md                # User documentation
├── TECHNICAL_DOCUMENTATION.md  # This file
│
├── document_processor.py    # Main orchestrator
├── file_scanner.py          # File detection
├── ocr_processor.py         # OCR handling
├── document_classifier.py   # Classification logic
├── excel_converter.py       # Excel generation
├── config_loader.py         # Configuration management
├── train_classifier.py      # ML model training
│
├── extractors/
│   ├── __init__.py
│   ├── po_extractor.py      # Purchase Order extraction
│   ├── stock_sales_extractor.py  # Stock & Sales extraction
│   └── text_parser.py       # TXT/CSV parsing
│
├── utils/
│   ├── __init__.py
│   ├── image_preprocessing.py  # OCR image preprocessing
│   └── logging_utils.py     # Logging system
│
└── models/                  # Generated ML models
    └── classifier_model.pkl (if trained)
```

---

## 12. Dependencies

### Core Dependencies:
- **Python 3.9+**
- **paddleocr** >= 2.7.0
- **pytesseract** >= 0.3.10
- **pdfplumber** >= 0.10.0
- **PyPDF2** >= 3.0.0
- **pandas** >= 2.0.0
- **openpyxl** >= 3.1.0
- **scikit-learn** >= 1.3.0
- **opencv-python** >= 4.8.0
- **Pillow** >= 10.0.0

### External Tools:
- **Tesseract OCR** (separate installation)
- **Poppler** (for pdf2image - separate installation)

---

## 13. Configuration Reference

### OCR Configuration:
```json
{
  "ocr": {
    "primary": "paddleocr",        // Primary OCR engine
    "fallback": "tesseract",       // Fallback OCR engine
    "dpi": 300,                    // Image resolution for OCR
    "preprocessing": {
      "deskew": true,              // Correct document rotation
      "binarize": true,            // Convert to black/white
      "noise_removal": true,       // Remove artifacts
      "confidence_threshold": 0.6  // Minimum OCR confidence
    }
  }
}
```

### Classification Configuration:
```json
{
  "classification": {
    "po_keywords": [...],          // Keywords for PO detection
    "stock_sales_keywords": [...], // Keywords for Stock detection
    "filename_patterns": {
      "po": [...],                 // Regex patterns for PO filenames
      "stock_sales": [...]         // Regex patterns for Stock filenames
    },
    "combined_confidence_threshold": 0.3  // Minimum confidence for classification
  }
}
```

### Path Configuration:
```json
{
  "paths": {
    "input_folder": "EmailAttachments",
    "output_folder": "Output",
    "po_output": "Output/PurchaseOrders",
    "stock_output": "Output/StockSalesReports",
    "log_file": "Output/ProcessingLog.json"
  }
}
```

---

## 14. Logging & Monitoring

### Log Files:
1. **ProcessingLog.json**: Structured JSON log with all processing details
2. **ProcessingLog_conversion_log.csv**: Human-readable CSV with key metrics

### Log Entry Structure:
- Timestamp
- File information (name, type, path)
- Classification (type, confidence)
- Conversion status (possible, items extracted, sections found)
- Output information (file path, errors)

### Statistics:
- Total files processed
- Success/error/skip counts
- Classification breakdown (PO vs Stock)
- Success rate percentage

---

## 15. Known Limitations & Future Improvements

### Current Limitations:
1. **Single-threaded**: Files processed sequentially
2. **Memory**: Large PDFs may consume significant memory
3. **OCR Accuracy**: Depends on scan quality
4. **Table Detection**: Some complex layouts may not extract perfectly

### Future Improvements:
1. **Parallel Processing**: Multi-threaded file processing
2. **Better OCR**: Fine-tuning preprocessing for specific document types
3. **Enhanced ML**: More training data for better classification
4. **Validation Rules**: Additional data quality checks
5. **Batch Optimization**: Process files in optimized batches

---

## 16. Troubleshooting Guide

### Common Issues:

**Issue**: Low classification confidence
- **Solution**: Add more keywords to `config.json`, train ML model

**Issue**: Missing columns in output
- **Solution**: Check table detection in extractor logs, adjust pdfplumber settings

**Issue**: OCR errors on scanned documents
- **Solution**: Improve image quality, adjust preprocessing settings, check DPI

**Issue**: Empty Excel files
- **Solution**: Check extraction logs, verify table structure matches expected format

---

## 17. Development Guidelines

### Adding New Extractors:
1. Create new class inheriting extraction pattern
2. Implement `extract_from_pdf()` and `extract_from_text()` methods
3. Register in `DocumentProcessor._extract_data()`
4. Add Excel conversion method if needed

### Testing:
- Use sample files from `EmailAttachments` folder
- Check logs for detailed processing information
- Verify Excel output structure matches expectations

### Code Style:
- Follow PEP 8
- Use type hints
- Document all public methods
- Log important operations

---

*Last Updated: January 2025*
*Version: 1.0*


