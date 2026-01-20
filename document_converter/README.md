# Document Classification and Conversion System

An offline, AI-powered system for classifying and converting documents from the EmailAttachments folder. The system automatically classifies Purchase Orders and Stock & Sales reports, then converts PDFs, TXT, CSV, and scanned files to structured Excel format.

## Features

- **Automatic Classification**: Hybrid rule-based + ML approach to classify Purchase Orders vs Stock & Sales reports
- **Multi-format Support**: Processes PDF (digital and scanned), TXT, CSV, XLS/XLSX, and image files (JPG, PNG)
- **Advanced OCR**: Uses PaddleOCR (primary) and Tesseract (fallback) for maximum accuracy on scanned documents
- **Intelligent Data Extraction**: Extracts structured data from documents with vendor-specific handling
- **Excel Export**: Creates formatted Excel files with multiple sheets, proper formatting, and metadata
- **Easy to Use**: Simple GUI interface or CLI mode
- **Fully Offline**: No internet connection required after initial setup

## Installation

### Prerequisites

1. **Python 3.9 or higher** - Download from [python.org](https://www.python.org/downloads/)

2. **Tesseract OCR Engine** - Required for OCR fallback
   - Windows: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install and note the installation path (default: `C:\Program Files\Tesseract-OCR`)

3. **Poppler** (for PDF to image conversion)
   - Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Extract and add `bin` folder to PATH environment variable

### Setup Steps

1. **Clone or download** this project to your local machine

2. **Install Python dependencies**:
   ```bash
   cd document_converter
   pip install -r requirements.txt
   ```

3. **Configure Tesseract path** (if not in PATH):
   - Edit `document_converter/ocr_processor.py` if needed
   - Or set environment variable: `TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

4. **Verify EmailAttachments folder**:
   - Ensure your `EmailAttachments` folder is in the project root directory
   - Or update the path in `config.json`

## Usage

### GUI Mode (Recommended)

Launch the graphical interface:

```bash
python -m document_converter.main --gui
```

Or:

```bash
python document_converter/main_gui.py
```

**GUI Steps**:
1. Click "Browse..." to select your EmailAttachments folder
2. Click "Scan Files" to detect all supported files
3. Review the file list
4. Click "Process All Files" to start conversion
5. Monitor progress in the progress bar
6. View results in the Output folder

### CLI Mode

Process files from command line:

```bash
python -m document_converter.main --input EmailAttachments
```

Or:

```bash
python document_converter/main.py --input EmailAttachments
```

### Training the ML Classifier (Optional)

To improve classification accuracy with your specific documents:

```bash
python document_converter/train_classifier.py --input EmailAttachments --max-files 100
```

This will:
- Analyze files in the EmailAttachments folder
- Train a classifier model using rule-based classifications
- Save the model to `document_converter/models/classifier_model.pkl`

The system will automatically use this model if available.

## Configuration

Edit `document_converter/config.json` to customize:

- **OCR Settings**: DPI, preprocessing options, confidence thresholds
- **Classification Keywords**: Add vendor-specific keywords for better classification
- **Output Paths**: Customize where Excel files are saved
- **Excel Formatting**: Header colors, sheet names, etc.

### Example Configuration

```json
{
  "classification": {
    "po_keywords": [
      "purchase order",
      "po number",
      "your-custom-keyword"
    ],
    "stock_sales_keywords": [
      "stock statement",
      "opening qty",
      "your-custom-keyword"
    ]
  }
}
```

## Output Structure

Processed files are organized in the `Output/` folder:

```
Output/
├── PurchaseOrders/
│   ├── filename_PO.xlsx
│   └── ...
├── StockSalesReports/
│   ├── filename_SS.xlsx
│   └── ...
└── ProcessingLog.json
```

### Excel File Format

**Purchase Orders**:
- Summary sheet with PO details (number, date, vendor, total)
- Items table with description, quantity, price, amount

**Stock & Sales Reports**:
- Metadata sheet (period, source file, statistics)
- Summary sheet (all items)
- Separate sheets per section/vendor

All Excel files include:
- Formatted headers (blue background, white text)
- Borders on all cells
- Auto-adjusted column widths
- Right-aligned numeric columns

## Troubleshooting

### OCR Issues

**Problem**: "Tesseract not found"
- **Solution**: Install Tesseract OCR and ensure it's in PATH or configure the path in code

**Problem**: Poor OCR accuracy on scanned documents
- **Solution**: 
  - Check image quality (should be at least 300 DPI)
  - Enable preprocessing options in `config.json`
  - Ensure good lighting and contrast in source images

### Classification Issues

**Problem**: Files classified incorrectly
- **Solution**: 
  - Train ML model with more examples: `python train_classifier.py --max-files 200`
  - Add keywords to `config.json` specific to your vendor formats
  - Check classification confidence scores in ProcessingLog.json

### PDF Processing Issues

**Problem**: "Failed to extract text from PDF"
- **Solution**:
  - For scanned PDFs, ensure pdf2image and Poppler are installed correctly
  - Check PDF is not password-protected
  - For digital PDFs, ensure PyPDF2 can read the file format

### Memory Issues

**Problem**: "Out of memory" errors with large files
- **Solution**:
  - Process files in smaller batches
  - Reduce OCR DPI in config (default 300)
  - Close other applications to free memory

## File Type Support

| Format | Digital | Scanned | Notes |
|--------|---------|---------|-------|
| PDF | ✅ | ✅ (OCR) | Best support |
| TXT | ✅ | N/A | Direct parsing |
| CSV | ✅ | N/A | Direct conversion |
| XLS/XLSX | ✅ | N/A | Read and convert |
| JPG/PNG | N/A | ✅ (OCR) | Image OCR required |

## Project Structure

```
document_converter/
├── main.py                    # CLI entry point
├── main_gui.py               # GUI interface
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── document_processor.py     # Main processing orchestrator
├── file_scanner.py           # File detection
├── ocr_processor.py          # OCR handling
├── document_classifier.py    # Classification logic
├── excel_converter.py        # Excel generation
├── config_loader.py          # Configuration management
├── train_classifier.py       # ML model training
├── extractors/
│   ├── po_extractor.py       # Purchase Order extraction
│   ├── stock_sales_extractor.py  # Stock & Sales extraction
│   └── text_parser.py        # TXT/CSV parsing
├── models/
│   └── classifier_model.pkl  # Trained ML model (generated)
└── utils/
    ├── image_preprocessing.py
    └── logging_utils.py
```

## Handover Notes

### For Non-Developers

1. **Initial Setup**: Follow Installation steps above
2. **Daily Use**: Simply run the GUI and process your EmailAttachments folder
3. **Adding Keywords**: Edit `config.json` to add new classification keywords (no coding needed)
4. **Viewing Results**: Check `Output/` folder for converted Excel files
5. **Logs**: Check `Output/ProcessingLog.json` for processing history

### For Developers

1. **Extending Extractors**: Add new extractor classes in `extractors/` folder
2. **Custom Classification**: Modify `document_classifier.py` to add new rules
3. **Training Models**: Run `train_classifier.py` after adding new document types
4. **Configuration**: All settings in `config.json` - no code changes needed

## Support

For issues or questions:
1. Check `ProcessingLog.json` for error details
2. Review configuration in `config.json`
3. Ensure all dependencies are installed correctly
4. Verify input file formats are supported

## License

This is an internal tool for document processing. All dependencies are open-source and listed in `requirements.txt`.


