# How to Run the Document Classification System

## Quick Start

### Option 1: GUI Mode (Easiest - Recommended)

**From the project root directory** (where `run_gui.py` is located):

```bash
python run_gui.py
```

This will launch the graphical interface where you can:
1. Browse and select your EmailAttachments folder
2. Scan files
3. Process all files with a click
4. View progress and results

### Option 2: CLI Mode

**From the project root directory**:

```bash
python run_cli.py --input EmailAttachments
```

Or specify a different folder:
```bash
python run_cli.py --input "path/to/your/folder"
```

### Option 3: Using Python Module (from project root)

```bash
# GUI mode
python -m document_converter --gui

# CLI mode
python -m document_converter --input EmailAttachments
```

## Important Notes

### ⚠️ Run from the CORRECT Directory

You must run the scripts from the **project root directory** (where `run_gui.py` is located), NOT from inside the `document_converter/` folder.

**Correct location:**
```
D:\neeraj-pillai-stockist>  ← Run from here
├── run_gui.py
├── run_cli.py
├── document_converter/
│   ├── main.py
│   └── ...
└── EmailAttachments/
```

**Wrong location:**
```
D:\neeraj-pillai-stockist\document_converter>  ← Don't run from here
```

### Installation First

Before running, make sure you've installed dependencies:

```bash
pip install -r document_converter/requirements.txt
```

Also install:
- **Tesseract OCR** (for OCR functionality)
- **Poppler** (for PDF to image conversion)

See `document_converter/README.md` for detailed installation instructions.

## Training the ML Model (Optional)

To improve classification accuracy:

```bash
python train_model.py --input EmailAttachments --max-files 100
```

This will create a trained model at `document_converter/models/classifier_model.pkl`

## Troubleshooting

### "ModuleNotFoundError: No module named 'document_converter'"

**Solution:** Make sure you're running from the project root directory, not from inside `document_converter/` folder.

### "No module named 'paddleocr'" or other import errors

**Solution:** Install dependencies:
```bash
pip install -r document_converter/requirements.txt
```

### GUI doesn't open

**Solution:** Check if tkinter is available:
```bash
python -c "import tkinter; print('tkinter OK')"
```

If it fails, install tkinter (usually comes with Python, but on some Linux systems you may need to install it separately).

## Output

After processing, check the `Output/` folder:
- `Output/PurchaseOrders/` - Purchase Order Excel files
- `Output/StockSalesReports/` - Stock & Sales Excel files
- `Output/ProcessingLog.json` - Processing log with statistics


