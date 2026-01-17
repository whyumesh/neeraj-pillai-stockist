"""
Simple launcher for GUI mode
Run this from the project root directory
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from document_converter.main_gui import DocumentConverterGUI

if __name__ == '__main__':
    app = DocumentConverterGUI()
    app.run()

