"""
Module entry point for running as: python -m document_converter
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Document Classification and Conversion System')
    parser.add_argument('--gui', '-g', action='store_true', help='Launch GUI interface')
    parser.add_argument('--input', '-i', type=str, default='EmailAttachments', help='Input folder path')
    
    args = parser.parse_args()
    
    if args.gui:
        from .main_gui import DocumentConverterGUI
        app = DocumentConverterGUI()
        app.run()
    else:
        from .main import main as cli_main
        sys.argv = ['main'] + (['--input', args.input] if args.input != 'EmailAttachments' else [])
        cli_main()

if __name__ == '__main__':
    main()

