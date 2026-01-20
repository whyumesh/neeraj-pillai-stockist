"""
Main entry point for document converter (CLI)
"""
import sys
from pathlib import Path
import argparse
import logging

from .document_processor import DocumentProcessor
from .config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Document Classification and Conversion System'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='attachments',
        help='Input folder path (default: attachments)'
    )
    parser.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Launch GUI interface'
    )
    
    args = parser.parse_args()
    
    if args.gui:
        # Launch GUI
        from .main_gui import DocumentConverterGUI
        app = DocumentConverterGUI()
        app.run()
    else:
        # CLI mode
        input_folder = Path(args.input)
        
        if not input_folder.exists():
            logger.error(f"Input folder does not exist: {input_folder}")
            sys.exit(1)
        
        logger.info(f"Processing folder: {input_folder}")
        
        config = ConfigLoader()
        processor = DocumentProcessor(config)
        
        result = processor.process_folder(input_folder)
        
        # Print statistics
        stats = result['statistics']
        logger.info("\n" + "="*50)
        logger.info("Processing Complete!")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Stock & Sales Reports: {stats['stock_sales_reports']}")
        logger.info(f"Other Documents: {stats['other_documents']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        logger.info("="*50)


if __name__ == '__main__':
    main()

