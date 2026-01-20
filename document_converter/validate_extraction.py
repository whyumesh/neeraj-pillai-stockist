"""
Extraction Validation Script
Compares extracted item counts with manual counts and validates data completeness.
"""
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_converter.file_scanner import FileScanner
from document_converter.ocr_processor import OCRProcessor
from document_converter.extractors.stock_sales_extractor import StockSalesExtractor
from document_converter.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_extraction(pdf_path: Path) -> Dict[str, Any]:
    """
    Validate extraction for a single PDF file
    
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating extraction for: {pdf_path.name}")
    
    result = {
        'filename': pdf_path.name,
        'extraction_successful': False,
        'items_extracted': 0,
        'sections_found': 0,
        'items_per_section': {},
        'validation_issues': [],
        'text_length': 0,
        'has_item_description_header': False,
        'has_numeric_fields': False,
    }
    
    try:
        # Extract using StockSalesExtractor
        extractor = StockSalesExtractor()
        extracted_data = extractor.extract_from_pdf(str(pdf_path))
        
        if not extracted_data:
            result['validation_issues'].append('No data extracted')
            return result
        
        result['extraction_successful'] = True
        result['items_extracted'] = len(extracted_data.get('items', []))
        result['sections_found'] = len(extracted_data.get('sections', []))
        
        # Count items per section
        items = extracted_data.get('items', [])
        for item in items:
            section = item.get('section', 'UNSPECIFIED')
            result['items_per_section'][section] = result['items_per_section'].get(section, 0) + 1
        
        # Extract text for analysis
        try:
            ocr_processor = OCRProcessor(ConfigLoader())
            with open(pdf_path, 'rb') as f:
                import PyPDF2
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            result['text_length'] = len(text)
            
            # Check for key indicators in text
            text_lower = text.lower()
            result['has_item_description_header'] = 'item description' in text_lower
            result['has_numeric_fields'] = any(keyword in text_lower for keyword in [
                'opening qty', 'receipt qty', 'issue qty', 'closing qty'
            ])
            
            # Validation checks
            if result['items_extracted'] == 0:
                result['validation_issues'].append('No items extracted')
            
            if result['has_item_description_header'] and result['items_extracted'] == 0:
                result['validation_issues'].append('Header found but no items extracted - possible parsing issue')
            
            if result['has_numeric_fields'] and result['items_extracted'] < 5:
                result['validation_issues'].append(f'Very few items extracted ({result["items_extracted"]}) despite numeric fields present')
            
            # Check for items with missing descriptions
            items_without_desc = sum(1 for item in items if not item.get('Item Description') and not item.get('item_description'))
            if items_without_desc > 0:
                result['validation_issues'].append(f'{items_without_desc} items missing descriptions')
            
            # Check for items with all zero values (might be filtered incorrectly)
            items_all_zero = 0
            numeric_fields = ['Opening Qty', 'Opening Value', 'Receipt Qty', 'Receipt Value', 
                            'Issue Qty', 'Issue Value', 'Closing Qty', 'Closing Value', 'Dump Qty']
            for item in items:
                all_zero = True
                for field in numeric_fields:
                    value = item.get(field, 0)
                    if isinstance(value, (int, float)) and value != 0:
                        all_zero = False
                        break
                if all_zero:
                    items_all_zero += 1
            
            if items_all_zero > len(items) * 0.5:
                result['validation_issues'].append(f'High percentage of items with all zero values ({items_all_zero}/{len(items)})')
            
        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            result['validation_issues'].append(f'Text extraction failed: {str(e)}')
        
    except Exception as e:
        logger.error(f"Error validating extraction for {pdf_path.name}: {e}")
        result['validation_issues'].append(f'Extraction error: {str(e)}')
    
    return result


def validate_folder(input_folder: Path, max_files: int = None) -> Dict[str, Any]:
    """
    Validate extraction for all files in a folder
    
    Returns:
        Dictionary with validation results for all files
    """
    logger.info(f"Validating extraction for files in: {input_folder}")
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return {}
    
    scanner = FileScanner(input_folder)
    files = scanner.scan_files()
    
    # Filter for PDF files only
    pdf_files = [f for f in files if f['file_type'] == 'pdf']
    
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    logger.info(f"Found {len(pdf_files)} PDF files to validate")
    
    results = {
        'total_files': len(pdf_files),
        'validated_files': 0,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'total_items_extracted': 0,
        'files_with_issues': 0,
        'file_results': [],
        'summary_statistics': {
            'avg_items_per_file': 0,
            'avg_sections_per_file': 0,
            'files_with_zero_items': 0,
            'files_with_few_items': 0,  # Less than 5 items
        }
    }
    
    for i, file_info in enumerate(pdf_files):
        file_path = file_info['path']
        logger.info(f"Processing {i+1}/{len(pdf_files)}: {file_path.name}")
        
        file_result = validate_extraction(file_path)
        results['file_results'].append(file_result)
        results['validated_files'] += 1
        
        if file_result['extraction_successful']:
            results['successful_extractions'] += 1
            results['total_items_extracted'] += file_result['items_extracted']
            
            if file_result['items_extracted'] == 0:
                results['summary_statistics']['files_with_zero_items'] += 1
            elif file_result['items_extracted'] < 5:
                results['summary_statistics']['files_with_few_items'] += 1
        else:
            results['failed_extractions'] += 1
        
        if file_result['validation_issues']:
            results['files_with_issues'] += 1
    
    # Calculate summary statistics
    if results['successful_extractions'] > 0:
        results['summary_statistics']['avg_items_per_file'] = results['total_items_extracted'] / results['successful_extractions']
        total_sections = sum(r['sections_found'] for r in results['file_results'] if r['extraction_successful'])
        results['summary_statistics']['avg_sections_per_file'] = total_sections / results['successful_extractions']
    
    return results


def generate_report(results: Dict[str, Any], output_path: Path = None):
    """Generate extraction validation report"""
    if not results:
        logger.error("No results to report")
        return
    
    stats = results['summary_statistics']
    
    logger.info("\n" + "="*80)
    logger.info("EXTRACTION VALIDATION REPORT")
    logger.info("="*80)
    logger.info(f"\nTotal files validated: {results['total_files']}")
    logger.info(f"Successful extractions: {results['successful_extractions']}")
    logger.info(f"Failed extractions: {results['failed_extractions']}")
    logger.info(f"Files with issues: {results['files_with_issues']}")
    logger.info(f"\nTotal items extracted: {results['total_items_extracted']}")
    logger.info(f"Average items per file: {stats['avg_items_per_file']:.1f}")
    logger.info(f"Average sections per file: {stats['avg_sections_per_file']:.1f}")
    logger.info(f"\nFiles with zero items: {stats['files_with_zero_items']}")
    logger.info(f"Files with few items (<5): {stats['files_with_few_items']}")
    
    # Show files with issues
    files_with_issues = [r for r in results['file_results'] if r['validation_issues']]
    if files_with_issues:
        logger.warning(f"\n⚠️  Files with validation issues ({len(files_with_issues)}):")
        for file_result in files_with_issues[:20]:  # Show first 20
            logger.warning(f"  - {file_result['filename']}:")
            logger.warning(f"    Items extracted: {file_result['items_extracted']}")
            for issue in file_result['validation_issues']:
                logger.warning(f"    ⚠️  {issue}")
        if len(files_with_issues) > 20:
            logger.warning(f"  ... and {len(files_with_issues) - 20} more files with issues")
    
    # Quality assessment
    logger.info(f"\nQuality Assessment:")
    if stats['avg_items_per_file'] < 5:
        logger.warning(f"  ⚠️  Low average items per file ({stats['avg_items_per_file']:.1f}). Extraction may be incomplete.")
    if stats['files_with_zero_items'] > results['total_files'] * 0.1:
        logger.warning(f"  ⚠️  High percentage of files with zero items ({stats['files_with_zero_items']}/{results['total_files']}).")
    if results['files_with_issues'] > results['total_files'] * 0.2:
        logger.warning(f"  ⚠️  High percentage of files with issues ({results['files_with_issues']}/{results['total_files']}).")
    
    # Save detailed report to JSON
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nDetailed report saved to: {output_path}")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Extraction Quality')
    parser.add_argument('--input', type=str, default='attachments test',
                       help='Input folder path (default: attachments test)')
    parser.add_argument('--output', type=str, 
                       default='document_converter/models/extraction_validation_report.json',
                       help='Output report path')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to validate')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_path = Path(args.output)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Validate extraction
    results = validate_folder(input_folder, max_files=args.max_files)
    
    if not results:
        logger.error("Validation failed - no results generated")
        return
    
    # Generate report
    generate_report(results, output_path)
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

