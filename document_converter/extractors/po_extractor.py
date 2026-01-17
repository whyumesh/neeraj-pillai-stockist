"""
Purchase Order data extractor
"""
import re
from typing import Dict, List, Any, Optional
import logging
import pdfplumber

logger = logging.getLogger(__name__)


class PurchaseOrderExtractor:
    """Extract data from Purchase Order documents"""
    
    def __init__(self):
        """Initialize PO extractor"""
        pass
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract PO data from text content
        
        Args:
            text: Extracted text content
            
        Returns:
            Dictionary with extracted PO data
        """
        result = {
            "po_number": self._extract_po_number(text),
            "date": self._extract_date(text),
            "vendor": self._extract_vendor(text),
            "items": self._extract_items(text),
            "total_amount": self._extract_total_amount(text)
        }
        
        return result
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract PO data from PDF using pdfplumber with multiple strategies
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted PO data
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract text from all pages
                text = ""
                all_tables = []
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Strategy 1: Try strict line-based table extraction
                    try:
                        page_tables = page.extract_tables(table_settings={
                            "vertical_strategy": "lines_strict",
                            "horizontal_strategy": "lines_strict",
                            "explicit_vertical_lines": [],
                            "explicit_horizontal_lines": [],
                            "snap_tolerance": 3,
                            "join_tolerance": 3
                        })
                        if page_tables:
                            all_tables.extend(page_tables)
                            logger.debug(f"Found {len(page_tables)} tables using lines_strict strategy")
                    except Exception as e:
                        logger.debug(f"lines_strict strategy failed: {e}")
                    
                    # Strategy 2: Try text-based table extraction (if Strategy 1 failed)
                    if not page_tables:
                        try:
                            page_tables = page.extract_tables(table_settings={
                                "vertical_strategy": "text",
                                "horizontal_strategy": "text"
                            })
                            if page_tables:
                                all_tables.extend(page_tables)
                                logger.debug(f"Found {len(page_tables)} tables using text strategy")
                        except Exception as e:
                            logger.debug(f"text strategy failed: {e}")
                    
                    # Strategy 3: Try default extraction (if both failed)
                    if not page_tables:
                        try:
                            page_tables = page.extract_tables()
                            if page_tables:
                                all_tables.extend(page_tables)
                                logger.debug(f"Found {len(page_tables)} tables using default strategy")
                        except Exception as e:
                            logger.debug(f"default strategy failed: {e}")
                
                # Extract metadata from text
                result = self.extract_from_text(text)
                
                # Extract items - try tables first, then fallback to text
                items = []
                if all_tables:
                    items = self._extract_items_from_tables(all_tables)
                    logger.info(f"Extracted {len(items)} items from tables")
                
                # Fallback to text-based extraction if table extraction failed or returned empty
                if not items:
                    items = self._extract_items(text)
                    logger.info(f"Extracted {len(items)} items from text (fallback)")
                
                result["items"] = items
                
                # Validate extraction
                self._validate_extraction(result, pdf_path, len(text))
                
                return result
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}", exc_info=True)
            # Return empty structure instead of failing completely
            return {
                "po_number": None,
                "date": None,
                "vendor": None,
                "items": [],
                "total_amount": None
            }
    
    def _extract_po_number(self, text: str) -> Optional[str]:
        """Extract Purchase Order number"""
        patterns = [
            r'purchase\s+order\s+(?:no|number|#)?\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'po\s+(?:no|number|#)?\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'order\s+number\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'po[_\-]?(\d+)',
            r'order[_\-]?(\d+)'
        ]
        
        text_lower = text[:2000].lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract order date"""
        # Common date patterns
        date_patterns = [
            r'date\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'order\s+date\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text[:1000])
            if matches:
                return matches[0]
        
        return None
    
    def _extract_vendor(self, text: str) -> Optional[str]:
        """Extract vendor/supplier name"""
        # Look for common vendor indicators
        patterns = [
            r'vendor\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'supplier\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'from\s*[:\-]?\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:500], re.IGNORECASE)
            if match:
                vendor = match.group(1).strip()
                if len(vendor) < 100:  # Reasonable length
                    return vendor
        
        return None
    
    def _extract_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract items from text"""
        items = []
        
        # Look for item tables
        lines = text.split('\n')
        item_started = False
        
        for line in lines:
            line = line.strip()
            
            # Detect start of items section
            if not item_started:
                if re.search(r'item|product|description|qty|quantity|price|amount', line, re.IGNORECASE):
                    item_started = True
                    continue
            
            if item_started and line:
                # Try to parse item line
                item = self._parse_item_line(line)
                if item:
                    items.append(item)
        
        return items
    
    def _parse_item_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single item line"""
        # Split by multiple spaces or tabs
        parts = re.split(r'\s{2,}|\t+', line)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) < 3:
            return None
        
        # Try to identify item name, quantity, price
        item = {}
        
        # First parts usually item name/description
        item["description"] = " ".join(parts[:-2])
        
        # Last parts usually numeric (qty, price)
        numeric_parts = []
        for part in parts[-2:]:
            # Try to extract numbers
            numbers = re.findall(r'[\d,]+\.?\d*', part.replace(',', ''))
            if numbers:
                numeric_parts.append(float(numbers[0].replace(',', '')))
        
        if len(numeric_parts) >= 1:
            item["quantity"] = numeric_parts[0]
        if len(numeric_parts) >= 2:
            item["price"] = numeric_parts[1]
            item["amount"] = item.get("quantity", 0) * item["price"]
        
        return item if item else None
    
    def _extract_items_from_tables(self, tables: List) -> List[Dict[str, Any]]:
        """
        Extract items from tables - COMPLETELY TEMPLATE-FREE, PRESERVE EXACT STRUCTURE
        
        This method extracts tables exactly as they appear in the PDF, preserving:
        - All columns in exact order
        - Original header names
        - No mapping to predefined fields
        """
        items = []
        
        for table_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                logger.debug(f"Table {table_idx} skipped: too small or empty")
                continue
            
            # Handle multi-row headers
            header_rows = self._detect_header_rows(table)
            logger.debug(f"Table {table_idx}: Detected {len(header_rows)} header rows")
            
            # Build column headers from all header rows
            headers = self._build_headers_from_rows(table, header_rows)
            
            if not headers:
                logger.warning(f"Table {table_idx}: No headers detected")
                continue
            
            logger.debug(f"Table {table_idx} headers ({len(headers)} columns): {headers[:5]}...")
            
            # Data starts after header rows
            data_start = len(header_rows)
            
            # Extract items preserving exact column structure
            for row_idx, row in enumerate(table[data_start:], start=data_start):
                if not row or all(not cell or str(cell).strip() == "" for cell in row):
                    continue
                
                # Skip total/summary rows
                row_text = " ".join([str(cell) for cell in row if cell]).lower()
                if any(keyword in row_text for keyword in ['total', 'sum', 'grand total', 'subtotal', 'summary']):
                    continue
                
                # Create item - PRESERVE ALL COLUMNS EXACTLY AS THEY APPEAR
                item = {}
                
                # Map each column by its header name (preserve exact order)
                for col_idx, header in enumerate(headers):
                    if col_idx < len(row):
                        cell_value = row[col_idx]
                        
                        # Handle None/empty values
                        if cell_value is None:
                            cell_value = ""
                        else:
                            cell_value = str(cell_value).strip()
                        
                        # Use header as key (preserve original header text)
                        if header:
                            # Keep original header name
                            key = header
                            
                            # ALWAYS add the value, even if empty (these are valid in POs)
                            # Try to preserve data type
                            try:
                                # Check if numeric (including "-" and "0")
                                if cell_value in ['-', '—', '']:
                                    item[key] = ""
                                elif re.match(r'^[\d,.\-]+$', cell_value.replace(',', '').replace('-', '')):
                                    # Try to convert to number
                                    numeric_value = self._to_number_safe(cell_value)
                                    item[key] = numeric_value
                                else:
                                    item[key] = cell_value
                            except:
                                item[key] = cell_value
                        else:
                            # No header, use column index
                            item[f"Column_{col_idx}"] = cell_value
                
                # Add item if it has any keys (even if values are empty)
                # Don't skip items with empty values - they're valid data
                if item and len(item) > 0:
                    items.append(item)
                else:
                    logger.debug(f"Row {row_idx} skipped: no data found. Item keys: {list(item.keys())}")
        
        logger.info(f"Extracted {len(items)} items from {len(tables)} tables with {len(headers) if headers else 0} columns")
        return items
    
    def _detect_header_rows(self, table: List) -> List[int]:
        """Detect which rows are headers"""
        if not table or len(table) < 2:
            return [0]
        
        header_rows = [0]  # First row is usually header
        
        # Check if second row looks like a header
        if len(table) > 1:
            row1 = table[1]
            row1_text = " ".join([str(cell) for cell in row1 if cell]).lower()
            
            header_keywords = ['qty', 'quantity', 'price', 'amount', 'rate', 'unit', 'description', 'item']
            has_header_keywords = any(kw in row1_text for kw in header_keywords)
            numbers_count = len(re.findall(r'\d+', row1_text))
            
            if has_header_keywords and numbers_count < 3:
                header_rows.append(1)
        
        return header_rows
    
    def _build_headers_from_rows(self, table: List, header_rows: List[int]) -> List[str]:
        """Build column headers from multiple header rows (handles sub-columns)"""
        if not header_rows:
            return []
        
        header_data = [table[i] for i in header_rows if i < len(table)]
        if not header_data:
            return []
        
        num_cols = max(len(row) for row in header_data) if header_data else 0
        if num_cols == 0:
            return []
        
        headers = []
        for col_idx in range(num_cols):
            header_parts = []
            
            for header_row in header_data:
                if col_idx < len(header_row) and header_row[col_idx]:
                    part = str(header_row[col_idx]).strip()
                    if part and part.lower() not in ['', 'none']:
                        header_parts.append(part)
            
            if header_parts:
                if len(header_parts) > 1:
                    header = " - ".join(header_parts)
                else:
                    header = header_parts[0]
                headers.append(header)
            else:
                headers.append(f"Column_{col_idx}")
        
        return headers
    
    def _to_number_safe(self, value_str: str) -> Any:
        """Safely convert string to number"""
        try:
            cleaned = value_str.replace(',', '').replace('₹', '').replace('$', '').strip()
            numbers = re.findall(r'[\d.]+', cleaned)
            if numbers:
                return float(numbers[0])
        except:
            pass
        return value_str  # Return original if conversion fails
    
    def _find_column(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by keywords (basic)"""
        for i, header in enumerate(headers):
            if any(kw in header for kw in keywords):
                return i
        return None
    
    def _find_column_flexible(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index with flexible matching (case-insensitive, partial matches)"""
        headers_lower = [h.lower() for h in headers]
        
        # Try exact match first
        for i, header in enumerate(headers_lower):
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower == header or kw_lower in header or header in kw_lower:
                    return i
        
        # Try partial word matching
        for i, header in enumerate(headers_lower):
            header_words = re.split(r'[\s\-_/]+', header)
            for kw in keywords:
                kw_lower = kw.lower()
                kw_words = re.split(r'[\s\-_/]+', kw_lower)
                # Check if any keyword word matches any header word
                if any(kw_word in header_words or header_word in kw_words 
                       for kw_word in kw_words for header_word in header_words):
                    return i
        
        return None
    
    def _validate_extraction(self, result: Dict[str, Any], pdf_path: str, text_length: int):
        """Validate extraction results and log warnings"""
        items = result.get("items", [])
        
        if not items:
            logger.warning(f"No items extracted from {pdf_path}. Text length: {text_length}")
            if text_length < 100:
                logger.warning(f"Very short text extracted - PDF may be scanned or corrupted")
        else:
            # Check item quality
            items_with_desc = sum(1 for item in items if item.get("description"))
            items_with_qty = sum(1 for item in items if item.get("quantity"))
            items_with_price = sum(1 for item in items if item.get("price"))
            
            logger.info(f"Extraction validation: {len(items)} items, "
                       f"{items_with_desc} with description, "
                       f"{items_with_qty} with quantity, "
                       f"{items_with_price} with price")
            
            if items_with_desc < len(items) * 0.5:
                logger.warning(f"Less than 50% of items have descriptions - extraction may be incomplete")
    
    def _extract_total_amount(self, text: str) -> Optional[float]:
        """Extract total amount"""
        patterns = [
            r'total\s+(?:amount|value)?\s*[:\-]?\s*[₹$]?\s*([\d,]+\.?\d*)',
            r'grand\s+total\s*[:\-]?\s*[₹$]?\s*([\d,]+\.?\d*)',
            r'total\s*[:\-]?\s*[₹$]?\s*([\d,]+\.?\d*)'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return float(match.group(1).replace(',', ''))
                except:
                    pass
        
        return None
