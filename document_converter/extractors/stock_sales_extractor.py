"""
Stock & Sales Report data extractor
"""
import re
from typing import Dict, List, Any, Optional
import logging
import pdfplumber

logger = logging.getLogger(__name__)


class StockSalesExtractor:
    """Extract data from Stock & Sales reports"""
    
    NUM_FIELDS = [
        "opening_qty", "opening_value",
        "receipt_qty", "receipt_value",
        "issue_qty", "issue_value",
        "closing_qty", "closing_value",
        "dump_qty", "oct_qty", "nexp_qty"
    ]
    
    def __init__(self):
        """Initialize Stock & Sales extractor"""
        pass
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract stock & sales data from text
        
        Args:
            text: Extracted text content
            
        Returns:
            Dictionary with extracted data
        """
        result = {
            "sections": self._extract_sections(text),
            "period": self._extract_period(text),
            "items": self._extract_items(text)
        }
        
        return result
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract from PDF using pdfplumber with multiple strategies
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted data
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                all_tables = []
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Strategy 1: Try strict line-based table extraction
                    page_tables = []
                    try:
                        page_tables = page.extract_tables(table_settings={
                            "vertical_strategy": "lines_strict",
                            "horizontal_strategy": "lines_strict",
                            "snap_tolerance": 3,
                            "join_tolerance": 3
                        })
                        if page_tables:
                            all_tables.extend(page_tables)
                            logger.debug(f"Found {len(page_tables)} tables using lines_strict strategy")
                    except Exception as e:
                        logger.debug(f"lines_strict strategy failed: {e}")
                    
                    # Strategy 2: Try text-based (if Strategy 1 failed)
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
                    
                    # Strategy 3: Try default (if both failed)
                    if not page_tables:
                        try:
                            page_tables = page.extract_tables()
                            if page_tables:
                                all_tables.extend(page_tables)
                                logger.debug(f"Found {len(page_tables)} tables using default strategy")
                        except Exception as e:
                            logger.debug(f"default strategy failed: {e}")
                
                # Extract sections and period from text
                result = {
                    "sections": self._extract_sections(text),
                    "period": self._extract_period(text),
                    "items": []
                }
                
                # Extract items - try tables first, then fallback to text
                items_from_table = []
                if all_tables:
                    # Validate table structure - if table has very few columns, it's likely malformed
                    valid_tables = []
                    for table in all_tables:
                        if table and len(table) > 2:
                            # Check if table has reasonable number of columns (at least 3)
                            max_cols = max(len(row) for row in table if row) if table else 0
                            if max_cols >= 3:
                                valid_tables.append(table)
                    
                    if valid_tables:
                        items_from_table = self._extract_items_from_tables(valid_tables, text)
                        if items_from_table:
                            logger.info(f"Extracted {len(items_from_table)} items from {len(valid_tables)} valid tables")
                
                # Always try text-based extraction (for space-delimited formats)
                items_from_text = self._extract_items(text)
                if items_from_text:
                    logger.info(f"Extracted {len(items_from_text)} items from text")
                
                # Use text-based extraction if it found more items, or if table extraction found very few
                # Text-based is more reliable for space-delimited formats
                if items_from_text and len(items_from_text) >= 3:
                    result["items"] = items_from_text
                elif items_from_table and len(items_from_table) > len(items_from_text):
                    result["items"] = items_from_table
                else:
                    result["items"] = items_from_text if items_from_text else items_from_table
                
                # Validate extraction
                self._validate_extraction(result, pdf_path, len(text))
                
                return result
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}", exc_info=True)
            return {
                "sections": [],
                "period": None,
                "items": []
            }
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract section headers"""
        sections = []
        
        # Pattern: COMPANY NAME (SECTION_NAME)
        section_pattern = r'^[A-Z][A-Z0-9 \-\.\&/]+?\(([A-Z0-9 \-\.\&/]+)\)\s*$'
        
        for line in text.split('\n'):
            match = re.match(section_pattern, line.strip())
            if match:
                section_name = match.group(1).strip()
                if section_name not in sections:
                    sections.append(section_name)
        
        return sections
    
    def _extract_period(self, text: str) -> Optional[str]:
        """Extract reporting period"""
        patterns = [
            r'period\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'for\s+the\s+period\s+(.+?)(?:\n|$)',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}',
            r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s+to\s+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _extract_items(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract items with stock quantities - IMPROVED PARSING
        
        Handles space-delimited format where data is in fixed positions
        """
        items = []
        lines = text.split('\n')
        
        in_items_block = False
        current_section = "UNSPECIFIED"
        header_line_idx = None
        sub_header_line_idx = None
        
        # First pass: find header structure
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Detect section header
            section_match = re.match(r'^[A-Z][A-Z0-9 \-\.\&/]+?\(([A-Z0-9 \-\.\&/]+)\)\s*$', line_stripped)
            if section_match:
                current_section = section_match.group(1).strip()
                continue
            
            # Detect main header row (ITEM DESCRIPTION OPENING RECEIPT...)
            if re.search(r'item\s+description', line_stripped, re.IGNORECASE) and \
               re.search(r'opening|receipt|issue|closing', line_stripped, re.IGNORECASE):
                header_line_idx = i
                in_items_block = True
                continue
            
            # Detect sub-header row (QTY. VALUE QTY. VALUE...)
            if header_line_idx is not None and i == header_line_idx + 1:
                if re.search(r'qty|value', line_stripped, re.IGNORECASE):
                    sub_header_line_idx = i
                    # Parse headers to understand structure
                    headers = self._parse_header_structure(lines[header_line_idx], lines[sub_header_line_idx] if sub_header_line_idx else None)
                    continue
            
            # Stop on separator lines or TOTAL
            if re.match(r'^[-=]+$', line_stripped) or re.match(r'^\s*total\b', line_stripped, re.IGNORECASE):
                if in_items_block and header_line_idx is not None:
                    # We're past headers, data should start next
                    continue
                else:
                    in_items_block = False
                    continue
            
            # Parse item line if we're in items block and past headers
            if in_items_block and header_line_idx is not None and i > (sub_header_line_idx or header_line_idx):
                # Skip empty lines
                if not line_stripped or len(line_stripped) < 10:
                    continue
                
                # Skip lines that look like headers, metadata, or separators
                if re.search(r'phone|gstin|stock.*sales|analysis|super.*market|enterprise|mittal|moga|punjab', line_stripped, re.IGNORECASE):
                    continue
                
                # Skip separator lines
                if re.match(r'^[-=_\s]+$', line_stripped):
                    continue
                
                # Skip TOTAL rows (all caps with "TOTAL" keyword)
                if re.search(r'^\s*TOTAL\b', line_stripped, re.IGNORECASE):
                    continue
                
                # Skip section headers - lines that are ALL CAPS, short (less than 50 chars), and have no numbers
                # Section headers like "ABBOTT GASTO", "ABBOTT HEALTHCARE" are usually short and all caps
                if (line_stripped.isupper() and 
                    len(line_stripped) < 50 and 
                    not re.search(r'\d', line_stripped) and
                    not re.search(r'[a-z]', line_stripped)):
                    # This might be a section header - try to extract section name
                    # But don't add it as an item
                    if not re.search(r'\b(TOTAL|SUMMARY|GRAND)\b', line_stripped, re.IGNORECASE):
                        # Update current section if it looks like a section name
                        potential_section = line_stripped.strip()
                        if len(potential_section) > 3:  # Reasonable section name length
                            current_section = potential_section
                    continue
                
                # Skip section headers (all caps with parentheses)
                if re.match(r'^[A-Z][A-Z0-9 \-\.\&/]+?\([A-Z0-9 \-\.\&/]+\)\s*$', line_stripped):
                    # This is a section header, update current section
                    section_match = re.match(r'^[A-Z][A-Z0-9 \-\.\&/]+?\(([A-Z0-9 \-\.\&/]+)\)\s*$', line_stripped)
                    if section_match:
                        current_section = section_match.group(1).strip()
                    continue
                
                # Validate line has enough numeric values (at least 5-6 to be a valid item row)
                # Count numeric tokens in the line
                numeric_count = len(re.findall(r'\b[\d,.\-]+\b', line_stripped))
                if numeric_count < 5:
                    # Not enough numeric values - likely not an item row
                    continue
                
                # Parse the item line
                item = self._parse_item_line_improved(line_stripped, lines[header_line_idx] if header_line_idx else None)
                if item and (item.get("Item Description") or item.get("item_description")):
                    # Additional validation: item description should not be too short or all caps single word
                    desc = item.get("Item Description") or item.get("item_description", "")
                    if desc:
                        # Skip if description is a single all-caps word (likely a section header)
                        desc_parts = desc.split()
                        if len(desc_parts) == 1 and desc.isupper() and len(desc) < 30:
                            continue
                        # Skip if description looks like a section header pattern
                        if re.match(r'^[A-Z\s]+$', desc) and len(desc) < 50 and not re.search(r'\d', desc):
                            continue
                    
                    item["section"] = current_section
                    items.append(item)
        
        return items
    
    def _parse_header_structure(self, main_header: str, sub_header: Optional[str] = None) -> List[str]:
        """Parse header structure to understand column layout"""
        # This helps understand the data structure
        # Main header: "ITEM DESCRIPTION OPENING RECEIPT ISSUE CLOSING DUMP"
        # Sub header: "QTY. VALUE QTY. VALUE QTY. VALUE QTY. VALUE QTY."
        return []
    
    def _parse_item_line_improved(self, line: str, header_line: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Improved item line parsing for space-delimited format
        
        Format: ITEM_NAME PCS QTY VALUE QTY VALUE QTY VALUE QTY VALUE QTY
        Example: "AQUAVIRON CAP 10 CAP PCS 7 1054.13 - 0.00 2 301.18 5 752.95 5"
        """
        if not line or len(line) < 10:
            return None
        
        # Skip separator lines
        if re.match(r'^[-=]+$', line):
            return None
        
        # Split by multiple spaces (preserve structure)
        # The format is: ITEM_NAME... PCS QTY VALUE QTY VALUE QTY VALUE QTY VALUE QTY
        parts = re.split(r'\s{2,}|\s+', line.strip())
        
        if len(parts) < 5:
            return None
        
        item = {}
        
        # Find where numeric data starts
        # Format: ITEM_NAME... PCS QTY VALUE QTY VALUE ...
        # Strategy: Look for unit (PCS, CS, etc.) that is IMMEDIATELY followed by numeric data
        # The unit should be the LAST word before the numeric sequence starts
        # IMPORTANT: Item descriptions can contain numbers (e.g., "10 CAP"), so we need to find
        # the unit word that comes RIGHT BEFORE the numeric data sequence
        
        numeric_start_idx = None
        
        # Find the unit word (PCS, CS, etc.) that marks the end of item description
        # Look from the END backwards to find the LAST unit word before numeric data
        # This handles cases like "AQUAVIRON CAP 10 CAP PCS 7 1054.13..." where "10" is in description
        unit_words = ['PCS', 'CS', 'BOX', 'STRIP', 'TAB', 'INJ', 'SYP', 'ML', 'GM', 'G', 'JAR', 'STR', 'STP', 'PIC']
        # Note: 'CAP' is excluded from unit_words because it can be part of item name (e.g., "10 CAP")
        
        # Strategy: Find the last occurrence of a unit word, then check if next tokens are numeric
        # Work backwards to find the rightmost unit word
        for i in range(len(parts) - 1, 0, -1):  # Start from end, go backwards
            part_upper = parts[i].upper()
            # Check if this is a unit word (but not CAP, as it's often in item names)
            if part_upper in unit_words:
                # Check if next part(s) form a numeric sequence
                if i + 1 < len(parts):
                    # Check next 3-4 parts to see if they form a numeric pattern
                    next_parts = parts[i+1:i+5] if i+5 <= len(parts) else parts[i+1:]
                    numeric_count = 0
                    for next_part in next_parts:
                        next_part = next_part.strip()
                        if (self._is_number_token(next_part) or 
                            next_part in ['-', '—'] or 
                            re.match(r'^[\d,.\-]+$', next_part.replace(',', '').replace('-', ''))):
                            numeric_count += 1
                        else:
                            break  # Stop if we hit non-numeric
                    
                    # If we found at least 2-3 consecutive numeric values, this is the start
                    if numeric_count >= 2:
                        numeric_start_idx = i + 1
                        break
        
        # If still not found, try with CAP as unit (but be more careful)
        if numeric_start_idx is None:
            for i in range(len(parts) - 1, 0, -1):
                if parts[i].upper() == 'CAP':
                    # For CAP, we need to be more strict - check if it's followed by PCS or number
                    if i + 1 < len(parts):
                        next_part = parts[i + 1].strip().upper()
                        # If next is PCS, then numeric data starts after PCS
                        if next_part == 'PCS' and i + 2 < len(parts):
                            next_next = parts[i + 2].strip()
                            if (self._is_number_token(next_next) or 
                                next_next in ['-', '—'] or 
                                re.match(r'^[\d,.\-]+$', next_next.replace(',', '').replace('-', ''))):
                                numeric_start_idx = i + 2
                                break
        
        # If not found, try to find pattern: word followed by number (but be careful with item names)
        # Look for a sequence of at least 5 consecutive numeric values
        if numeric_start_idx is None:
            # Count consecutive numeric values from the end backwards
            # We expect 9 numeric values, so find where that sequence starts
            consecutive_numeric = 0
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i].strip()
                if (self._is_number_token(part) or 
                    part in ['-', '—'] or 
                    re.match(r'^[\d,.\-]+$', part.replace(',', '').replace('-', ''))):
                    consecutive_numeric += 1
                    if consecutive_numeric >= 5:  # Found sequence of at least 5 numbers
                        # Work backwards to find the start
                        numeric_start_idx = i - (consecutive_numeric - 1)
                        break
                else:
                    consecutive_numeric = 0
        
        if numeric_start_idx is None or numeric_start_idx == 0:
            return None
        
        # Validate: numeric_start_idx should be at least 2 (item name should have at least 2 words)
        if numeric_start_idx < 2:
            return None
        
        # Item description is everything before numeric data (including unit if present)
        # But exclude the unit from description if it's right before numbers
        desc_end = numeric_start_idx
        if desc_end > 0 and parts[desc_end - 1].upper() in ['PCS', 'CS', 'BOX', 'STRIP', 'TAB', 'CAP', 'INJ', 'SYP']:
            desc_end = desc_end - 1  # Exclude unit from description
        
        item["item_description"] = " ".join(parts[:desc_end])
        
        # Extract numeric values - we expect exactly 9 values
        # Format: OPENING_QTY OPENING_VALUE RECEIPT_QTY RECEIPT_VALUE ISSUE_QTY ISSUE_VALUE CLOSING_QTY CLOSING_VALUE DUMP_QTY
        numeric_parts = parts[numeric_start_idx:]
        
        # Clean and extract exactly 9 numeric values
        cleaned_numeric = []
        for part in numeric_parts:
            # Check if it's a number, dash, or numeric string
            if part in ['-', '—', '']:
                cleaned_numeric.append('-')
            elif self._is_number_token(part):
                cleaned_numeric.append(part)
            elif re.match(r'^[\d,.\-]+$', part.replace(',', '').replace('-', '')):
                cleaned_numeric.append(part)
            
            # Stop at 9 values (expected count)
            if len(cleaned_numeric) >= 9:
                break
        
        # Ensure we have exactly 9 values (pad with '-' if needed)
        while len(cleaned_numeric) < 9:
            cleaned_numeric.append('-')
        cleaned_numeric = cleaned_numeric[:9]  # Take only first 9
        
        # Map to CORRECT column names - ONLY ONE SET, NO DUPLICATES
        # Use standard column names that match the PDF structure
        item_description = item.get("item_description", "")
        
        # Create item with ONLY the correct columns, no duplicates
        result_item = {
            "Item Description": item_description,
            "Opening Qty": self._to_number(cleaned_numeric[0]),
            "Opening Value": self._to_number(cleaned_numeric[1]),
            "Receipt Qty": self._to_number(cleaned_numeric[2]),
            "Receipt Value": self._to_number(cleaned_numeric[3]),
            "Issue Qty": self._to_number(cleaned_numeric[4]),
            "Issue Value": self._to_number(cleaned_numeric[5]),
            "Closing Qty": self._to_number(cleaned_numeric[6]),
            "Closing Value": self._to_number(cleaned_numeric[7]),
            "Dump Qty": self._to_number(cleaned_numeric[8])
        }
        
        return result_item
    
    def _parse_item_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse item line with quantities (legacy method - calls improved version)"""
        return self._parse_item_line_improved(line, None)
    
    def _is_number_token(self, token: str) -> bool:
        """Check if token is numeric"""
        token = token.strip()
        if token in {"-", "—", ""}:
            return True
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token))
    
    def _to_number(self, token: str) -> float:
        """Convert token to number"""
        token = token.strip()
        if token in {"-", "—", ""}:
            return 0.0
        try:
            return float(token.replace(",", ""))
        except ValueError:
            return 0.0
    
    def _extract_items_from_tables(self, tables: List, text: str = "") -> List[Dict[str, Any]]:
        """
        Extract items from tables - COMPLETELY TEMPLATE-FREE, PRESERVE EXACT STRUCTURE
        
        This method extracts tables exactly as they appear in the PDF, including:
        - Multi-row headers (e.g., "Opening Qty" with sub-columns)
        - All columns in exact order
        - No mapping to predefined fields
        
        Args:
            tables: List of extracted tables
            text: Full text for section detection
            
        Returns:
            List of items with preserved column structure
        """
        items = []
        current_section = "UNSPECIFIED"
        
        # Extract sections from text for later assignment
        sections_list = self._extract_sections(text)
        
        for table_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                logger.debug(f"Table {table_idx} skipped: too small or empty")
                continue
            
            # Handle multi-row headers - check if first 2-3 rows are headers
            header_rows = self._detect_header_rows(table)
            logger.debug(f"Table {table_idx}: Detected {len(header_rows)} header rows")
            
            # Build column headers from all header rows (handle sub-columns)
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
                
                # Skip section headers - rows that are all caps, short, and have few/no numbers
                row_text_upper = " ".join([str(cell) for cell in row if cell])
                if (row_text_upper.isupper() and 
                    len(row_text_upper) < 50 and 
                    not re.search(r'\d', row_text_upper) and
                    len([c for c in row if c and str(c).strip()]) < 5):
                    # This might be a section header - update section but don't add as item
                    potential_section = row_text_upper.strip()
                    if len(potential_section) > 3 and len(potential_section) < 50:
                        current_section = potential_section
                    continue
                
                # Skip rows with too few columns or mostly empty
                non_empty_cells = [c for c in row if c and str(c).strip()]
                if len(non_empty_cells) < 3:
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
                        
                        # Use header as key - NORMALIZE to standard column names
                        if header:
                            # Normalize header to standard column name
                            normalized_header = self._normalize_header_name(header)
                            
                            # ALWAYS add the value, even if empty or "-" (these are valid in stock reports)
                            # Try to preserve data type
                            try:
                                # Check if numeric (including "-" and "0")
                                if cell_value in ['-', '—', '']:
                                    item[normalized_header] = 0.0
                                elif re.match(r'^[\d,.\-]+$', cell_value.replace(',', '').replace('-', '')):
                                    # Try to convert to number
                                    numeric_value = self._to_number(cell_value)
                                    item[normalized_header] = numeric_value
                                else:
                                    item[normalized_header] = cell_value
                            except:
                                item[normalized_header] = cell_value
                        else:
                            # No header, use column index
                            item[f"Column_{col_idx}"] = cell_value
                
                # Add section
                item["section"] = current_section
                
                # Add item if it has item_description or any non-empty values
                # Don't skip items with zeros - they're valid data
                has_description = bool(item.get("Item Description") or item.get("item_description"))
                has_any_data = any(v for v in item.values() if v != "" and v is not None and str(v).strip() != "")
                
                if has_description or has_any_data:
                    items.append(item)
                else:
                    logger.debug(f"Row {row_idx} skipped: no data found. Item keys: {list(item.keys())}")
        
        logger.info(f"Extracted {len(items)} items from {len(tables)} tables with {len(headers) if headers else 0} columns")
        return items
    
    def _detect_header_rows(self, table: List) -> List[int]:
        """
        Detect which rows are headers (usually first 1-3 rows)
        
        Returns:
            List of row indices that are headers
        """
        if not table or len(table) < 2:
            return [0]
        
        header_rows = [0]  # First row is usually header
        
        # Check if second row looks like a header (has text but few numbers)
        if len(table) > 1:
            row1 = table[1]
            row1_text = " ".join([str(cell) for cell in row1 if cell]).lower()
            
            # If row has header-like keywords and few numbers, it's a header
            header_keywords = ['qty', 'quantity', 'value', 'amount', 'opening', 'receipt', 'issue', 'closing']
            has_header_keywords = any(kw in row1_text for kw in header_keywords)
            numbers_count = len(re.findall(r'\d+', row1_text))
            
            if has_header_keywords and numbers_count < 3:
                header_rows.append(1)
                
                # Check third row too
                if len(table) > 2:
                    row2 = table[2]
                    row2_text = " ".join([str(cell) for cell in row2 if cell]).lower()
                    numbers_count2 = len(re.findall(r'\d+', row2_text))
                    if numbers_count2 < 2:
                        header_rows.append(2)
        
        return header_rows
    
    def _build_headers_from_rows(self, table: List, header_rows: List[int]) -> List[str]:
        """
        Build column headers from multiple header rows (handles sub-columns)
        
        Example:
            Row 0: ["Item Description", "Opening", "", "Receipt", ""]
            Row 1: ["", "Qty", "Value", "Qty", "Value"]
            Result: ["Item Description", "Opening Qty", "Opening Value", 
                     "Receipt Qty", "Receipt Value"]
        
        Returns:
            List of normalized header names
        """
        if not header_rows:
            return []
        
        # Get header rows
        header_data = [table[i] for i in header_rows if i < len(table)]
        
        if not header_data:
            return []
        
        num_cols = max(len(row) for row in header_data) if header_data else 0
        if num_cols == 0:
            return []
        
        # Build headers column by column
        headers = []
        for col_idx in range(num_cols):
            header_parts = []
            
            # Collect header parts from each header row
            for header_row in header_data:
                if col_idx < len(header_row) and header_row[col_idx]:
                    part = str(header_row[col_idx]).strip()
                    if part and part.lower() not in ['', 'none']:
                        header_parts.append(part)
            
            # Combine header parts intelligently
            if header_parts:
                # If we have parent header (e.g., "Opening") and sub-header (e.g., "Qty")
                # Combine them properly: "Opening" + "Qty" = "Opening Qty"
                if len(header_parts) > 1:
                    # First part is usually the category (Opening, Receipt, etc.)
                    # Second part is usually the type (Qty, Value)
                    parent = header_parts[0]
                    sub = header_parts[1]
                    
                    # Normalize to standard format
                    parent_lower = parent.lower()
                    sub_lower = sub.lower()
                    
                    # Build standard header name
                    if 'item' in parent_lower or 'description' in parent_lower or 'product' in parent_lower:
                        header = "Item Description"
                    elif 'qty' in sub_lower or 'quantity' in sub_lower:
                        header = f"{parent} Qty"
                    elif 'value' in sub_lower or 'amount' in sub_lower:
                        header = f"{parent} Value"
                    else:
                        header = f"{parent} {sub}"
                else:
                    header = header_parts[0]
                
                # Normalize the header to standard names
                header = self._normalize_header_name(header)
                headers.append(header)
            else:
                # No header for this column
                headers.append(f"Column_{col_idx}")
        
        return headers
    
    def _normalize_header_name(self, header: str) -> str:
        """
        Normalize header name to standard column names
        
        Maps various header formats to standard names:
        - "Opening Qty - Qty" -> "Opening Qty"
        - "Opening Qty - Value" -> "Opening Value"
        - "opening_qty" -> "Opening Qty"
        - "OPENING_QTY" -> "Opening Qty"
        """
        if not header:
            return header
        
        header_lower = header.lower().strip()
        
        # Map to standard column names
        # Item Description
        if any(kw in header_lower for kw in ['item', 'description', 'product', 'name']):
            return "Item Description"
        
        # Opening Qty
        if 'opening' in header_lower and ('qty' in header_lower or 'quantity' in header_lower):
            return "Opening Qty"
        
        # Opening Value
        if 'opening' in header_lower and ('value' in header_lower or 'amount' in header_lower):
            return "Opening Value"
        
        # Receipt Qty
        if 'receipt' in header_lower and ('qty' in header_lower or 'quantity' in header_lower):
            return "Receipt Qty"
        
        # Receipt Value
        if 'receipt' in header_lower and ('value' in header_lower or 'amount' in header_lower):
            return "Receipt Value"
        
        # Issue Qty
        if 'issue' in header_lower and ('qty' in header_lower or 'quantity' in header_lower):
            return "Issue Qty"
        
        # Issue Value
        if 'issue' in header_lower and ('value' in header_lower or 'amount' in header_lower):
            return "Issue Value"
        
        # Closing Qty
        if 'closing' in header_lower and ('qty' in header_lower or 'quantity' in header_lower):
            return "Closing Qty"
        
        # Closing Value
        if 'closing' in header_lower and ('value' in header_lower or 'amount' in header_lower):
            return "Closing Value"
        
        # Dump Qty
        if 'dump' in header_lower and ('qty' in header_lower or 'quantity' in header_lower):
            return "Dump Qty"
        
        # If no match, return cleaned version
        return header.strip()
    
    def _find_column(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by keywords"""
        for i, header in enumerate(headers):
            if any(kw in header for kw in keywords):
                return i
        return None
    
    def _validate_extraction(self, result: Dict[str, Any], pdf_path: str, text_length: int):
        """Validate extraction results and log warnings"""
        items = result.get("items", [])
        sections = result.get("sections", [])
        
        if not items:
            logger.warning(f"No items extracted from {pdf_path}. Text length: {text_length}")
            if text_length < 100:
                logger.warning(f"Very short text extracted - PDF may be scanned or corrupted")
        else:
            # Check item quality
            items_with_desc = sum(1 for item in items if item.get("item_description"))
            avg_columns = sum(len(item.keys()) for item in items) / len(items) if items else 0
            
            logger.info(f"Extraction validation: {len(items)} items, "
                       f"{items_with_desc} with description, "
                       f"{len(sections)} sections, "
                       f"avg {avg_columns:.1f} columns per item")
            
            if items_with_desc < len(items) * 0.5:
                logger.warning(f"Less than 50% of items have descriptions - extraction may be incomplete")
            
            if avg_columns < 3:
                logger.warning(f"Very few columns per item ({avg_columns:.1f}) - data may be incomplete")

