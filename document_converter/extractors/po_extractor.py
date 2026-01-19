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
                    
                    page_tables = []
                    
                    # Strategy 1: Try text-based extraction first (better for preserving cell text integrity)
                    # Text strategy preserves cell boundaries better and reduces fragmentation
                    try:
                        page_tables = page.extract_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 3,
                            "join_tolerance": 3
                        })
                        if page_tables:
                            # Clean up cell values - join fragmented text in cells
                            page_tables = self._clean_table_cells(page_tables)
                            all_tables.extend(page_tables)
                            logger.debug(f"Found {len(page_tables)} tables using text strategy")
                    except Exception as e:
                        logger.debug(f"text strategy failed: {e}")
                    
                    # Strategy 2: Try lines_strict if text strategy didn't work or found poor results
                    if not page_tables or (page_tables and any(len(t) < 3 for t in page_tables)):
                        try:
                            page_tables_lines = page.extract_tables(table_settings={
                                "vertical_strategy": "lines_strict",
                                "horizontal_strategy": "lines_strict",
                                "explicit_vertical_lines": [],
                                "explicit_horizontal_lines": [],
                                "snap_tolerance": 3,
                                "join_tolerance": 3
                            })
                            if page_tables_lines:
                                # Clean up cell values
                                page_tables_lines = self._clean_table_cells(page_tables_lines)
                                # Use lines strategy if it found better tables
                                if not page_tables or len(page_tables_lines) > len(page_tables):
                                    all_tables.extend(page_tables_lines)
                                    logger.debug(f"Found {len(page_tables_lines)} tables using lines_strict strategy")
                                elif page_tables:
                                    # Keep original but also add lines-based if different
                                    for t in page_tables_lines:
                                        if t not in page_tables:
                                            all_tables.append(t)
                        except Exception as e:
                            logger.debug(f"lines_strict strategy failed: {e}")
                    
                    # Strategy 3: Try default extraction (if both failed)
                    if not page_tables:
                        try:
                            page_tables = page.extract_tables()
                            if page_tables:
                                # Clean up cell values
                                page_tables = self._clean_table_cells(page_tables)
                                all_tables.extend(page_tables)
                                logger.debug(f"Found {len(page_tables)} tables using default strategy")
                        except Exception as e:
                            logger.debug(f"default strategy failed: {e}")
                
                # Extract metadata from text
                result = self.extract_from_text(text)
                
                # Extract items - PRIMARY METHOD: tables (most accurate)
                # Text extraction is unreliable and picks up metadata - avoid it
                items = []
                if all_tables:
                    items = self._extract_items_from_tables(all_tables)
                    logger.info(f"Extracted {len(items)} items from {len(all_tables)} tables")
                
                # Only use text extraction as last resort if NO tables found at all
                # This ensures we only get data from actual tables, not from metadata/text
                # Text extraction is less reliable and may miss columns or hallucinate data
                if not items and not all_tables:
                    logger.warning(f"No tables found in PDF, attempting text extraction as last resort")
                    logger.warning(f"Text extraction may be less accurate - columns may be missing or misaligned")
                    items = self._extract_items(text)
                    logger.info(f"Extracted {len(items)} items from text (fallback)")
                elif not items and all_tables:
                    logger.warning(f"Tables found ({len(all_tables)}) but no items extracted - check table filtering logic")
                    logger.warning(f"Tables may have been filtered out due to strict validation")
                
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
        """
        Extract items from text - PO-SPECIFIC FORMAT
        
        PO format: Single-row headers like "Sr. Product Name Pack Qty Free Remarks"
        Each item line: "1 KENACORT 0.1% 7.5GM 1*7.5GM 200 0"
        
        This is DIFFERENT from Stock & Sales which has multi-row headers and 9 numeric values.
        """
        items = []
        lines = text.split('\n')
        
        in_items_block = False
        header_line_idx = None
        header_columns = []  # Store actual PO column names from header
        
        # First pass: find PO header structure (single-row, not multi-row like Stock & Sales)
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Detect PO header row - must have item number (Sr/SN/No) AND product AND qty
            has_item_number = bool(re.search(r'\b(sr\.?|sno\.?|sn\.?|no\.?)\b', line_stripped, re.IGNORECASE))
            has_product = bool(re.search(r'\b(product|item|description|name)\b', line_stripped, re.IGNORECASE))
            has_qty = bool(re.search(r'\b(qty\.?|quantity)\b', line_stripped, re.IGNORECASE))
            has_rate_or_amount = bool(re.search(r'\b(rate|price|amount|pack|packing|free)\b', line_stripped, re.IGNORECASE))
            
            # PO header should have: (Sr/SN/No) + Product + Qty + (Rate/Amount/Pack)
            if (has_item_number or has_product) and has_qty and has_rate_or_amount:
                # Additional check: should have few numbers (headers don't have many numbers)
                numbers_count = len(re.findall(r'\b\d+\b', line_stripped))
                if numbers_count < 5:  # PO headers typically have 0-2 numbers (like "Sr.", "No 1")
                    header_line_idx = i
                    in_items_block = True
                    # Parse header to get actual PO column names
                    # PO headers use variable spacing - use intelligent parsing
                    header_columns = self._parse_po_header_line(line_stripped)
                    logger.debug(f"Found PO header at line {i}: {header_columns}")
                    continue
            
            # Stop on separator lines or TOTAL
            if re.match(r'^[-=]+$', line_stripped) or re.match(r'^\s*(total|sub\s+total|net\s+amount)\b', line_stripped, re.IGNORECASE):
                if in_items_block and header_line_idx is not None:
                    # We're past headers, data should start next
                    continue
                else:
                    in_items_block = False
                    continue
            
            # Parse item line if we're in items block and past headers
            if in_items_block and header_line_idx is not None and i > header_line_idx:
                # Skip empty lines
                if not line_stripped or len(line_stripped) < 5:
                    continue
                
                # Skip lines that look like company info/metadata - STRICT FILTERING (like Stock & Sales)
                skip_patterns = [
                    'purchase order', 'po no', 'po number', 'order no', 'order number',
                    'date:', 'page no', 'page:', 'page 1', 'page 2',
                    'phone', 'phone no', 'phone:', 'mobile', 'mob:', 'tel:', 'telephone',
                    'email', 'email id', 'e-mail',
                    'gst', 'gst no', 'gst:', 'gstin', 'gst number',
                    'address', 'street', 'road', 'pincode', 'pin code', 'city', 'state',
                    'terms', 'terms and conditions', 'conditions',
                    'for,', 'dear sir', 'dear madam',
                    'vendor', 'supplier', 'company name',
                    'order information', 'order details',
                    'supply dt', 'supply date', 'delivery date',
                    'bank', 'account', 'ifsc',
                    'note:', 'notes:', 'remarks:', 'remark:',
                    'authorized', 'signature', 'stamp'
                ]
                if any(skip_word in line_stripped.lower() for skip_word in skip_patterns):
                    continue
                
                # Skip separator lines
                if re.match(r'^[-=_\s]+$', line_stripped):
                    continue
                
                # Skip TOTAL/SUMMARY rows (PO-specific)
                if re.search(r'^\s*(total|sub\s+total|net\s+amount|grand\s+total)\b', line_stripped, re.IGNORECASE):
                    continue
                
                # Skip section headers (all caps company names like "ABBOTT INDIA LTD (NEURO LIFE)")
                if re.match(r'^[A-Z\s\(\)]+$', line_stripped) and len(line_stripped) < 100 and not re.search(r'\d', line_stripped):
                    continue
                
                # Skip if line looks like a duplicate header (has header keywords but no item data)
                if re.search(r'\b(sr\.?|sno\.?|sn\.?|no\.?|product|item|description|name|qty|quantity|price|amount|rate|pack|free|remark)\b', 
                           line_stripped, re.IGNORECASE):
                    # Check if it has item numbers (like "1", "2", etc.) - if not, likely duplicate header
                    has_item_row_number = bool(re.search(r'^\s*\d+\s+[A-Z]', line_stripped))
                    if not has_item_row_number:
                        continue
                
                # Validate line has enough content (at least 2 parts)
                parts = re.split(r'\s{2,}', line_stripped)  # Try 2+ spaces first
                if len(parts) < 2:
                    parts = line_stripped.split()  # Fallback to single space
                
                if len(parts) < 2:
                    continue
                
                # Check if line looks like an item row (starts with number, has product-like text)
                # PO item rows typically start with item number (1, 2, 3, etc.)
                if not re.match(r'^\s*\d+', line_stripped):
                    # Might still be an item, but less likely - continue for now
                    pass
                
                # Parse the item line - use PO header columns if available
                item = self._parse_item_line(line_stripped, header_columns if header_columns else None)
                if item:
                    # Validate item has at least some data
                    has_description = any(k in item for k in ['description', 'Product Name', 'Product', 'DESCRIPTION'])
                    has_values = any(v for v in item.values() if v and str(v).strip() and str(v).strip() not in ['', '-'])
                    if has_description or has_values:
                        items.append(item)
        
        return items
    
    def _parse_po_header_line(self, header_line: str) -> List[str]:
        """
        Parse PO header line intelligently - handles multi-word column names
        
        Examples:
        - "Sr. Product Name Pack Qty Free Remarks" -> ["Sr.", "Product Name", "Pack", "Qty", "Free", "Remarks"]
        - "No Product Name Packing Qty Free Rate Amount" -> ["No", "Product Name", "Packing", "Qty", "Free", "Rate", "Amount"]
        - "SN. PRODUCT DESCRIPTION PACKINGNICK QTY FREE MRP RATE AMOUNT SCHM REMARK" -> ["SN.", "PRODUCT DESCRIPTION", "PACKINGNICK", "QTY", "FREE", "MRP", "RATE", "AMOUNT", "SCHM", "REMARK"]
        """
        if not header_line or len(header_line) < 5:
            return []
        
        header_line = header_line.strip()
        
        # Try to split by 2+ spaces first (more reliable if PDF has proper spacing)
        parts_2spaces = re.split(r'\s{2,}', header_line)
        if len(parts_2spaces) >= 3:
            # Clean parts and return
            headers = [re.sub(r'\(cid:\d+\)', '', p.strip()) for p in parts_2spaces if p.strip()]
            return headers
        
        # If 2+ space split didn't work well, try intelligent parsing
        # Split by single space, then merge adjacent words that form column names
        words = header_line.split()
        headers = []
        i = 0
        
        while i < len(words):
            word = words[i]
            found_multiword = False
            
            # Check for "Product Name" or "Product Description" (multi-word column names)
            if i + 1 < len(words) and word.lower() == 'product':
                if words[i + 1].lower() in ['name', 'description']:
                    # Combine "Product Name" or "Product Description"
                    combined = f"{word} {words[i + 1]}"
                    headers.append(combined)
                    i += 2
                    found_multiword = True
            
            if not found_multiword:
                # Single-word column name - use as-is
                # Clean word (remove periods, normalize case)
                clean_word = word
                # Preserve original formatting (case, punctuation)
                headers.append(clean_word)
                i += 1
        
        # Clean headers (remove cid markers)
        headers = [re.sub(r'\(cid:\d+\)', '', h.strip()) for h in headers if h.strip()]
        
        return headers
    
    def _parse_po_data_line_smart(self, line: str, header_columns: List[str]) -> List[str]:
        """
        Parse PO data line smartly - aligns parts with header columns
        
        Handles cases where:
        - Product names have multiple words (e.g., "KENACORT 0.1% 7.5GM")
        - Columns have blanks
        - Variable spacing
        
        Strategy: Use position hints from header structure and data patterns
        """
        if not line or not header_columns:
            return []
        
        line_stripped = line.strip()
        
        # Count expected numeric columns (usually Qty, Free, Rate, Amount, etc.)
        numeric_column_patterns = [r'\bqty', r'\bfree', r'\brate', r'\bamount', r'\bmrp', r'\bprice']
        num_numeric_cols = sum(1 for h in header_columns 
                              if any(re.search(p, h.lower()) for p in numeric_column_patterns))
        
        # Find numeric values in the line (these anchor the column positions)
        all_words = line_stripped.split()
        numeric_indices = []
        for i, word in enumerate(all_words):
            if re.match(r'^[\d,.\-]+$', word.replace(',', '').replace('-', '')):
                numeric_indices.append(i)
        
        # Strategy: Work backwards from numeric values
        # The last num_numeric_cols words should map to numeric columns
        # Everything before that is product name, pack, etc.
        
        parts = []
        
        if len(numeric_indices) >= num_numeric_cols:
            # We have enough numeric values - use them as anchors
            # Everything after the last numeric anchor goes to numeric columns
            # Everything before goes to descriptive columns
            
            # Find where numeric sequence starts
            if len(numeric_indices) >= num_numeric_cols:
                # Last num_numeric_cols words are numeric columns
                numeric_start_word_idx = numeric_indices[-num_numeric_cols] if numeric_indices else len(all_words) - num_numeric_cols
                
                # Descriptive columns are everything before numeric sequence
                desc_words = all_words[:numeric_start_word_idx]
                numeric_words = all_words[numeric_start_word_idx:]
                
                # Try to align descriptive words with descriptive columns
                # Usually: Sr (1 word) + Product Name (multiple words) + Pack (1-2 words)
                desc_col_count = len(header_columns) - num_numeric_cols
                
                if desc_col_count == 1:
                    # Single descriptive column (Product Name)
                    parts.append(" ".join(desc_words))
                elif desc_col_count == 2:
                    # Two descriptive columns (Sr + Product Name, or Product Name + Pack)
                    if desc_words and re.match(r'^\d+$', desc_words[0]):
                        # First word is Sr number
                        parts.append(desc_words[0])
                        parts.append(" ".join(desc_words[1:]) if len(desc_words) > 1 else "")
                    else:
                        # Try to split product name and pack
                        # Usually pack is last 1-2 words before numeric values
                        if len(desc_words) >= 2:
                            parts.append(" ".join(desc_words[:-1]))  # Product name
                            parts.append(desc_words[-1])  # Pack
                        else:
                            parts.append(" ".join(desc_words))
                            parts.append("")
                elif desc_col_count == 3:
                    # Three descriptive columns (Sr + Product Name + Pack)
                    if desc_words and re.match(r'^\d+$', desc_words[0]):
                        parts.append(desc_words[0])  # Sr
                        if len(desc_words) >= 2:
                            # Product name is middle words, pack is last word(s)
                            parts.append(" ".join(desc_words[1:-1]) if len(desc_words) > 2 else desc_words[1])  # Product name
                            parts.append(desc_words[-1] if len(desc_words) > 1 else "")  # Pack
                        else:
                            parts.append("")
                            parts.append("")
                    else:
                        # No Sr number, try to split Product Name and Pack
                        if len(desc_words) >= 2:
                            parts.append("")
                            parts.append(" ".join(desc_words[:-1]))
                            parts.append(desc_words[-1])
                        else:
                            parts.append("")
                            parts.append(" ".join(desc_words))
                            parts.append("")
                else:
                    # More descriptive columns - use first words as-is
                    parts.extend(desc_words[:desc_col_count])
                    while len(parts) < desc_col_count:
                        parts.append("")
                
                # Add numeric values
                parts.extend(numeric_words[:num_numeric_cols])
                
                # Ensure we have correct number of parts
                while len(parts) < len(header_columns):
                    parts.append("")
        else:
            # Fallback: simple split by 2+ spaces or single space
            parts_2spaces = re.split(r'\s{2,}', line_stripped)
            if len(parts_2spaces) >= len(header_columns) - 2:
                parts = [p.strip() for p in parts_2spaces if p.strip()]
            else:
                # Single space split - will need adjustment
                parts = [w.strip() for w in all_words if w.strip()]
            
            # Adjust parts count to match header columns
            if len(parts) < len(header_columns):
                # Too few parts - pad with empty strings
                parts.extend([""] * (len(header_columns) - len(parts)))
            elif len(parts) > len(header_columns):
                # Too many parts - combine descriptive columns
                # Usually first few columns are descriptive (Product Name might have multiple words)
                desc_col_count = len(header_columns) - num_numeric_cols
                if desc_col_count > 0:
                    # Combine first (len(parts) - num_numeric_cols - desc_col_count + 1) parts into product name
                    extra_parts = len(parts) - len(header_columns)
                    if extra_parts > 0 and desc_col_count > 0:
                        # Combine extra parts with first descriptive column (usually Product Name)
                        combined_desc = " ".join(parts[:extra_parts + 1])
                        parts = [combined_desc] + parts[extra_parts + 1:]
        
        # Ensure we have exactly len(header_columns) parts
        parts = parts[:len(header_columns)]
        while len(parts) < len(header_columns):
            parts.append("")
        
        return parts
    
    def _parse_item_line(self, line: str, header_columns: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Parse a single item line - PO-SPECIFIC FORMAT
        
        PO format examples:
        - "1 KENACORT 0.1% 7.5GM 1*7.5GM 200 0"
        - "1 DIGENE GEL MINT [200ML] 200ML 108 12 123.94 13385.52"
        - "1 VERTIN OD 24 TAB 15S ABBNLF 10 0 627.22 430.09 4300.90"
        
        If header_columns provided (e.g., ["Sr.", "Product Name", "Pack", "Qty", "Free", "Rate", "Amount"]),
        maps values directly to those column names by position.
        
        This is DIFFERENT from Stock & Sales which has fixed 9 numeric values.
        """
        if not line or len(line) < 5:
            return None
        
        # Skip separator lines
        if re.match(r'^[-=]+$', line):
            return None
        
        # Parse data line - need to align with header structure
        # PO format uses variable spacing, need intelligent parsing
        item = {}
        
        if header_columns and len(header_columns) > 0:
            # We have header columns - align data parts with header columns
            # Use intelligent parsing to handle variable spacing and multi-word product names
            
            # Try splitting by 2+ spaces first (preserves column structure better)
            parts_by_2spaces = re.split(r'\s{2,}', line.strip())
            parts_by_2spaces = [p.strip() for p in parts_by_2spaces if p.strip()]
            
            # If 2+ space split gives us close to header column count, use it
            if len(parts_by_2spaces) > 0 and abs(len(parts_by_2spaces) - len(header_columns)) <= 1:
                parts = parts_by_2spaces
            else:
                # Need smarter parsing - single space split with alignment
                # This handles cases where product names have multiple words
                parts = self._parse_po_data_line_smart(line, header_columns)
            
            # Map parts to header columns
            # Handle case where parts count doesn't match header count exactly
            for col_idx, header in enumerate(header_columns):
                header_clean = re.sub(r'\(cid:\d+\)', '', header.strip())
                
                if col_idx < len(parts):
                    value = parts[col_idx]
                    value_str = str(value).strip() if value else ""
                    
                    # Convert numeric values if they look numeric
                    if value_str and re.match(r'^[\d,.\-]+$', value_str.replace(',', '').replace('-', '')):
                        numeric_value = self._to_number_safe(value_str)
                        item[header_clean] = numeric_value
                    else:
                        # Keep as string (including empty strings for blanks)
                        item[header_clean] = value_str
                else:
                    # Column exists in header but not in data - preserve blank
                    item[header_clean] = ""
            
            # ALWAYS return item if it has keys (even if values are empty/blanks)
            # Blanks are valid data and should be preserved
            # Only skip if item is completely empty (no keys)
            if item and len(item) > 0:
                return item
        else:
            # Fallback: no header columns provided - try to infer structure
            # PO format typically: ITEM_NUMBER PRODUCT_NAME PACK QTY FREE RATE AMOUNT [REMARKS]
            
            # Split by 2+ spaces first, then single space if needed
            parts_2spaces = re.split(r'\s{2,}', line.strip())
            if len(parts_2spaces) >= 2:
                parts = [p.strip() for p in parts_2spaces if p.strip()]
            else:
                # Single space split
                parts = line.strip().split()
            
            if len(parts) < 2:
                return None
            
            # Find numeric parts from the end - be conservative, don't assume structure
            numeric_start_idx = None
            for i in range(len(parts) - 1, 0, -1):
                part = parts[i]
                # Only consider a part numeric if it's clearly numeric (not ambiguous)
                if re.match(r'^[\d,.\-]+$', part.replace(',', '').replace('-', '')) and len(part.replace(',', '').replace('-', '')) > 0:
                    numeric_start_idx = i
                else:
                    break
            
            # If we found numeric values, split into description and numeric parts
            # Be conservative - only extract what we're confident about
            if numeric_start_idx is not None and numeric_start_idx > 0:
                desc_parts = parts[:numeric_start_idx]
                numeric_parts = parts[numeric_start_idx:]
                
                # First part might be item number - but don't assume
                if desc_parts:
                    # Try to determine structure: usually first is item number, rest is product name
                    if len(desc_parts) > 1 and re.match(r'^\d+$', desc_parts[0]):
                        item["Sr."] = int(desc_parts[0])
                        item["Product Name"] = " ".join(desc_parts[1:])
                    else:
                        # Don't assume it's an item number if it doesn't look like one
                        item["Product Name"] = " ".join(desc_parts)
                
                # Map numeric parts to common PO columns - but only if we're confident
                # Don't hallucinate column names that might not exist
                if len(numeric_parts) >= 1:
                    item["Qty"] = self._to_number_safe(numeric_parts[0])
                if len(numeric_parts) >= 2:
                    item["Free"] = self._to_number_safe(numeric_parts[1])
                if len(numeric_parts) >= 3:
                    item["Rate"] = self._to_number_safe(numeric_parts[2])
                if len(numeric_parts) >= 4:
                    item["Amount"] = self._to_number_safe(numeric_parts[3])
                
                # Only return if we have meaningful data (product name or clear item number)
                if item.get("Product Name") or (item.get("Sr.") and item.get("Product Name")):
                    return item
            else:
                # No clear numeric pattern - be very conservative
                # Only extract if we can clearly identify structure
                if len(parts) >= 3:
                    # Check if first part looks like item number
                    first_is_number = re.match(r'^\d+$', parts[0])
                    # Check if last part looks numeric
                    last_is_numeric = re.match(r'^[\d,.\-]+$', parts[-1].replace(',', '').replace('-', ''))
                    
                    if first_is_number or last_is_numeric:
                        # Only extract if structure is clear
                        if first_is_number:
                            item["Sr."] = parts[0]
                            if len(parts) > 2:
                                item["Product Name"] = " ".join(parts[1:-1]) if last_is_numeric else " ".join(parts[1:])
                            else:
                                item["Product Name"] = parts[1] if len(parts) > 1 else ""
                        
                        if last_is_numeric and len(parts) >= 2:
                            item["Qty"] = self._to_number_safe(parts[-1])
                            if len(parts) >= 3:
                                item["Amount"] = self._to_number_safe(parts[-1])
                        
                        # Only return if we have product name
                        if item.get("Product Name"):
                            return item
        
        return None
    
    def _extract_from_combined_row(self, row: List, headers: List[str]) -> List[Dict[str, Any]]:
        """
        Extract items from a combined row where all items are in one row, newline-separated
        This handles special PDF layouts where items are combined in cells with newlines
        
        Example: Row has ['1\n2\n3', 'PRODUCT1\nPRODUCT2\nPRODUCT3', '10\n20\n30']
        """
        items = []
        
        if not row or not headers:
            return items
        
        # Split each column by newlines to get individual item values
        column_values = []
        for col_idx, header in enumerate(headers):
            cell_value = str(row[col_idx]) if col_idx < len(row) and row[col_idx] else ""
            # Split by newlines and clean
            values = [v.strip() for v in cell_value.split('\n') if v.strip()]
            column_values.append(values)
        
        # Determine number of items (max length of all columns)
        num_items = max(len(col_vals) for col_vals in column_values) if column_values else 0
        
        if num_items == 0:
            return items
        
        # Create items by combining values from each column
        for item_idx in range(num_items):
            item = {}
            for col_idx, header in enumerate(headers):
                header_clean = re.sub(r'\(cid:\d+\)', '', header.strip()) if header else ""
                # Get value for this item from this column
                if col_idx < len(column_values):
                    col_vals = column_values[col_idx]
                    if item_idx < len(col_vals):
                        value = col_vals[item_idx]
                    else:
                        value = ""  # Missing value for this column
                else:
                    value = ""
                
                # Convert numeric values if they look numeric
                if value:
                    value_str = str(value).strip()
                    if re.match(r'^[\d,.\-]+$', value_str.replace(',', '').replace('-', '')):
                        numeric_value = self._to_number_safe(value_str)
                        item[header_clean] = numeric_value
                    else:
                        item[header_clean] = value_str
                else:
                    item[header_clean] = ""
            
            # ALWAYS add item if it has keys (even if all values are empty/blanks)
            # This preserves row structure - blanks are valid data
            if item and len(item) > 0:
                items.append(item)
        
        return items
    
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
            
            # Validate this is likely an item table (not company info table) - PO-SPECIFIC
            # Check if table has reasonable structure for PO items
            # Strategy: Check for item table headers AND item data rows (not just metadata)
            
            # Check for PO item headers in first few rows - more lenient detection
            has_po_item_header = False
            header_row_idx = None
            for i in range(min(6, len(table))):
                row_text = " ".join([str(cell) for cell in table[i] if cell]).lower() if table[i] else ""
                
                # More lenient: need at least 2 of these indicators
                has_item_number = bool(re.search(r'\b(sr\.?|sno\.?|sn\.?|no\.?)\b', row_text, re.IGNORECASE))
                has_product = bool(re.search(r'\b(product|item|description|name)\b', row_text, re.IGNORECASE))
                has_qty = bool(re.search(r'\b(qty\.?|quantity)\b', row_text, re.IGNORECASE))
                has_rate_amount = bool(re.search(r'\b(rate|amount|price|pack|packing)\b', row_text, re.IGNORECASE))
                
                # Count how many indicators we have
                indicator_count = sum([has_item_number, has_product, has_qty, has_rate_amount])
                
                # If we have at least 2 indicators, it's likely a header row
                if indicator_count >= 2:
                    has_po_item_header = True
                    header_row_idx = i
                    break
            
            # Check if table has actual item data rows - more lenient
            has_item_data = False
            if header_row_idx is not None and header_row_idx + 1 < len(table):
                # Check rows after header for item data (should have some content)
                for row_idx in range(header_row_idx + 1, min(header_row_idx + 5, len(table))):
                    row = table[row_idx]
                    if not row:
                        continue
                    # Count non-empty cells
                    non_empty_cells = [cell for cell in row if cell and str(cell).strip()]
                    if len(non_empty_cells) >= 2:  # At least 2 non-empty cells indicates data row
                        has_item_data = True
                        break
            
            # If header detection was lenient but we didn't find data, try to use first row as header anyway
            # Some tables might not have clear headers but still have valid data
            if not has_po_item_header and len(table) >= 2:
                # Check if first row could be header (has some text but not too much data)
                first_row_text = " ".join([str(cell) for cell in table[0] if cell]).lower() if table[0] else ""
                first_row_numbers = len(re.findall(r'\b\d+\b', first_row_text))
                # If first row has few numbers (< 3) and multiple cells, it might be a header
                if first_row_numbers < 3 and len([c for c in table[0] if c]) >= 2:
                    has_po_item_header = True
                    header_row_idx = 0
                    # Check if second row has data
                    if len(table) > 1:
                        second_row_non_empty = [c for c in table[1] if c and str(c).strip()]
                        if len(second_row_non_empty) >= 2:
                            has_item_data = True
            
            # Only skip if we're confident it's NOT an item table
            # Be more conservative - if table has structure, try to extract it
            if not has_po_item_header and len(table) < 3:
                logger.debug(f"Table {table_idx} skipped: too small and no clear header")
                continue
            
            # Additional check: Item table should have at least 2-3 rows (header + data)
            if len(table) < 2:
                logger.debug(f"Table {table_idx} skipped: too small (less than 2 rows)")
                continue
            
            # Handle multi-row headers
            header_rows = self._detect_header_rows(table)
            logger.debug(f"Table {table_idx}: Detected {len(header_rows)} header rows")
            
            # Build column headers from all header rows
            headers = self._build_headers_from_rows(table, header_rows)
            
            if not headers:
                logger.warning(f"Table {table_idx}: No headers detected")
                continue
            
            # Validate headers look like item table headers - but be lenient
            # Some tables might have unusual header names but still be valid
            headers_text = " ".join([h.lower() for h in headers if h])
            item_keywords = ['product', 'item', 'description', 'qty', 'quantity', 'price', 'amount', 'rate', 'pack', 
                           'sr', 'sno', 'sn', 'no', 'name', 'free', 'mrp', 'remark']
            # Only skip if headers clearly don't match AND we didn't already detect it as an item table
            if not has_po_item_header and not any(kw in headers_text for kw in item_keywords):
                # Very strict check - only skip if we're sure it's not an item table
                if len(table) > 5:  # Large tables without item keywords are likely not item tables
                    logger.debug(f"Table {table_idx} skipped: headers don't look like item table headers")
                    continue
            
            logger.debug(f"Table {table_idx} headers ({len(headers)} columns): {headers[:5]}...")
            
            # Data starts after header rows
            data_start = max(header_rows) + 1 if header_rows else len(header_rows)
            
            # Check if items are in a single cell (newline-separated) - special PDF format
            # This happens when all items are combined in one cell with newlines
            combined_items_cell = None
            combined_row_idx = None
            
            # Look for rows with newline-separated data in first column (special format)
            for row_idx in range(data_start, min(data_start + 3, len(table))):
                row = table[row_idx]
                if not row or len(row) == 0:
                    continue
                # Check if first cell has multiple newlines (indicates combined items)
                first_cell = str(row[0]) if row[0] else ""
                if '\n' in first_cell and first_cell.count('\n') >= 2:
                    # Likely combined items in this cell
                    combined_items_cell = row
                    combined_row_idx = row_idx
                    break
            
            # Extract items preserving exact column structure
            if combined_items_cell:
                # Special format: all items in one row, newline-separated
                # Need to split each column by newlines and map together
                logger.debug(f"Table {table_idx}: Detected combined items format - splitting by newlines")
                items_from_combined = self._extract_from_combined_row(combined_items_cell, headers)
                items.extend(items_from_combined)
                continue
            
            # Normal format: each row is an item
            for row_idx, row in enumerate(table[data_start:], start=data_start):
                if not row or all(not cell or str(cell).strip() == "" for cell in row):
                    continue
                
                # Skip total/summary rows
                row_text = " ".join([str(cell) for cell in row if cell]).lower()
                if any(keyword in row_text for keyword in ['total', 'sum', 'grand total', 'subtotal', 'summary', 
                                                           'net amount', 'sub total']):
                    continue
                
                # Skip rows that look like company info or metadata - STRICT FILTERING (like Stock & Sales)
                skip_keywords = [
                    'purchase order', 'po no', 'po number', 'order no', 'order number',
                    'date:', 'page no', 'page:', 'page 1', 'page 2',
                    'phone', 'phone no', 'phone:', 'mobile', 'mob:', 'tel:', 'telephone',
                    'email', 'email id', 'e-mail',
                    'gst', 'gst no', 'gst:', 'gstin', 'gst number',
                    'address', 'street', 'road', 'pincode', 'pin code', 'city', 'state',
                    'terms', 'terms and conditions', 'conditions',
                    'for,', 'dear sir', 'dear madam',
                    'order information', 'order details',
                    'supply dt', 'supply date', 'delivery date',
                    'vendor', 'supplier', 'company name',
                    'bank', 'account', 'ifsc', 'swift',
                    'note:', 'notes:', 'remarks:', 'remark:',
                    'authorized', 'signature', 'stamp'
                ]
                if any(skip_word in row_text for skip_word in skip_keywords):
                    logger.debug(f"Row {row_idx} skipped: contains metadata keywords")
                    continue
                
                # Skip rows that are all caps company names (section headers) - like Stock & Sales
                row_text_upper_check = " ".join([str(cell) for cell in row if cell])
                if (row_text_upper_check.isupper() and 
                    len(row_text_upper_check) < 100 and 
                    not re.search(r'\d', row_text_upper_check) and
                    not re.search(r'\b(sr|sno|sn|no|product|item|qty|rate|amount|pack)\b', row_text, re.IGNORECASE)):
                    logger.debug(f"Row {row_idx} skipped: appears to be section header/company name")
                    continue
                
                # Skip rows with too few columns or mostly empty
                non_empty_cells = [c for c in row if c and str(c).strip()]
                if len(non_empty_cells) < 2:  # At least 2 non-empty cells needed
                    continue
                
                # Skip if row looks like a header (all text, no numbers) - might be a duplicate header
                if len(non_empty_cells) > 0:
                    row_has_numbers = any(re.search(r'\d', str(c)) for c in non_empty_cells)
                    if not row_has_numbers and len(non_empty_cells) <= len(headers):
                        # Likely a duplicate header row
                        continue
                
                # Create item - PRESERVE ALL COLUMNS EXACTLY AS THEY APPEAR
                item = {}
                
                # Map each column by its header name (preserve exact order)
                for col_idx, header in enumerate(headers):
                    # Use header as key (preserve original header text)
                    if header:
                        key = header
                    else:
                        # No header, use column index
                        key = f"Column_{col_idx}"
                    
                    # Get cell value
                    if col_idx < len(row):
                        cell_value = row[col_idx]
                    else:
                        cell_value = None
                    
                    # Handle None/empty values - preserve blanks correctly
                    if cell_value is None:
                        item[key] = ""  # Preserve blank
                    else:
                        cell_value_str = str(cell_value).strip()
                        
                        # Preserve data type and handle special cases
                        try:
                            # Check if cell is empty or blank
                            if not cell_value_str or cell_value_str == '':
                                item[key] = ""  # Preserve blank
                            elif cell_value_str in ['-', '—', 'N/A', 'n/a']:
                                item[key] = ""  # Treat dash/placeholder as blank
                            elif re.match(r'^[\d,.\-]+$', cell_value_str.replace(',', '').replace('-', '')):
                                # Try to convert to number (preserve as number for Excel formatting)
                                numeric_value = self._to_number_safe(cell_value_str)
                                item[key] = numeric_value
                            else:
                                # Keep as string (preserve original value exactly)
                                item[key] = cell_value_str
                        except Exception as e:
                            # On error, preserve the original value as string
                            logger.debug(f"Error processing cell [{row_idx}, {col_idx}]: {e}")
                            item[key] = cell_value_str if cell_value_str else ""
                
                # ALWAYS add item if it has any keys (even if all values are empty)
                # This preserves the row structure and blank rows are valid data
                if item and len(item) > 0:
                    items.append(item)
                else:
                    logger.debug(f"Row {row_idx} skipped: no item created")
        
        logger.info(f"Extracted {len(items)} items from {len(tables)} tables")
        return items
    
    def _detect_header_rows(self, table: List) -> List[int]:
        """
        Detect which rows are headers - PO-SPECIFIC
        
        PO headers are typically single-row with: Sr/SN/No + Product + Qty + Rate/Amount
        Examples: "Sr. Product Name Pack Qty Free Remarks"
                  "No Product Name Packing Qty Free Rate Amount"
                  "SN. PRODUCT DESCRIPTION PACKINGNICK QTY FREE MRP RATE AMOUNT SCHM REMARK"
        """
        if not table or len(table) < 2:
            return [0]
        
        # PO-specific header patterns
        # Must have item number indicator (Sr/SN/No) AND product/item AND qty/quantity
        po_header_keywords_required = [
            (['sr', 'sno', 'sn', 'no', 's\\.'], ['product', 'item', 'description'], ['qty', 'quantity', 'qty\\.']),
            (['sr\\.', 'sn\\.'], ['product', 'item'], ['qty', 'quantity']),
        ]
        
        # Optional keywords (rate, amount, pack, etc.)
        po_header_keywords_optional = ['rate', 'price', 'amount', 'pack', 'packing', 'free', 'mrp', 'remark', 'remarks', 'schm']
        
        # Look for the row that has PO item table headers (not company info)
        # Check first 6 rows to find the actual header row
        header_rows = []
        
        for i in range(min(6, len(table))):
            row = table[i]
            if not row:
                continue
            
            row_text = " ".join([str(cell) for cell in row if cell]).lower()
            
            # Skip rows that look like company info (addresses, phone, email, etc.)
            skip_patterns = ['phone', 'email', 'gst', 'address', 'street', 'road', 'pincode', 'pin:', 
                           'mob:', 'tel:', 'purchase order', 'po no', 'po number', 'date:', 'page',
                           'order information', 'order no', 'supply dt', 'for,', 'dear sir']
            if any(skip_word in row_text for skip_word in skip_patterns):
                continue
            
            # Check if this row matches PO header pattern
            # PO headers typically have: (Sr/SN/No) + (Product/Item) + (Qty) + (Rate/Amount/Pack)
            # CRITICAL: Must check for actual header words, not fragments or company names
            
            # Look for complete header words - be more specific
            has_item_number = bool(re.search(r'\b(sr\.?|sno\.?|sn\.?|no\.?|serial)\b', row_text, re.IGNORECASE))
            # IMPORTANT: "Name of Product" or "Product Name" is a key indicator
            has_product = bool(re.search(r'\b(name\s+of\s+product|product\s+name|product|item|description|name)\b', row_text, re.IGNORECASE))
            has_qty = bool(re.search(r'\b(qty\.?|quantity|qnty)\b', row_text, re.IGNORECASE))
            has_packing = bool(re.search(r'\b(packing|pack)\b', row_text, re.IGNORECASE))
            has_rate_or_amount = bool(re.search(r'\b(rate|price|amount)\b', row_text, re.IGNORECASE))
            has_remark = bool(re.search(r'\b(remark|remarks|free\s+remark|free)\b', row_text, re.IGNORECASE))
            
            # Count header keywords
            keyword_matches = sum(1 for kw in po_header_keywords_optional if kw in row_text)
            numbers_count = len(re.findall(r'\b\d+\b', row_text))
            
            # Skip rows that look like company names or addresses (all caps short words, or long phrases without header keywords)
            # Check if row looks like a company name row (common pattern: OTT HEALTH, CARE PVT LTD, etc.)
            row_cells = [str(c).strip() for c in row if c]
            is_likely_company_name = False
            if row_cells:
                # If row has very few cells (1-3) and all are short words or company-like, skip
                if len(row_cells) <= 3:
                    all_short = all(len(c) < 15 for c in row_cells)
                    all_upper = all(c.isupper() for c in row_cells if c)
                    if all_short and all_upper:
                        # Could be company name - check if it doesn't have header keywords
                        if not has_item_number and not has_qty and not has_product:
                            is_likely_company_name = True
            
            # Skip company name rows
            if is_likely_company_name:
                continue
            
            # PO header should have:
            # - Item number indicator (Sr/SN/No) OR product keyword (especially "Name of Product")
            # - Qty keyword OR Packing keyword
            # - Few numbers (< 3, headers don't have many numbers)
            
            is_po_header = False
            
            # STRONG MATCH: Has both item number and product keywords (most reliable)
            if (has_item_number and has_product) and numbers_count < 3:
                is_po_header = True
            
            # STRONG MATCH: Has "Name of Product" specifically (this is a key indicator)
            if re.search(r'\bname\s+of\s+product\b', row_text, re.IGNORECASE) and numbers_count < 3:
                is_po_header = True
            
            # MODERATE MATCH: Has Qty and (Product or Packing)
            if has_qty and (has_product or has_packing) and numbers_count < 3:
                is_po_header = True
            
            # MODERATE MATCH: Has item number and qty
            if has_item_number and has_qty and numbers_count < 3:
                is_po_header = True
            
            # FALLBACK: Multiple header keywords (3+) with few numbers
            if not is_po_header:
                total_po_keywords = (1 if has_item_number else 0) + (1 if has_product else 0) + \
                                   (1 if has_qty else 0) + (1 if has_packing else 0) + \
                                   (1 if has_rate_or_amount else 0) + (1 if has_remark else 0)
                if total_po_keywords >= 3 and numbers_count < 3:
                    is_po_header = True
            
            if is_po_header:
                header_rows.append(i)
                # PO headers are typically single-row, but check if next row looks like continuation
                # (unlikely for PO, but check anyway)
                if i + 1 < len(table):
                    next_row = table[i + 1]
                    next_row_text = " ".join([str(cell) for cell in next_row if cell]).lower()
                    # If next row has header keywords but no numbers, might be sub-header
                    next_keyword_matches = sum(1 for kw in po_header_keywords_optional if kw in next_row_text)
                    next_numbers = len(re.findall(r'\b\d+\b', next_row_text))
                    if next_keyword_matches >= 1 and next_numbers < 2:
                        # Might be a sub-header, but PO typically doesn't have multi-row headers
                        # Skip for now - PO headers are usually single-row
                        pass
                break
        
        # Fallback: if no header found, use first row
        if not header_rows:
            header_rows = [0]
        
        return header_rows
    
    def _build_headers_from_rows(self, table: List, header_rows: List[int]) -> List[str]:
        """
        Build column headers from multiple header rows (handles sub-columns)
        IMPROVED to clean and normalize headers
        """
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
                    # Clean header - remove newlines, extra spaces, common prefixes
                    part = re.sub(r'\s+', ' ', part)  # Normalize whitespace
                    part = re.sub(r'\n+', ' ', part)  # Remove newlines
                    if part and part.lower() not in ['', 'none', 'none']:
                        header_parts.append(part)
            
            if header_parts:
                # Combine header parts intelligently
                if len(header_parts) > 1:
                    # Join with space if parts are short, otherwise with " - "
                    if all(len(p) < 10 for p in header_parts):
                        header = " ".join(header_parts)
                    else:
                        header = " - ".join(header_parts)
                else:
                    header = header_parts[0]
                
                # Clean up header
                header = header.strip()
                headers.append(header)
            else:
                headers.append(f"Column_{col_idx}")
        
        return headers
    
    def _clean_table_cells(self, tables: List) -> List:
        """
        Clean table cells - join fragmented text that got split across cells
        This fixes cases where pdfplumber fragments text (e.g., "Phone" -> "Pho ne")
        However, we'll be conservative - only fix obvious issues and preserve structure
        
        Args:
            tables: List of tables (each table is a list of rows, each row is a list of cells)
            
        Returns:
            Cleaned tables with properly preserved cell values
        """
        cleaned_tables = []
        
        for table in tables:
            if not table:
                cleaned_tables.append(table)
                continue
            
            cleaned_table = []
            for row in table:
                if not row:
                    cleaned_table.append(row)
                    continue
                
                # Clean each cell - normalize whitespace but preserve structure
                cleaned_row = []
                for cell_value in row:
                    if cell_value is None:
                        cleaned_row.append("")
                    else:
                        cell_str = str(cell_value).strip()
                        # Normalize whitespace (multiple spaces to single space)
                        cell_str = re.sub(r'\s+', ' ', cell_str)
                        # Remove newlines but keep the text
                        cell_str = cell_str.replace('\n', ' ')
                        cleaned_row.append(cell_str)
                
                cleaned_table.append(cleaned_row)
            
            cleaned_tables.append(cleaned_table)
        
        return cleaned_tables
    
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
