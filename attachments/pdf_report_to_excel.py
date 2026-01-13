
# pdf_report_to_excel.py
import os
import re
import sys
import glob
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
import pandas as pd

# ---------- Helpers ----------
NUM_FIELDS = [
    "opening_qty", "opening_value",
    "receipt_qty", "receipt_value",
    "issue_qty", "issue_value",
    "closing_qty", "closing_value",
    "dump_qty", "oct_qty", "nexp_qty"
]

def is_number_token(tok: str) -> bool:
    """Return True if token is numeric or a placeholder '-'."""
    tok = tok.strip()
    if tok == "-" or tok == "—":
        return True
    # allow integers and floats
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", tok))

def to_number(tok: str) -> float:
    """Convert token to float; '-' or empty -> 0.0"""
    tok = tok.strip()
    if tok in {"-", "—", ""}:
        return 0.0
    try:
        return float(tok)
    except ValueError:
        # sometimes commas or stray chars
        tok = tok.replace(",", "")
        try:
            return float(tok)
        except ValueError:
            return 0.0

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        # normalize spaces
        parts.append(txt)
    return "\n".join(parts)

SECTION_HEADER_RE = re.compile(
    r"^[A-Z][A-Z0-9 \-\.\&/]+?\([A-Z0-9 \-\.\&/]+\)\s*$"
)

ITEMS_BLOCK_START_RE = re.compile(r"\bITEM DESCRIPTION\b", re.IGNORECASE)
TOTAL_LINE_RE = re.compile(r"^\s*TOTAL\b", re.IGNORECASE)

def split_into_sections(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Split the document into sections by header lines.
    Returns list of (section_header, section_lines)
    """
    sections = []
    current_header = "UNSPECIFIED"
    buffer = []

    for ln in lines:
        if SECTION_HEADER_RE.match(ln.strip()):
            # flush previous
            if buffer:
                sections.append((current_header, buffer))
                buffer = []
            current_header = ln.strip()
        else:
            buffer.append(ln)
    if buffer:
        sections.append((current_header, buffer))
    return sections

def normalize_section_name(header: str) -> str:
    """
    Extract the parenthetical part as the section name, else use header.
    e.g. 'ABBOTT HEALTHCARE (CRITICAL-CARDIOLOGY)' -> 'CRITICAL-CARDIOLOGY'
    """
    m = re.search(r"\(([^\)]+)\)", header)
    if m:
        return m.group(1).strip()
    return header.strip()

def parse_item_line(ln: str) -> Dict:
    """
    Parse one item line into a dict with name + numeric fields.
    Return {} if the line doesn't look like an item row.
    """
    raw = ln.strip()
    if not raw or TOTAL_LINE_RE.match(raw):
        return {}

    # Collapse multiple spaces to single; keep tokens
    tokens = re.split(r"\s+", raw)

    # Find index of first numeric token; everything before is item name
    first_num_idx = None
    for i, tok in enumerate(tokens):
        if is_number_token(tok):
            first_num_idx = i
            break

    if first_num_idx is None or first_num_idx == 0:
        # No numeric part -> likely header/footer or noise
        return {}

    name_tokens = tokens[:first_num_idx]
    num_tokens = tokens[first_num_idx:]

    # We expect up to 11 numeric tokens (some may be missing);
    # pad with '-' to reach 11
    if len(num_tokens) < len(NUM_FIELDS):
        num_tokens = num_tokens + ["-"] * (len(NUM_FIELDS) - len(num_tokens))
    else:
        # sometimes extra trailing tokens; keep first 11
        num_tokens = num_tokens[:len(NUM_FIELDS)]

    values = list(map(to_number, num_tokens))

    item = {"item_description": " ".join(name_tokens)}
    for key, val in zip(NUM_FIELDS, values):
        item[key] = val
    return item

def extract_items_from_section(section_lines: List[str]) -> List[Dict]:
    """Find the items block and parse rows until a TOTAL line."""
    items = []
    in_items_block = False
    for ln in section_lines:
        # Detect start of items block
        if not in_items_block and ITEMS_BLOCK_START_RE.search(ln):
            in_items_block = True
            continue

        if in_items_block:
            # Stop on TOTAL line; but skip capturing totals in items
            if TOTAL_LINE_RE.match(ln.strip()):
                # We could parse totals separately if needed
                # For now, ignore totals
                in_items_block = False
                continue

            row = parse_item_line(ln)
            if row:
                items.append(row)
    return items

def process_pdf(pdf_path: str) -> Dict[str, pd.DataFrame]:
    text = extract_text_from_pdf(pdf_path)
    # Split into lines; remove empty noise lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    sections = split_into_sections(lines)

    all_rows = []
    per_section_tables: Dict[str, List[Dict]] = {}

    for header, section_lines in sections:
        section_name = normalize_section_name(header)
        rows = extract_items_from_section(section_lines)
        if rows:
            # Attach section name to rows
            for r in rows:
                r["section"] = section_name
            all_rows.extend(rows)
            per_section_tables.setdefault(section_name, []).extend(rows)

    # Build DataFrames
    dfs: Dict[str, pd.DataFrame] = {}
    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        dfs["Summary"] = summary_df

    for sec, rows in per_section_tables.items():
        dfs[sec] = pd.DataFrame(rows)

    return dfs

def write_excel(dfs: Dict[str, pd.DataFrame], out_path: str):
    if not dfs:
        print(f"[WARN] No tables parsed; skipping write: {out_path}")
        return
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Summary first (if present), then other sections
        if "Summary" in dfs:
            dfs["Summary"].to_excel(writer, sheet_name="Summary", index=False)
        for sheet, df in dfs.items():
            if sheet == "Summary":
                continue
            # Excel sheet name limit 31 chars
            safe = sheet[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    print(f"[OK] Wrote: {out_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_report_to_excel.py <folder_or_file_path>")
        sys.exit(1)

    in_path = sys.argv[1]
    pdf_files = []
    if os.path.isdir(in_path):
        pdf_files = glob.glob(os.path.join(in_path, "*.pdf"))
    elif os.path.isfile(in_path) and in_path.lower().endswith(".pdf"):
        pdf_files = [in_path]
    else:
        print("Provide a PDF file path or a folder containing PDF files.")
        sys.exit(1)

    if not pdf_files:
        print("No PDF files found.")
        sys.exit(0)

    for pdf in pdf_files:
        dfs = process_pdf(pdf)
        out = os.path.splitext(pdf)[0] + ".xlsx"
        write_excel(dfs, out)

if __name__ == "__main__":
    main()
