#!/usr/bin/env python3
"""
Scan EmailAttachments-style filenames and list which stockists sent files in each month.

Expected filename pattern (same as SmartStock ingestion):
  {stockist_code}_{YYYYMMDD}_{HHMMSS}_...

Month is taken from the YYYYMMDD segment (calendar month of the receive timestamp in the name).

Outputs:
  - stockists_by_month.csv         columns: year_month, stockist_code, file_count
  - stockists_by_month_summary.csv columns: year_month, unique_stockists, total_files
  - stdout: per-month totals

Usage:
  python scripts/stockists_by_month_from_attachments.py
  python scripts/stockists_by_month_from_attachments.py --email-attachments "D:/smartstock/EmailAttachments"
  python scripts/stockists_by_month_from_attachments.py --year 2026
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# SmartStock attachment naming: CODE_YYYYMMDD_HHMMSS_rest...
FNAME_RE = re.compile(r"^(\d+)_(\d{8})_\d{6}_", re.IGNORECASE)


def scan_folder(
    email_dir: Path,
    year_filter: int | None,
) -> tuple[dict[tuple[int, int], dict[str, int]], int, int]:
    """
    Returns:
      (month_key -> {stockist_code -> count}), matched_files, skipped_files
      month_key = (year, month)
    """
    counts: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matched = 0
    skipped = 0

    if not email_dir.is_dir():
        raise FileNotFoundError(str(email_dir))

    for p in email_dir.iterdir():
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if not m:
            skipped += 1
            continue
        code, ymd = m.group(1), m.group(2)
        if len(ymd) != 8 or not ymd.isdigit():
            skipped += 1
            continue
        y = int(ymd[:4])
        mo = int(ymd[4:6])
        if year_filter is not None and y != year_filter:
            continue
        counts[(y, mo)][code] += 1
        matched += 1

    return counts, matched, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description="List stockists (by code) that sent attachment files in each month."
    )
    ap.add_argument(
        "--email-attachments",
        type=Path,
        default=Path("EmailAttachments"),
        help="Folder with attachment files (default: ./EmailAttachments)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Where to write CSV files (default: current directory)",
    )
    ap.add_argument(
        "--year",
        type=int,
        default=None,
        help="Only include files whose YYYYMMDD year matches (e.g. 2026)",
    )
    args = ap.parse_args()

    try:
        counts, matched, skipped = scan_folder(args.email_attachments, args.year)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bom = "\ufeff"

    # Flatten to rows: year_month, stockist_code, file_count
    rows: list[tuple[str, str, int]] = []
    for (y, mo) in sorted(counts.keys()):
        ym = f"{y:04d}-{mo:02d}"
        for code in sorted(counts[(y, mo)].keys(), key=int):
            n = counts[(y, mo)][code]
            rows.append((ym, code, n))

    p1 = args.out_dir / "stockists_by_month.csv"
    with p1.open("w", encoding="utf-8", newline="") as f:
        f.write(bom)
        w = csv.writer(f)
        w.writerow(["year_month", "stockist_code", "file_count"])
        for ym, code, n in rows:
            w.writerow([ym, code, n])
    print(f"Wrote {p1.resolve()} ({len(rows)} stockist-month rows)")

    # Summary: one line per month
    p2 = args.out_dir / "stockists_by_month_summary.csv"
    with p2.open("w", encoding="utf-8", newline="") as f:
        f.write(bom)
        w = csv.writer(f)
        w.writerow(["year_month", "unique_stockists", "total_files"])
        for (y, mo) in sorted(counts.keys()):
            ym = f"{y:04d}-{mo:02d}"
            d = counts[(y, mo)]
            w.writerow([ym, len(d), sum(d.values())])
    print(f"Wrote {p2.resolve()}")

    print("\nPer month (unique stockists / total files):")
    for (y, mo) in sorted(counts.keys()):
        ym = f"{y:04d}-{mo:02d}"
        d = counts[(y, mo)]
        print(f"  {ym}: {len(d)} stockist(s), {sum(d.values())} file(s)")

    print(f"\nParsed filenames (matched pattern): {matched}")
    print(f"Skipped (no CODE_YYYYMMDD_HHMMSS_ pattern): {skipped}")
    if args.year is not None:
        print(f"Year filter: {args.year}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
