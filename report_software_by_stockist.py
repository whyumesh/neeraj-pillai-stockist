#!/usr/bin/env python3
"""
Map each stockist code to its report-generator software from SmartStock template JSON.

Reads: data/stockist_templates/*.json (or --templates DIR)

Outputs:
  - software_by_stockist.csv   one row per stockist (deduped by code)
  - software_summary.csv       one row per (software, stockist_code) for pivoting
  - stdout: counts by software

Usage:
  python scripts/report_software_by_stockist.py
  python scripts/report_software_by_stockist.py --templates "D:/smartstock/data/stockist_templates"
  python scripts/report_software_by_stockist.py --out-dir ./reports
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_template(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def collect_stockists(templates_dir: Path) -> dict[str, dict]:
    """
    For each numeric stockist prefix, pick primary {code}.json when present;
    else first {code}_*.json by name. Record software + name + all variant paths.
    """
    by_code: dict[str, dict] = {}
    all_files = sorted(templates_dir.glob("*.json"))

    # group paths by stockist code (prefix before first _)
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in all_files:
        stem = p.stem
        prefix = stem.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        groups[prefix].append(p)

    for code in sorted(groups.keys(), key=int):
        paths = sorted(groups[code], key=lambda x: x.name)
        primary = templates_dir / f"{code}.json"
        chosen = primary if primary in paths else paths[0]
        data = _load_template(chosen)
        if not data:
            continue
        software = (data.get("software") or "").strip() or "(no software field)"
        name = (data.get("stockist_name") or "").strip()
        by_code[code] = {
            "stockist_code": code,
            "stockist_name": name,
            "software": software,
            "primary_template": str(chosen),
            "all_template_files": ";".join(str(p) for p in paths),
            "template_count": len(paths),
        }
    return by_code


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Classify stockist codes by report software (from template JSON)."
    )
    ap.add_argument(
        "--templates",
        type=Path,
        default=Path("data/stockist_templates"),
        help="Folder containing stockist template .json files (default: data/stockist_templates)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Where to write CSV files (default: current directory)",
    )
    args = ap.parse_args()

    td = args.templates
    if not td.is_dir():
        print(f"ERROR: templates folder not found: {td.resolve()}", file=sys.stderr)
        return 1

    by_code = collect_stockists(td)
    if not by_code:
        print("ERROR: no valid template JSON found.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bom = "\ufeff"

    # --- software_by_stockist.csv
    p1 = args.out_dir / "software_by_stockist.csv"
    with p1.open("w", encoding="utf-8", newline="") as f:
        f.write(bom)
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stockist_code",
                "stockist_name",
                "software",
                "template_count",
                "primary_template",
                "all_template_files",
            ],
        )
        w.writeheader()
        for row in sorted(by_code.values(), key=lambda r: int(r["stockist_code"])):
            w.writerow(row)
    print(f"Wrote {p1.resolve()} ({len(by_code)} stockists)")

    # --- software_summary.csv (software -> stockist rows; easy pivot / filter)
    p2 = args.out_dir / "software_summary.csv"
    by_software: dict[str, list[str]] = defaultdict(list)
    for r in by_code.values():
        by_software[r["software"]].append(r["stockist_code"])

    with p2.open("w", encoding="utf-8", newline="") as f:
        f.write(bom)
        w = csv.writer(f)
        w.writerow(["software", "stockist_code", "stockist_name"])
        for sw in sorted(by_software.keys(), key=str.lower):
            for r in sorted(
                (by_code[c] for c in by_software[sw]), key=lambda x: int(x["stockist_code"])
            ):
                w.writerow([sw, r["stockist_code"], r["stockist_name"]])
    print(f"Wrote {p2.resolve()}")

    # --- stdout summary
    print("\nStockists per report software:")
    for sw in sorted(by_software.keys(), key=str.lower):
        codes = by_software[sw]
        print(f"  {sw}: {len(codes)} stockist(s)")
    print(f"\nTotal unique software labels: {len(by_software)}")
    print(f"Total templated stockist codes: {len(by_code)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
