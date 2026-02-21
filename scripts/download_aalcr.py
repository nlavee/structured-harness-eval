#!/usr/bin/env python3
"""Download and set up the AA-LCR dataset from HuggingFace.

Downloads the AA-LCR dataset files from
https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR
and unpacks them into the expected data directory layout:

    data/AA-LCR/
    ├── AA-LCR_Dataset.csv
    ├── README.md
    └── AA-LCR_extracted-text/
        └── lcr/
            ├── Academia/
            ├── Company_Documents/
            ├── Government_Consultations/
            ├── Industry_Reports/
            ├── Legal/
            ├── Marketing/
            └── Survey_Reports/

Usage:
    python scripts/download_aalcr.py          # default: data/AA-LCR
    python scripts/download_aalcr.py --dest /path/to/output
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download and set up the AA-LCR dataset from HuggingFace."
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/AA-LCR",
        help="Destination directory (default: data/AA-LCR)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    repo_id = "ArtificialAnalysis/AA-LCR"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # ── 1. Download CSV ─────────────────────────────────────────────────────
    csv_dest = dest / "AA-LCR_Dataset.csv"
    if csv_dest.exists():
        print(f"✓ CSV already exists: {csv_dest}")
    else:
        print("Downloading AA-LCR_Dataset.csv ...")
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="AA-LCR_Dataset.csv",
            repo_type="dataset",
        )
        shutil.copy2(csv_path, csv_dest)
        print(f"✓ Saved CSV to {csv_dest}")

    # ── 2. Download README ──────────────────────────────────────────────────
    readme_dest = dest / "README.md"
    if readme_dest.exists():
        print(f"✓ README already exists: {readme_dest}")
    else:
        print("Downloading README.md ...")
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
        )
        shutil.copy2(readme_path, readme_dest)
        print(f"✓ Saved README to {readme_dest}")

    # ── 3. Download and extract the document texts ──────────────────────────
    extracted_dir = dest / "AA-LCR_extracted-text"
    lcr_dir = extracted_dir / "lcr"

    if lcr_dir.exists() and any(lcr_dir.iterdir()):
        print(f"✓ Extracted texts already exist: {lcr_dir}")
    else:
        print("Downloading AA-LCR_extracted-text.zip ...")
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename="extracted_text/AA-LCR_extracted-text.zip",
            repo_type="dataset",
        )

        # Also save a copy of the zip for reference
        zip_dest = dest / "AA-LCR_extracted-text.zip"
        if not zip_dest.exists():
            shutil.copy2(zip_path, zip_dest)

        print("Extracting document texts ...")
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted_dir)
        print(f"✓ Extracted to {extracted_dir}")

    # ── 4. Fix known filename encoding issues ───────────────────────────────
    #
    # Some files extracted from the zip have mojibake filenames
    # (UTF-8 bytes interpreted as Windows-1252). Fix them so they match
    # the filenames referenced in the CSV.
    #
    fixes = _find_encoding_fixes(lcr_dir)
    if fixes:
        print(f"Fixing {len(fixes)} filename encoding issue(s) ...")
        for old_path, new_path in fixes:
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                print(f"  Renamed: {old_path.name} → {new_path.name}")
        print("✓ Filename fixes applied")
    else:
        print("✓ No filename encoding fixes needed")

    # ── 5. Clean up Zone.Identifier files (Windows metadata) ────────────────
    zone_files = list(lcr_dir.rglob("*:Zone.Identifier"))
    if zone_files:
        print(f"Removing {len(zone_files)} Zone.Identifier files ...")
        for zf in zone_files:
            zf.unlink()
        print("✓ Zone.Identifier files removed")

    # ── 6. Verify ───────────────────────────────────────────────────────────
    print()
    _verify(dest)


def _find_encoding_fixes(lcr_dir: Path):
    """Find files with mojibake filenames and return (old, new) path pairs."""
    import unicodedata

    # Known mojibake → correct Unicode mappings (UTF-8 bytes read as CP1252)
    mojibake_map = {
        "\u0393\u00c7\u00d6": "\u2019",   # ΓÇÖ → ' (right single quote)
        "\u0393\u00c7\u00f6": "\u2014",   # ΓÇö → — (em dash)
        "\u0393\u00c7\u00f4": "\u2013",   # ΓÇô → – (en dash)
        "s\u2560\u00ba": "\u015f",           # s╠º → ş (s with cedilla, deep mojibake)
    }

    fixes = []
    for p in lcr_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name

        # Check for mojibake patterns
        new_name = name
        for bad, good in mojibake_map.items():
            new_name = new_name.replace(bad, good)

        # Check for decomposed Unicode that should be precomposed (NFC)
        nfc_name = unicodedata.normalize("NFC", new_name)

        if nfc_name != name:
            fixes.append((p, p.with_name(nfc_name)))

    return fixes


def _verify(dest: Path):
    """Verify that all document files referenced in the CSV exist on disk."""
    import csv

    csv_path = dest / "AA-LCR_Dataset.csv"
    lcr_dir = dest / "AA-LCR_extracted-text" / "lcr"

    if not csv_path.exists():
        print("⚠ Cannot verify: CSV not found")
        return
    if not lcr_dir.exists():
        print("⚠ Cannot verify: extracted text directory not found")
        return

    total = 0
    missing = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames = row["data_source_filenames"].split(";")
            category = row["document_category"]
            set_id = row["document_set_id"]
            for fn in filenames:
                total += 1
                doc_path = lcr_dir / category / set_id / fn
                if not doc_path.is_file():
                    missing += 1
                    if missing <= 5:
                        print(f"  MISSING: {category}/{set_id}/{fn}")

    if missing == 0:
        print(f"✓ All {total} document files verified")
    else:
        if missing > 5:
            print(f"  ... and {missing - 5} more")
        print(f"⚠ {missing}/{total} document files missing")

    print(f"\nDataset ready at: {dest}")
    print(f"  CSV:       {csv_path}")
    print(f"  Documents: {lcr_dir}")


if __name__ == "__main__":
    main()
