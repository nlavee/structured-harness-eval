#!/usr/bin/env python3
"""Download and set up the FinWorkBench/Finch dataset from HuggingFace.

Downloads the Finch dataset from
https://huggingface.co/datasets/FinWorkBench/Finch
and sets up the expected data directory layout:

    data/Finch/
    ├── finch_tasks.json       # Task metadata (all 172 tasks)
    ├── source_files/          # Downloaded source files per task
    │   ├── 0/
    │   │   ├── 0_src_0.xlsx
    │   │   └── ...
    │   └── ...
    └── reference_files/       # Downloaded reference output files per task
        ├── 0/
        │   ├── 0_ref_0.xlsx
        │   └── ...
        └── ...

Usage:
    python scripts/download_finch.py                    # default: data/Finch
    python scripts/download_finch.py --dest /path/out   # custom location
    python scripts/download_finch.py --skip-files       # metadata only (fast)
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download and set up the FinWorkBench/Finch dataset from HuggingFace."
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/Finch",
        help="Destination directory (default: data/Finch)",
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Only download task metadata, skip source/reference file downloads",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit download to first N tasks (for testing)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library is required. Install with:")
        print("  pip install datasets")
        sys.exit(1)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset from HuggingFace ─────────────────────────────────
    tasks_path = dest / "finch_tasks.json"

    if tasks_path.exists():
        print(f"[1/4] Task metadata already exists: {tasks_path}")
        with open(tasks_path, encoding="utf-8") as f:
            tasks = json.load(f)
    else:
        print("[1/4] Loading Finch dataset from HuggingFace ...")
        ds = load_dataset("FinWorkBench/Finch", split="test")
        tasks = [dict(row) for row in ds]

        # Serialize — handle non-JSON-serializable types
        for task in tasks:
            for key, val in task.items():
                if hasattr(val, "tolist"):
                    task[key] = val.tolist()

        with open(tasks_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved {len(tasks)} tasks to {tasks_path}")

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
        print(f"  (Limited to first {args.max_tasks} tasks)")

    print(f"  Total tasks: {len(tasks)}")

    if args.skip_files:
        print("\n[2/4] Skipping source file downloads (--skip-files)")
        print("[3/4] Skipping reference file downloads (--skip-files)")
    else:
        # ── 2. Download source files ─────────────────────────────────────
        source_dir = dest / "source_files"
        print(f"\n[2/4] Downloading source files to {source_dir} ...")
        src_stats = _download_task_files(
            tasks,
            url_key="source_files_urls",
            name_key="source_files",
            output_dir=source_dir,
        )
        print(f"  Downloaded: {src_stats['downloaded']}, Skipped: {src_stats['skipped']}, "
              f"Failed: {src_stats['failed']}")

        # ── 3. Download reference files ──────────────────────────────────
        ref_dir = dest / "reference_files"
        print(f"\n[3/4] Downloading reference files to {ref_dir} ...")
        ref_stats = _download_task_files(
            tasks,
            url_key="reference_file_urls",
            name_key=None,  # derive names from reference_outputs.files
            output_dir=ref_dir,
            ref_outputs_key="reference_outputs",
        )
        print(f"  Downloaded: {ref_stats['downloaded']}, Skipped: {ref_stats['skipped']}, "
              f"Failed: {ref_stats['failed']}")

    # ── 4. Verify ────────────────────────────────────────────────────────
    print(f"\n[4/4] Verifying ...")
    _verify(dest, tasks, skip_files=args.skip_files)


def _parse_list_field(raw) -> list:
    """Parse a field that may be a list, JSON string, or None."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                return [str(x) for x in parsed] if isinstance(parsed, list) else [raw]
            except json.JSONDecodeError:
                pass
        return [x.strip() for x in raw.split(";") if x.strip()]
    return [str(raw)]


def _download_task_files(tasks, url_key, name_key, output_dir, ref_outputs_key=None):
    """Download files for each task into task-specific subdirectories."""
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for task in tasks:
        task_id = str(task.get("id", ""))
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        urls = _parse_list_field(task.get(url_key))

        # Get filenames
        if name_key:
            names = _parse_list_field(task.get(name_key))
        elif ref_outputs_key:
            ref_raw = task.get(ref_outputs_key)
            if isinstance(ref_raw, dict):
                names = _parse_list_field(ref_raw.get("files"))
            elif isinstance(ref_raw, str):
                try:
                    parsed = json.loads(ref_raw)
                    names = _parse_list_field(parsed.get("files")) if isinstance(parsed, dict) else []
                except (json.JSONDecodeError, AttributeError):
                    names = []
            else:
                names = []
        else:
            names = []

        for i, url in enumerate(urls):
            if not url or not url.startswith("http"):
                continue

            # Determine filename
            if i < len(names) and names[i]:
                fname = names[i]
            else:
                fname = os.path.basename(url.split("?")[0]) or f"file_{i}"

            local_path = task_dir / fname
            if local_path.exists():
                stats["skipped"] += 1
                continue

            try:
                urllib.request.urlretrieve(url, str(local_path))
                stats["downloaded"] += 1
            except Exception as e:
                print(f"  WARN: Failed to download {url}: {e}")
                stats["failed"] += 1

    return stats


def _verify(dest, tasks, skip_files=False):
    """Verify dataset integrity."""
    tasks_path = dest / "finch_tasks.json"

    if not tasks_path.exists():
        print("  FAIL: finch_tasks.json not found")
        return

    print(f"  Tasks metadata: {len(tasks)} tasks")

    # Count task types
    type_counts = {}
    biz_counts = {}
    for task in tasks:
        task_type = task.get("task_type", "Unknown")
        biz_type = task.get("business_type", "Unknown")
        if isinstance(task_type, str):
            for tt in task_type.split(","):
                tt = tt.strip()
                if tt:
                    type_counts[tt] = type_counts.get(tt, 0) + 1
        biz_counts[biz_type] = biz_counts.get(biz_type, 0) + 1

    print(f"\n  Task types ({len(type_counts)}):")
    for tt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {tt}: {count}")

    print(f"\n  Business types ({len(biz_counts)}):")
    for bt, count in sorted(biz_counts.items(), key=lambda x: -x[1]):
        print(f"    {bt}: {count}")

    if not skip_files:
        source_dir = dest / "source_files"
        ref_dir = dest / "reference_files"

        src_count = sum(1 for _ in source_dir.rglob("*") if _.is_file()) if source_dir.exists() else 0
        ref_count = sum(1 for _ in ref_dir.rglob("*") if _.is_file()) if ref_dir.exists() else 0
        print(f"\n  Source files on disk: {src_count}")
        print(f"  Reference files on disk: {ref_count}")

        # Check for tasks with missing source files
        missing_src = 0
        for task in tasks:
            task_id = str(task.get("id", ""))
            names = _parse_list_field(task.get("source_files"))
            for fname in names:
                if not (source_dir / task_id / fname).is_file():
                    missing_src += 1
        if missing_src:
            print(f"  WARNING: {missing_src} source files missing")
        else:
            print(f"  All source files verified")

    print(f"\nDataset ready at: {dest}")


if __name__ == "__main__":
    main()
