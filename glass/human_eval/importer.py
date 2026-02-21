"""
Human evaluation importer.

Reads a completed annotation CSV back in, validates completeness,
and attaches human_label to the corresponding EvalResult objects.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from glass.judges.base import EvalResult
from glass.storage.run_store import RunStore

logger = logging.getLogger(__name__)


class ImportValidationError(Exception):
    """Raised when the imported labels CSV is incomplete or malformed."""

    pass


def import_human_labels(
    labels_path: Path,
    results: List[EvalResult],
    store: RunStore,
) -> Tuple[List[EvalResult], Dict[str, int]]:
    """
    Read human labels CSV and attach labels to EvalResult objects.

    Args:
        labels_path: Path to the completed annotation CSV.
        results: All EvalResult objects for this run (mutated in place).
        store: RunStore for persisting updated EvalResults.

    Returns:
        (updated_results, label_map) where label_map is {sample_id: label}.

    Raises:
        ImportValidationError: If required rows are missing labels or malformed.
    """
    if not Path(labels_path).exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Read CSV
    rows = []
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "human_label" not in (reader.fieldnames or []):
            raise ImportValidationError(f"Labels CSV missing 'human_label' column. Found: {reader.fieldnames}")
        for i, row in enumerate(reader, start=2):  # start=2 (row 1 is header)
            if not row.get("sample_id"):
                raise ImportValidationError(f"Row {i}: missing sample_id")
            label_raw = row.get("human_label", "").strip()
            if label_raw == "":
                raise ImportValidationError(
                    f"Row {i} (sample_id={row['sample_id']}): human_label is blank. "
                    "All exported rows must be labelled before import."
                )
            if label_raw not in {"0", "1"}:
                raise ImportValidationError(
                    f"Row {i} (sample_id={row['sample_id']}): human_label must be 0 or 1, got {label_raw!r}"
                )
            rows.append((row["sample_id"], row.get("system_name", ""), int(label_raw)))

    # Build label map {(system_name, sample_id): label}
    label_map: Dict[Tuple[str, str], int] = {(sname, sid): label for sid, sname, label in rows}

    # Attach labels and persist updated EvalResults
    updated = 0
    for result in results:
        key = (result.system_name, result.sample_id)
        if key in label_map:
            result.human_label = label_map[key]
            store.save_eval_result(result)
            updated += 1

    logger.info(
        "[GLASS] Imported %d human labels from %s (%d results updated)",
        len(rows),
        labels_path,
        updated,
    )
    print(f"[GLASS] Imported {len(rows)} human labels → {updated} EvalResults updated")

    # Return simple {sample_id: label} dict for downstream use
    simple_map = {sid: label for sid, _, label in rows}
    return results, simple_map
