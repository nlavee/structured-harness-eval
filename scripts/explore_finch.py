#!/usr/bin/env python3
"""Explore the Finch dataset to identify chain-able task groups.

This is Step 1.2 of the Accounting Evaluation Pipeline plan.
It loads all Finch tasks, groups them by business domain and shared
source file patterns, scores each group's suitability for sequential
chaining, and outputs an annotated report.

Output:
    chains/candidates.json — ranked chain candidate groups with:
      - business_type, task count, workflow type coverage
      - task IDs with their types and source file patterns
      - chain_score (higher = more suitable for chaining)

Usage:
    python scripts/explore_finch.py                          # default data/Finch
    python scripts/explore_finch.py --data-dir /path/to/data
    python scripts/explore_finch.py --verbose                # print full report
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Workflow types in the canonical ordering from accounting.md
# Tasks should flow: data_entry → calculation → validation → reporting
WORKFLOW_ORDER = {
    "Data Entry/Import": 0,
    "Data Entry": 0,
    "Web Search": 0,
    "Cross-Sheet/File Retrieval": 1,
    "Calculation & Financial Modeling": 2,
    "Calculation": 2,
    "Validation/Review": 3,
    "Validation": 3,
    "Structuring/Formatting": 4,
    "Structuring": 4,
    "Summary/Visualization": 5,
    "Report & Analysis": 6,
    "Report": 6,
    "Trading & Risk Management": 2,
    "Accounts Payable/Receivable": 2,
}

# Minimum workflow type categories a group needs to be chain-worthy
# (data entry/calc + validation/reporting = at least 2 of the 4 macro-categories)
MACRO_CATEGORIES = {
    "data_entry": {"Data Entry/Import", "Data Entry", "Web Search"},
    "calculation": {"Calculation & Financial Modeling", "Calculation",
                    "Cross-Sheet/File Retrieval", "Trading & Risk Management",
                    "Accounts Payable/Receivable"},
    "validation": {"Validation/Review", "Validation"},
    "reporting": {"Report & Analysis", "Report", "Summary/Visualization",
                  "Structuring/Formatting", "Structuring"},
}


def main():
    parser = argparse.ArgumentParser(
        description="Explore Finch dataset for chain candidate identification."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/Finch",
        help="Path to Finch data directory (default: data/Finch)",
    )
    parser.add_argument(
        "--output", type=str, default="chains/candidates.json",
        help="Output path for chain candidates (default: chains/candidates.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed analysis to stdout",
    )
    args = parser.parse_args()

    # Load tasks
    tasks_path = os.path.join(args.data_dir, "finch_tasks.json")
    if not os.path.isfile(tasks_path):
        print(f"ERROR: finch_tasks.json not found at {tasks_path}")
        print(f"  Run: python scripts/download_finch.py --skip-files")
        sys.exit(1)

    with open(tasks_path, encoding="utf-8") as f:
        raw_tasks = json.load(f)

    print(f"Loaded {len(raw_tasks)} Finch tasks\n")

    # ── 1. Parse tasks ───────────────────────────────────────────────────
    tasks = []
    for raw in raw_tasks:
        task = _parse_task(raw)
        tasks.append(task)

    # ── 2. Group by business_type ────────────────────────────────────────
    biz_groups = defaultdict(list)
    for task in tasks:
        biz_groups[task["business_type"]].append(task)

    print("=== Business Type Groups ===\n")
    for biz, group in sorted(biz_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {biz}: {len(group)} tasks")
        if args.verbose:
            for t in group:
                print(f"    [{t['id']}] types={t['task_types']}  files={len(t['source_files'])}")

    # ── 3. Within each business type, find sub-groups sharing source patterns ─
    print("\n=== Sub-groups by Shared Source File Patterns ===\n")
    all_subgroups = []

    for biz, group in sorted(biz_groups.items(), key=lambda x: -len(x[1])):
        subgroups = _find_source_subgroups(group)
        for sg in subgroups:
            sg["business_type"] = biz
            all_subgroups.append(sg)

        if args.verbose:
            for sg in subgroups:
                print(f"  [{biz}] Subgroup (shared_pattern={sg['shared_pattern']}, "
                      f"tasks={len(sg['tasks'])})")
                for tid in sg["task_ids"]:
                    t = next(t for t in group if t["id"] == tid)
                    print(f"    [{tid}] {', '.join(t['task_types'][:3])}")

    # ── 4. Score each subgroup for chain suitability ─────────────────────
    print("\n=== Chain Candidates (ranked by score) ===\n")
    candidates = []

    for sg in all_subgroups:
        score_info = _score_chain_candidate(sg)
        sg.update(score_info)
        candidates.append(sg)

    candidates.sort(key=lambda x: -x["chain_score"])

    for i, c in enumerate(candidates[:20]):
        macro_str = ", ".join(c["macro_categories_covered"])
        print(f"  #{i+1} [{c['business_type']}] score={c['chain_score']:.2f}  "
              f"tasks={len(c['task_ids'])}  "
              f"macro_cats=[{macro_str}]  "
              f"pattern={c['shared_pattern']}")
        if args.verbose:
            for tid in c["task_ids"]:
                task = next((t for t in tasks if t["id"] == tid), None)
                if task:
                    print(f"      [{tid}] {', '.join(task['task_types'][:3])}  "
                          f"src_files={task['source_files'][:2]}")

    # ── 5. Build the full report ─────────────────────────────────────────
    print(f"\n=== Summary ===\n")
    print(f"  Total subgroups found: {len(candidates)}")
    viable = [c for c in candidates if c["chain_score"] >= 3.0]
    print(f"  Viable candidates (score >= 3.0): {len(viable)}")
    good = [c for c in candidates if c["chain_score"] >= 5.0]
    print(f"  Strong candidates (score >= 5.0): {len(good)}")

    # Overall dataset statistics
    all_types = Counter()
    for t in tasks:
        for tt in t["task_types"]:
            all_types[tt] += 1

    print(f"\n  Task type distribution:")
    for tt, count in all_types.most_common():
        print(f"    {tt}: {count}")

    # ── 6. Write output ──────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build output structure
    output = {
        "metadata": {
            "total_tasks": len(tasks),
            "total_subgroups": len(candidates),
            "viable_candidates": len(viable),
            "strong_candidates": len(good),
            "task_type_distribution": dict(all_types.most_common()),
            "business_type_counts": {k: len(v) for k, v in sorted(biz_groups.items(), key=lambda x: -len(x[1]))},
        },
        "candidates": [
            {
                "rank": i + 1,
                "business_type": c["business_type"],
                "shared_pattern": c["shared_pattern"],
                "chain_score": round(c["chain_score"], 2),
                "task_count": len(c["task_ids"]),
                "task_ids": c["task_ids"],
                "macro_categories_covered": c["macro_categories_covered"],
                "workflow_type_coverage": c["workflow_type_coverage"],
                "tasks_detail": [
                    {
                        "id": tid,
                        "task_types": next((t["task_types"] for t in tasks if t["id"] == tid), []),
                        "source_files": next((t["source_files"] for t in tasks if t["id"] == tid), []),
                        "instruction_preview": next(
                            (t["instruction"][:200] for t in tasks if t["id"] == tid), ""
                        ),
                    }
                    for tid in c["task_ids"]
                ],
                "notes": "",  # placeholder for manual annotation
            }
            for i, c in enumerate(candidates)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nChain candidates written to: {output_path}")
    print(f"Next step: manually review candidates and annotate 'notes' field")
    print(f"           for the top 10 groups suitable for sequential chaining.")


def _parse_task(raw: dict) -> dict:
    """Parse a raw Finch task into a simplified dict for analysis."""
    task_id = str(raw.get("id", ""))

    # Parse task types
    task_type_raw = raw.get("task_type", "")
    if isinstance(task_type_raw, str):
        task_types = [t.strip() for t in task_type_raw.split(",") if t.strip()]
    elif isinstance(task_type_raw, list):
        task_types = [str(t) for t in task_type_raw]
    else:
        task_types = []

    # Parse source files
    source_raw = raw.get("source_files")
    if isinstance(source_raw, list):
        source_files = [str(x) for x in source_raw]
    elif isinstance(source_raw, str):
        source_files = _parse_json_or_split(source_raw)
    else:
        source_files = []

    return {
        "id": task_id,
        "instruction": raw.get("instruction_en", ""),
        "task_types": task_types,
        "business_type": raw.get("business_type", "Unknown"),
        "task_constraints": raw.get("task_constraints", ""),
        "source_files": source_files,
        "source_file_stem": _extract_file_stem_pattern(source_files),
    }


def _parse_json_or_split(raw: str) -> list:
    """Parse a string that might be JSON array or semicolon-separated."""
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


def _extract_file_stem_pattern(source_files: list) -> str:
    """Extract a common stem pattern from source file names.

    e.g. ["42_src_0.xlsx", "42_src_1.xlsx"] → "42_src"
    This helps identify tasks that share the same underlying data.
    """
    if not source_files:
        return ""

    stems = []
    for f in source_files:
        name = os.path.basename(f)
        # Strip extension and trailing _N index
        stem = os.path.splitext(name)[0]
        stem = re.sub(r"_\d+$", "", stem)
        stems.append(stem)

    # Find the longest common prefix
    if len(stems) == 1:
        return stems[0]

    prefix = os.path.commonprefix(stems)
    return prefix.rstrip("_") if prefix else stems[0]


def _find_source_subgroups(group: list) -> list:
    """Within a business_type group, find sub-groups sharing source file patterns.

    Two tasks are considered related if:
    1. They share the same file stem pattern, OR
    2. They share any source file name (exact match)
    """
    # Group by file stem pattern
    stem_groups = defaultdict(list)
    for task in group:
        stem = task["source_file_stem"]
        if stem:
            stem_groups[stem].append(task)

    # Also detect shared exact filenames across tasks
    file_to_tasks = defaultdict(set)
    for task in group:
        for f in task["source_files"]:
            file_to_tasks[os.path.basename(f)].add(task["id"])

    # Merge overlapping groups (union-find style)
    task_to_group = {}
    group_counter = 0

    for stem, tasks_in_stem in stem_groups.items():
        # Check if any task in this stem group already belongs to a group
        existing_groups = {task_to_group[t["id"]] for t in tasks_in_stem if t["id"] in task_to_group}

        if existing_groups:
            target = min(existing_groups)
        else:
            target = group_counter
            group_counter += 1

        for t in tasks_in_stem:
            task_to_group[t["id"]] = target

        # Merge any other groups into target
        for eg in existing_groups:
            if eg != target:
                for tid, gid in task_to_group.items():
                    if gid == eg:
                        task_to_group[tid] = target

    # Also group tasks with no stem pattern that share exact files
    for fname, task_ids in file_to_tasks.items():
        if len(task_ids) > 1:
            task_ids_list = list(task_ids)
            existing_groups = {task_to_group[tid] for tid in task_ids_list if tid in task_to_group}
            if existing_groups:
                target = min(existing_groups)
            else:
                target = group_counter
                group_counter += 1

            for tid in task_ids_list:
                task_to_group[tid] = target

    # Assign ungrouped tasks their own group
    for task in group:
        if task["id"] not in task_to_group:
            task_to_group[task["id"]] = group_counter
            group_counter += 1

    # Collect groups
    final_groups = defaultdict(list)
    for task in group:
        gid = task_to_group[task["id"]]
        final_groups[gid].append(task)

    # Build subgroup summaries
    subgroups = []
    for gid, tasks_in_group in final_groups.items():
        stems = set()
        for t in tasks_in_group:
            if t["source_file_stem"]:
                stems.add(t["source_file_stem"])

        subgroups.append({
            "task_ids": [t["id"] for t in tasks_in_group],
            "tasks": tasks_in_group,
            "shared_pattern": ", ".join(sorted(stems)) if stems else "(no shared pattern)",
        })

    return subgroups


def _score_chain_candidate(sg: dict) -> dict:
    """Score a subgroup's suitability for sequential chaining.

    Scoring criteria:
    1. Task count: more tasks = more periods possible (weight: 2)
    2. Macro-category coverage: spans data_entry→calculation→validation→reporting (weight: 3)
    3. Workflow type diversity: variety of task types (weight: 1)
    4. Shared pattern strength: tasks that share data are more chainable (weight: 1)

    Returns dict with chain_score and supporting metrics.
    """
    tasks = sg["tasks"]
    task_ids = sg["task_ids"]

    # Gather all task types across the group
    all_types = set()
    for t in tasks:
        all_types.update(t["task_types"])

    # Determine macro-category coverage
    macro_covered = []
    for macro_name, macro_types in MACRO_CATEGORIES.items():
        if all_types & macro_types:
            macro_covered.append(macro_name)

    # Scoring
    task_count_score = min(len(task_ids) / 3.0, 3.0)  # max 3 points, saturates at 9 tasks
    macro_coverage_score = len(macro_covered) * 1.5    # 0-6 points (4 categories × 1.5)
    type_diversity_score = min(len(all_types) / 3.0, 2.0)  # max 2 points
    pattern_score = 1.0 if sg["shared_pattern"] != "(no shared pattern)" else 0.0

    chain_score = task_count_score + macro_coverage_score + type_diversity_score + pattern_score

    return {
        "chain_score": chain_score,
        "macro_categories_covered": macro_covered,
        "workflow_type_coverage": sorted(all_types),
        "task_count_score": round(task_count_score, 2),
        "macro_coverage_score": round(macro_coverage_score, 2),
        "type_diversity_score": round(type_diversity_score, 2),
        "pattern_score": round(pattern_score, 2),
    }


if __name__ == "__main__":
    main()
