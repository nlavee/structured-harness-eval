import json
from pathlib import Path
from typing import Set, Tuple


class CheckpointManager:
    def __init__(self, runs_dir: str, run_id: str):
        self.run_dir = Path(runs_dir) / run_id
        self.checkpoint_file = self.run_dir / "checkpoint.json"
        self.completed: Set[Tuple[str, str]] = set()  # (system_name, sample_id)
        self._load()

    def _load(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                # stored as list of [system, sample]
                for item in data:
                    self.completed.add(tuple(item))

    def mark_complete(self, system_name: str, sample_id: str):
        self.completed.add((system_name, sample_id))
        self._save()

    def is_complete(self, system_name: str, sample_id: str) -> bool:
        return (system_name, sample_id) in self.completed

    def _save(self):
        with open(self.checkpoint_file, "w") as f:
            json.dump(list(self.completed), f)
