import json
import logging
from pathlib import Path

from glass.judges.base import EvalResult
from glass.systems.base import RawOutput

logger = logging.getLogger(__name__)


class RunStore:
    def __init__(self, runs_dir: str, run_id: str):
        self.run_dir = Path(runs_dir) / run_id
        self.inference_dir = self.run_dir / "inference"
        self.evaluation_dir = self.run_dir / "evaluation"

        self.inference_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_output(self, output: RawOutput) -> None:
        system_dir = self.inference_dir / output.system_name
        system_dir.mkdir(exist_ok=True)

        file_path = system_dir / f"sample_{output.sample_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

        logger.debug("Saved raw output → %s", file_path)

    def load_raw_output(self, system_name: str, sample_id: str) -> RawOutput:
        file_path = self.inference_dir / system_name / f"sample_{sample_id}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RawOutput(**data)

    def save_eval_result(self, result: EvalResult) -> None:
        # Save flattened structure if needed, or structured
        # PLAN says: runs/{run_id}/evaluation/sample_{id}.json containing results for ALL systems or
        # separate?
        # PLAN Design:
        # evaluation/
        #   sample_001.json

        # But wait, EvalResult has `system_name`. If we save per sample, we might overwrite or need a list.
        # Let's check PLAN structure.
        # "evaluation/sample_001.json"
        # And EvalResult has "system_name".
        # This implies either sample_001.json contains a list of results for that sample across systems,
        # OR we save as evaluation/{system_name}/sample_{id}.json.

        # Looking at PLAN "Run Artifact Structure":
        # evaluation/
        #   sample_001.json

        # If I save EvalResult (which is for ONE system), I can't overwrite sample_001.json.
        # I will deviate slightly to ensure no data loss: evaluation/{system_name}/sample_{id}.json
        # OR I append to a list in sample_{id}.json.

        # Appending is risky for race conditions (though we are sequential).
        # Let's check AP-24 etc.
        # Phase 3 loads all EvalResults.

        # Let's stick to the robust pattern: evaluation/{system_name}/sample_{id}.json
        # It mirrors inference structure and is cleaner.

        system_dir = self.evaluation_dir / result.system_name
        system_dir.mkdir(exist_ok=True)

        file_path = system_dir / f"sample_{result.sample_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

    def load_all_eval_results(self) -> list[EvalResult]:
        results = []
        for system_dir in self.evaluation_dir.iterdir():
            if system_dir.is_dir():
                for file_path in system_dir.glob("*.json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        results.append(EvalResult(**json.load(f)))
        return results
