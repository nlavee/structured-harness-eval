import datetime
import json
import logging
import os
from pathlib import Path
import traceback

from rich.progress import Progress

from glass.config.schema import Config
from glass.datasets.registry import get_dataset_class
from glass.judges.base import EvalResult
from glass.judges.llm import JudgeAPIError
from glass.judges.strategies.rotation import RotationStrategy
from glass.metrics.registry import get_metric_class
from glass.reports.csv_writer import write_results_csv
from glass.reports.statistics_report import generate_statistics_report
from glass.reports.summary import generate_summary
from glass.storage.checkpoint import CheckpointManager
from glass.storage.manifest import create_manifest
from glass.storage.run_store import RunStore
from glass.systems.registry import get_system_class
from glass.tui import get_console

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config, run_id: str = None, re_evaluate_source: str = None):
        self.config = config
        self.ui = get_console()
        self.re_evaluate_source = re_evaluate_source
        self.runs_dir = Path(config.output.runs_dir)

        self.run_id = self._determine_run_id(run_id)
        self.store = RunStore(config.output.runs_dir, self.run_id)
        self.checkpoint = CheckpointManager(config.output.runs_dir, self.run_id)

        self._setup_run_directory()

    def _determine_run_id(self, run_id: str = None) -> str:
        if run_id:
            return run_id
        if self.re_evaluate_source:
            return self._calculate_branched_run_id(self.re_evaluate_source)
        if self.config.experiment.run_id:
            return self.config.experiment.run_id
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{self.config.experiment.name}"

    def _calculate_branched_run_id(self, source_id: str) -> str:
        import re
        match = re.search(r"^(.*?)(?:_(\d+))?$", source_id)
        base_name = match.group(1) if match else source_id

        max_suffix = 0
        if self.runs_dir.exists():
            for d in self.runs_dir.iterdir():
                if d.is_dir() and d.name.startswith(f"{base_name}_"):
                    suffix_str = d.name[len(base_name)+1:]
                    if suffix_str.isdigit():
                        max_suffix = max(max_suffix, int(suffix_str))
        return f"{base_name}_{max_suffix + 1}"

    def _setup_run_directory(self):
        if self.re_evaluate_source:
            self._inherit_from_source()

        log_path = self.ui.setup_file_logging(self.store.run_dir, level=self.config.output.log_level)
        logger.info("File logging → %s", log_path)

        manifest_path = self.store.run_dir / "manifest.json"
        if not manifest_path.exists():
            self.store.run_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(create_manifest(self.config.model_dump()), f, indent=2)

        with open(self.store.run_dir / "config.yaml", "w") as f:
            import yaml
            yaml.dump(self.config.model_dump(), f)

    def _inherit_from_source(self):
        source_dir = self.runs_dir / self.re_evaluate_source
        if not source_dir.exists():
            raise FileNotFoundError(f"Source run '{self.re_evaluate_source}' not found for re-evaluation.")
        
        import shutil
        self.store.run_dir.mkdir(parents=True, exist_ok=True)
        
        source_inference = source_dir / "inference"
        if source_inference.exists():
            logger.info("Inheriting inference artifacts from %s", self.re_evaluate_source)
            shutil.copytree(source_inference, self.store.run_dir / "inference", dirs_exist_ok=True)
            
        for file_name in ["manifest.json", "glass.log"]:
            source_file = source_dir / file_name
            if source_file.exists():
                shutil.copy2(source_file, self.store.run_dir / file_name)

    def run(self):
        samples, systems = self._load_data_and_systems()
        system_names = [s.config.name for s in systems]

        self.ui.show_banner(
            experiment=self.config.experiment.name,
            run_id=self.run_id,
            dataset=f"{self.config.dataset.name} ({len(samples)} samples)",
            systems=system_names,
        )

        self._run_inference(samples, systems)
        self._run_evaluation(samples, systems)
        self._run_statistics()
        self._run_reporting()

        results = self.store.load_all_eval_results()
        quick_stats = self._build_quick_stats(results, system_names)
        self.ui.show_completion(self.store.run_dir, stats=quick_stats)

    def _load_data_and_systems(self):
        ds_cls = get_dataset_class(self.config.dataset.name)
        dataset = ds_cls()
        dataset.load(dataset_config=self.config.dataset)
        samples = dataset.get_samples()

        if self.config.dataset.domains:
            samples = [s for s in samples if s.domain in self.config.dataset.domains]
        if self.config.dataset.samples:
            import random
            random.seed(self.config.experiment.seed)
            if len(samples) > self.config.dataset.samples:
                samples = random.sample(samples, self.config.dataset.samples)

        systems = [get_system_class(sys_config.type)(sys_config) for sys_config in self.config.systems]
        return samples, systems

    def _run_inference(self, samples, systems):
        if self.re_evaluate_source:
            self.ui.phase_header(1, "Inference", f"Skipped — Inherited from {self.re_evaluate_source}")
            return

        self.ui.phase_header(1, "Inference", f"{len(samples)} samples × {len(systems)} systems")
        progress = self.ui.make_progress()
        with progress:
            task_id = progress.add_task("Inference", total=len(samples) * len(systems))
            for sample in samples:
                for system in systems:
                    if self.checkpoint.is_complete(system.config.name, sample.sample_id):
                        self.ui.task_skip(system.config.name, sample.sample_id, "checkpoint")
                        logger.debug("Skipped %s / sample_%s (checkpoint)", system.config.name, sample.sample_id)
                        progress.advance(task_id)
                        continue

                    raw_output = system.generate(sample)
                    self.store.save_raw_output(raw_output)
                    self.checkpoint.mark_complete(system.config.name, sample.sample_id)

                    if raw_output.error_type == "timeout":
                        self.ui.task_warning(system.config.name, sample.sample_id, f"TIMEOUT ({self.config.systems[0].timeout_s}s)")
                    elif raw_output.error_type:
                        self.ui.task_error(system.config.name, sample.sample_id, f"{raw_output.error_type}  exit={raw_output.exit_code}")
                    else:
                        self.ui.task_success(system.config.name, sample.sample_id, f"{raw_output.latency_s:.1f}s")
                    progress.advance(task_id)

    def _run_evaluation(self, samples, systems):
        self.ui.phase_header(2, "Evaluation")
        metric_map = {name: get_metric_class(name)() for name in self.config.metrics}
        judge_strategy = RotationStrategy(self.config.judges)

        progress = self.ui.make_progress()
        with progress:
            task_id = progress.add_task("Evaluation", total=len(samples) * len(systems))
            for sample in samples:
                for system in systems:
                    try:
                        raw_output = self.store.load_raw_output(system.config.name, sample.sample_id)
                    except FileNotFoundError:
                        self.ui.task_warning(system.config.name, sample.sample_id, "No raw output — skipping eval")
                        logger.warning("No raw output for %s / sample_%s — skipping eval", system.config.name, sample.sample_id)
                        progress.advance(task_id)
                        continue

                    judge = judge_strategy.assign_judge(system.config.name)
                    judge_outputs = {}
                    
                    if raw_output.error_type:
                        logger.warning("Skipping Judge for %s/%s — SUT error_type='%s'", system.config.name, sample.sample_id, raw_output.error_type)
                        self.ui.task_warning(system.config.name, sample.sample_id, f"Judge skipped — SUT {raw_output.error_type}")

                    metric_results = {}
                    for m_name, m_inst in metric_map.items():
                        try:
                            metric_results[m_name] = m_inst.compute(raw_output, sample, judge=judge, judge_outputs=judge_outputs)
                        except Exception as e:
                            logger.error("Metric %s failed on %s/%s: %s\n%s", m_name, system.config.name, sample.sample_id, e, traceback.format_exc())
                            metric_results[m_name] = None
                            if m_inst.requires_judge:
                                judge_outputs[m_name] = f"Metric Exception: {e}"
                                
                    if not raw_output.error_type and metric_results.get("judge_score") is not None:
                        judge_model_str = f"judge={judge.provider}/{judge.model}" if hasattr(judge, "provider") and hasattr(judge, "model") else ""
                        self.ui.task_success(system.config.name, sample.sample_id, judge_model_str)

                    judge_model_str = f"{judge.provider}/{judge.model}" if hasattr(judge, "provider") and hasattr(judge, "model") else None

                    eval_result = EvalResult(
                        sample_id=sample.sample_id,
                        system_name=system.config.name,
                        domain=sample.domain,
                        judge_model=judge_model_str,
                        metrics=metric_results,
                        judge_outputs=judge_outputs,
                    )
                    self.store.save_eval_result(eval_result)
                    progress.advance(task_id)

    def _run_statistics(self):
        self.ui.phase_header(3, "Statistics")
        results = self.store.load_all_eval_results()
        stats_path = self.store.run_dir / "statistics.json"
        generate_statistics_report(results=results, config=self.config, output_path=stats_path)
        self.ui.file_saved("statistics.json", stats_path)

    def _run_reporting(self):
        self.ui.phase_header(4, "Reporting")
        results = self.store.load_all_eval_results()

        csv_path = self.store.run_dir / "results.csv"
        write_results_csv(results, csv_path)
        self.ui.file_saved("results.csv", csv_path)

        summary_path = self.store.run_dir / "summary.md"
        generate_summary(results, summary_path, stats_path=self.store.run_dir / "statistics.json")
        self.ui.file_saved("summary.md", summary_path)

    def _build_quick_stats(self, results: list[EvalResult], system_names: list[str]) -> dict:
        metrics_of_interest = ["exact_match", "judge_score", "error_rate", "latency_s"]
        available = [m for m in metrics_of_interest if m in self.config.metrics]

        if not available: return {}

        stats = {"systems": system_names, "metrics": {}}
        for m in available:
            stats["metrics"][m] = {}
            for sys in system_names:
                vals = [r.metrics.get(m) for r in results if r.system_name == sys and r.metrics.get(m) is not None]
                stats["metrics"][m][sys] = sum(vals) / len(vals) if vals else None
        return stats
