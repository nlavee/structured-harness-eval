import datetime
import logging
import traceback

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
    def __init__(self, config: Config, run_id: str = None):
        self.config = config
        self.ui = get_console()

        # Phase 0: Setup
        if run_id:
            self.run_id = run_id
        elif config.experiment.run_id:
            self.run_id = config.experiment.run_id
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{ts}_{config.experiment.name}"

        self.store = RunStore(config.output.runs_dir, self.run_id)
        self.checkpoint = CheckpointManager(config.output.runs_dir, self.run_id)

        # Setup file logging into the run directory
        log_path = self.ui.setup_file_logging(
            self.store.run_dir, level=config.output.log_level
        )
        logger.info("File logging → %s", log_path)

        # Write manifest if new run
        manifest_path = self.store.run_dir / "manifest.json"
        if not manifest_path.exists():
            self.store.run_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                import json

                json.dump(create_manifest(config.model_dump()), f, indent=2)

            # Save frozen config
            with open(self.store.run_dir / "config.yaml", "w") as f:
                import yaml

                yaml.dump(config.model_dump(), f)

    def run(self):
        ui = self.ui

        # Load Dataset
        ds_cls = get_dataset_class(self.config.dataset.name)
        dataset = ds_cls()
        dataset.load(dataset_config=self.config.dataset)
        samples = dataset.get_samples()

        # Filter samples if config
        if self.config.dataset.domains:
            samples = [s for s in samples if s.domain in self.config.dataset.domains]
        if self.config.dataset.samples:
            import random

            random.seed(self.config.experiment.seed)
            if len(samples) > self.config.dataset.samples:
                samples = random.sample(samples, self.config.dataset.samples)

        # Instantiate Systems
        systems = []
        for sys_config in self.config.systems:
            sys_cls = get_system_class(sys_config.type)
            systems.append(sys_cls(sys_config))

        system_names = [s.config.name for s in systems]

        # Show banner
        ui.show_banner(
            experiment=self.config.experiment.name,
            run_id=self.run_id,
            dataset=f"{self.config.dataset.name} ({len(samples)} samples)",
            systems=system_names,
        )

        total_inference = len(samples) * len(systems)

        # ── Phase 1: Inference ──────────────────────────────────────────── #
        ui.phase_header(1, "Inference", f"{len(samples)} samples × {len(systems)} systems")

        progress = ui.make_progress()
        with progress:
            task_id = progress.add_task("Inference", total=total_inference)

            for sample in samples:
                for system in systems:
                    if self.checkpoint.is_complete(system.config.name, sample.sample_id):
                        ui.task_skip(system.config.name, sample.sample_id, "checkpoint")
                        logger.debug("Skipped %s / sample_%s (checkpoint)", system.config.name, sample.sample_id)
                        progress.advance(task_id)
                        continue

                    raw_output = system.generate(sample)
                    self.store.save_raw_output(raw_output)
                    self.checkpoint.mark_complete(system.config.name, sample.sample_id)

                    if raw_output.error_type == "timeout":
                        ui.task_warning(
                            system.config.name,
                            sample.sample_id,
                            f"TIMEOUT ({self.config.systems[0].timeout_s}s)",
                        )
                    elif raw_output.error_type:
                        ui.task_error(
                            system.config.name,
                            sample.sample_id,
                            f"{raw_output.error_type}  exit={raw_output.exit_code}",
                        )
                    else:
                        ui.task_success(
                            system.config.name,
                            sample.sample_id,
                            f"{raw_output.latency_s:.1f}s",
                        )

                    progress.advance(task_id)

        # ── Phase 2: Evaluation ─────────────────────────────────────────── #
        ui.phase_header(2, "Evaluation")

        # Instantiate metrics once (AP-25: avoid mutable state accumulation)
        metric_map = {name: get_metric_class(name)() for name in self.config.metrics}

        # Instantiate Judge Strategy
        judge_strategy = RotationStrategy(self.config.judges)

        total_eval = len(samples) * len(systems)
        progress = ui.make_progress()
        with progress:
            task_id = progress.add_task("Evaluation", total=total_eval)

            for sample in samples:
                for system in systems:
                    try:
                        raw_output = self.store.load_raw_output(system.config.name, sample.sample_id)
                    except FileNotFoundError:
                        ui.task_warning(
                            system.config.name,
                            sample.sample_id,
                            "No raw output — skipping eval",
                        )
                        logger.warning(
                            "No raw output for %s / sample_%s — skipping eval",
                            system.config.name,
                            sample.sample_id,
                        )
                        progress.advance(task_id)
                        continue

                    # Get Judge
                    judge = judge_strategy.assign_judge(system.config.name)
                    judge_outputs = {}
                    
                    if raw_output.error_type:
                        logger.warning(
                            "Skipping Judge for %s/%s — SUT error_type='%s'",
                            system.config.name,
                            sample.sample_id,
                            raw_output.error_type,
                        )
                        ui.task_warning(
                            system.config.name,
                            sample.sample_id,
                            f"Judge skipped — SUT {raw_output.error_type}",
                        )

                    # Compute All Metrics (AP-3: failures return None, not 0.0)
                    metric_results = {}
                    for m_name, m_inst in metric_map.items():
                        try:
                            # Pass judge instance and judge_outputs container down to metrics via kwargs
                            metric_results[m_name] = m_inst.compute(
                                raw_output, 
                                sample, 
                                judge=judge, 
                                judge_outputs=judge_outputs
                            )
                        except Exception as e:
                            logger.error(
                                "Metric %s failed on %s/%s: %s\n%s",
                                m_name,
                                system.config.name,
                                sample.sample_id,
                                e,
                                traceback.format_exc(),
                            )
                            metric_results[m_name] = None  # AP-3: never silently zero
                            
                            # Record error for any judge-dependent metric (AP-24: no hardcoded names)
                            if m_inst.requires_judge:
                                judge_outputs[m_name] = f"Metric Exception: {e}"
                                
                    # Set UI success status if judge score didn't crash/error
                    if not raw_output.error_type and metric_results.get("judge_score") is not None:
                        judge_model_str = ""
                        if hasattr(judge, "provider") and hasattr(judge, "model"):
                            judge_model_str = f"judge={judge.provider}/{judge.model}"
                        ui.task_success(
                            system.config.name,
                            sample.sample_id,
                            judge_model_str,
                        )

                    judge_model_str = None
                    if hasattr(judge, "provider") and hasattr(judge, "model"):
                        judge_model_str = f"{judge.provider}/{judge.model}"

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

        # ── Phase 3: Statistics ─────────────────────────────────────────── #
        ui.phase_header(3, "Statistics")
        results = self.store.load_all_eval_results()
        stats_path = self.store.run_dir / "statistics.json"
        generate_statistics_report(results=results, config=self.config, output_path=stats_path)
        ui.file_saved("statistics.json", stats_path)

        # ── Phase 4: Reporting ──────────────────────────────────────────── #
        ui.phase_header(4, "Reporting")

        # CSV
        csv_path = self.store.run_dir / "results.csv"
        write_results_csv(results, csv_path)
        ui.file_saved("results.csv", csv_path)

        # Summary
        summary_path = self.store.run_dir / "summary.md"
        generate_summary(results, summary_path, stats_path=stats_path)
        ui.file_saved("summary.md", summary_path)

        # ── Completion ──────────────────────────────────────────────────── #
        # Build quick stats for display
        quick_stats = self._build_quick_stats(results, system_names)
        ui.show_completion(self.store.run_dir, stats=quick_stats)

    def _build_quick_stats(self, results: list[EvalResult], system_names: list[str]) -> dict:
        """Build a quick stats dict for the completion table."""
        metrics_of_interest = ["exact_match", "judge_score", "error_rate", "latency_s"]
        available = [m for m in metrics_of_interest if m in self.config.metrics]

        if not available:
            return {}

        stats = {"systems": system_names, "metrics": {}}
        for metric_name in available:
            stats["metrics"][metric_name] = {}
            for sys_name in system_names:
                values = [
                    r.metrics.get(metric_name)
                    for r in results
                    if r.system_name == sys_name and r.metrics.get(metric_name) is not None
                ]
                if values:
                    stats["metrics"][metric_name][sys_name] = sum(values) / len(values)
                else:
                    stats["metrics"][metric_name][sys_name] = None

        return stats
