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

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config, run_id: str = None):
        self.config = config

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

        # Phase 1: Inference
        print(f"\n[GLASS] Starting Phase 1: Inference ({len(samples)} samples, {len(systems)} systems)")
        for sample in samples:
            for system in systems:
                if self.checkpoint.is_complete(system.config.name, sample.sample_id):
                    print(f"[GLASS] Skipping {system.config.name} on sample {sample.sample_id} (Completed)")
                    continue

                print(f"[GLASS] Running {system.config.name} on sample {sample.sample_id}")
                raw_output = system.generate(sample)
                self.store.save_raw_output(raw_output)
                self.checkpoint.mark_complete(system.config.name, sample.sample_id)

        # Phase 2: Evaluation
        print("\n[GLASS] Starting Phase 2: Evaluation")

        # Instantiate metrics once (AP-25: avoid mutable state accumulation)
        metric_map = {name: get_metric_class(name)() for name in self.config.metrics}

        # Instantiate Judge Strategy
        judge_strategy = RotationStrategy(self.config.judges)

        # Iterate over all completed outputs
        for sample in samples:
            for system in systems:
                try:
                    raw_output = self.store.load_raw_output(system.config.name, sample.sample_id)
                except FileNotFoundError:
                    print(
                        f"[GLASS] Warning: No raw output for {system.config.name} "
                        f"sample {sample.sample_id} — skipping eval"
                    )
                    continue

                # Compute Deterministic Metrics (AP-3: failures return None, not 0.0)
                metric_results = {}
                for m_name, m_inst in metric_map.items():
                    try:
                        metric_results[m_name] = m_inst.compute(raw_output, sample)
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

                # Judge Evaluation (AP-15: skip judges if SUT errored)
                judge = judge_strategy.assign_judge(system.config.name)
                judge_outputs = {}

                if raw_output.error_type:
                    # AP-15: undefined, not 0.0
                    metric_results["judge_score"] = None
                    metric_results["hallucination_rate"] = None
                else:
                    try:
                        score, explanation = judge.evaluate_correctness(
                            sample.question, sample.gold_answer, raw_output.output
                        )
                        metric_results["judge_score"] = score
                        judge_outputs["correctness"] = explanation
                    except JudgeAPIError as e:
                        logger.error(
                            "Judge correctness failed for %s/%s: %s",
                            system.config.name,
                            sample.sample_id,
                            e,
                        )
                        metric_results["judge_score"] = None
                        judge_outputs["correctness"] = f"ERROR: {e}"

                    try:
                        hr, hr_details = judge.evaluate_hallucination(raw_output.output, sample.context_prompt)
                        metric_results["hallucination_rate"] = hr
                        judge_outputs["hallucination"] = hr_details
                    except JudgeAPIError as e:
                        logger.error(
                            "Judge hallucination failed for %s/%s: %s",
                            system.config.name,
                            sample.sample_id,
                            e,
                        )
                        metric_results["hallucination_rate"] = None
                        judge_outputs["hallucination"] = f"ERROR: {e}"

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
                print(f"[GLASS] Evaluated {system.config.name} sample {sample.sample_id}")

        # Phase 3: Statistics (AP-4: computed on complete result set, never inline)
        print("\n[GLASS] Starting Phase 3: Statistics")
        results = self.store.load_all_eval_results()
        stats_path = self.store.run_dir / "statistics.json"
        generate_statistics_report(results=results, config=self.config, output_path=stats_path)
        print(f"[GLASS] Saved statistics → {stats_path}")

        # Phase 4: Reporting
        print("\n[GLASS] Starting Phase 4: Reporting")

        # CSV
        write_results_csv(results, self.store.run_dir / "results.csv")
        print(f"[GLASS] Saved results.csv → {self.store.run_dir / 'results.csv'}")

        # Summary (passes stats path so it can embed key stats)
        generate_summary(results, self.store.run_dir / "summary.md", stats_path=stats_path)
        print(f"[GLASS] Saved summary → {self.store.run_dir / 'summary.md'}")

        print(f"\n[GLASS] Run complete. Artifacts → {self.store.run_dir}")
