from glass.datasets.base import EvaluationSample
from glass.judges.llm import JudgeAPIError
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput
import logging

logger = logging.getLogger(__name__)


@register("judge_score")
class JudgeScoreMetric(BaseMetric):
    """LLM-as-judge correctness evaluation (CORRECT / INCORRECT).

    This is a primary metric for hypothesis testing per PLAN.md.
    Requires a judge instance passed via kwargs by the pipeline.
    """

    @property
    def requires_judge(self) -> bool:
        return True

    @property
    def category(self) -> str:
        return "correctness"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None

        judge = kwargs.get("judge")
        judge_outputs = kwargs.get("judge_outputs", {})
        system_name = output.system_name

        try:
            score, explanation = judge.evaluate_correctness(
                sample.question, sample.gold_answer, output.output
            )
            judge_outputs["correctness"] = explanation
            return score
        except JudgeAPIError as e:
            logger.error(
                "Judge correctness failed for %s/%s: %s",
                system_name,
                sample.sample_id,
                e,
            )
            judge_outputs["correctness"] = f"ERROR: {e}"
            return None
