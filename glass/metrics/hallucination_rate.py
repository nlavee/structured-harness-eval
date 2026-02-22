from glass.datasets.base import EvaluationSample
from glass.judges.llm import JudgeAPIError
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput
import logging

logger = logging.getLogger(__name__)


@register("hallucination_rate")
class HallucinationRateMetric(BaseMetric):
    """LLM-as-judge hallucination evaluation.

    Per-sentence NLI classification: SUPPORTED / CONTRADICTED / UNVERIFIED.
    Score = (Contradicted + Unverified) / Total sentences.

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
            hr, hr_details = judge.evaluate_hallucination(
                output.output, sample.context_prompt
            )
            judge_outputs["hallucination"] = hr_details
            return hr
        except JudgeAPIError as e:
            logger.error(
                "Judge hallucination failed for %s/%s: %s",
                system_name,
                sample.sample_id,
                e,
            )
            judge_outputs["hallucination"] = f"ERROR: {e}"
            return None
