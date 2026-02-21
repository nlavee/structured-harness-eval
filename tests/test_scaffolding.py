from glass.config.schema import Config, SystemConfig
from glass.datasets.base import EvaluationSample
from glass.judges.base import EvalResult, Judge
from glass.metrics.base import BaseMetric
from glass.systems.base import RawOutput, SystemUnderTest


def test_config_schema():
    config_data = {
        "experiment": {"name": "test_exp", "seed": 42},
        "dataset": {"name": "aa_lcr"},
        "systems": [{"name": "test_sys", "type": "claude", "command": ["claude"]}],
        "metrics": ["exact_match"],
        "judges": {
            "strategy": "fixed",
            "fixed": {"provider": "openai", "model": "gpt-4"},
            "templates": {"correctness": "tmpl1", "hallucination": "tmpl2"},
        },
        "statistics": {"primary_test": "wilcoxon", "secondary_test": "ttest"},
        "output": {},
    }
    config = Config(**config_data)
    assert config.experiment.name == "test_exp"
    assert config.systems[0].name == "test_sys"


def test_evaluation_sample():
    sample = EvaluationSample(
        sample_id="1",
        domain="Legal",
        question="Q",
        gold_answer="A",
        context_prompt="Context",
        input_tokens=100,
        metadata={},
    )
    assert sample.sample_id == "1"


def test_raw_output():
    output = RawOutput(
        sample_id="1",
        system_name="sys",
        command=["claude", "--print"],
        output="out",
        latency_s=1.0,
        exit_code=0,
        stderr="",
        timestamp="2023-01-01",
    )
    assert output.output == "out"
    # AP-8: command field required
    assert output.command == ["claude", "--print"]


def test_eval_result():
    result = EvalResult(
        sample_id="1",
        system_name="sys",
        domain="Legal",
        metrics={"score": 1.0},
        judge_outputs={"correctness": "Correct"},
    )
    assert result.metrics["score"] == 1.0


def test_abc_instantiation():
    class ConcreteSystem(SystemUnderTest):
        def generate(self, sample):
            return RawOutput(
                sample_id=sample.sample_id,
                system_name=self.config.name,
                command=["stub"],
                output="test",
                latency_s=0.1,
                exit_code=0,
                stderr="",
                timestamp="now",
            )

    cfg = SystemConfig(name="test", type="stub")
    sys_instance = ConcreteSystem(cfg)
    assert isinstance(sys_instance, SystemUnderTest)

    class ConcreteMetric(BaseMetric):
        def compute(self, prediction, sample):
            return 1.0

    met = ConcreteMetric()
    assert isinstance(met, BaseMetric)

    class ConcreteJudge(Judge):
        def evaluate_correctness(self, q, g, p):
            return 1.0, "ok"

        def evaluate_hallucination(self, p, c):
            return 0.0, "ok"

    jud = ConcreteJudge()
    assert isinstance(jud, Judge)
