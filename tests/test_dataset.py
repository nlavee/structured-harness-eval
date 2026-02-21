from unittest.mock import patch

from glass.datasets.aalcr import AALCRAdapter
from glass.datasets.registry import get_dataset_class


def test_registry():
    cls = get_dataset_class("aa_lcr")
    assert cls == AALCRAdapter


@patch("glass.datasets.aalcr.datasets.load_dataset")
def test_aalcr_load(mock_load):
    # Mock return value
    mock_load.return_value = [
        {
            "id": "1",
            "domain": "Legal",
            "question": "Q1",
            "answer": "A1",
            "context": "12345678",  # 8 chars
            "extra": "meta",
        }
    ]

    adapter = AALCRAdapter()
    adapter.load()
    samples = adapter.get_samples()

    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "1"
    assert s.domain == "Legal"
    assert s.question == "Q1"
    assert s.gold_answer == "A1"
    assert s.context_prompt == "12345678"
    assert s.input_tokens == 2  # 8 // 4 = 2
    assert s.metadata["extra"] == "meta"
