import os
import tempfile
from unittest.mock import patch

import pytest

from glass.datasets.aalcr import AALCRAdapter, _build_prompt, _build_documents_text
from glass.datasets.registry import get_dataset_class

in_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"


def test_registry():
    cls = get_dataset_class("aa_lcr")
    assert cls == AALCRAdapter


def test_build_documents_text():
    docs = ["Hello world", "Second doc"]
    result = _build_documents_text(docs)
    assert "BEGIN DOCUMENT 1:" in result
    assert "Hello world" in result
    assert "END DOCUMENT 1" in result
    assert "BEGIN DOCUMENT 2:" in result
    assert "Second doc" in result
    assert "END DOCUMENT 2" in result


def test_build_prompt():
    docs = ["Doc content A", "Doc content B"]
    prompt = _build_prompt("What is the answer?", docs)

    # Verify prompt template structure
    assert prompt.startswith("BEGIN INPUT DOCUMENTS")
    assert "END INPUT DOCUMENTS" in prompt
    assert "BEGIN DOCUMENT 1:" in prompt
    assert "Doc content A" in prompt
    assert "END DOCUMENT 1" in prompt
    assert "BEGIN DOCUMENT 2:" in prompt
    assert "Doc content B" in prompt
    assert "END DOCUMENT 2" in prompt
    assert "Answer the following question using the input documents provided above." in prompt
    assert "START QUESTION" in prompt
    assert "What is the answer?" in prompt
    assert "END QUESTION" in prompt

    # Verify document order (doc 1 before doc 2)
    assert prompt.index("BEGIN DOCUMENT 1:") < prompt.index("BEGIN DOCUMENT 2:")


def test_build_prompt_document_order():
    """Verify documents appear in the prompt in the same order as provided."""
    docs = ["Third", "First", "Second"]
    prompt = _build_prompt("Q?", docs)
    # "Third" should be DOCUMENT 1, "First" should be DOCUMENT 2
    idx_1 = prompt.index("BEGIN DOCUMENT 1:")
    idx_2 = prompt.index("BEGIN DOCUMENT 2:")
    idx_3 = prompt.index("BEGIN DOCUMENT 3:")
    assert idx_1 < idx_2 < idx_3
    assert "Third" in prompt[idx_1:idx_2]
    assert "First" in prompt[idx_2:idx_3]


@pytest.mark.skipif(in_github_actions, reason="Depends on local files excluded from git")
def test_aalcr_load_from_local():
    """Test that AALCRAdapter loads from local CSV and builds proper prompts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure matching the expected layout:
        # tmpdir/AA-LCR_extracted-text/lcr/TestCat/test_set/
        extracted_dir = os.path.join(tmpdir, "AA-LCR_extracted-text")
        lcr_dir = os.path.join(extracted_dir, "lcr")
        doc_dir = os.path.join(lcr_dir, "TestCat", "test_set")
        os.makedirs(doc_dir)

        # Create document files
        with open(os.path.join(doc_dir, "doc_a.txt"), "w") as f:
            f.write("Content of document A")
        with open(os.path.join(doc_dir, "doc_b.txt"), "w") as f:
            f.write("Content of document B")

        # Create CSV at tmpdir/AA-LCR_Dataset.csv
        # (grandparent of lcr_dir = tmpdir)
        csv_path = os.path.join(tmpdir, "AA-LCR_Dataset.csv")
        with open(csv_path, "w") as f:
            f.write(",document_category,document_set_id,question_id,question,answer,data_source_filenames,data_source_urls,input_tokens\n")
            f.write('0,TestCat,test_set,1,"What is the answer?",The answer is 42,doc_a.txt;doc_b.txt,http://example.com,1000\n')

        # Create a mock dataset_config
        class MockConfig:
            dataset_folder = lcr_dir

        adapter = AALCRAdapter()
        adapter.load(dataset_config=MockConfig())
        samples = adapter.get_samples()

        assert len(samples) == 1
        s = samples[0]
        assert s.sample_id == "1"
        assert s.domain == "TestCat"
        assert s.question == "What is the answer?"
        assert s.gold_answer == "The answer is 42"
        assert s.input_tokens == 1000

        # Verify prompt structure
        assert "BEGIN INPUT DOCUMENTS" in s.context_prompt
        assert "BEGIN DOCUMENT 1:" in s.context_prompt
        assert "Content of document A" in s.context_prompt
        assert "END DOCUMENT 1" in s.context_prompt
        assert "BEGIN DOCUMENT 2:" in s.context_prompt
        assert "Content of document B" in s.context_prompt
        assert "END DOCUMENT 2" in s.context_prompt
        assert "END INPUT DOCUMENTS" in s.context_prompt
        assert "START QUESTION" in s.context_prompt
        assert "What is the answer?" in s.context_prompt
        assert "END QUESTION" in s.context_prompt

        # Verify metadata
        assert s.metadata["document_category"] == "TestCat"
        assert s.metadata["document_set_id"] == "test_set"
        assert s.metadata["data_source_filenames"] == ["doc_a.txt", "doc_b.txt"]


@pytest.mark.skipif(in_github_actions, reason="Depends on local files excluded from git")
def test_aalcr_missing_csv_exits_early():
    """Test that missing local data causes an early exit with FileNotFoundError."""

    class MockConfig:
        dataset_folder = "/nonexistent/path/to/lcr"

    adapter = AALCRAdapter()
    with pytest.raises(FileNotFoundError, match="AA-LCR"):
        adapter.load(dataset_config=MockConfig())


@pytest.mark.skipif(in_github_actions, reason="Depends on local files excluded from git")
def test_aalcr_missing_extracted_dir_exits_early():
    """Test that missing extracted text dir causes early exit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CSV but not the extracted text dir
        csv_path = os.path.join(tmpdir, "AA-LCR_Dataset.csv")
        with open(csv_path, "w") as f:
            f.write(",document_category\n")

        class MockConfig:
            dataset_folder = os.path.join(tmpdir, "nonexistent_lcr")

        adapter = AALCRAdapter()
        with pytest.raises(FileNotFoundError, match="extracted text directory not found"):
            adapter.load(dataset_config=MockConfig())


@pytest.mark.skipif(in_github_actions, reason="Depends on local files excluded from git")
def test_aalcr_missing_document_file():
    """Test that a missing document file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lcr_dir = os.path.join(tmpdir, "lcr")
        doc_dir = os.path.join(lcr_dir, "TestCat", "test_set")
        os.makedirs(doc_dir)

        # Create only one of two expected documents
        with open(os.path.join(doc_dir, "doc_a.txt"), "w") as f:
            f.write("Content A")

        csv_path = os.path.join(tmpdir, "AA-LCR_Dataset.csv")
        with open(csv_path, "w") as f:
            f.write(",document_category,document_set_id,question_id,question,answer,data_source_filenames,data_source_urls,input_tokens\n")
            f.write('0,TestCat,test_set,1,"Q?",A,doc_a.txt;doc_missing.txt,http://example.com,100\n')

        class MockConfig:
            dataset_folder = lcr_dir

        adapter = AALCRAdapter()
        with pytest.raises(FileNotFoundError, match="Document not found"):
            adapter.load(dataset_config=MockConfig())
