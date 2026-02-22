import logging
import traceback
from typing import Tuple

import nltk

from glass.judges.base import Judge
from glass.judges.prompts import AA_LCR_EQUALITY_V1, NLI_SENTENCE_V1

logger = logging.getLogger(__name__)


class JudgeAPIError(Exception):
    """Raised when a judge API call fails. Caught by pipeline to set metric=None (AP-26)."""

    pass


class LLMJudge(Judge):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        # Lazy NLTK download — only triggers on first instantiation
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass  # Fallback sentence splitter used in evaluate_hallucination

    def evaluate_correctness(self, question: str, gold_answer: str, prediction: str) -> Tuple[float, str]:
        prompt = AA_LCR_EQUALITY_V1.format(question=question, gold_answer=gold_answer, prediction=prediction)
        response = self._call_llm(prompt)  # raises JudgeAPIError on failure

        # INCORRECT takes precedence to avoid false-positives from responses that say
        # "... is INCORRECT" before then saying "CORRECT would be..."
        if "INCORRECT" in response.upper():
            score = 0.0
        elif "CORRECT" in response.upper():
            score = 1.0
        else:
            # Ambiguous response: treat as incorrect but record raw text for audit (AP-12)
            logger.warning("Judge returned ambiguous correctness response: %r", response[:200])
            score = 0.0

        return score, response

    def evaluate_hallucination(self, prediction: str, context: str) -> Tuple[float, str]:
        # AP-27: Sentence tokenization with graceful fallback
        try:
            sentences = nltk.sent_tokenize(prediction)
        except Exception:
            sentences = [s.strip() for s in prediction.split(". ") if s.strip()]

        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return None, "[]"

        results = []
        for sent in sentences:
            prompt = NLI_SENTENCE_V1.format(context=context, sentence=sent)
            resp = self._call_llm(prompt)  # raises JudgeAPIError on failure
            results.append(resp)

        count = sum(1 for r in results if "CONTRADICTED" in r.upper() or "UNVERIFIED" in r.upper())
        score = count / len(results)
        return score, str(results)

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider. Raises JudgeAPIError on any failure (AP-26)."""
        try:
            if self.provider == "openai":
                import openai

                client = openai.OpenAI()
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )
                except openai.BadRequestError as e:
                    if "temperature" in str(e):
                        # Some models (e.g. gpt-5-mini) only support default temperature
                        resp = client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                        )
                    else:
                        raise
                return resp.choices[0].message.content or ""
            elif self.provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0,
                )
                return resp.content[0].text
            elif self.provider == "google":
                import google.generativeai as genai

                model = genai.GenerativeModel(self.model)
                resp = model.generate_content(prompt)
                return resp.text
            else:
                raise JudgeAPIError(f"Unknown provider: {self.provider!r}")
        except JudgeAPIError:
            raise
        except Exception as e:
            # AP-26: Log full traceback; re-raise as JudgeAPIError so pipeline
            # can set metric=None instead of silently recording 0.0
            logger.error(
                "Judge API call failed (provider=%s, model=%s): %s\n%s",
                self.provider,
                self.model,
                e,
                traceback.format_exc(),
            )
            raise JudgeAPIError(f"{self.provider}/{self.model}: {e}") from e
