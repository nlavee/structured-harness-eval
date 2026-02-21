import string
import unicodedata


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    return " ".join(s.split())
