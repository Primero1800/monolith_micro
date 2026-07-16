import re

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for duplicate-ticket matching"""
    without_punctuation = _PUNCTUATION_RE.sub("", text.lower())
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()
