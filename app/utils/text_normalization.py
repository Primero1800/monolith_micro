import re

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for duplicate-ticket matching"""
    without_punctuation = _PUNCTUATION_RE.sub("", text.lower())
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()


def is_degenerate_text(normalized_text: str) -> bool:
    """True if there is nothing meaningful left to classify (empty or no letters at all)"""
    return not normalized_text or not any(ch.isalpha() for ch in normalized_text)
