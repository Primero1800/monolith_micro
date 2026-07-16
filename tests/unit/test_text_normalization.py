import pytest

from app.utils.text_normalization import is_degenerate_text, normalize_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Здравствуйте! Хочу снять квартиру.", "здравствуйте хочу снять квартиру"),
        ("  много   пробелов  ", "много пробелов"),
        ("UPPER CASE", "upper case"),
        ("текст,,, с --- пунктуацией!!!", "текст с пунктуацией"),
        ("", ""),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    """normalize_text lowercases, strips punctuation, and collapses whitespace"""
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "normalized, expected",
    [
        ("", True),
        ("   ", True),
        ("123 456", True),
        ("!!! ...", True),
        ("привет", False),
        ("а", False),
        ("123 текст", False),
    ],
)
def test_is_degenerate_text(normalized: str, expected: bool) -> None:
    """is_degenerate_text is True only for empty/whitespace/punctuation-only/no-letter text"""
    assert is_degenerate_text(normalized) is expected
