import pytest

from app.common.enums import TicketCategoryEnum
from app.utils.category_matching import match_single_category
from app.utils.text_normalization import normalize_text


@pytest.mark.parametrize(
    "raw_text, expected",
    [
        ("хочу снять квартиру", TicketCategoryEnum.RENT),
        ("хочу сдать квартиру", TicketCategoryEnum.RENT),
        ("хочу купить квартиру", TicketCategoryEnum.SALE),
        ("хочу продать квартиру", TicketCategoryEnum.SALE),
        ("запишите меня на просмотр", TicketCategoryEnum.VIEWING),
        ("вопрос по ипотеке", TicketCategoryEnum.CONSULTATION),
        ("это просто мошенничество", TicketCategoryEnum.COMPLAINT),
    ],
)
def test_match_single_category_finds_the_one_matching_category(
    raw_text: str, expected: TicketCategoryEnum
) -> None:
    """A text containing exactly one category's keyword resolves to that category"""
    assert match_single_category(normalize_text(raw_text)) == expected


def test_match_single_category_returns_none_when_no_keyword_matches() -> None:
    """Text with no recognized category keyword falls through (caller sends it to the LLM)"""
    assert match_single_category(normalize_text("добрый день, у меня вопрос")) is None


def test_match_single_category_returns_none_when_multiple_categories_match() -> None:
    """An ambiguous text matching two categories' keywords is deliberately not guessed"""
    text = normalize_text("хочу снять и купить квартиру")
    assert match_single_category(text) is None
