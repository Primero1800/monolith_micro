from app.common.enums import TicketCategoryEnum

FAST_PATH_MAX_LENGTH = 50

_CATEGORY_KEYWORDS: dict[TicketCategoryEnum, tuple[str, ...]] = {
    TicketCategoryEnum.RENT: ("аренда", "снять", "сдать"),
    TicketCategoryEnum.SALE: ("продажа", "купить", "продать"),
    TicketCategoryEnum.VIEWING: ("просмотр", "посмотреть"),
    TicketCategoryEnum.CONSULTATION: ("документ", "ипотек", "юридич"),
    TicketCategoryEnum.COMPLAINT: (
        "жалоба",
        "недоволен",
        "ужасно",
        "мошенни",
        "верните",
    ),
}


def match_single_category(normalized_text: str) -> TicketCategoryEnum | None:
    """Return the one category whose keywords match, or None if zero or several match

    Deliberately conservative: only a single, unambiguous match counts —
    conflicting or absent signals fall through to the LLM instead of guessing.
    """
    matches = {
        category
        for category, keywords in _CATEGORY_KEYWORDS.items()
        if any(keyword in normalized_text for keyword in keywords)
    }
    if len(matches) == 1:
        return matches.pop()
    return None
