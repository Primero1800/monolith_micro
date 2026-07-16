from app.services.prompt_service import PromptService


def test_get_ticket_classification_prompt_mentions_all_categories() -> None:
    """The system prompt lists every TicketCategoryEnum value so the LLM can pick one"""
    prompt = PromptService.get_ticket_classification_prompt()

    for category in ("rent", "sale", "viewing", "consultation", "complaint", "other"):
        assert category in prompt


def test_get_ticket_classification_prompt_is_stable() -> None:
    """The prompt is a fixed constant, not regenerated per call"""
    assert (
        PromptService.get_ticket_classification_prompt()
        == PromptService.get_ticket_classification_prompt()
    )
