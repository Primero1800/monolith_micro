import logging

from app.common.logging import log_decorator
from app.services.base import BaseService


class TicketService(BaseService):
    """Orchestrates ticket classification: dedup check, regex fast-path, LLM fallback"""

    @log_decorator(level=logging.INFO)
    async def analyze(self, text: str) -> None:
        """Classify and summarize a ticket text, chaining the classification steps"""
        raise NotImplementedError
