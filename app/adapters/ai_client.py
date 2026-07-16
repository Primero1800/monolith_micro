import abc
import logging
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import aiohttp
from aiohttp import ClientSession

from app.common.logging import log_decorator, logger
from app.core.aiohttp_exception_handler import external_request_exception_handler
from app.core.config import settings


class Message(TypedDict):
    """Single chat message with a role and text content"""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ChatResult:
    """Chat completion result: text content plus token usage for DB tracking"""

    content: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AIClientAbstract(abc.ABC):
    """Abstract base class for AI client implementations"""

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        """Send messages to the LLM and return the parsed completion"""
        pass


class UniversalChatClient(AIClientAbstract):
    """OpenAI-compatible `/chat/completions` client.

    Mistral, DeepSeek and OpenAI all mirror the same request/response shape, so
    switching provider is just a matter of pointing LLM_BASE_URL/LLM_API_KEY/LLM_MODEL
    at a different one in `.env` — no code change needed.
    """

    def __init__(self, aiohttp_session: aiohttp.ClientSession) -> None:
        """Bind the shared aiohttp session and read connection settings from `.env`"""
        self.session: ClientSession = aiohttp_session
        self._base_url = settings.LLM_BASE_URL.rstrip("/")
        self._model = settings.LLM_MODEL
        self._timeout = aiohttp.ClientTimeout(total=settings.LLM_TIMEOUT_SEC)
        self._headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json",
        }

    @log_decorator(level=logging.DEBUG)
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        """Send a chat/completions request to the configured OpenAI-compatible provider

        :return:
            ChatResult with empty content and zeroed token counts if the request failed
            (the network layer already logged the failure and returned None)
        """
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        result = await self.__post("/chat/completions", payload)
        if result is None:
            return ChatResult(
                content=None, prompt_tokens=0, completion_tokens=0, total_tokens=0
            )

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Unexpected LLM response shape: {result}")
            content = None

        usage = result.get("usage") or {}
        return ChatResult(
            content=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    @external_request_exception_handler(is_raise=False)
    async def __post(self, path: str, payload: dict[str, Any]) -> Any:
        """POST payload to the LLM endpoint; returns None on request failure instead of raising"""
        url = f"{self._base_url}{path}"
        async with self.session.post(
            url, json=payload, headers=self._headers, timeout=self._timeout
        ) as response:
            response.raise_for_status()
            return await response.json()
