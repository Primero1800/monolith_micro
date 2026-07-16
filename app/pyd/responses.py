from typing import Any

from pydantic import BaseModel


class HTTPExceptionResponse(BaseModel):
    """Schema for HTTP exception response"""

    detail: str
    headers: dict[str, Any] | None = None
    status_code: int
