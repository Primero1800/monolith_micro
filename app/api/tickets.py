import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.common.exceptions import IntegrityDataException
from app.common.logging import log_decorator
from app.dependencies.services import get_ticket_service_without_session
from app.pyd.tickets import TicketAnalyzeRequest, TicketAnalyzeResponse
from app.services.ticket_service import TicketService

router = APIRouter(
    prefix="/api/v1/tickets",
    tags=["Tickets"],
)


@router.post(
    "/analyze",
    response_model=TicketAnalyzeResponse,
    status_code=status.HTTP_200_OK,
)
@log_decorator(level=logging.INFO)
async def analyze_ticket(
    payload: TicketAnalyzeRequest,
    ticket_service: Annotated[
        TicketService, Depends(get_ticket_service_without_session)
    ],
) -> Any:
    """Classify and summarize a support ticket synchronously"""
    try:
        return await ticket_service.analyze(payload.text)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Classification service not implemented yet",
        ) from exc
    except IntegrityDataException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        ) from exc
