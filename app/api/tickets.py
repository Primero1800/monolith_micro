import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.common.exceptions import IntegrityDataException
from app.common.logging import log_decorator
from app.dependencies.services import (
    get_ticket_service,
    get_ticket_service_without_session,
)
from app.pyd.tickets import (
    TicketAdminResponse,
    TicketAnalyzeRequest,
    TicketAnalyzeResponse,
    TicketDraftResponse,
)
from app.services.ticket_service import TicketService

router = APIRouter(
    prefix="/api/v1/tickets",
    tags=["Tickets"],
)

admin_router = APIRouter(
    prefix="/api/v1/admin/tickets",
    tags=["Admin"],
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
    except IntegrityDataException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        ) from exc


@router.post(
    "",
    response_model=TicketDraftResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@log_decorator(level=logging.INFO)
async def create_ticket(
    payload: TicketAnalyzeRequest,
    ticket_service: Annotated[TicketService, Depends(get_ticket_service)],
) -> Any:
    """Queue a ticket for asynchronous classification; poll GET /{ticket_id} for the result"""
    try:
        return await ticket_service.create_draft(payload.text)
    except IntegrityDataException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        ) from exc


@router.get(
    "/{ticket_id}",
    response_model=TicketAnalyzeResponse,
    status_code=status.HTTP_200_OK,
)
@log_decorator(level=logging.INFO)
async def get_ticket(
    ticket_id: int,
    ticket_service: Annotated[TicketService, Depends(get_ticket_service)],
) -> Any:
    """Fetch a ticket's public status and result by id"""
    ticket = await ticket_service.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found"
        )
    return ticket


@admin_router.get(
    "/{ticket_id}",
    response_model=TicketAdminResponse,
    status_code=status.HTTP_200_OK,
)
@log_decorator(level=logging.INFO)
async def get_ticket_admin(
    ticket_id: int,
    ticket_service: Annotated[TicketService, Depends(get_ticket_service)],
) -> Any:
    """Fetch a ticket's full record by id, including internal technical fields

    No authentication/authorization — deliberate simplification for this
    prototype, see README for the security caveat.
    """
    ticket = await ticket_service.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found"
        )
    return ticket
