from app.dependencies.services import get_base_deps_standalone
from app.services.ticket_scheduler_service import TicketSchedulerService


async def run_ticket_scheduler() -> None:
    """Build fresh deps and run one tick of the ticket scheduler

    Called by APScheduler on its configured interval.
    """
    base_deps = await get_base_deps_standalone()
    await TicketSchedulerService(base_deps).run()
