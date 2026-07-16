import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI

from app.common.logging import logger
from app.core.config import settings
from app.core.database import initialize_db
from app.dependencies.infrastructure import get_aiohttp_session
from app.utils.ticket_scheduler_func import run_ticket_scheduler


class AppLifecycle:
    """Manages initialization and cleanup of application components"""

    def __init__(self, app: FastAPI) -> None:
        """Bind the FastAPI app instance and reset the aiohttp session/scheduler slots"""
        self.app = app
        self.aiohttp_session: aiohttp.ClientSession | None = None
        self._scheduler: AsyncIOScheduler | None = None

    async def on_startup(self) -> None:
        """Run all startup procedures"""
        await self._initialize_core()
        self.aiohttp_session = await get_aiohttp_session()

        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            func=run_ticket_scheduler,
            trigger=IntervalTrigger(seconds=settings.SCHEDULER_TICK_SEC),
            id="ticket_scheduler",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "[ticket_scheduler] APScheduler started, interval=%ds",
            settings.SCHEDULER_TICK_SEC,
        )

    async def on_shutdown(self) -> None:
        """Gracefully stop all services and close connections"""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
        if self.aiohttp_session:
            await self.aiohttp_session.close()
        logger.info("Shutting down the APP")

    async def _initialize_core(self) -> None:
        """Initialize core DB components"""
        try:
            await initialize_db()
            logger.info("Core application components initialized.")
        except Exception as e:
            logger.critical(f"FATAL: Core initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Core application initialization failed: {e}") from e
