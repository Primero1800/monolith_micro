from dataclasses import dataclass

from app.adapters.ai_client import AIClientAbstract
from app.uow import UnitOfWork


@dataclass
class BaseDeps:
    """Infrastructure dependencies shared across all services"""

    uow_factory: UnitOfWork
    ai_client: AIClientAbstract


class BaseService:
    """Base service providing shared infrastructure dependencies to subclasses"""

    def __init__(
        self,
        base_deps: BaseDeps,
        uow: UnitOfWork | None = None,
    ) -> None:
        """Unpack shared infrastructure dependencies and bind the optional UoW session"""
        self.uow_factory = base_deps.uow_factory
        self.uow = uow
        self.ai_client = base_deps.ai_client
