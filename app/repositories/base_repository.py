from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Base repository for creating repositories for all database instances"""

    _session: AsyncSession

    def __init__(self, session: AsyncSession) -> None:
        """Store the session this repository issues its queries on"""
        self._session = session
