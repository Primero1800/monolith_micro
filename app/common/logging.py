import inspect
import logging
import logging.handlers
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from app.core.config import settings

WHITE_COLOR = "\033[0m"
BLACK_COLOR = "\033[30m"
RED_COLOR = "\033[31m"
GREEN_COLOR = "\033[32m"
YELLOW_COLOR = "\033[33m"
BLUE_COLOR = "\033[34m"


class CustomFormatter(logging.Formatter):
    """Custom log formatter with terminal color support"""

    def __init__(self, fmt: str, datefmt: str):
        """Store the format and date format strings for the base Formatter"""
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Colorize the level name and message before formatting"""
        level_colors = {
            logging.INFO: GREEN_COLOR,
            logging.DEBUG: YELLOW_COLOR,
            logging.WARNING: BLUE_COLOR,
            logging.ERROR: RED_COLOR,
            logging.CRITICAL: RED_COLOR,
        }
        level_color = level_colors.get(record.levelno, WHITE_COLOR)
        record.levelname = f"{level_color}{record.levelname}{WHITE_COLOR}"
        if record.levelno in (logging.DEBUG, logging.WARNING):
            record.msg = f"{YELLOW_COLOR}{record.msg}"
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            record.msg = f"{RED_COLOR}{record.msg}"
        return super().format(record)


class FileFormatter(logging.Formatter):
    """Log formatter that removes terminal colors for file storage"""

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and strip ANSI color codes for file storage"""
        pattern = r"\033\[\d{1,2}m"
        message = super().format(record)
        return re.sub(pattern, "", message)


fmt = "%(asctime)s [%(levelname)s] %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)
logger.propagate = True

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter(fmt, datefmt))
logger.addHandler(handler)

root_dir = os.path.dirname(os.path.abspath(__file__))
app_root = os.path.dirname(root_dir)
file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(root_dir, "..", "logs", "app_log.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
)
file_handler.setFormatter(FileFormatter(fmt, datefmt))
logger.addHandler(file_handler)

logging.root.setLevel(settings.LOG_LEVEL)


T = TypeVar("T", bound=Awaitable[Any])


def log_decorator(
    level: int = logging.INFO,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log function entry, exit, and execution time"""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Wrap func to log its start/completion and execution time"""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            """Log entry, run func, then log completion (with timing at INFO level)"""
            calling_module = inspect.getmodule(func)
            module_file = (
                getattr(calling_module, "__file__", None) or func.__code__.co_filename
            )
            filename = os.path.relpath(module_file, start=app_root)
            _, lineno = inspect.getsourcelines(func)
            start_time = time.time()
            if level == logging.INFO:
                log_message(
                    f"{filename}:{lineno} :: {GREEN_COLOR}Started "
                    f"{WHITE_COLOR}-> {YELLOW_COLOR}def {func.__name__}() "
                    f"{WHITE_COLOR}:: {BLUE_COLOR}Timer {RED_COLOR}0.00 {BLUE_COLOR}seconds",
                    level,
                )
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                log_message(
                    f"{filename}:{lineno} :: {GREEN_COLOR}Completed "
                    f"{WHITE_COLOR}-> {YELLOW_COLOR}def {func.__name__}() "
                    f"{WHITE_COLOR}:: {BLUE_COLOR}Execution time {RED_COLOR}"
                    f"{execution_time:.2f} {BLUE_COLOR}seconds",
                    level,
                )
            else:
                log_message(
                    f"{filename}:{lineno} :: Started -> def {func.__name__}()", level
                )
                result = await func(*args, **kwargs)
                log_message(
                    f"{filename}:{lineno} :: Completed -> def {func.__name__}()", level
                )
            return result  # type: ignore

        return wrapper  # type: ignore

    return decorator


def log_message(message: str, level: int) -> None:
    """Dispatch message to the logger at the given level

    :raise:
        ValueError: if level is not one of the supported logging levels
    """
    if level == logging.INFO:
        logger.info(message)
    elif level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    else:
        raise ValueError(f"Invalid log level: {level}")
