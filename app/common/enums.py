from enum import Enum


class TicketStatusEnum(str, Enum):
    """Ticket processing lifecycle status"""

    DRAFT = "draft"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class TicketCategoryEnum(str, Enum):
    """Ticket category — real estate agency contact center request types"""

    RENT = "rent"
    SALE = "sale"
    VIEWING = "viewing"
    CONSULTATION = "consultation"
    COMPLAINT = "complaint"
    OTHER = "other"


class TicketPriorityEnum(str, Enum):
    """Ticket priority level"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
