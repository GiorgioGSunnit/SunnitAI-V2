### per raffinare lo schema, usare in secondo momento

from enum import Enum


class OrganizationType(str, Enum):
    GOVERNMENT = "Government"
    PRIVATE = "Private"
    BANK = "Bank"
    COMMITTEE = "Committee"


class EventType(str, Enum):
    TENDER = "Tender"
    AUCTION = "Auction"
    BANKRUPTCY = "Bankruptcy"
    LIQUIDATION = "Liquidation"
    ESTABLISHMENT = "Establishment"


class EventStatus(str, Enum):
    OPEN = "Open"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"
    EXTENDED = "Extended"


class PersonRole(str, Enum):
    LIQUIDATOR = "Liquidator"
    SHAREHOLDER = "Shareholder"
    BOARD_MEMBER = "Board Member"
    JUDGE = "Judge"


class LegalDocType(str, Enum):
    DECREE = "Decree"
    LAW = "Law"
    MINISTERIAL_DECISION = "Ministerial Decision"
