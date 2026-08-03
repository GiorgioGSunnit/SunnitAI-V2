"""Collapse the same offer arriving through more than one channel.

An aggregator relays quotes issued by insurers we also contact directly, so one
underlying offer can show up twice. Nothing is deleted — the duplicate is
*linked* to a primary and marked, so the results screen can show one row while
the audit trail still records that both channels returned it.

Direct wins over aggregator: when the same offer arrives both ways, the copy
that came straight from the insurer is the primary, because it is the one whose
purchase link and quote reference the customer will actually use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.quotes import NormalizedQuoteData


@dataclass
class DedupeResult:
    #: quote id -> id of the primary it duplicates.
    duplicate_to_primary: dict[str, str] = field(default_factory=dict)
    #: primary id -> the channels the same offer arrived through.
    channels_by_primary: dict[str, list[str]] = field(default_factory=dict)

    @property
    def duplicate_ids(self) -> set[str]:
        return set(self.duplicate_to_primary)

    def is_duplicate(self, quote_id: str) -> bool:
        return quote_id in self.duplicate_to_primary


def _rank(quote: NormalizedQuoteData) -> tuple[int, str]:
    """Lower sorts first. Direct channels outrank aggregator relays."""
    return (0 if quote.source_channel == "direct" else 1, quote.provider_id)


def deduplicate(quotes: list[tuple[str, NormalizedQuoteData]]) -> DedupeResult:
    """Group quotes by their offer signature and pick a primary per group."""
    groups: dict[tuple, list[tuple[str, NormalizedQuoteData]]] = {}
    for quote_id, quote in quotes:
        groups.setdefault(quote.dedupe_signature(), []).append((quote_id, quote))

    result = DedupeResult()
    for members in groups.values():
        ordered = sorted(members, key=lambda item: _rank(item[1]))
        primary_id, primary = ordered[0]
        channels = [primary.source_channel]
        for quote_id, quote in ordered[1:]:
            result.duplicate_to_primary[quote_id] = primary_id
            if quote.source_channel not in channels:
                channels.append(quote.source_channel)
        result.channels_by_primary[primary_id] = channels
    return result
