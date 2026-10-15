"""On/off switches for retrieval and answer behaviour, read from the environment.

Read on every call (.env is loaded at startup, so a change needs a restart).
Only true/false values count (1/0, yes/no, on/off also accepted); anything
else falls back to the default, so a typo never switches a feature on.

  LEGACY_LAW_HINT_FALLBACK     true:  today's document lock (round 3, item 1)
  ISSUE_SPOTTING_ENABLED       false (item 2)
  APPLICABILITY_CHECK_ENABLED  false (item 3)
  STRATEGY_DRAFT_RULES_ENABLED false (item 4)
  SCOPED_DOTTRINA_ENABLED      false (item 5)
"""

import logging
import os

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    word = value.strip().lower()
    if word in _TRUE:
        return True
    if word in _FALSE:
        return False
    logger.warning("%s=%r is not true/false: using %s", name, value, default)
    return default
