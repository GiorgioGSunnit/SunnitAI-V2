"""The fixed cuts on the USER'S OWN TEXT, behind one test switch (flag USER_TEXT_UNCAPPED, default off).

Off (the default): every cut is as it was -- cap(n) is n, most(n) is n. On: the user's text is never cut by length or
by count on the case-first path -- cap(n) is None (a slice [:None] takes everything), most(n) is sys.maxsize. A switch
for tests: with it on a very long case can overflow a call's context (the model's window is 32,768 tokens,
section_fit.MODEL_CONTEXT_TOKENS); the call helper then retries with a smaller answer budget or the step fails and is
logged, which is what such a test is meant to show.

What it lifts (each site reads cap / most): the case text (defensive_generation._FACTS_CHARS, 12,000 characters), the
numbered sentences' length and number (issue_records.SENTENCE_CHARS 260, SENTENCES_MAX 80), the heads of the case given
to the grader (2,000), the search distill (6,000), the candidate choice (6,000), the fact issues (8,000), the
opposing-arguments check (4,000), the plan patches (6,000), the rulings link (1,500), the sentences quoted to the writer
(issue_records.WRITER_SENTENCE_CHARS 160), a question's length (intake.QUESITO_CHARS 300), the frame's text
(data/case_frame.json «max_text_chars» 40,000) and §2's cut of the user's sentences (lawyer_method.FACT_CHARS 220).
Not lifted: the text sent to the embedding model (defensive_generation, 4,000: that model's own input limit), §2's
number of facts (12 / 8 / 8) and every cut on text the model or the code wrote.
"""
import sys
from typing import Optional

from .feature_flags import enabled as _flag_enabled

FLAG = "USER_TEXT_UNCAPPED"


def enabled() -> bool:
    return _flag_enabled(FLAG)


def cap(n: int) -> Optional[int]:
    """A slice's end for the user's text: n, or None (no cut) with the switch on."""
    return None if enabled() else n


def most(n: int) -> int:
    """A count's or a length's ceiling for the user's text: n, or no ceiling with the switch on."""
    return sys.maxsize if enabled() else n
