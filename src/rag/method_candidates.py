"""METHOD_CANDIDATES_ENABLED: articles always added as candidates for a case's
area and for its procedural posture, then filtered by the applicability check.

r13 (21 Oct): the lawyer's review of held-out drafts found the issue a case is
built around missing because nothing searched for it -- the general-part
article (an attempt, art. 56 c.p.) or the procedural one (the execution of an
administrative judgment, artt. 112-114 c.p.a.) that decides the case shares no
words with its facts. These lists hold the method articles of each area and
the procedural articles of each posture, so they are always among the
candidates; the applicability check (APPLICABILITY_CHECK_ENABLED, required)
then drops the ones the facts exclude.

DRAFT, pending the lawyer's validation: `python scripts/method_candidates_table.py`
prints them as a table (with --check, each article's heading in the corpus).
Areas and postures are the case frame's own labels (CASE_FRAME_ENABLED:
"areas", "posture_type"). An article is (number, code): a sigla from
defensive_generation._CODE_PREFIXES, or a law by name as _fetch_any resolves
it ("Codice del Consumo", "Legge sul Procedimento Amministrativo 2026").
"""

from typing import Dict, Iterable, List, Tuple

Ref = Tuple[str, str]

# L. 241/1990 by the document's own name: prod holds two copies, and "l. 241/1990"
# resolves (by number and year) to "Legge 7 agosto 1990, n. 241 ...", which has
# only art. 21-septies (checked 21 Oct).
L241 = "Legge sul Procedimento Amministrativo 2026"


def _refs(code: str, numbers: str) -> List[Ref]:
    return [(n, code) for n in numbers.split()]


# area -> (label, articles)
AREA_ARTICLES: Dict[str, Tuple[str, List[Ref]]] = {
    "penale": ("Diritto penale, parte generale", _refs(
        "c.p.", "40 41 42 43 47 49 52 54 56 61 62 62-bis 81 110 131-bis 133")),
    "contratti": ("Obbligazioni e contratti", _refs(
        "c.c.", "1175 1176 1218 1223 1227 1337 1362 1366 1375 1418 1453 1455 1460 2697 2946")),
    "responsabilita_civile": ("Responsabilità extracontrattuale", _refs(
        "c.c.", "2043 2046 2049 2050 2051 2052 2054 2055 2056 2059 1227 2697 2947")),
    "locazione": ("Locazione", _refs("c.c.", "1571 1575 1576 1578 1587 1590 1591 1453 1455 1460")
                  + _refs("c.p.c.", "447-bis 657 658")),
    "successioni": ("Successioni", _refs(
        "c.c.", "456 457 459 460 470 474 475 476 480 484 485 490 519 521 752 754")),
    "diritti_reali": ("Diritti reali, possesso, condominio", _refs(
        "c.c.", "832 844 948 949 1027 1031 1051 1079 1102 1117 1140 1158 1168 1170")),
    "famiglia": ("Famiglia", _refs(
        "c.c.", "143 147 151 156 177 179 316 316-bis 337-bis 337-ter 337-sexies 433 438")),
    "consumo": ("Consumo", _refs("Codice del Consumo", "3 33 34 36 49 52 128 129 130 132 133")),
    "amministrativo": ("Diritto amministrativo", _refs(L241, "2 3 7 10-bis 21-septies 21-octies 21-nonies")
                       + _refs("c.p.a.", "29 30 34")),
}

# posture -> (label, articles)
POSTURE_ARTICLES: Dict[str, Tuple[str, List[Ref]]] = {
    "precontenzioso": ("Controversia civile prima del giudizio",
                       _refs("c.c.", "1219 1454 2943") + _refs("c.p.c.", "633 696-bis")),
    "giudizio_civile_primo_grado": ("Giudizio civile di primo grado in corso",
                                    _refs("c.p.c.", "115 116 163 167 183") + _refs("c.c.", "2697")),
    "sentenza_civile_primo_grado": ("Sentenza civile di primo grado (appello)",
                                    _refs("c.p.c.", "323 325 327 339 342 345 348-bis")),
    "sentenza_civile_appello": ("Sentenza civile d'appello (cassazione)", _refs("c.p.c.", "325 327 360 366 369")),
    "esecuzione_civile": ("Esecuzione forzata civile", _refs("c.p.c.", "474 480 615 617")),
    "indagini_preliminari": ("Indagini preliminari", _refs(
        "c.p.p.", "335 369 369-bis 408 409 410 415-bis 438 444 464-bis")),
    "giudizio_penale_primo_grado": ("Giudizio penale di primo grado in corso", _refs(
        "c.p.p.", "129 438 444 464-bis 530 533")),
    "condanna_primo_grado": ("Condanna in primo grado (appello)", _refs(
        "c.p.p.", "568 581 585 591 593 595 597 603 605")),
    "sentenza_penale_appello": ("Sentenza penale d'appello (cassazione)", _refs(
        "c.p.p.", "569 585 591 606 609 610 620 623")),
    "provvedimento_amministrativo": ("Provvedimento amministrativo da impugnare", _refs("c.p.a.", "29 30 41 55")),
    "silenzio_amministrazione": ("Silenzio o inerzia dell'amministrazione",
                                 _refs("c.p.a.", "31 117") + _refs(L241, "2")),
    "giudicato_amministrativo_non_eseguito": ("Giudicato amministrativo non eseguito (ottemperanza)",
                                              _refs("c.p.a.", "112 113 114 117")),
}

# The analysis' own "area", for a case without a frame (or a frame with no areas).
_ANALYSIS_AREA_TO_AREAS = {"penale": ["penale"], "amministrativo": ["amministrativo"]}


def candidate_refs(areas: Iterable[str], posture_type: str = "") -> List[Ref]:
    """The articles for these areas and this posture, deduplicated, area lists
    first in the given order, then the posture's."""
    refs: List[Ref] = []
    for area in areas:
        refs += AREA_ARTICLES.get(area, ("", []))[1]
    refs += POSTURE_ARTICLES.get(posture_type, ("", []))[1]
    return list(dict.fromkeys(refs))


def areas_for(frame_areas: Iterable[str], analysis_area: str) -> List[str]:
    """The frame's areas when it gave any, else the analysis' own area mapped."""
    areas = [a for a in frame_areas if a in AREA_ARTICLES]
    return areas or _ANALYSIS_AREA_TO_AREAS.get((analysis_area or "").strip().lower(), [])


def table_rows() -> List[Tuple[str, str, str, str]]:
    """(kind, key, label, articles) per list, for the lawyer's table."""
    def joined(refs: List[Ref]) -> str:
        by_code: Dict[str, List[str]] = {}
        for number, code in refs:
            by_code.setdefault(code, []).append(number)
        return "; ".join(f"{code} {', '.join(numbers)}" for code, numbers in by_code.items())
    return ([("area", key, label, joined(refs)) for key, (label, refs) in AREA_ARTICLES.items()]
            + [("fase", key, label, joined(refs)) for key, (label, refs) in POSTURE_ARTICLES.items()])
