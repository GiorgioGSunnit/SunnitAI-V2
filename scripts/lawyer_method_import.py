#!/usr/bin/env python3
"""v70: the lawyer's method (docs/lawyer/metodo.md) into data/lawyer_method.json, verbatim -- his RUOLO, his FASI 1-10,
the section titles of his STRUTTURA DELL'OUTPUT (the parenthesis of a line kept as the section's «note»), his STILE,
his REGOLE DI ONESTÀ, his CONTROLLO FINALE (the coherence checklist) and one module per ambito («Moduli per ambito»).
His text is only re-cut: the bullets' marks and the invisible word joiners of his file go, the words stay. The
tables his text does not cover (reforms, litisconsorzio, stragiudiziale, circumstances, riti alternativi) are kept from
the data file as they are -- Code's drafts, listed in docs/copy_review.md.

  python scripts/lawyer_method_import.py               # rewrites data/lawyer_method.json
  python scripts/lawyer_method_import.py --check       # exit 1 when the data file is out of sync with metodo.md
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "lawyer" / "metodo.md"
TARGET = ROOT / "data" / "lawyer_method.json"
# the code's key for each line of STRUTTURA DELL'OUTPUT, in his order
SECTION_KEYS = ("sintesi", "fatti", "questione", "norme", "orientamenti", "soluzioni", "obiezioni", "processuali",
                "conclusioni", "verificare")
# the code's key for each «MODULO: …» heading
MODULE_KEYS = {
    "DIRITTO PENALE SOSTANZIALE": "penale_sostanziale",
    "PROCEDURA PENALE": "procedura_penale",
    "DIRITTO CIVILE (CONTRATTI, OBBLIGAZIONI, RESPONSABILITÀ)": "civile",
    "DIRITTI REALI, SUCCESSIONI, FAMIGLIA": "diritti_reali_famiglia",
    "PROCEDURA CIVILE": "procedura_civile",
    "DIRITTO AMMINISTRATIVO": "amministrativo",
    "DIRITTO COSTITUZIONALE": "costituzionale",
    "DIRITTO DEL LAVORO": "lavoro",
    "DIRITTO COMMERCIALE, SOCIETARIO, CRISI D'IMPRESA": "commerciale",
    "DIRITTO TRIBUTARIO": "tributario",
    "DIRITTO UE E INTERNAZIONALE PRIVATO": "ue",
}
# which modules a posture loads: by its rito, then by the frame's areas (pool_gates.AREA_CODES' names) -- his «Uso»:
# the general prompt, plus every module of the case's ambiti (his «Più ambiti»: e.g. lavoro + procedura civile)
MODULES_BY_RITO = {
    "penale": ["penale_sostanziale", "procedura_penale"],
    "penale_gdp": ["penale_sostanziale", "procedura_penale"],
    "civile": ["procedura_civile"],
    "famiglia": ["diritti_reali_famiglia", "procedura_civile"],
    "esecuzione": ["procedura_civile"],
    "lavoro": ["lavoro", "procedura_civile"],
    "amministrativo": ["amministrativo"],
}
MODULES_BY_AREA = {
    "civile": ["civile"], "contratti": ["civile"], "responsabilita_civile": ["civile"], "consumo": ["civile"],
    "locazione": ["civile"], "diritti_reali": ["diritti_reali_famiglia"], "successioni": ["diritti_reali_famiglia"],
    "famiglia": ["diritti_reali_famiglia"], "lavoro": ["lavoro"], "tributario": ["tributario"],
    "amministrativo": ["amministrativo"], "societario": ["commerciale"], "commerciale": ["commerciale"],
    "crisi": ["commerciale"], "fallimentare": ["commerciale"], "costituzionale": ["costituzionale"],
    "ue": ["ue"], "internazionale": ["ue"],
}
# a civil posture with no area naming its matter: the general civil module
CIVIL_DEFAULT = ["civile"]

_JOINER = "⁠"
_NUMBERED_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$")


def clean(line: str) -> str:
    """His line without the file's marks: the word joiners, a bullet, the spaces around them."""
    line = line.replace(_JOINER, "").strip()
    return re.sub(r"^[•·]\s*", "", line).strip()


def _between(lines: List[str], start: str, stops: tuple) -> List[str]:
    """The cleaned, non-empty lines after the first line starting with `start`, up to one starting with a stop."""
    out, inside = [], False
    for raw in lines:
        line = clean(raw)
        if not inside:
            inside = line.startswith(start)
            continue
        if any(line.startswith(s) for s in stops):
            break
        if line:
            out.append(line)
    return out


def parse(text: str) -> Dict[str, Any]:
    lines = text.splitlines()
    ruolo = _between(lines, "RUOLO", ("CASO",))
    fasi = []
    current = None
    for raw in lines:
        line = clean(raw)
        m = re.match(r"^FASE (\d+) - (.+)$", line)
        if m:
            current = {"n": int(m.group(1)), "title": m.group(2).strip(), "lines": []}
            fasi.append(current)
            continue
        if line.startswith("STRUTTURA DELL'OUTPUT"):
            break
        if current is not None and line:
            current["lines"].append(line)
    sections = []
    for line in _between(lines, "STRUTTURA DELL'OUTPUT", ("STILE",)):
        m = _NUMBERED_RE.match(line)
        if not m:
            continue
        body = m.group(2).rstrip(".").strip()
        note = ""
        p = re.search(r"\s*\(([^()]*)\)\s*$", body)
        if p:
            note, body = p.group(1).strip(), body[:p.start()].strip()
        sections.append({"key": SECTION_KEYS[len(sections)], "title": body, **({"note": note} if note else {})})
    stile = _between(lines, "STILE", ("REGOLE DI ONESTÀ",))
    onesta = _between(lines, "REGOLE DI ONESTÀ", ("CONTROLLO FINALE",))
    controllo = [re.sub(r"^\[\s*\]\s*", "", x) for x in _between(lines, "CONTROLLO FINALE", ("Come usarlo",))]
    modules: Dict[str, Dict[str, Any]] = {}
    current = None
    started = False
    for raw in lines:
        line = clean(raw)
        if line.startswith("Moduli per ambito"):
            started = True
            continue
        if not started:
            continue
        if line.startswith("Note d"):
            break
        if line.startswith("MODULO:"):
            name = line[len("MODULO:"):].strip()
            current = {"title": line, "lines": []}
            modules[MODULE_KEYS[name]] = current
            continue
        if current is None or not line or (_NUMBERED_RE.match(line) and not line.startswith("Art.")):
            continue
        current["lines"].append(line)
    return {"ruolo": ruolo[0] if ruolo else "", "fasi": fasi, "sections": sections, "stile": " ".join(stile),
            "honesty_rules": onesta, "controllo_finale": controllo, "modules": modules}


ABOUT = ("v70: the lawyer's method as protocol (src/rag/lawyer_method.py, LAWYER_METHOD). From «ruolo» to «modules» "
         "the lawyer's own words, verbatim from docs/lawyer/metodo.md (scripts/lawyer_method_import.py: re-run it when "
         "he changes his text; tests/test_v70.py checks the sync): ruolo, fasi (his FASI 1-10), sections (the titles of "
         "his STRUTTURA DELL'OUTPUT, each line's parenthesis as «note»; the keys are the code's), stile, honesty_rules "
         "(his REGOLE DI ONESTÀ, in every writer call with his FASE 7, his STILE and the modules), controllo_finale (his "
         "CONTROLLO FINALE: the coherence checklist), modules (one per ambito, «MODULO: …»; loaded by modules_by_rito "
         "and modules_by_area -- the code's choice). The tables after them are Code's drafts, not his (listed in "
         "docs/copy_review.md for the PO): reforms (ratione temporis), litisconsorzio (by matter), stragiudiziale (by "
         "family and role), circumstances (aggravanti and attenuanti as fattispecie, each element graded against the "
         "case's sentences; 62 n. 6 with both alternatives of the article's text), riti_alternativi (the imputato's "
         "subordinate line, gated on the charged offence's pena edittale).")
DRAFT_KEYS = ("reforms", "litisconsorzio", "stragiudiziale", "circumstances", "riti_alternativi")


def build(existing: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = {"_about": ABOUT, **parsed, "modules_by_rito": MODULES_BY_RITO, "modules_by_area": MODULES_BY_AREA,
           "civil_default": CIVIL_DEFAULT}
    for key in DRAFT_KEYS:
        if key in existing:
            out[key] = existing[key]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    existing = json.loads(TARGET.read_text(encoding="utf-8")) if TARGET.is_file() else {}
    parsed = parse(SOURCE.read_text(encoding="utf-8"))
    if len(parsed["sections"]) != len(SECTION_KEYS) or len(parsed["modules"]) != len(MODULE_KEYS):
        print(f"metodo.md: {len(parsed['sections'])} sections, {len(parsed['modules'])} modules", file=sys.stderr)
        return 2
    data = build(existing, parsed)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        same = all(existing.get(k) == data[k] for k in parsed)
        print("in sync" if same else "out of sync: run scripts/lawyer_method_import.py")
        return 0 if same else 1
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}: {len(data['sections'])} sections, {len(data['fasi'])} fasi, {len(data['modules'])} modules, "
          f"{len(data['honesty_rules'])} honesty rules, {len(data['controllo_finale'])} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
