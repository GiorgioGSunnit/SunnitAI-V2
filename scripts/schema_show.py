"""v79.11 (B): the lawyer's review surface for the element schemas (src/rag/element_schemas.py) -- offline: it reads
data/article_schemas.jsonl (ARTICLE_SCHEMAS_FILE) and data/article_schemas_reviewed.jsonl (ARTICLE_SCHEMAS_REVIEWED_FILE),
writes only the latter (--edit, --import-notes), never Neo4j, never a model. Run from the app folder:

    venv/bin/python scripts/schema_show.py --article "582 c.p."            # the schema as readable tables
    venv/bin/python scripts/schema_show.py --article "582 c.p." --markdown # the same, in markdown
    venv/bin/python scripts/schema_show.py --edit corrections.yaml         # the lawyer's corrections -> reviewed line
    venv/bin/python scripts/schema_show.py --import-notes                  # docs/lawyer/note_inquadramento.md -> overlay
    venv/bin/python scripts/schema_show.py --stale | --coverage | --quality
    venv/bin/python scripts/schema_show.py --top 50 --out schemi_da_leggere.md

--article (--doc to choose the document): header (label, rubrica, tipo, roles with their conditions, rapporto, source
generated/reviewed, stale, version, model, date), the elements, the logic as an outline (TUTTI / ALMENO UNO / NON / SE …
ALLORA …), eccezioni, conseguenza, each comma speciale, collegati, the checks raised, the lawyer's notes («[nota
dell'avvocato]»). A reviewed schema is shown when there is one, else the generated one.
--edit FILE.yaml: article (label), reviewer (required), doc, date (default today), and any of tipo, destinatario,
legittimato, soggetto_passivo, rapporto, logic, eccezioni, conseguenza, commi_speciali, collegati (replaced), elementi (the
whole list replaced), elementi_patch ({E2: {kind: …}, E5: null}: changed, removed; «co2.E1» for a comma speciale's),
add_elementi (a list), note (the reviewer's own words). It starts from the reviewed line if there is one, else the
generated line (by doc and article, else by label), and writes the reviewed line (reviewed: true, reviewer, review_date,
source_hash of the line it started from, generated_schema = the model's schema kept verbatim, corrections, origin per
element «modello»/«avvocato», the notes kept, the checks re-run against the text). It never refuses to save -- the
lawyer's word -- but prints a warning for each check raised. A stale review (the text changed after it) keeps its old
source_hash, so the runtime keeps the text path, unless the YAML says «testo_attuale: true»: the lawyer reviewed it
against the text now -- the generated line's current_source_hash / current_text (article_cards_bulk records them).
--import-notes [FILE]: the lawyer's «note di inquadramento» (src/rag/article_notes.parse) into the overlay, one line per
article (62 n. 2 and n. 6 both under «62 c.p.», with their numero), a reviewed schema kept; an earlier import of the same
file replaced (idempotent). A notes-only line has reviewed: false and no schema.
--stale: reviewed lines whose source_hash is no longer the article's text's (the generated line's current_source_hash,
which article_cards_bulk --schemas records for a reviewed label, else its source_hash): the text changed.
--coverage: by code, articles with a usable schema / an unusable one / carded without one (--cards).
--top N --out FILE: the N most used articles (article_cards_bulk.frequency over --runs; the generated file's order when
there are none), each as --markdown with the official text first. --quality: the measures of element_schemas.
quality_report over the reviewed lines.
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
NOTES_FILE = os.path.join("docs", "lawyer", "note_inquadramento.md")
RUNS = os.path.join("diagnose_long_runs", "*", "cases", "*.json")
FIELDS = ("tipo", "destinatario", "legittimato", "soggetto_passivo", "rapporto", "logic", "eccezioni", "conseguenza",
          "commi_speciali", "collegati")
EDIT_KEYS = {"article", "doc", "reviewer", "date", "elementi", "elementi_patch", "add_elementi", "note", "testo_attuale",
             *FIELDS}
GROUNDING_IT = {"explicit": "esplicito", "inferred": "implicito"}
NOTE_MARK = "«[nota dell'avvocato]»"


def _offline_env() -> None:
    """The src.rag imports build their clients at import time: a placeholder key lets them load offline (no call is
    made from here)."""
    for name, value in (("LLM_API_KEY", "offline"), ("OPENAI_API_KEY", "offline"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "offline")):
        os.environ.setdefault(name, value)


def _es():
    from src.rag import element_schemas
    return element_schemas


def _bulk():
    spec = importlib.util.spec_from_file_location("article_cards_bulk", os.path.join(ROOT, "scripts",
                                                                                     "article_cards_bulk.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(label: str) -> Tuple[str, str]:
    """(number, code) of a label («582 c.p.» -> ("582", "c.p.")), the code read as its sigla."""
    es = _es()
    norm = es.norm_label(label)
    number, _, code = norm.partition(" ")
    return number, code


# --- the lines ---------------------------------------------------------------------------------------------------------

def lines_for(label: str, doc: Optional[str] = None) -> Dict[str, Any]:
    """{"label", "generated": the generated line (by doc and article, else by label) or None, "overlay": the overlay's
    line or None}."""
    es = _es()
    key = es.norm_label(label)
    number, _ = _split(label)
    gen = es.generated().get((str(doc), number.lower())) if doc else None
    if gen is None:
        gen = es.generated_by_label().get(key)
    return {"label": key, "generated": gen, "overlay": es.reviewed().get(key)}


def is_reviewed(line: Optional[Dict[str, Any]]) -> bool:
    return bool(line) and line.get("reviewed") is True and isinstance(line.get("schema"), dict)


def current_hash(gen: Optional[Dict[str, Any]]) -> Optional[str]:
    """v79.11: the article's text's hash now as the generated file knows it -- current_source_hash (the generator
    records it beside a reviewed label's line, never made again) else the line's source_hash."""
    return (gen or {}).get("current_source_hash") or (gen or {}).get("source_hash")


def current_text(gen: Optional[Dict[str, Any]]) -> str:
    return (gen or {}).get("current_text") or (gen or {}).get("text") or ""


def is_stale(overlay: Optional[Dict[str, Any]], gen: Optional[Dict[str, Any]]) -> bool:
    """A reviewed line whose source_hash is no longer the article's text's (current_hash: the text changed after the
    review)."""
    return is_reviewed(overlay) and bool(current_hash(gen)) and overlay.get("source_hash") != current_hash(gen)


def view(label: str, doc: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """What --article shows: the reviewed line if any, else the generated one, with the overlay's notes; None when
    neither has a schema nor a note."""
    found = lines_for(label, doc)
    gen, overlay = found["generated"], found["overlay"]
    notes = list((overlay or {}).get("note_di_inquadramento") or [])
    if is_reviewed(overlay):
        line, source = overlay, "reviewed"
    elif gen:
        line, source = gen, "generated"
    elif notes:
        line, source = {"label": found["label"]}, "notes"
    else:
        return None
    stale_now = is_stale(overlay, gen)
    text = (current_text(gen) if stale_now else "") or line.get("text") or (gen or {}).get("text") or ""
    return {"label": found["label"], "line": line, "source": source, "stale": stale_now,
            "text": text, "notes": notes, "generated": gen, "overlay": overlay}


# --- the rendering -----------------------------------------------------------------------------------------------------

def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " / ".join(_cell(v) for v in value)
    return " ".join(str(value).split())


def table(headers: List[str], rows: List[List[Any]], markdown: bool) -> List[str]:
    rows = [[_cell(c) for c in r] for r in rows]
    if markdown:
        esc = lambda c: c.replace("|", "\\|")       # noqa: E731
        return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + \
            ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()   # noqa: E731
    return [line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows]


def roles(entries: Any) -> str:
    out = []
    for e in entries or []:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, dict):
            cond = " ".join(str(e.get("condizione") or "").split())
            out.append(f"{e.get('ruolo')}" + (f" ({cond})" if cond else ""))
    return "; ".join(out) or "—"


def outline(node: Any, texts: Dict[str, str], depth: int = 0, markdown: bool = False) -> List[str]:
    """The logic tree as an indented outline: TUTTI / ALMENO UNO / NON / SE … ALLORA … ALTRIMENTI …"""
    pad = "  " * depth + ("- " if markdown else "")
    if depth > 40:
        return [pad + "…"]
    if not isinstance(node, dict):
        return [pad + f"(nodo non valido: {node!r})"]
    if "ref" in node and "op" not in node:
        rid = str(node.get("ref"))
        return [pad + rid + (f" — {texts[rid]}" if rid in texts else " — (elemento mancante)")]
    op = node.get("op")
    if op in ("all_of", "any_of"):
        out = [pad + ("TUTTI:" if op == "all_of" else "ALMENO UNO:")]
        for a in node.get("args") or []:
            out += outline(a, texts, depth + 1, markdown)
        return out
    if op == "not":
        return [pad + "NON:"] + outline(node.get("arg"), texts, depth + 1, markdown)
    if op == "conditional":
        out = [pad + "SE:"] + outline(node.get("if"), texts, depth + 1, markdown)
        out += [pad + "ALLORA:"] + outline(node.get("then"), texts, depth + 1, markdown)
        if node.get("else") is not None:
            out += [pad + "ALTRIMENTI:"] + outline(node.get("else"), texts, depth + 1, markdown)
        return out
    return [pad + f"(op sconosciuto: {op!r})"]


def _heading(title: str, markdown: bool, level: int = 3) -> List[str]:
    return ["", "#" * level + " " + title, ""] if markdown else ["", title.upper(), ""]


def _grounding(el: Dict[str, Any]) -> str:
    g = GROUNDING_IT.get(el.get("grounding"), str(el.get("grounding") or ""))
    why = " ".join(str(el.get("spiegazione") or "").split())
    return g + (f": {why}" if why else "")


def _schema_body(schema: Dict[str, Any], markdown: bool, origin: Dict[str, str], scope: str = "",
                 level: int = 3) -> List[str]:
    L: List[str] = []
    elementi = [e for e in schema.get("elementi") or [] if isinstance(e, dict)]
    texts = {str(e.get("id")): " ".join(str(e.get("testo") or "").split())[:90] for e in elementi}
    with_origin = bool(origin)
    rows = []
    for e in elementi:
        kind = str(e.get("kind") or "") + (f" ({e.get('kind_note')})" if e.get("kind_note") else "")
        joined = (e.get("connective") or "") + (f": {' + '.join(e.get('parti') or [])}" if e.get("parti") else "")
        row = [e.get("id"), e.get("testo"), kind, e.get("comma"), _grounding(e), e.get("branches") or [], joined]
        if with_origin:
            eid = f"co{scope}.{e.get('id')}" if scope else str(e.get("id"))
            row.append(origin.get(eid, ""))
        rows.append(row)
    L += _heading("Elementi", markdown, level)
    headers = ["ID", "Testo", "Tipo di elemento", "Comma", "Fondamento", "Alternative", "Connettivo/parti"]
    L += table(headers + (["Origine"] if with_origin else []), rows, markdown) if rows else ["(nessun elemento)"]
    L += _heading("Logica", markdown, level)
    L += outline(schema.get("logic"), texts, 0, markdown) if schema.get("logic") is not None else ["(nessuna logica)"]
    eccezioni = [x for x in schema.get("eccezioni") or [] if isinstance(x, dict)]
    L += _heading("Eccezioni", markdown, level)
    L += table(["Testo", "Comma", "Fondamento"], [[x.get("testo"), x.get("comma"), _grounding(x)] for x in eccezioni],
               markdown) if eccezioni else ["(nessuna)"]
    cons = schema.get("conseguenza")
    if isinstance(cons, dict) and cons:
        L += _heading("Conseguenza", markdown, level)
        L.append(f"{_cell(cons.get('testo'))} ({cons.get('tipo') or '—'}, comma {cons.get('comma') or '—'})")
    return L


def render(v: Dict[str, Any], markdown: bool = False, text_first: bool = False, level: int = 2) -> str:
    """The schema of a view (view()) as readable tables, Italian headings; markdown with --markdown."""
    es = _es()
    line = v["line"]
    schema = line.get("schema") if isinstance(line.get("schema"), dict) else {}
    title = f"Art. {v['label']}" + (f" — {line.get('rubrica')}" if line.get("rubrica") else "")
    L = ([f"{'#' * level} {title}", ""] if markdown else [title, "=" * len(title), ""])
    if text_first:
        L += _heading("Testo ufficiale", markdown, level + 1)
        body = v.get("text") or "(testo non disponibile)"
        L += [f"> {x}" if x.strip() else ">" for x in body.split("\n")] if markdown else body.split("\n")
        L.append("")
    source = {"reviewed": f"rivisto da {line.get('reviewer') or '—'} il {line.get('review_date') or '—'}",
              "generated": "generato dal modello", "notes": "solo note (nessuno schema)"}[v["source"]]
    gen = v.get("generated") or {}
    head = [["Articolo", v["label"]], ["Rubrica", line.get("rubrica") or gen.get("rubrica") or "—"],
            ["Documento", line.get("doc") or gen.get("doc") or "—"],
            ["Tipo", ", ".join(schema.get("tipo") or []) or "—"],
            ["Destinatario", roles(schema.get("destinatario"))], ["Legittimato", roles(schema.get("legittimato"))],
            ["Soggetto passivo", roles(schema.get("soggetto_passivo"))], ["Rapporto", schema.get("rapporto") or "—"],
            ["Fonte", source], ["Testo cambiato dopo la revisione", "sì (superato)" if v["stale"] else "no"],
            ["Versione dello schema", line.get("schema_version", "—")],
            ["Modello", line.get("model") or gen.get("model") or "—"],
            ["Creato il", line.get("created_at") or gen.get("created_at") or "—"]]
    L += table(["Voce", "Valore"], head, markdown)
    if schema:
        L += _schema_body(schema, markdown, line.get("origin") or {}, "", level + 1)
        special = [cs for cs in schema.get("commi_speciali") or [] if isinstance(cs, dict)]
        L += _heading("Commi speciali", markdown, level + 1)
        if not special:
            L.append("(nessuno)")
        for cs in special:
            k = es.norm_comma(cs.get("comma"))
            sub = cs.get("schema") if isinstance(cs.get("schema"), dict) else {}
            L += _heading(f"Comma {k or '?'} — {cs.get('kind') or '—'}" +
                          (f" (tipo: {', '.join(sub.get('tipo') or [])})" if sub.get("tipo") else ""),
                          markdown, level + 2)
            L += _schema_body(sub, markdown, line.get("origin") or {}, k, level + 3)
        L += _heading("Articoli collegati", markdown, level + 1)
        L.append(", ".join(schema.get("collegati") or []) or "(nessuno)")
        checks = es.check(schema, v["text"]) if v.get("text") else (line.get("checks") or {})
        raised = [(k, checks.get(k)) for k in es.BLOCKING + es.NON_BLOCKING if checks.get(k)]
        L += _heading("Controlli", markdown, level + 1)
        if not v.get("text"):
            L.append("(testo non disponibile: i controlli sono quelli registrati con la riga)")
        if raised:
            for k, items in raised:
                kind = "bloccante" if k in es.BLOCKING else "segnalato"
                values = items if isinstance(items, list) else [items]
                L.append(f"{'- ' if markdown else ''}{k} ({kind}): " + "; ".join(_cell(x) for x in values))
        else:
            L.append("nessun controllo sollevato")
    L += _heading("Note dell'avvocato", markdown, level + 1)
    if v["notes"]:
        for n in v["notes"]:
            numero = f" (n. {n.get('numero')})" if n.get("numero") else ""
            L.append(f"{'- ' if markdown else ''}{NOTE_MARK}{numero} {_cell(n.get('testo'))}")
    else:
        L.append("(nessuna)")
    if line.get("note"):
        L += ["", f"Nota del revisore: {_cell(line.get('note'))}"]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(L)).rstrip() + "\n"


# --- writing the overlay -----------------------------------------------------------------------------------------------

def _read_raw(path: str) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    out = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    rec = None
                out.append((raw.rstrip("\n"), rec if isinstance(rec, dict) else None))
    return out


def _label_of_line(line: Dict[str, Any]) -> str:
    es = _es()
    if line.get("label"):
        return es.norm_label(line["label"])
    if line.get("article") is not None and (line.get("code") or line.get("doc")):
        return es.norm_label(es.label_of(line["article"], line.get("code") or line.get("doc")))
    return ""


def write_overlay(updates: Dict[str, Optional[Dict[str, Any]]], path: Optional[str] = None) -> str:
    """The reviewed file with each label's line replaced in place (one line per label; a new label appended; None
    removes it); lines this tool cannot read kept as they are. Written atomically."""
    es = _es()
    path = path or es.reviewed_file()
    out: List[str] = []
    done = set()
    for raw, rec in _read_raw(path):
        label = _label_of_line(rec) if rec else ""
        if not label or label not in updates:
            out.append(raw)
            continue
        if label in done:
            continue                        # an earlier duplicate of the label: one line per label
        done.add(label)
        if updates[label] is not None:
            out.append(json.dumps(updates[label], ensure_ascii=False))
    for label, rec in updates.items():
        if label not in done and rec is not None:
            out.append(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("".join(x + "\n" for x in out))
    os.replace(path + ".tmp", path)
    return path


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _sub(schema: Dict[str, Any], eid: str) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """(the schema an element id belongs to, its scope, its own id): «co2.E1» a comma speciale's."""
    es = _es()
    m = re.match(r"^co([^.]+)\.(.+)$", str(eid))
    if not m:
        return schema, "", str(eid)
    for cs in schema.get("commi_speciali") or []:
        if isinstance(cs, dict) and es.norm_comma(cs.get("comma")) == es.norm_comma(m.group(1)) and \
                isinstance(cs.get("schema"), dict):
            return cs["schema"], es.norm_comma(m.group(1)), m.group(2)
    return None, "", str(eid)


def _all_ids(schema: Dict[str, Any]) -> List[str]:
    es = _es()
    out = [str(e.get("id")) for e in schema.get("elementi") or [] if isinstance(e, dict)]
    for cs in schema.get("commi_speciali") or []:
        if isinstance(cs, dict) and isinstance(cs.get("schema"), dict):
            k = es.norm_comma(cs.get("comma"))
            out += [f"co{k}.{e.get('id')}" for e in cs["schema"].get("elementi") or [] if isinstance(e, dict)]
    return out


def _next_id(schema: Dict[str, Any]) -> str:
    taken = {str(e.get("id")) for e in schema.get("elementi") or [] if isinstance(e, dict)}
    n = 1
    while f"E{n}" in taken:
        n += 1
    return f"E{n}"


def edit(spec: Dict[str, Any], today: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    """v79.11 (B): the lawyer's corrections (a parsed --edit YAML) applied and written as the article's reviewed line;
    (the line, the warnings). ValueError when the YAML lacks article or reviewer or there is no line to start from."""
    es = _es()
    if not isinstance(spec, dict):
        raise ValueError("il file YAML deve contenere un oggetto (article, reviewer, …)")
    if not str(spec.get("article") or "").strip():
        raise ValueError("manca «article» (per esempio «582 c.p.»)")
    reviewer = " ".join(str(spec.get("reviewer") or "").split())
    if not reviewer:
        raise ValueError("manca «reviewer»: chi rivede lo schema")
    warnings = [f"chiave sconosciuta ignorata: {k}" for k in spec if k not in EDIT_KEYS]
    found = lines_for(str(spec["article"]), spec.get("doc"))
    label, gen, overlay = found["label"], found["generated"], found["overlay"]
    rebased = False
    if is_reviewed(overlay):
        base = overlay
        schema = _copy(overlay["schema"])
        generated_schema = overlay.get("generated_schema")
        corrections = list(overlay.get("corrections") or [])
        origin = dict(overlay.get("origin") or {})
        if is_stale(overlay, gen) and spec.get("testo_attuale") is True:
            rebased = True
            warnings.append("il testo dell'articolo è cambiato dopo la revisione: con «testo_attuale: true» la "
                            "revisione vale per il testo attuale (controlli rifatti su quel testo)")
        elif is_stale(overlay, gen):
            warnings.append("il testo dell'articolo è cambiato dopo la revisione: la revisione resta superata "
                            "(source_hash della revisione precedente; «testo_attuale: true» se l'hai rivista sul "
                            "testo attuale)")
    elif gen:
        base = gen
        schema = _copy(gen["schema"]) if isinstance(gen.get("schema"), dict) else {}
        generated_schema = _copy(gen["schema"]) if isinstance(gen.get("schema"), dict) else None
        corrections = []
        origin = {eid: "modello" for eid in _all_ids(schema)}
    else:
        raise ValueError(f"nessuno schema generato né rivisto per {label}: niente da cui partire")
    text = (current_text(gen) if rebased else "") or base.get("text") or (gen or {}).get("text") or ""
    new: List[Dict[str, Any]] = []
    for field in FIELDS:
        if field in spec:
            value = _copy(spec[field])
            if schema.get(field) != value:
                new.append({"field": field, "from": schema.get(field), "to": value, "by": reviewer})
            schema[field] = value
    if "elementi" in spec:
        value = _copy(spec["elementi"] or [])
        before = {str(e.get("id")): e for e in schema.get("elementi") or [] if isinstance(e, dict)}
        if schema.get("elementi") != value:
            new.append({"field": "elementi", "from": schema.get("elementi"), "to": value, "by": reviewer})
        schema["elementi"] = value
        for e in value if isinstance(value, list) else []:
            if isinstance(e, dict):
                eid = str(e.get("id"))
                origin[eid] = origin.get(eid, "modello") if before.get(eid) == e else "avvocato"
        for eid in [k for k in origin if "." not in k and k not in {str(e.get("id")) for e in value
                                                                     if isinstance(e, dict)}]:
            origin.pop(eid)
    for eid, patch in (spec.get("elementi_patch") or {}).items():
        target, scope, own = _sub(schema, str(eid))
        els = (target or {}).get("elementi") or []
        idx = next((i for i, e in enumerate(els) if isinstance(e, dict) and str(e.get("id")) == own), None)
        if target is None or idx is None:
            warnings.append(f"elementi_patch: {eid} non è un elemento dello schema, ignorato")
            continue
        if patch is None:
            new.append({"element": str(eid), "from": els[idx], "to": None, "by": reviewer})
            del els[idx]
            origin.pop(str(eid), None)
            continue
        if not isinstance(patch, dict):
            warnings.append(f"elementi_patch: {eid} deve essere un oggetto o null, ignorato")
            continue
        for k, v in patch.items():
            v = _copy(v)
            if els[idx].get(k) != v:
                new.append({"element": str(eid), "field": k, "from": els[idx].get(k), "to": v, "by": reviewer})
                els[idx][k] = v
                origin[str(eid)] = "avvocato"
    for e in spec.get("add_elementi") or []:
        if not isinstance(e, dict):
            warnings.append(f"add_elementi: {e!r} non è un elemento, ignorato")
            continue
        e = _copy(e)
        if not str(e.get("id") or "").strip() or str(e["id"]) in {str(x.get("id")) for x in schema.get("elementi") or []
                                                                  if isinstance(x, dict)}:
            e["id"] = _next_id(schema)
        schema.setdefault("elementi", []).append(e)
        origin[str(e["id"])] = "avvocato"
        new.append({"element": str(e["id"]), "from": None, "to": e, "by": reviewer})
    if text:
        schema = es.ground(schema, text)
        checks = es.check(schema, text)
    else:
        checks = es.empty_checks()
        warnings.append("testo dell'articolo non disponibile: controlli non eseguiti")
    for k in es.BLOCKING + es.NON_BLOCKING:
        if checks.get(k):
            items = checks[k] if isinstance(checks[k], list) else [checks[k]]
            warnings.append(f"controllo {k}: " + "; ".join(str(x) for x in items))
    number, code = _split(label)
    review_date = str(spec.get("date") or today or date.today().isoformat())
    line = {"label": label, "doc": spec.get("doc") or base.get("doc") or (gen or {}).get("doc") or "",
            "article": base.get("article") or (gen or {}).get("article") or number,
            "code": base.get("code") or (gen or {}).get("code") or code,
            "rubrica": base.get("rubrica") or (gen or {}).get("rubrica") or "", "reviewed": True,
            "reviewer": reviewer, "review_date": review_date,
            "source_hash": current_hash(gen) if rebased else base.get("source_hash"),
            "schema_version": base.get("schema_version", es.SCHEMA_VERSION), "schema": schema,
            "generated_schema": generated_schema, "corrections": corrections + new, "origin": origin,
            "note_di_inquadramento": list((overlay or {}).get("note_di_inquadramento") or []), "checks": checks}
    if spec.get("note") is not None:
        line["note"] = str(spec["note"])
    elif (overlay or {}).get("note"):
        line["note"] = overlay["note"]
    if text:
        line["text"] = text
    write_overlay({label: line})
    return line, warnings


def edit_file(path: str, today: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    import yaml
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    return edit(spec, today)


def _fonte(path: str) -> str:
    full = os.path.abspath(path)
    rel = os.path.relpath(full, ROOT)
    return rel if not rel.startswith("..") else path


def import_notes(path: str = NOTES_FILE, today: Optional[str] = None) -> Dict[str, Any]:
    """v79.11 (B): the lawyer's notes file into the overlay -- per article label, its notes (the article's, then each
    numero's) replacing those an earlier import of the same file left; a reviewed schema kept; a notes-only line
    reviewed: false and no schema; notes of an article no longer in the file removed. Unchanged notes keep their
    import date (idempotent). {"articles": [labels], "changed": [labels], "path"}."""
    es = _es()
    from src.rag import article_notes
    full = path if os.path.isabs(path) else os.path.join(ROOT, path) if not os.path.isfile(path) else path
    with open(full, encoding="utf-8") as fh:
        parsed = article_notes.parse(fh.read())
    fonte = _fonte(full)
    when = today or date.today().isoformat()
    current = es.reviewed()
    updates: Dict[str, Optional[Dict[str, Any]]] = {}
    labels = []
    for raw_label, entry in parsed.items():
        label = es.norm_label(raw_label)
        labels.append(label)
        notes = ([{"testo": entry["nota"], "numero": None}] if entry.get("nota") else []) + \
            [{"testo": t, "numero": str(n)} for n, t in (entry.get("numeri") or {}).items() if t]
        old = current.get(label)
        kept = [n for n in (old or {}).get("note_di_inquadramento") or [] if n.get("fonte") != fonte]
        earlier = [n for n in (old or {}).get("note_di_inquadramento") or [] if n.get("fonte") == fonte]
        if [(n.get("testo"), n.get("numero")) for n in earlier] == [(n["testo"], n["numero"]) for n in notes]:
            continue                                    # the same notes: nothing to change
        fresh = [{**n, "autore": "avvocato", "fonte": fonte, "importato": when} for n in notes]
        if old:
            line = {**old, "note_di_inquadramento": kept + fresh}
        else:
            number, code = _split(label)
            gen = es.generated_by_label().get(label) or {}
            line = {"label": label, "doc": gen.get("doc") or "", "article": number, "code": code, "reviewed": False,
                    "note_di_inquadramento": fresh}
        updates[label] = line
    for label, old in current.items():
        if label in labels or not any(n.get("fonte") == fonte for n in old.get("note_di_inquadramento") or []):
            continue
        kept = [n for n in old.get("note_di_inquadramento") or [] if n.get("fonte") != fonte]
        empty = not kept and not is_reviewed(old) and not isinstance(old.get("schema"), dict)
        updates[label] = None if empty else {**old, "note_di_inquadramento": kept}
    if updates:
        write_overlay(updates)
    return {"articles": labels, "changed": list(updates), "path": es.reviewed_file()}


# --- the listings ------------------------------------------------------------------------------------------------------

def stale() -> List[Dict[str, Any]]:
    """The reviewed lines whose source_hash differs from the article's text's (current_hash of the generated line)."""
    es = _es()
    gens = es.generated_by_label()
    out = []
    for label, line in es.reviewed().items():
        gen = gens.get(label)
        if is_stale(line, gen):
            out.append({"label": label, "reviewed_hash": line.get("source_hash"), "generated_hash": current_hash(gen),
                        "reviewer": line.get("reviewer"), "review_date": line.get("review_date")})
    return out


def _code_order(code: str) -> Tuple[int, str]:
    order = ("c.c.", "c.p.", "c.p.c.", "c.p.p.")
    return (order.index(code) if code in order else len(order), code)


def coverage(cards_path: Optional[str] = None) -> Dict[str, Counter]:
    """{code: Counter(usable, unusable, without, reviewed)}: the generated lines with a usable / an unusable schema, the
    carded articles (and failed lines) without one; reviewed = the overlay's reviewed labels of the code."""
    es = _es()
    from src.rag import article_cards
    cards_path = cards_path or os.getenv(article_cards.FILE_ENV) or article_cards.DEFAULT_FILE
    bulk = _bulk()
    cards = bulk.read_file(cards_path)
    gens = es.generated()
    out: Dict[str, Counter] = defaultdict(Counter)
    code_cache: Dict[str, str] = {}

    def code_of(doc: str, line: Optional[Dict[str, Any]] = None) -> str:
        if line and line.get("code"):
            return es._code(line["code"])
        if doc not in code_cache:
            code_cache[doc] = es._code(doc) or doc
        return code_cache[doc]
    for key, line in gens.items():
        c = out[code_of(key[0], line)]
        if not isinstance(line.get("schema"), dict):
            c["without"] += 1
        else:
            c["usable" if es.usable(line) else "unusable"] += 1
    for doc, number in cards:
        if (str(doc), str(number).strip().lower()) not in gens:
            out[code_of(doc)]["without"] += 1
    for label, line in es.reviewed().items():
        if is_reviewed(line):
            out[label.partition(" ")[2]]["reviewed"] += 1
    return dict(out)


def coverage_lines(cov: Dict[str, Counter], markdown: bool = False) -> List[str]:
    rows, total = [], Counter()
    for code in sorted(cov, key=_code_order):
        c = cov[code]
        total.update(c)
        rows.append([code, c["usable"], c["unusable"], c["without"], c["reviewed"]])
    rows.append(["totale", total["usable"], total["unusable"], total["without"], total["reviewed"]])
    return table(["Codice", "Schema usabile", "Schema inutilizzabile", "Senza schema (schedati)", "Rivisti"], rows,
                 markdown)


def top_labels(n: int, runs: str = RUNS) -> Tuple[List[Dict[str, Any]], str]:
    """The n most used articles' lines (generated, or the overlay's reviewed one): article_cards_bulk.frequency over the
    runs -- the generator's order --, then the generated file's order; (lines, how the order was made)."""
    es = _es()
    gens, by_label, overlay = es.generated(), es.generated_by_label(), es.reviewed()
    paths = glob.glob(runs if os.path.isabs(runs) else os.path.join(ROOT, runs)) if runs else []
    counted = _bulk().frequency(paths) if paths else Counter()
    out: List[str] = []
    for (doc, number, code), _ in counted.most_common():
        line = gens.get((str(doc), str(number).lower())) if doc else None
        label = es.norm_label(es.label_of(number, code))
        if line is None:
            line = by_label.get(label)
        if line is None and not is_reviewed(overlay.get(label)):
            continue
        label = _label_of_line(line) if line else label
        if label not in out:
            out.append(label)
        if len(out) >= n:
            break
    how = f"frequenza in {len(paths)} casi" if counted else "ordine del file generato"
    if len(out) < n:
        for line in gens.values():
            label = _label_of_line(line)
            if label and label not in out:
                out.append(label)
            if len(out) >= n:
                break
        if counted:
            how += ", poi l'ordine del file generato"
    return [view(label) for label in out if view(label)], how


def top_pack(n: int, out_path: Optional[str], runs: str = RUNS) -> str:
    views, how = top_labels(n, runs)
    L = ["# Schemi degli elementi: pacchetto di lettura", "",
         f"{len(views)} articoli, i più usati ({how}); per ciascuno il testo ufficiale, poi lo schema.", ""]
    for v in views:
        L += ["---", "", render(v, markdown=True, text_first=True, level=2)]
    text = "\n".join(L).rstrip() + "\n"
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


QUALITY_ROWS = (("element_precision", "precisione degli elementi"), ("element_recall", "richiamo degli elementi"),
                ("logic_accuracy", "logica esatta"), ("exception_recall", "richiamo delle eccezioni"),
                ("comma_accuracy", "comma esatto"), ("grounding_accuracy", "fondamento esatto"),
                ("correction_rate", "schemi corretti dall'avvocato"), ("n", "schemi rivisti confrontati"),
                ("grounding_rate_auto", "elementi espliciti fondati (automatico, tutti i generati)"))


def quality_lines(markdown: bool = False) -> List[str]:
    report = _es().quality_report()
    rows = [[label, "—" if report.get(key) is None else report.get(key)] for key, label in QUALITY_ROWS]
    L = table(["Misura", "Valore"], rows, markdown)
    if report.get("note"):
        L += ["", report["note"]]
    return L


# --- main --------------------------------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--article", help='the label, e.g. "582 c.p."')
    action.add_argument("--edit", metavar="YAML")
    action.add_argument("--import-notes", nargs="?", const=NOTES_FILE, metavar="FILE")
    action.add_argument("--stale", action="store_true")
    action.add_argument("--coverage", action="store_true")
    action.add_argument("--top", type=int, metavar="N")
    action.add_argument("--quality", action="store_true")
    parser.add_argument("--doc", help="--article: the document's exact name")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--out", help="--top: the markdown file to write (default: printed)")
    parser.add_argument("--runs", default=RUNS, help="--top: diagnose_long case files (glob)")
    parser.add_argument("--cards", help="--coverage: the cards file (default: src/rag/article_cards' file)")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    args = parser.parse_args(argv)
    if os.path.isfile(args.env):
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env)
        except ImportError:
            pass
    _offline_env()
    if args.article:
        v = view(args.article, args.doc)
        if v is None:
            print(f"Nessuno schema né nota per {args.article}.")
            return 1
        print(render(v, markdown=args.markdown), end="")
        return 0
    if args.edit:
        try:
            line, warnings = edit_file(args.edit)
        except ValueError as exc:
            print(f"Non salvato: {exc}", file=sys.stderr)
            return 2
        print(f"Salvato: {line['label']} rivisto da {line['reviewer']} il {line['review_date']}, "
              f"{len(line['corrections'])} correzioni ({_es().reviewed_file()})")
        for w in warnings:
            print(f"ATTENZIONE: {w}")
        if not any(w.startswith("controllo ") for w in warnings):
            print("Controlli: nessuno sollevato.")
        return 0
    if args.import_notes:
        got = import_notes(args.import_notes)
        print(f"{len(got['articles'])} articoli con note: {', '.join(got['articles'])}; "
              f"{len(got['changed'])} righe cambiate ({got['path']})")
        return 0
    if args.stale:
        found = stale()
        if not found:
            print("Nessuna revisione superata.")
        else:
            print(f"{len(found)} revisioni superate (il testo è cambiato):")
            print("\n".join(table(["Articolo", "Hash rivisto", "Hash attuale", "Revisore", "Data"],
                                  [[x["label"], x["reviewed_hash"], x["generated_hash"], x["reviewer"],
                                    x["review_date"]] for x in found], args.markdown)))
        return 0
    if args.coverage:
        print("\n".join(coverage_lines(coverage(args.cards), args.markdown)))
        return 0
    if args.top is not None:
        text = top_pack(args.top, args.out, args.runs)
        print(f"File: {os.path.abspath(args.out)}" if args.out else text, end="\n" if args.out else "")
        return 0
    print("\n".join(quality_lines(args.markdown)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
