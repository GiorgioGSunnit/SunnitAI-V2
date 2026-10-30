"""ANCHORED_WRITING_ENABLED (src/rag/anchored_writing.py), the procedure vocabulary, and fetch_audit's provenance by
section and kind. Every article here is invented; no model, no database."""
import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest  # noqa: E402

from src.rag import anchored_writing as aw  # noqa: E402
from src.rag import comma_material as cm  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402
from src.rag import practical_checklist  # noqa: E402

import fetch_audit as fa  # noqa: E402

C1 = "Il custode immaginario risponde dei danni cagionati dalle cose che ha in custodia, salvo il caso fortuito."
C2 = "La denuncia del danno deve pervenire entro sessanta giorni dalla scoperta, a pena di decadenza dal diritto."
RULING = "La responsabilità del custode ha natura oggettiva e cede solo davanti al caso fortuito provato."
REF = "Cass. civ., sez. III, n. 12345/2099"


def _rows():
    return [{"_article": ("9001", "c.c."), "s": {"plain_text": f"{C1}\n{C2}"}, cm.COMMI_KEY: [C1, C2],
             cm.FROM_KEY: "sections"},
            {"_article": ("9100", "c.p.c."), "s": {"plain_text": "Il ricorso si propone al giudice del luogo."}}]


def _plan(passi=None):
    return {"parte_assistita": "Tizio", "posizione": "danneggiato", "obiettivo": "", "questione_centrale": "",
            "risposte": [], "subordinate": [], "punti_deboli": [], "questioni": [
        {"titolo": "Decadenza dalla denuncia", "norme": [("9001", "c.c.")], "pronunce": [REF],
         "argomento": "La denuncia è arrivata entro sessanta giorni dalla scoperta.", "obiezione": "", "replica": "",
         **({"passi": passi} if passi is not None else {})}]}


def _catalogue(passi=None):
    plan = _plan(passi)
    cat = aw.build(plan, _rows(), {REF: [{"_passage": RULING}]}, procedural=[("9100", "c.p.c.")],
                   terms=[{"ref": "9002 c.c.", "text": "Il termine per agire è di un anno dalla consegna."}])
    return plan, cat


def test_the_plan_s_passi_are_read_and_numbered_a_norm_without_one_gets_its_best_comma():
    known = {("9001", "c.c.")}
    passi = aw.parse_passi(["art. 9001 c.c. co. 1", REF, "art. 7777 c.c."], known, {REF: [{}]})
    assert passi == [{"ref": ("9001", "c.c."), "comma": 1}, {"ruling": REF}]
    plan, cat = _catalogue(passi)
    assert plan["questioni"][0]["ancore"] == ["A1", "A2"]
    assert cat.passages["A1"]["label"] == "art. 9001 c.c., co. 1" and cat.passages["A1"]["text"] == C1
    assert cat.passages["A2"]["ruling"] == REF
    plan, cat = _catalogue([])                                # no passo: the comma that matches the argument
    assert cat.passages["A1"]["text"] == C2 and cat.passages["A1"]["comma"] == 2
    assert [cat.passages[i]["label"] for i in cat.procedural_ids] == ["art. 9100 c.p.c.", "art. 9002 c.c."]
    assert "(norme: art. 9001 c.c.; pronunce: " in ptw._plan_text(plan) and "passi: A1, A2" in ptw._plan_text(plan)


def test_check_anchor_citation_and_overlap():
    _, cat = _catalogue([])            # A1 = C2 (9001 co. 2), A2 = the ruling
    good = "La denuncia del danno deve pervenire entro sessanta giorni dalla scoperta (art. 9001 c.c.) [A1]."
    assert aw.check(good, ["A1"], cat) is None
    assert aw.check("La denuncia va fatta entro sessanta giorni dalla scoperta (art. 9005 c.c.) [A1].", ["A1"], cat) \
        == "citation not its anchor's"
    assert aw.check("Il proprietario risponde sempre per colpa grave presunta (art. 9001 c.c.) [A1].", ["A1"], cat) \
        == "overlap"
    assert aw.check("Vedi [A9].", ["A9"], cat) == "unknown anchor"
    assert aw.check("Il custode risponde sempre (art. 9001 c.c.).", [], cat) == "unanchored citation"
    assert aw.check("Tizio ha scoperto il danno il 3 marzo e lo ha denunciato subito.", [], cat) is None
    assert aw.check(f"La responsabilità del custode ha natura oggettiva ({REF}) [A2].", ["A2"], cat) is None
    assert aw.check("La responsabilità del custode ha natura oggettiva (n. 999/2099) [A2].", ["A2"], cat) \
        == "citation not its anchor's"


def test_verify_rewrites_once_from_the_passage_else_removes_and_strips_anchors(monkeypatch):
    events = []
    monkeypatch.setattr(aw, "vlog", lambda event, data=None, **_: events.append((event, data)))
    _, cat = _catalogue([])

    def rewrite(messages, max_tokens, **k):
        human = messages[1].content
        assert messages[0].content == aw.REWRITE_SYSTEM and aw.REWRITE_MARK in human
        items = [ln for ln in human.split(aw.REWRITE_MARK + "\n", 1)[1].splitlines() if ln.strip()]
        out = []
        for ln in items:
            n = int(ln.split(".", 1)[0])
            if "colpa grave" in ln:
                out.append({"i": n, "frase": "La denuncia del danno deve pervenire entro sessanta giorni dalla "
                                             "scoperta, a pena di decadenza (art. 9001 c.c.)"})
            else:
                out.append({"i": n, "frase": "Il proprietario risponde per colpa (art. 9001 c.c.)."})  # still failing
        return out
    monkeypatch.setattr(dg, "_chat_json", rewrite)
    keep = "La denuncia del danno deve pervenire entro sessanta giorni dalla scoperta (art. 9001 c.c.) [A1]."
    fixable = "Il proprietario risponde per colpa grave presunta (art. 9001 c.c.) [A1]."
    hopeless = "Il custode risponde sempre del danno anche senza colpa (art. 9001 c.c.)."
    nowhere = "La prescrizione è decennale (art. 9999 c.c.)."
    standard = "**Profilo penale**\n\nIl fatto è reato. Norme: art. 589 c.p."
    text = " ".join([keep, fixable, hopeless, nowhere, "Tizio ha scoperto il danno il 3 marzo."])
    out = aw.verify([("Norme e giurisprudenza", text), ("Profili processuali e termini", standard)], cat,
                    exempt=[standard])
    body = out[0][1]
    assert "[A" not in body and "sessanta giorni dalla scoperta (art. 9001 c.c.)." in body
    assert "a pena di decadenza (art. 9001 c.c.)." in body
    assert hopeless not in body and nowhere not in body and "Tizio ha scoperto il danno il 3 marzo." in body
    assert out[1][1] == standard
    log = [d for e, d in events if e == "anchored_writing"][-1]
    assert log["totals"]["rewritten"] == 1 and log["totals"]["removed"] == 2 and log["totals"]["anchored"] == 2
    why = {e["sentence"][:12]: (e["why"], e["how"]) for e in log["entries"]}
    assert why[nowhere[:12]] == ("unanchored citation", "removed")         # no passage of 9999: removed directly
    assert why[fixable[:12]] == ("overlap", "rewritten")


def test_sections_get_their_passages_vocabulary_and_rule_only_with_anchors(monkeypatch):
    plan, cat = _catalogue([])
    analysis = {"frame": {"posture_type": "indagini_preliminari", "areas": ["penale"]}}
    jobs = ptw.section_jobs(plan, "fatti", "richiesta", analysis, _rows(), {REF: [{"_passage": RULING}]}, "",
                            anchors=cat)
    by = {title: (instruction, material) for title, instruction, material, _ in jobs}
    for title in ptw.SECTION_TITLES:
        assert aw.ANCHOR_RULE in by[title][0] and f"[A1] art. 9001 c.c., co. 2: {C2}" in by[title][1]
    assert "Risposte ai quesiti" in by and "[A3] art. 9100 c.p.c." in by["Profili processuali e termini"][1]
    assert "[A3]" not in by["Conclusioni"][1]
    assert practical_checklist.VOCABULARY_TITLE in by["Profili processuali e termini"][1]
    assert "incidente probatorio" in by["Profili processuali e termini"][1]
    plain = ptw.section_jobs(plan, "fatti", "richiesta", analysis, _rows(), {}, "")
    assert all(aw.BLOCK_TITLE not in material for _, _, material, _ in plain)


def test_the_plan_prompt_asks_for_passi_only_with_the_flag(monkeypatch):
    assert '"passi"' not in ptw._plan_system()
    monkeypatch.setenv(aw.FLAG, "true")
    assert '"passi": ["art. N codice co. K"' in ptw._plan_system() and aw.PLAN_RULE in ptw._plan_system()
    plan, _ = ptw.clean_plan({"questioni": [{"titolo": "t", "argomento": "a", "norme": ["art. 9001 c.c."],
                                             "passi": ["art. 9001 c.c. co. 2"]}]}, {("9001", "c.c.")}, {})
    assert plan["questioni"][0]["passi"] == [{"ref": ("9001", "c.c."), "comma": 2}]


def test_procedure_vocabulary_by_stage():
    assert "opposizione all'esecuzione" in practical_checklist.procedure_vocabulary("esecuzione_civile")
    assert "ricorso per cassazione" in practical_checklist.procedure_vocabulary("sentenza_penale_appello")
    assert practical_checklist.procedure_vocabulary("altro") == []


# --- fetch_audit: provenance by section and kind ----------------------------------------------------------------

def test_provenance_breakdown_by_section_and_kind():
    answer = ("**1. Risposte ai quesiti**\n\nUna frase.\n\n**3. Norme e giurisprudenza**\n\nAltra frase.\n\n"
              "**6. Conclusioni**\n\nUltima.")
    assert [n for n, _ in fa.answer_sections(answer)] == ["Risposte ai quesiti", "Norme e giurisprudenza", "Conclusioni"]
    sentences = [{"section": "Risposte ai quesiti", "kind": "regola", "verdict": "assente"},
                 {"section": "Risposte ai quesiti", "kind": "procedura", "verdict": "supportata"},
                 {"stage": "section_conclusioni", "kind": "massima", "verdict": "assente"},       # an old audit.json
                 {"stage": "code", "kind": "istituto", "verdict": "contraddetta"},
                 {"section": "Conclusioni", "kind": "altro", "verdict": None}]
    cases = [{"run": "gr-on", "provenance": {"sentences": sentences}}]
    data = fa.provenance_by_section(cases)["gr-on"]
    assert data["Risposte ai quesiti"]["regola_assente"] == 1 and data["Risposte ai quesiti"]["all"] == 2
    assert data["Conclusioni"]["massima_assente"] == 1 and data["(unassigned)"]["all_contraddetta"] == 1
    table = "\n".join(fa.provenance_breakdown(cases))
    assert "| Risposte ai quesiti | 1 / 1 (100%) | - | 1 / 1 (0%)" not in table
    assert "| Risposte ai quesiti | 1 / 1 (100%) | - | 0 / 1 (0%) | - | 1 / 2 (50%) | 0 |" in table
    assert "| **total** |" in table
