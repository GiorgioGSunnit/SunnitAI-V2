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


def test_verify_rewrites_a_checked_article_s_statement_else_keeps_it_and_strips_anchors(monkeypatch):
    events = []
    monkeypatch.setattr(aw, "vlog", lambda event, data=None, **_: events.append((event, data)))
    _, cat = _catalogue([])
    cat.checked = [("9001", "c.c.")]         # graded 2 or more: never removed

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
    assert hopeless in body                     # the rewrite still fails: kept, its checked citation with it
    assert "prescrizione" not in body and "Tizio ha scoperto il danno il 3 marzo." in body
    assert out[1][1] == standard
    log = [d for e, d in events if e == "anchored_writing"][-1]
    t = log["totals"]
    assert (t["rewritten"], t["removed"], t["kept_checked"], t["anchored"]) == (1, 1, 1, 2)
    assert t["removed_without_citation"] == 1 and t["overlap_failures"] == {"0-5%": 1}
    why = {e["sentence"][:12]: (e["why"], e["action"], e["how"]) for e in log["entries"]}
    # 9999 is outside the checked material: its citation dropped, "La prescrizione è decennale." no longer reads
    assert why[nowhere[:12]] == ("unanchored citation", "outside", "removed")
    assert why[fixable[:12]] == ("overlap", "checked", "rewritten")
    assert why[hopeless[:12]] == ("unanchored citation", "checked", "kept")
    assert next(e for e in log["entries"] if e["sentence"] == nowhere)["cited"] == ["9999 c.c."]


def test_a_statement_citing_only_outside_the_checked_material_loses_the_citation(monkeypatch):
    events = []
    monkeypatch.setattr(aw, "vlog", lambda event, data=None, **_: events.append((event, data)))
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: pytest.fail("no call: nothing to rewrite"))
    _, cat = _catalogue([])
    cat.checked = [("9100", "c.p.c.")]
    sentence = "Il proprietario risponde sempre per colpa grave presunta (art. 9001 c.c.) [A1]."
    out = aw.verify([("Norme e giurisprudenza", sentence)], cat)
    assert out[0][1] == "Il proprietario risponde sempre per colpa grave presunta."
    entry = [d for e, d in events if e == "anchored_writing"][-1]["entries"][0]
    assert (entry["action"], entry["how"], entry["dropped"]) == ("outside", "citation dropped", ["9001 c.c."])


def test_overlap_counts_a_paraphrase_naming_the_rubrica():
    _, cat = _catalogue([])
    cat.passages["A1"]["rubrica"] = "Denuncia dei vizi immaginari"
    paraphrase = "Chi compra deve sempre fare la denuncia dei vizi presto, altrimenti perde ogni tutela [A1]."
    ratio, rubrica = aw.overlap(paraphrase, [cat.passages["A1"]])
    assert ratio < aw.MIN_OVERLAP and rubrica
    assert aw.check(paraphrase, ["A1"], cat) is None
    assert aw._reads_whole("Il custode risponde del danno.") and not aw._reads_whole("Il custode risponde ai sensi.")
    assert aw._bucket(0.07) == "5-10%" and aw._bucket(0.3) == ">=25%"


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


# --- two kinds of sentence ------------------------------------------------------------------------------------------

CASE = "Tizio ha acquistato una lavatrice da Caio. Dopo tre mesi la lavatrice si è rotta e Tizio ha scritto a Caio."


@pytest.mark.parametrize("sentence,kind", [
    ("La giurisprudenza ritiene che il venditore risponda dei vizi occulti.", "case_law"),
    ("Secondo la Cassazione il termine decorre dalla scoperta.", "case_law"),
    ("L'art. 9001 c.c. prevede che il custode risponda del danno.", "law"),
    ("Ai sensi dell'art. 9001 c.c., il custode risponde salvo il caso fortuito.", "law"),
    ("I requisiti della garanzia sono la denuncia e il termine.", "law"),
    ("Caio non ha provato il caso fortuito, quindi risponde del danno (art. 9001 c.c.).", "application"),
    ("Nel caso di specie il ritardo è imputabile al venditore.", "application"),
    ("Dopo tre mesi la lavatrice si è rotta e ha scritto subito.", "application"),
    ("Il termine è breve e va rispettato con attenzione.", None),
])
def test_classify_the_two_kinds(sentence, kind):
    assert aw.classify(sentence, CASE) == kind


def test_drop_citations_takes_out_only_the_citation():
    assert aw.drop_citations("Caio risponde del danno (art. 9005 c.c.).", [("9005", "c.c.")]) == "Caio risponde del danno."
    assert aw.drop_citations("Caio risponde ai sensi dell'art. 9005 c.c. del danno subito.", [("9005", "c.c.")]) == \
        "Caio risponde del danno subito."
    kept = "Caio risponde (artt. 9001 e 9005 c.c.)."
    assert aw.drop_citations(kept, [("9005", "c.c.")]) == kept          # a run naming a kept article stays


def test_applications_are_kept_case_law_needs_a_ruling_anchor_of_its_section(monkeypatch):
    events = []
    monkeypatch.setattr(aw, "vlog", lambda event, data=None, **_: events.append((event, data)))
    _, cat = _catalogue([])            # A1 = 9001 co. 2 (C2), A2 = the ruling
    cat.section_ids = {"Applicazione al caso": ["A1", "A2"], "Conclusioni": ["A1"]}
    cat.checked = [("9001", "c.c.")]
    seen = []

    def rewrite(messages, max_tokens, **k):
        human = messages[1].content
        items = [ln for ln in human.split(aw.REWRITE_MARK + "\n", 1)[1].splitlines() if ln.strip()]
        seen.extend(items)
        out = []
        for ln in items:
            n = int(ln.split(".", 1)[0])
            if "[prima il tipo]" in ln:
                out.append({"i": n, "tipo": "applicazione", "frase": ""})
            elif "senza giurisprudenza" in ln:
                out.append({"i": n, "tipo": "generale", "frase": "La denuncia del danno deve pervenire entro "
                                                                 "sessanta giorni dalla scoperta (art. 9001 c.c.)."})
            else:
                out.append({"i": n, "tipo": "generale", "frase": ""})
        return out
    monkeypatch.setattr(dg, "_chat_json", rewrite)
    application = "Caio, che non ha provato il caso fortuito, risponde del danno a Tizio (art. 9005 c.c.)."
    unclear = "Il guasto si è verificato entro il termine utile (art. 9001 c.c.)."
    ruling_ok = f"La responsabilità del custode ha natura oggettiva secondo la Cassazione ({REF}) [A2]."
    generic = "La giurisprudenza ritiene che la denuncia vada fatta entro sessanta giorni (art. 9001 c.c.) [A1]."
    generic_nothing = "La giurisprudenza è costante nel tutelare il consumatore in ogni caso."
    text = " ".join([application, unclear, ruling_ok, generic, generic_nothing])
    materials = {"Applicazione al caso": MATERIAL_9001, "Conclusioni": "PIANO"}
    out = aw.verify([("Applicazione al caso", text),
                     ("Conclusioni", f"La responsabilità del custode ha natura oggettiva ({REF}) [A2].")],
                    cat, case_text=CASE, materials=materials)
    body = out[0][1]
    assert "Caio, che non ha provato il caso fortuito, risponde del danno a Tizio." in body     # citation only gone
    assert "Il guasto si è verificato entro il termine utile (art. 9001 c.c.)." in body       # the call: application
    assert f"natura oggettiva secondo la Cassazione ({REF})." in body
    assert "giurisprudenza ritiene" not in body and "sessanta giorni dalla scoperta (art. 9001 c.c.)." in body
    assert generic_nothing not in body          # case law, no ruling anchor, nothing to restate it from: removed
    assert out[1][1] == ""                      # a ruling passage not given to Conclusioni: removed
    totals = [d for e, d in events if e == "anchored_writing"][-1]["totals"]
    assert totals["citations_dropped"] == 1 and totals["kept_as_application"] == 1 and totals["application"] == 1
    assert totals["removed_case_law"] == 2
    assert any("[prima il tipo]" in ln for ln in seen) and any("senza giurisprudenza" in ln for ln in seen)


MATERIAL_9001 = f"TESTI DELLE NORME:\n[art. 9001 c.c.]\n{C1} {C2}"


def test_a_section_without_rulings_gets_no_ruling_passage():
    plan, cat = _catalogue([])
    analysis = {"frame": {"posture_type": "altro", "areas": []}}
    ptw.section_jobs(plan, "fatti", "richiesta", analysis, _rows(), {REF: [{"_passage": RULING}]}, "", anchors=cat)
    assert "A2" in cat.section_ids["Norme e giurisprudenza"] and "A2" not in cat.section_ids["Conclusioni"]


def test_fetch_audit_counts_applications_apart():
    sentences = [{"sentence": f"frase {i}", "section": "Applicazione al caso", "kind": kind, "verdict": verdict}
                 for i, (kind, verdict) in enumerate([("applicazione", "assente"), ("massima", "assente"),
                                                      ("regola", "supportata")])]
    data = fa.provenance_by_section([{"run": "r", "provenance": {"sentences": sentences}}])["r"]["Applicazione al caso"]
    assert data["all"] == 2 and data["all_assente"] == 1 and data["applicazione_assente"] == 1
    out = {"granularity": {"claims": []}, "provenance": {"sentences": sentences}, "understanding": {}}
    fa.attribute(out)
    assert out["provenance"]["defects"] == 1
    assert '"applicazione"' in fa.PROVENANCE_SYSTEM
