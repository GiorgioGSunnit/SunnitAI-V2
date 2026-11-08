"""v48: context and speed (passage cap, section fit and refit, _looks_cut, opposing-argument patch, NORME_SPLIT, the grade-1
cap), role-fit (client_role conditions, ROLE_FIT), the term engine on usucapione, the main remedy's remedies, art. 696-bis,
the rubrica stamp, and diagnose_long's timing and case checks. The v47 dumps' own strings where they matter (h3 Leonardo,
l3's art. 229 Codice degli Appalti). No model, no database."""
import importlib.util
import json
import os
import pathlib

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import anchored_writing as aw  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import fact_signals as fs  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402
from src.rag import role_fit  # noqa: E402
from src.rag import section_fit as sf  # noqa: E402

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)
_ev_spec = importlib.util.spec_from_file_location("eval_narrative", HERE / "eval_narrative.py")
ev = importlib.util.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(ev)
_os_spec = importlib.util.spec_from_file_location("oversized_sections", HERE.parent / "scripts" / "oversized_sections.py")
oversized = importlib.util.module_from_spec(_os_spec)
_os_spec.loader.exec_module(oversized)

FLAGS = ("PLAN_PATCH_REDO", "NORME_SPLIT", "ROLE_FIT", "PLAN_RULE_INCLUDE", "REVIEW_FIXES_ENABLED", "WRITER_RULES_ENABLED",
         "REGIME_CANDIDATES", "RUBRICA_TEXT_CHECK", "ANCHORED_WRITING_ENABLED", "WRITER_LOOP_GUARD", "MODEL_CONTEXT_TOKENS")


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def events(monkeypatch):
    seen = []
    log = lambda name, data=None, **k: seen.append((name, data))  # noqa: E731
    monkeypatch.setattr(dg, "vlog", log)
    monkeypatch.setattr(aw, "vlog", log)
    return seen


# --- A1. the passage cap ---------------------------------------------------------------------------------------------

# l3 v47: art. 229 Codice degli Appalti 2026, co. 10, holding the annexes after it: 108k characters
ANNEX = "10. Il programma triennale dei lavori pubblici riporta la previsione degli interventi. " + \
        "Allegato I.5, elementi per la programmazione dei lavori; schede e tabelle; " * 1500


def test_a_whole_comma_is_capped_around_its_start_and_marked(events):
    cat = aw.Catalogue()
    pid = cat.add(("article", ("229", "Codice degli Appalti 2026"), 10), "article",
                  "art. 229 Codice degli Appalti 2026, co. 10", ANNEX, whole=True)
    text = cat.passages[pid]["text"]
    assert len(text) <= aw.PASSAGE_CAP + 2 and text.endswith("[…]") and text.startswith("10. Il programma triennale")
    log = next(d for n, d in events if n == "passage_capped")
    assert log["chars"] > 100000 and log["kept"] == len(text)


def test_a_short_whole_passage_is_untouched(events):
    cat = aw.Catalogue()
    pid = cat.add(("article", ("2051", "c.c.")), "article", "art. 2051 c.c.", "Ciascuno è responsabile del danno.",
                  whole=True)
    assert cat.passages[pid]["text"] == "Ciascuno è responsabile del danno." and not events


# --- A2/A3. the section fit and refit --------------------------------------------------------------------------------

PLAN = {"parte_assistita": "signora Bianchi", "posizione": "proprietaria", "obiettivo": "", "questione_centrale": "",
        "risposte": [], "subordinate": [], "punti_deboli": [], "pronunce_contrarie": [],
        "plan_source": {"229 Codice degli Appalti 2026": "rule"},
        "questioni": [{"titolo": "Obblighi dell'amministratore", "norme": [("1130", "c.c."), ("1131", "c.c."),
                                                                         ("229", "Codice degli Appalti 2026")],
                       "pronunce": ["Cass. civ., sez. II, ord. n. 1/2026"], "argomento": "A.", "obiezione": "",
                       "replica": "", "passi": []}]}
RULE = [("229", "Codice degli Appalti 2026")]


def material(rule_chars=3000, other_chars=3000, ruling_chars=3000):
    return "\n\n".join([
        "CASO:\nLa signora Bianchi subisce un danno dallo scarico condominiale.",
        "PIANO:\nParte assistita: signora Bianchi\n1. Obblighi dell'amministratore (norme: art. 1130 c.c., art. 1131 c.c.)",
        "TESTI DELLE NORME:\n[art. 1130 c.c.]\nRubrica: Attribuzioni.\n" + "a" * 500
        + "\n\n[art. 229 Codice degli Appalti 2026]\nRubrica: Entrata in vigore.\n" + "b" * rule_chars,
        f"PASSI DELLE PRONUNCE:\n[Cass. civ., sez. II, ord. n. 1/2026]\n{'r' * ruling_chars}",
        "PASSI CON ANCORA:\n[A1] art. 1130 c.c.: " + "c" * 400 + "\n[A2] art. 1131 c.c.: " + "d" * other_chars
        + "\n[A3] art. 229 Codice degli Appalti 2026, co. 10: " + "e" * rule_chars
        + "\n[A4] Cass. civ., sez. II, ord. n. 1/2026: " + "f" * 300])


def test_a_prompt_within_the_budget_is_untouched():
    out, log = sf.fit("Norme e giurisprudenza", "sistema", material(), 1000, PLAN, RULE)
    assert out == material() and log["fits"] and not log["dropped"]


def test_rule_rows_go_first_then_extra_passages_then_rulings_never_the_case_or_the_plan():
    text = material()
    whole = sf._estimate("sistema") + sf._estimate(text)
    out, log = sf.fit("Norme e giurisprudenza", "sistema", text, 0, PLAN, RULE, budget=whole - 1500)
    assert [d["kind"] for d in log["dropped"]] == ["rule row", "rule row"]
    assert "229 Codice" not in out and "[A2]" in out and "CASO:" in out and "PIANO:" in out
    out, log = sf.fit("Norme e giurisprudenza", "sistema", text, 0, PLAN, RULE, budget=whole - 3000)
    kinds = [d["kind"] for d in log["dropped"]]
    assert kinds[:2] == ["rule row", "rule row"] and "extra passage" in kinds
    assert "[A2]" not in out and "[A1]" in out          # 1131 is not its issue's governing norm; 1130 is
    out, log = sf.fit("Norme e giurisprudenza", "sistema", text, 0, PLAN, RULE, budget=whole - 3800)
    assert "ruling" in [d["kind"] for d in log["dropped"]] and "PASSI DELLE PRONUNCE" not in out
    assert "CASO:\nLa signora Bianchi" in out and "PIANO:" in out


def test_nothing_left_to_drop_is_impossible():
    with pytest.raises(sf.SectionTooLarge):
        sf.fit("Norme e giurisprudenza", "sistema", material(), 0, PLAN, RULE, budget=10)


def test_a_dropped_passage_leaves_the_section_s_anchors():
    cat = aw.Catalogue()
    cat.section_ids = {"Norme e giurisprudenza": ["A1", "A2", "A3", "A4"]}
    text = material()
    whole = sf._estimate("s") + sf._estimate(text)
    sf.fit("Norme e giurisprudenza", "s", text, 0, PLAN, RULE, anchors=cat, budget=whole - 1500)
    assert "A3" not in cat.section_ids["Norme e giurisprudenza"]


def _sections_plan():
    return {**PLAN, "questioni": [dict(q) for q in PLAN["questioni"]]}


CONTEXT_ERROR = ("Error code: 400 - This model's maximum context length is 32768 tokens. However, you requested 975 "
                 "output tokens and your prompt contains at least 31794 input tokens, for a total of at least 32769 "
                 "tokens.")


def test_an_impossible_fit_writes_the_parere_in_one_call(monkeypatch, events):
    monkeypatch.setenv("MODEL_CONTEXT_TOKENS", "1200")
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: pytest.fail("no section call when nothing fits"))
    jobs = [("Norme e giurisprudenza", "istr", "CASO:\n" + "x" * 9000, 400)]
    assert ptw.write_sections(_sections_plan(), jobs, "it") is None
    assert any(n == "plan_then_write_fallback" and d["step"] == "section fit" for n, d in events)


def test_a_context_error_is_refitted_and_tried_once_more_never_identical(monkeypatch, events):
    sent = []

    def chat(messages, max_tokens=None, **k):
        sent.append(messages[1].content)
        if len(sent) == 1:
            raise ValueError(CONTEXT_ERROR)
        return "La norma regola il caso."
    monkeypatch.setattr(dg, "_call_chat", chat)
    jobs = [("Norme e giurisprudenza", "istr", material(rule_chars=20000), 975)]
    out = ptw.write_sections(_sections_plan(), jobs, "it")
    assert out == [("Norme e giurisprudenza", "La norma regola il caso.")]
    assert len(sent) == 2 and sent[0] != sent[1] and "229 Codice" not in sent[1]
    fit = [d for n, d in events if n == "case_analysis_section_fit"]
    assert fit and fit[-1]["sections"][0]["refit"] and fit[-1]["sections"][0]["server_tokens"] == 31794


def test_a_context_error_with_nothing_to_drop_is_not_retried(monkeypatch, events):
    sent = []

    def chat(messages, max_tokens=None, **k):
        sent.append(messages[1].content)
        raise ValueError(CONTEXT_ERROR)
    monkeypatch.setattr(dg, "_call_chat", chat)
    jobs = [("Norme e giurisprudenza", "istr", "CASO:\nbreve", 975)]
    assert ptw.write_sections(_sections_plan(), jobs, "it") is None
    assert len(sent) == 1


# --- A4. _looks_cut --------------------------------------------------------------------------------------------------

def test_a_trailing_rule_line_is_no_cut():
    # h1 v47's one-call parere, section 1
    assert not ptw._looks_cut("Le lesioni sono un danno biologico temporaneo, riconoscibile ai fini del risarcimento.\n\n---")
    assert not ptw._looks_cut("- **Rischio basso** per la prova del danno economico.\n\n***\n")
    assert ptw._looks_cut("Il danno è stato causato da un caso fortuito o da una negligenza della\n\n---")


# --- A5. the opposing arguments patched into the plan -----------------------------------------------------------------

FIRST = {"parte_assistita": "Leonardo", "questione_centrale": "La servitù è già acquisita",
         "risposte": [{"quesito": "", "risposta": "Leonardo può agire in confessoria (art. 1079 c.c.)."}],
         "questioni": [{"titolo": "Servitù per destinazione del padre di famiglia", "norme": ["art. 1062 c.c."],
                        "argomento": "La servitù è sorta alla separazione dei fondi (art. 1062 c.c.)."}]}
OPPOSING = [{"argomento": "Il cancello è una misura di riservatezza legittima", "perche": "giova a Domitilla"}]


def _make_plan(monkeypatch, reply):
    plans, patches = [], []
    monkeypatch.setattr(ptw, "_plan_json", lambda messages: plans.append(messages) or FIRST)
    monkeypatch.setattr(dg, "_opposing_arguments", lambda *a, **k: list(OPPOSING))
    monkeypatch.setattr(ptw, "_issue_check", lambda issue, rows, headings: [
        {"articolo": "art. 1062 c.c.", "pertinente": True, "contraddice": False}])

    def chat_json(messages, **kwargs):
        if messages[0].content == ptw._OPPOSING_SYSTEM:
            patches.append(kwargs)
            return reply
        return []
    monkeypatch.setattr(dg, "_chat_json", chat_json)
    rows = [{"_article": ("1062", "c.c."), "_grade": 3, "s": {"plain_text": "La destinazione del padre di famiglia.",
                                                              "titolo_articolo": "R"}}]
    plan = ptw.make_plan("Leonardo passa sulla strada.", "", {}, "[art. 1062 c.c.]", "", "", rows, {})
    return plan, plans, patches


def test_opposing_arguments_become_objections_without_a_redo(monkeypatch, events):
    monkeypatch.setenv("PLAN_PATCH_REDO", "true")
    reply = [{"argomento_contrario": "Il cancello è una misura di riservatezza legittima", "questione": 1,
              "obiezione": "Domitilla dirà che il cancello tutela la sua riservatezza",
              "replica": "Il proprietario del fondo servente non può diminuire l'esercizio della servitù", "argomento": ""}]
    plan, plans, patches = _make_plan(monkeypatch, reply)
    assert len(plans) == 1 and patches[0]["max_tokens"] == 800
    q = plan["questioni"][0]
    assert "riservatezza" in q["obiezione"] and "fondo servente" in q["replica"] and "1062" in q["argomento"]
    log = next(d for n, d in events if n == "case_analysis_plan_patch")
    assert log["kind"] == "opposing" and log["then"] == "kept"


def test_an_unplaced_opposing_argument_gets_the_full_redo(monkeypatch, events):
    monkeypatch.setenv("PLAN_PATCH_REDO", "true")
    plan, plans, patches = _make_plan(monkeypatch, [{"questione": 9, "obiezione": "x", "replica": "y"}])
    assert len(plans) == 2 and len(patches) == 1
    assert next(d for n, d in events if n == "case_analysis_plan_patch")["then"] == "full redo"


def test_without_the_flag_the_opposing_redo_as_before(monkeypatch, events):
    plan, plans, patches = _make_plan(monkeypatch, [])
    assert len(plans) == 2 and not patches


# --- A6. NORME_SPLIT -------------------------------------------------------------------------------------------------

NINE = [(str(n), "c.c.") for n in range(1051, 1060)]


def _nine_plan():
    return {**PLAN, "plan_source": {}, "questioni": [{"titolo": "Servitù", "norme": list(NINE), "pronunce": [],
                                                      "argomento": "A.", "obiezione": "", "replica": "", "passi": []}]}


def _nine_material():
    texts = "\n\n".join(f"[art. {n} c.c.]\nRubrica: R{n}.\nTesto {n}." for n, _ in NINE)
    return (f"CASO:\nLeonardo.\n\nPIANO:\nParte assistita: Leonardo\n\nTESTI DELLE NORME:\n{texts}\n\n"
            "PASSI DELLE PRONUNCE:\n[Cass. civ., sez. II, ord. n. 7662/2026]\nPrincipio.")


def test_more_than_eight_articles_two_halves_over_disjoint_sets(monkeypatch, events):
    monkeypatch.setenv("NORME_SPLIT", "true")
    calls = []

    def chat(messages, max_tokens=None, **k):
        human = messages[1].content
        calls.append((human, messages[0].content, max_tokens))
        half = 2 if "[art. 1059 c.c.]" in human else 1
        ptw.last_reply.finish_reason = "length" if half == 2 else "stop"
        return "Prima parte completa." if half == 1 else "Seconda parte completa. E una frase troncat"
    monkeypatch.setattr(dg, "_call_chat", chat)
    jobs = [("Norme e giurisprudenza", "istr. Al massimo 1000 parole; chiudi sempre l'ultima frase.",
             _nine_material(), 2000)]
    out = ptw.write_sections(_nine_plan(), jobs, "it")
    assert out == [("Norme e giurisprudenza", "Prima parte completa.\n\nSeconda parte completa.")]
    first, second = sorted(calls, key=lambda c: "[art. 1059 c.c.]" in c[0])
    assert "[art. 1051 c.c.]" in first[0] and "[art. 1059 c.c.]" not in first[0] and "7662/2026" in first[0]
    assert "[art. 1059 c.c.]" in second[0] and "[art. 1051 c.c.]" not in second[0] and "7662/2026" not in second[0]
    assert "Al massimo 600 parole" in first[1] and first[2] == 1200 and "Non citare pronunce" in second[1]
    log = next(d for n, d in events if n == "case_analysis_norme_split")
    assert [h["trimmed"] for h in log["halves"]] == [False, True]


def test_eight_articles_or_the_flag_off_one_call(monkeypatch, events):
    calls = []
    monkeypatch.setattr(dg, "_call_chat", lambda messages, max_tokens=None, **k: calls.append(1) or "Testo.")
    jobs = [("Norme e giurisprudenza", "istr", _nine_material(), 2000)]
    ptw.write_sections(_nine_plan(), jobs, "it")
    assert len(calls) == 1
    monkeypatch.setenv("NORME_SPLIT", "true")
    plan = _nine_plan()
    plan["questioni"][0]["norme"] = NINE[:8]
    calls.clear()
    ptw.write_sections(plan, jobs, "it")
    assert len(calls) == 1


# --- A7. the grade-1 cap ---------------------------------------------------------------------------------------------

def _graded(number, grade, signals):
    return {"_article": (number, "c.c."), "_grade": grade, "_signals": signals,
            "s": {"plain_text": "Testo " * 50, "titolo_articolo": f"Rubrica {number}"}}


def test_at_most_four_grade_1_rows_by_signal_count(monkeypatch):
    plan = {"questioni": [{"titolo": "Servitù", "argomento": "Rubrica", "norme": [("1062", "c.c.")]}]}
    rows = [_graded("1062", 3, ["frame"])]
    rows += [_graded(str(n), 1, ["frame", "distill"]) for n in range(2000, 2005)]
    rows += [{**_graded("2009", 1, ["frame", "distill"]), "_vote": True}]          # three signals: first
    articles = "".join(f"[art. {r['_article'][0]} c.c.]" for r in rows)
    out, inserted, skipped = ptw.include_by_rule(plan, rows, articles, {}, {})
    ones = [e["article"] for e in inserted if e["grade"] == "1"]
    assert len(ones) == 4 and "2009 c.c." in ones                     # the most signals goes in
    capped = [e for e in skipped if e["reason"].startswith("grade-1 cap")]
    assert [e["article"] for e in capped] == ["2003 c.c.", "2004 c.c."] and all("tokens" in e for e in capped)


def test_the_grade_1_cap_holds_the_added_tokens(monkeypatch):
    monkeypatch.setattr(ptw, "RULE_INCLUDE_GRADE1_TOKENS", 1500)        # one row costs ~1,335
    plan = {"questioni": [{"titolo": "Servitù", "argomento": "Rubrica", "norme": [("1062", "c.c.")]}]}
    rows = [{**_graded(str(n), 1, ["frame", "distill"]), "s": {"plain_text": "Parola " * 400, "titolo_articolo": "R"}}
            for n in range(2000, 2004)]
    articles = "".join(f"[art. {r['_article'][0]} c.c.]" for r in rows)
    _, inserted, _ = ptw.include_by_rule(plan, rows, articles, {}, {})
    assert len(inserted) == 1


# --- B1/C2. client_role conditions -----------------------------------------------------------------------------------

# h3 v47 (Leonardo): the client claims the servitude, his fund is landlocked, his men passed until 2016
H3_FACTS = {"parties": [{"name": "Leonardo", "role": "cliente, proprietario"}, {"name": "Domitilla", "role": "proprietario"}],
            "client_side": "Leonardo (cliente, proprietario)",
            "possession": [{"who": "incaricati di Leonardo", "asset": "strada sterrata", "since": "", "until": "2016",
                            "use": "passaggio", "still_ongoing": "no"},
                           {"who": "Domitilla", "asset": "strada sterrata", "since": "2016", "until": "",
                            "use": "chiusura", "still_ongoing": "sì"}],
            "events": [{"date": "2003", "what": "Sempronio vende il terreno a Domitilla", "kind": "vendita"},
                       {"date": "2016", "what": "Domitilla appone un cancello chiuso a chiave",
                        "kind": "chiusura/ostruzione accesso"},
                       {"date": "2018", "what": "Leonardo si rivolge ad un avvocato", "kind": "azione giudiziaria"}],
            "property_history": [{"owner": "Sempronio", "to": "2003"}, {"owner": "Domitilla", "from": "2003"}],
            "access": [{"asset": "fondo di Leonardo", "landlocked": "sì"}],
            "requests": ["Leonardo chiede un parere legale per far valere il diritto di passaggio sulla strada sterrata."],
            "_regimes": True}
H3_FIRED = [{"id": "destinazione_padre_famiglia", "acquired_right": True,
             "institutions": ["servitù per destinazione del padre di famiglia"]}]


def _h3_analysis():
    return {"frame": {"areas": ["civile"], "tasks": ["Come tutelare il passaggio"]},
            "questioni": [{"titolo": "Servitù di passaggio", "istituti": [{"nome": "passaggio coattivo"},
                                                                       {"nome": "azione confessoria e negatoria"}]}],
            "fact_signals": {"facts": H3_FACTS, "fired": H3_FIRED}}


def test_leonardo_gets_no_negatoria_and_no_rivendicazione():
    assert fs.client_roles(H3_FACTS, H3_FIRED) >= {"claims_servitude", "possessor", "landlocked_owner"}
    assert "resists_servitude" not in fs.client_roles(H3_FACTS, H3_FIRED)
    excluded = []
    found = {r["id"]: r for r in fs.regime_candidates(_h3_analysis(), ["civile"], log=excluded)}
    assert found["azione_reale_immobiliare"]["refs"] == [("1079", "c.c.")]
    assert {e["article"] for e in excluded if e.get("article")} >= {"948 c.c.", "949 c.c."}
    assert found["passaggio_coattivo"]["refs"] == [("1053", "c.c."), ("1051", "c.c.")]
    assert found["passaggio_coattivo"]["commi"] == {"1051 c.c.": "last"}


def test_the_last_comma_of_1051_reaches_the_passages():
    from src.rag import comma_material as cm
    commi = ["(co. 1) Il proprietario, il cui fondo è circondato da fondi altrui, ha diritto di ottenere il passaggio.",
             "(co. 2) Il passaggio si deve stabilire in quella parte per cui l'accesso alla via pubblica è più breve.",
             "(co. 3) Esso può essere anche concesso per l'ampliamento di un passaggio esistente.",
             "(co. 4) Sono esenti da questa disposizione le case, i cortili, i giardini e le aie ad esse attinenti."]
    row = {"_article": ("1051", "c.c."), "_regime_comma": "last", cm.COMMI_KEY: commi, cm.FROM_KEY: "lines",
           "s": {"plain_text": " ".join(commi) * 3, "titolo_articolo": "Passaggio coattivo"}}
    plan = {"questioni": [{"titolo": "Passaggio coattivo", "argomento": "fondo circondato", "norme": [("1051", "c.c.")],
                           "pronunce": [], "passi": []}]}
    cat = aw.build(plan, [row], {})
    assert any("esenti" in p["text"] for p in cat.passages.values())


# --- B2. ROLE_FIT ----------------------------------------------------------------------------------------------------

def test_role_fit_asks_one_call_per_substantive_graded_article_and_tags_the_counterparty_s(monkeypatch, events):
    monkeypatch.setenv("ROLE_FIT", "true")
    asked = []

    def chat_json(messages, **kwargs):
        human = messages[1].content
        asked.append(human)
        assert messages[0].content == role_fit.SYSTEM and kwargs["max_tokens"] == role_fit.MAX_TOKENS
        if "art. 949" in human:
            return {"a_favore": "controparte", "perche": "È l'azione del proprietario contro chi vanta la servitù."}
        return {"a_favore": "cliente", "perche": "Fonda la pretesa di Leonardo."}
    monkeypatch.setattr(dg, "_chat_json", chat_json)
    plan = {"parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
            "subordinate": [], "punti_deboli": [],
            "questioni": [{"titolo": "Azione confessoria e negatoria", "norme": [("1079", "c.c."), ("949", "c.c."),
                                                                              ("102", "c.p.c."), ("1055", "c.c.")],
                           "pronunce": [], "argomento": "A.", "obiezione": "", "replica": ""}]}
    rows = [{"_article": ("1079", "c.c."), "_grade": 3, "s": {"plain_text": "Il titolare della servitù può...",
                                                              "titolo_articolo": "Accertamento della servitù"}},
            {"_article": ("949", "c.c."), "_grade": 2, "s": {"plain_text": "Il proprietario può agire per far "
                                                                           "dichiarare l'inesistenza di diritti...",
                                                             "titolo_articolo": "Azione negatoria"}},
            {"_article": ("102", "c.p.c."), "_grade": 3, "s": {"plain_text": "Litisconsorzio."}},
            {"_article": ("1055", "c.c."), "_grade": 1, "s": {"plain_text": "Cessazione dell'interclusione."}}]
    out, log = role_fit.check(plan, rows, "Leonardo è proprietario di terreni.", {"facts": H3_FACTS, "fired": H3_FIRED})
    assert len(asked) == 2                                   # 1079 and 949: not 102 c.p.c., not 1055 (grade 1)
    assert "PARTE ASSISTITA: Leonardo (cliente, proprietario)" in asked[0] and "vanta una servitù" in asked[0]
    assert out["norme_contrarie"] == ["949 c.c."] and {e["article"]: e["answer"] for e in log}["949 c.c."] == "controparte"
    text = ptw._plan_text(out)
    assert "art. 949 c.c. (norma della controparte)" in text and "art. 1079 c.c. (norma" not in text
    assert next(d for n, d in events if n == "case_analysis_role_fit")["adverse"] == ["949 c.c."]


def test_a_negatoria_presented_as_the_counterparty_s_argument_is_kept():
    ctx = {"servitude": True}
    assert ptw._wrong_statement("Leonardo può agire in negatoria (art. 949 c.c.).", ctx)
    assert not ptw._wrong_statement("La controparte potrebbe invocare l'azione negatoria (art. 949 c.c.), ma la "
                                    "servitù è acquisita.", ctx)


# --- C1. the usucapione start ----------------------------------------------------------------------------------------

def test_the_start_is_never_a_sale_a_contact_or_a_closure():
    terms = {t["id"]: t for t in fs.compute_terms(H3_FACTS, ["diritti_reali"])}
    usucapione = terms["usucapione_ordinaria"]
    assert usucapione["start_unknown"] and usucapione["status"] == "DA VERIFICARE"
    assert fs.START_UNKNOWN in usucapione["text"] and "2016" not in usucapione["text"] and "2003" not in usucapione["text"]
    bought = {**H3_FACTS, "possession": [{"who": "Leonardo", "asset": "strada sterrata", "since": "2003",
                                          "use": "passaggio", "still_ongoing": "sì"}]}
    assert fs.START_UNKNOWN in {t["id"]: t for t in fs.compute_terms(bought, ["diritti_reali"])}["usucapione_ordinaria"]["text"]


def test_a_stated_length_or_a_real_start_is_kept():
    # n9 v47: "Tizio possiede da più di 30 anni"
    n9 = {"parties": [{"name": "Sempronio", "role": "proprietario"}], "_regimes": True,
          "possession": [{"who": "Tizio", "asset": "stanza", "since": "da più di 30 anni", "still_ongoing": "sì"}]}
    entry = {t["id"]: t for t in fs.compute_terms(n9, ["diritti_reali"])}["usucapione_ordinaria"]
    assert not entry.get("start_unknown") and fs.START_UNKNOWN not in entry["text"]
    since = {**H3_FACTS, "possession": [{"who": "Leonardo", "asset": "strada", "since": "1990", "use": "passaggio",
                                         "still_ongoing": "sì"}]}
    entry = {t["id"]: t for t in fs.compute_terms(since, ["diritti_reali"])}["usucapione_ordinaria"]
    assert "1990" in entry["text"] and not entry.get("start_unknown")


def test_twenty_years_asserted_against_an_unknown_start_is_flagged():
    terms = fs.compute_terms(H3_FACTS, ["diritti_reali"])
    ctx = {"terms": [{k: t.get(k) for k in ("id", "ref", "length", "status", "from", "start_unknown")} for t in terms]}
    # the PO's red, Leonardo v46
    assert ptw._wrong_statement("La servitù apparente (strada visibile) è stata usata in modo continuo e ininterrotto "
                                "per oltre vent’anni, permettendo a Leonardo di accedere ai propri terreni.", ctx)
    assert not ptw._wrong_statement("Occorre accertare se il possesso dura da oltre vent'anni: è da verificare.", ctx)
    assert not ptw._wrong_statement("La servitù si estingue per non uso quando non se ne usa per oltre vent'anni.", ctx)
    assert "inizio del possesso da accertare" in ptw._TERMS_RULE_REGIMES


# --- D1. the main remedy's remedies ----------------------------------------------------------------------------------

ART_1079 = ("Il titolare della servitù può farne riconoscere in giudizio l'esistenza contro chi ne contesta l'esercizio e "
            "può far cessare gli eventuali impedimenti e turbative. Può anche chiedere la rimessione delle cose in "
            "pristino, oltre il risarcimento dei danni.")


def test_the_remedies_of_1079():
    assert ptw.remedies_of(ART_1079) == ["farne riconoscere in giudizio l'esistenza",
                                         "far cessare gli eventuali impedimenti e turbative",
                                         "rimessione delle cose in pristino", "risarcimento dei danni"]
    assert ptw.remedies_of("Il proprietario del fondo servente non può compiere alcuna cosa.") == []


def test_every_remedy_reaches_the_answers_and_the_material(events):
    plan = {"risposte": [{"quesito": "", "risposta": "Leonardo agisce in confessoria (art. 1079 c.c.) e chiede la "
                                                     "rimozione del cancello."}],
            "questioni": [{"titolo": "Servitù", "norme": [("1062", "c.c.")]},
                          {"titolo": "Azione confessoria", "norme": [("1079", "c.c.")]},
                          {"titolo": "In via subordinata: passaggio coattivo", "norme": [("1051", "c.c.")]}]}
    rows = [{"_article": ("1062", "c.c."), "s": {"plain_text": "La destinazione del padre di famiglia ha luogo."}},
            {"_article": ("1079", "c.c."), "s": {"plain_text": ART_1079}}]
    out, log = ptw.enumerate_remedies(plan, rows)
    assert log["article"] == "art. 1079 c.c." and "rimessione delle cose in pristino" in log["missing"]
    assert "rimessione delle cose in pristino" in out["risposte"][0]["risposta"]
    block = ptw._remedies_block(out)
    assert block.startswith(ptw.REMEDIES_TITLE) and "risarcimento dei danni" in block and ptw.REMEDIES_RULE in block


# --- D2. art. 696-bis c.p.c. -----------------------------------------------------------------------------------------

def test_leonardo_s_accertamento_cites_696_alone():
    fired = [{"id": "ostruzione", "tools": [{"id": "accertamento_preventivo", "why": "fissare lo stato dei luoghi"}]}]
    tools = fs.rule_tools([], fired, H3_FACTS)
    assert tools[0]["refs"] == [["696", "c.p.c."]] and "696-bis" not in tools[0]["text"]
    claim = {**H3_FACTS, "requests": ["il risarcimento del danno per l'interclusione"]}
    assert ["696-bis", "c.p.c."] in fs.rule_tools([], fired, claim)[0]["refs"]


# --- D3. the rubrica stamp -------------------------------------------------------------------------------------------

ART_1054 = ("Interclusione per effetto di alienazione o di divisione. Se il fondo è divenuto da ogni parte chiuso per "
            "effetto di alienazione a titolo oneroso, il proprietario ha diritto di ottenere dall'altro contraente il "
            "passaggio senza alcuna indennità. La stessa norma si applica in caso di divisione.")
# the PO's "falso DA VERIFICARE", Leonardo v46
V46 = ("In via subordinata, Leonardo può chiedere il passaggio coattivo ai sensi dell’art. 1051 c.c., in quanto il fondo "
       "è intercluso, e non è dovuta alcuna indennità in base all’art. 1054 c.c.")


def test_the_1054_clause_is_its_own_not_1051_s():
    assert dg._citing_clause(V46, ("1054", "c.c.")) == ("in quanto il fondo è intercluso, e non è dovuta alcuna "
                                                         "indennità in base all’art. 1054 c.c.")
    assert dg._text_covers(V46, ("1054", "c.c."), ART_1054)
    assert dg._citing_clause("Ai sensi dell'art. 1061 c.c. la servitù è apparente.", ("1061", "c.c.")).startswith("Ai sensi")


# --- E. diagnose_long, the eval checks, oversized_sections -----------------------------------------------------------

def test_llm_calls_are_counted_and_read_off_old_dumps(tmp_path, monkeypatch):
    from src.rag import ai_chat
    monkeypatch.setattr(ai_chat, "_call_chat", lambda messages, *a, **k: "ok")
    monkeypatch.setattr(dg, "_call_chat", ai_chat._call_chat)
    stats = dl._count_llm(dg)
    assert dg._call_chat([]) == "ok" and stats["calls"] == 1 and stats["errors"] == 0
    for i, (seconds, status) in enumerate(((2.5, "ok"), (0.5, "ERROR BadRequestError: x"))):
        (tmp_path / f"{i:02d}_section.txt").write_text(f"stage: s\nduration: {seconds}s\nstatus: {status}\n")
    assert dl.llm_from_dumps(str(tmp_path)) == {"calls": 2, "seconds": 3.0, "errors": 1, "from": "dumps"}
    assert dl.llm_from_dumps(str(tmp_path / "none")) == {}


def test_h3_checks_name_state_and_forbid():
    cases = json.loads((HERE / "eval_narrative_cases.json").read_text(encoding="utf-8"))
    items = cases["cases"] if isinstance(cases, dict) else cases
    h3 = next(c for c in items if c["id"] == "h3")
    assert h3["must_name"] == ["Tizio", "Caio", "Annibale"] and h3["checks_status"].startswith("draft")
    bad = ev.checks(h3, "Leonardo agisce in negatoria (art. 949 c.c.) contro Tizio e Caio.")
    assert bad["forbidden_cited"] == ["art. 949 cc"] and bad["must_name_missing"] == ["Annibale"]
    assert len(bad["must_state_missing"]) == 3
    good = ev.checks(h3, "Contro Tizio, Caio e Annibale (art. 102 c.p.c.): la rimessione in pristino, il risarcimento e "
                         "la trascrizione della domanda (art. 2653 c.c.).")
    assert not good["failures"]
    rec = {"id": "h3", "draft": "Tizio, Caio e Annibale."}
    assert dl.case_checks(rec, ev)["must_state_missing"]


def test_the_oversized_sections_report():
    rows = [{"document": "Codice degli Appalti 2026", "type": "primary", "section": "229.0.0",
             "rubrica": "Entrata in vigore", "chars": 108000, "after": "Allegato I.5 | elementi"},
            {"document": "Testo Unico sulla Sicurezza del Lavoro 2026", "type": "primary", "section": "306.0.0",
             "rubrica": "Disposizioni transitorie", "chars": 76000, "after": "ALLEGATO I"}]
    lines = oversized.report(rows, 20000)
    assert "2 Sections in 2 documents" in lines[2]
    assert lines.index("## Codice degli Appalti 2026") < lines.index("## Testo Unico sulla Sicurezza del Lavoro 2026")
    assert any("108,000" in ln and "Allegato I.5 / elementi" in ln for ln in lines)
    assert "MATCH" in oversized.QUERY and "SET" not in oversized.QUERY and "DELETE" not in oversized.QUERY


# --- v48b: the probe's texts, ROLE_FIT shadow, the action gate, --exclude-type ----------------------------------------

_probe_spec = importlib.util.spec_from_file_location("role_fit_probe", HERE.parent / "scripts" / "role_fit_probe.py")
probe = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(probe)
ART_949 = ("Il proprietario può agire per far dichiarare l'inesistenza di diritti affermati da altri sulla cosa, quando "
           "ha motivo di temerne pregiudizio.")


def test_the_probe_reads_949_s_text_off_material_and_the_applicability_prompt(tmp_path):
    # h3 v47: 949 was in no section's TESTI DELLE NORME; material.txt and 08_applicability.txt hold it
    (tmp_path / "llm").mkdir()
    (tmp_path / "material.txt").write_text(f"[art. 949 c.c.]\n>> provenance: institution\n(co. 1) {ART_949}\n\n")
    (tmp_path / "llm" / "08_applicability.txt").write_text(
        f"[A5] 949 c.c.\nRubrica: Azione negatoria.\nTrovato da: più fonti\nTesto: {ART_949}\n\n[A6] 1079 c.c.\n"
        "Rubrica: -\nTesto: Il titolare della servitù può farne riconoscere l'esistenza.\n")
    texts = probe.article_texts(str(tmp_path))
    assert texts["949 c.c."] == ("Azione negatoria.", ART_949)
    assert texts["1079 c.c."][0] == "" and texts["1079 c.c."][1].startswith("Il titolare")


def _h3_plan():
    return {"parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
            "subordinate": [], "punti_deboli": [],
            "questioni": [{"titolo": "Azione confessoria e negatoria", "norme": [("1079", "c.c."), ("949", "c.c."),
                                                                              ("1062", "c.c.")],
                           "pronunce": [], "argomento": "A.", "obiezione": "", "replica": ""}]}


def _h3_rows(text_949=ART_949):
    return [{"_article": ("1079", "c.c."), "_grade": 3, "s": {"plain_text": ART_1079}},
            {"_article": ("949", "c.c."), "_grade": 2, "s": {"plain_text": text_949, "titolo_articolo": "Azione negatoria"}},
            {"_article": ("1062", "c.c."), "_grade": 3, "s": {"plain_text": "La destinazione del padre di famiglia."}}]


def _answers(monkeypatch, asked):
    def chat_json(messages, **kwargs):
        human = messages[1].content
        asked.append(human)
        return {"a_favore": "controparte" if "art. 949" in human else "cliente", "perche": "p"}
    monkeypatch.setattr(dg, "_chat_json", chat_json)


def test_an_article_without_its_text_is_not_asked(monkeypatch, events):
    asked = []
    _answers(monkeypatch, asked)
    _, log = role_fit.check(_h3_plan(), _h3_rows(text_949=""), "caso", {"facts": H3_FACTS, "fired": H3_FIRED})
    assert {e["article"]: e["answer"] for e in log}["949 c.c."] == "no text" and len(asked) == 2
    assert all("NORMA: art." in h and "\n" in h.split("NORMA: art.")[1] for h in asked)


def test_shadow_asks_and_logs_but_tags_nothing(monkeypatch, events):
    monkeypatch.setenv("ROLE_FIT", "shadow")
    assert role_fit.mode() == "shadow" and not role_fit.enabled()
    asked = []
    _answers(monkeypatch, asked)
    plan = _h3_plan()
    out, log = role_fit.check(plan, _h3_rows(), "caso", {"facts": H3_FACTS, "fired": H3_FIRED}, shadow=True)
    assert out is plan and "norme_contrarie" not in out and len(asked) == 3
    by = {e["article"]: e for e in log}
    assert by["949 c.c."]["deterministic"] == "controparte" and by["949 c.c."]["agrees"]
    assert by["1079 c.c."]["deterministic"] == "cliente" and by["1079 c.c."]["agrees"]
    assert "deterministic" not in by["1062 c.c."]
    entry = next(d for n, d in events if n == "case_analysis_role_fit")
    assert entry["mode"] == "shadow" and entry["agreement"] == {"compared": 2, "agree": 2}
    assert "norma della controparte" not in ptw._plan_text(out)
    monkeypatch.setenv("ROLE_FIT", "true")
    assert role_fit.mode() == "on"


def test_the_action_gate_on_h3_949_adverse_1079_kept(monkeypatch, events):
    # h3 v47's own resolutions: "azione confessoria e negatoria" -> 949 c.c. (fact rule possessoria_preclusa)
    monkeypatch.setenv("REGIME_CANDIDATES", "true")
    analysis = {"fact_signals": {"facts": H3_FACTS, "fired": H3_FIRED}}
    rows = [{"_article": ("949", "c.c.")}, {"_article": ("1079", "c.c.")}]
    gated = dg._action_gate(analysis, "azione confessoria e negatoria", rows)
    assert set(gated) == {("949", "c.c.")} and "negatoria" in gated[("949", "c.c.")]
    assert list(analysis["action_gate"]) == ["949 c.c."]
    logged = {d["article"]: d["fits"] for n, d in events if n == "case_analysis_action_gate"}
    assert logged == {"949 c.c.": False, "1079 c.c.": True}
    assert dg._action_gate(analysis, "decadenza dall'azione di reintegrazione e di manutenzione",
                           [{"_article": ("1168", "c.c.")}]) == {}            # Leonardo possessed: his
    assert dg._action_gate(analysis, "servitù di passaggio", [{"_article": ("949", "c.c.")}]) == {}   # no action named
    monkeypatch.delenv("REGIME_CANDIDATES")
    assert dg._action_gate({"fact_signals": {"facts": H3_FACTS, "fired": H3_FIRED}}, "azione negatoria", rows) == {}


def test_the_owner_resisting_keeps_the_negatoria():
    roles = fs.client_roles({"parties": [{"name": "Sempronio", "role": "cliente, proprietario"}],
                             "client_side": "Sempronio", "requests": ["Sempronio contesta la servitù di passaggio"]})
    assert fs.action_gate("azione negatoria", ("949", "c.c."), roles)["fits"]
    assert not fs.action_gate("azione confessoria", ("1079", "c.c."), roles)["fits"]


def test_a_gated_plan_article_is_tagged_adverse_without_a_call(monkeypatch, events):
    analysis = {"action_gate": {"1062 c.c.": "azione negatoria: spetta a chi ..."}}
    plans = []
    monkeypatch.setattr(ptw, "_plan_json", lambda messages: plans.append(messages) or FIRST)
    monkeypatch.setattr(dg, "_opposing_arguments", lambda *a, **k: [])
    monkeypatch.setattr(ptw, "_issue_check", lambda issue, rows, headings: [
        {"articolo": "art. 1062 c.c.", "pertinente": True, "contraddice": False}])
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: pytest.fail("no call for the gate"))
    rows = [{"_article": ("1062", "c.c."), "_grade": 3, "s": {"plain_text": "La destinazione.", "titolo_articolo": "R"}}]
    plan = ptw.make_plan("Fatti.", "", analysis, "[art. 1062 c.c.]", "", "", rows, {})
    assert plan["norme_contrarie"] == ["1062 c.c."]
    assert "art. 1062 c.c. (norma della controparte)" in ptw._plan_text(plan)


def test_diagnose_long_reads_the_shadow_answers_against_the_forbidden_list():
    rec = {"id": "h3", "events": [
        ["case_analysis_role_fit", {"mode": "shadow", "adverse": ["949 c.c."], "agreement": {"compared": 2, "agree": 2},
                                    "articles": [{"article": "949 c.c.", "answer": "controparte"},
                                                 {"article": "1079 c.c.", "answer": "cliente"}]}, None],
        ["case_analysis_action_gate", {"article": "949 c.c.", "fits": False}, None]]}
    trace = dl.role_fit_trace(rec)
    assert trace["forbidden"] == [{"article": "949 c.c.", "answer": "controparte"}]
    assert trace["gate_adverse"] == ["949 c.c."] and trace["mode"] == "shadow" and trace["asked"] == 2
    assert dl.role_fit_trace({"id": "p1", "events": []}) == {}


def test_oversized_sections_leaves_a_document_type_out():
    assert "NOT toLower(coalesce(d.document_type, '')) IN $exclude" in oversized.QUERY
    lines = oversized.report([], 20000, ["ccnl"])
    assert "Document types left out: ccnl." in lines[2]
