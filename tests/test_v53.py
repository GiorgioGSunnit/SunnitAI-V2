"""v53, phase 1: RULE_QUOTE (the comma's own words), RENDER_COMPUTED (the procedural section by template, the action gate's
articles only as objections), TOPIC_STEP (the case's topics) and diagnose_long --topics-only's scoring. The article texts
are the codes' own; the cases synthetic. No model, no database."""
import json
import os
import re

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402
from src.rag import render_computed, rule_quote, topic_step  # noqa: E402

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

ART_1067 = ("Il proprietario del fondo dominante non può fare innovazioni che rendano più gravosa la condizione del fondo "
            "servente.\nIl proprietario del fondo servente non può compiere alcuna cosa che tenda a diminuire l'esercizio "
            "della servitù o a renderlo più incomodo.")
ART_1062 = ("La destinazione del padre di famiglia ha luogo quando consta, mediante qualunque genere di prova, che due "
            "fondi, attualmente divisi, sono stati posseduti dallo stesso proprietario, e che questi ha posto o lasciato "
            "le cose nello stato dal quale risulta la servitù.\nSe i due fondi cessarono di appartenere allo stesso "
            "proprietario, senza alcuna disposizione relativa alla servitù, questa s'intende stabilita attivamente e "
            "passivamente a favore e sopra ciascuno dei fondi separati.\nCAPO V – DELL'ESERCIZIO DELLE SERVITU'")
ART_1054 = ("Se il fondo è divenuto da ogni parte chiuso per effetto di alienazione a titolo oneroso, il proprietario ha "
            "diritto di ottenere dall'altro contraente il passaggio senza alcuna indennità.\nLa stessa norma si applica "
            "in caso di divisione.")
ART_1170 = ("Chi è stato molestato nel possesso di un immobile, di un diritto reale sopra un immobile o di un'universalità "
            "di mobili può, entro l'anno dalla turbativa, chiedere la manutenzione del possesso medesimo.\nL'azione è data "
            "se il possesso dura da oltre un anno, continuo e non interrotto, e non è stato acquistato violentemente o "
            "clandestinamente.")
ART_949 = ("Il proprietario può agire per far dichiarare l'inesistenza di diritti affermati da altri sulla cosa, quando ha "
           "motivo di temerne pregiudizio.\nSe sussistono anche turbative o molestie, il proprietario può chiedere che se "
           "ne ordini la cessazione, oltre la condanna al risarcimento del danno.")

DOMINANT, SERVIENT = {"dominante", "titolare", "chiede", "proprietario"}, {"servente", "concede"}


def _row(number, text, code="c.c."):
    return {"d": {"id": f"LEGAL_DOC::{code}", "name": "Codice Civile 2026"}, "_article": (number, code),
            "s": {"name": f"{number}.0.0", "plain_text": text}}


ROWS = [_row("1067", ART_1067), _row("1062", ART_1062), _row("1054", ART_1054), _row("949", ART_949)]
FACTS = {"parties": [{"name": "Leonardo", "role": "cliente, proprietario"},
                     {"name": "Domitilla", "role": "acquirente del fondo"}],
         "possession": [{"who": "Leonardo", "asset": "strada sterrata", "use": "passaggio"}],
         "property_history": [{"owner": "Leonardo", "asset": "terreni"}]}
ANALYSIS = {"parte_assistita": "Leonardo", "fact_signals": {"facts": FACTS, "fired": []}}
PLAN = {"parte_assistita": "Leonardo", "posizione": "proprietario del fondo intercluso", "obiettivo": "",
        "questione_centrale": "", "risposte": [], "subordinate": [], "punti_deboli": [],
        "questioni": [{"titolo": "Servitù per destinazione del padre di famiglia", "argomento": "", "obiezione": "",
                       "replica": "", "pronunce": [], "norme": [("1062", "c.c."), ("1067", "c.c."), ("1054", "c.c.")]}]}


# --- A. RULE_QUOTE ---------------------------------------------------------------------------------------------------

def test_the_client_side_selects_the_comma_that_binds_the_other_side():
    commi = rule_quote.commi_of(ART_1067)
    assert [k for k, _ in rule_quote.operative(commi, DOMINANT, SERVIENT)[0]] == [2]
    assert [k for k, _ in rule_quote.operative(commi, {"servente", "proprietario"}, {"dominante"})[0]] == [1]


def test_no_comma_naming_one_side_passes_every_comma_numbered_without_the_next_chapter():
    commi = rule_quote.commi_of(ART_1062)
    chosen, how = rule_quote.operative(commi, DOMINANT, SERVIENT)
    assert how == "all" and [k for k, _ in chosen] == [1, 2]
    assert not any("CAPO" in t for _, t in commi)


def test_a_comma_naming_no_role_stays_with_the_selected_one():
    chosen, how = rule_quote.operative(rule_quote.commi_of(ART_1170), {"possessore"}, {"proprietario"})
    assert how == "role" and [k for k, _ in chosen] == [1, 2]          # co. 2: the action's own requisites


def test_requisites_split_on_the_connectors_and_the_enumerations():
    first = rule_quote.commi_of(ART_1062)[0][1]
    assert rule_quote.requisites(first) == [
        "consta, mediante qualunque genere di prova, che due fondi, attualmente divisi, sono stati posseduti dallo "
        "stesso proprietario", "che questi ha posto o lasciato le cose nello stato dal quale risulta la servitù"]
    assert rule_quote.requisites(rule_quote.commi_of(ART_1054)[0][1]) == [
        "il fondo è divenuto da ogni parte chiuso per effetto di alienazione a titolo oneroso"]     # the main clause cut
    assert rule_quote.requisites(rule_quote.commi_of(ART_1170)[1][1]) == [
        "il possesso dura da oltre un anno, continuo e non interrotto",
        "non è stato acquistato violentemente o clandestinamente"]
    assert rule_quote.requisites(rule_quote.commi_of(ART_949)[1][1]) == ["sussistono anche turbative o molestie"]
    assert rule_quote.requisites("Salvo che il fatto costituisca più grave reato, è punito chiunque cagiona un danno.") \
        == ["salvo che il fatto costituisca più grave reato"]


def test_the_block_numbers_two_to_eight_requisites_and_quotes_the_rest_whole(monkeypatch):
    assert rule_quote.material_block(PLAN, ROWS, ANALYSIS) == ""                     # flag off
    monkeypatch.setenv(rule_quote.FLAG, "true")
    block = rule_quote.material_block(PLAN, ROWS, ANALYSIS)
    assert block.startswith(f"{rule_quote.TITLE}:\n") and block.endswith(rule_quote.RULE_TEXT)
    assert "co. 1: requisiti (dal testo del comma):\n    1. «consta, mediante" in block              # 1062: two
    assert "co. 2: «Se i due fondi cessarono" in block                                            # one: whole
    assert "art. 1067 c.c. — il comma che opera per Leonardo:\n  co. 2: «Il proprietario del fondo servente" in block
    assert "fondo dominante non può fare innovazioni" not in block
    assert "co. 1: «Se il fondo è divenuto da ogni parte chiuso" in block                         # 1054: quoted


def test_an_article_of_the_other_side_gets_the_commi_that_work_for_it(monkeypatch):
    monkeypatch.setenv(rule_quote.FLAG, "true")
    plan = {**PLAN, "norme_contrarie": ["1067 c.c."]}
    rule = next(r for r in rule_quote.article_rules(plan, ROWS, ANALYSIS) if r["label"] == "1067 c.c.")
    assert rule["adverse"] and rule["selected"] == [1]
    assert "art. 1067 c.c. (norma della controparte)" in rule_quote.material_block(plan, ROWS, ANALYSIS)


def _check(text, plan=PLAN):
    rules = rule_quote.article_rules(plan, ROWS, ANALYSIS)
    return rule_quote.check_text("Norme e giurisprudenza", text, rules, rule_quote.party_names(ANALYSIS, plan), [])


def test_a_rule_sentence_adding_a_requisite_becomes_the_comma_and_the_application_stays():
    text = ("L'art. 1054 c.c. richiede che il fondo sia divenuto intercluso per una vendita e che non esista alcuna altra "
            "strada percorribile con mezzi agricoli. Nel caso, il fondo di Leonardo è rimasto chiuso dopo la vendita.")
    out, log, judged = _check(text)
    assert judged == 1 and len(log) == 1 and log[0]["article"] == "1054 c.c." and log[0]["how"] == "quoted"
    assert out == ("L'art. 1054, co. 1, c.c. dispone: «Se il fondo è divenuto da ogni parte chiuso per effetto di "
                   "alienazione a titolo oneroso, il proprietario ha diritto di ottenere dall'altro contraente il passaggio "
                   "senza alcuna indennità.» Nel caso, il fondo di Leonardo è rimasto chiuso dopo la vendita.")
    assert "altra strada percorribile" in log[0]["sentence"]


def test_a_sentence_within_its_comma_a_quotation_and_a_sentence_naming_a_party_stay():
    within = ("L'art. 1067 c.c. vieta al proprietario del fondo servente di compiere alcuna cosa che tenda a diminuire "
              "l'esercizio della servitù o a renderlo più incomodo.")
    quoted = "L'art. 1062 c.c. stabilisce che «due fondi, attualmente divisi, sono stati posseduti dallo stesso proprietario»."
    party = "Ai sensi dell'art. 1054 c.c. Leonardo non deve alcuna indennità a Domitilla."
    objection = "La controparte potrebbe sostenere che ai sensi dell'art. 1054 c.c. sia dovuta un'indennità."
    for text in (within, quoted, party, objection):
        assert _check(text)[0] == text and not _check(text)[1]


def test_a_quotation_of_another_comma_gets_its_own_number():
    text = "L'art. 1062, co. 1, c.c. dispone che «questa s'intende stabilita attivamente e passivamente»."
    out, log, _ = _check(text)
    assert log[0]["how"] == "renumbered" and out == text.replace("co. 1", "co. 2")


def test_the_verb_belongs_to_the_nearest_citation_and_each_clause_is_judged_alone():
    text = ("L'art. 1067 c.c. vieta al proprietario del fondo servente di compiere alcuna cosa che tenda a diminuire "
            "l'esercizio della servitù; l'art. 1054 c.c. esclude ogni indennità per i terreni agricoli irrigui destinati a vigneto.")
    clauses = rule_quote.rule_clauses(text, rule_quote.article_rules(PLAN, ROWS, ANALYSIS), [])
    assert [r["label"] for r, _ in clauses] == ["1067 c.c.", "1054 c.c."]
    assert clauses[1][1].startswith("l'art. 1054 c.c. esclude") and "servente" not in clauses[1][1]
    out, log, judged = _check(text)
    assert judged == 2 and [e["article"] for e in log] == ["1054 c.c."]
    assert out == ("L'art. 1067 c.c. vieta al proprietario del fondo servente di compiere alcuna cosa che tenda a "
                   "diminuire l'esercizio della servitù. L'art. 1054, co. 1, c.c. dispone: «Se il fondo è divenuto da "
                   "ogni parte chiuso per effetto di alienazione a titolo oneroso, il proprietario ha diritto di ottenere "
                   "dall'altro contraente il passaggio senza alcuna indennità.»")     # the clause that passed stays
    text = ("L'art. 1067 c.c. vieta al proprietario del fondo servente di compiere alcuna cosa che tenda a diminuire "
            "l'esercizio della servitù, ma l’art. 1054 c.c. esclude ogni indennità per i terreni agricoli irrigui destinati a vigneto.")
    assert _check(text)[0].startswith("L'art. 1067 c.c. vieta al proprietario del fondo servente di compiere alcuna "
                                      "cosa che tenda a diminuire l'esercizio della servitù. L'art. 1054, co. 1, c.c.")


def test_a_rendered_section_is_never_checked(monkeypatch):
    text = "L'art. 1054 c.c. richiede che non esista alcuna altra strada percorribile con mezzi agricoli."
    sections = [("Norme e giurisprudenza", text), (render_computed.SECTION_TITLE, text)]
    out, log = rule_quote.check_sections(sections, PLAN, ROWS, ANALYSIS, [], skip=[render_computed.SECTION_TITLE])
    assert out[1][1] == text and out[0][1] != text and log["replaced"] == 1 and log["by_article"] == {"1054 c.c.": 1}


# --- B. RENDER_COMPUTED ----------------------------------------------------------------------------------------------

SIGNALS = {
    "facts": FACTS, "fired": [],
    "mediation": {"required": True, "subject": "diritti reali",
                  "text": "Mediazione obbligatoria (art. 5 d.lgs. 28/2010): la materia è diritti reali, il suo "
                          "esperimento è condizione di procedibilità della domanda."},
    "terms": [{"id": "possessoria", "ref": "artt. 1168 e 1170 c.c.", "text": "azione possessoria: termine annuale DECORSO"},
              {"id": "azione_confessoria", "ref": "art. 1079 c.c.", "text": "azione confessoria: non soggetta a prescrizione"}],
    "tools": [{"id": "provvedimento_urgenza", "refs": [["700", "c.p.c."]], "text": "Provvedimento d'urgenza (art. 700 c.p.c.).",
               "matched": ["l'ostacolo è ancora in atto"]},
              {"id": "trascrizione_domanda", "refs": [["2653", "c.c."]],
               "text": "Altre domande e atti soggetti a trascrizione. (art. 2653 c.c.).",
               "matched": ["trascrizione della domanda giudiziale (azione confessoria) ai sensi dell'art. 2653, n. 1, c.c."]},
              {"id": "competenza_territoriale", "refs": [["21", "c.p.c."]],
               "text": "Foro per le cause relative a diritti reali (art. 21 c.p.c.).", "matched": ["servitù"]},
              {"id": "litisconsorzio", "refs": [["102", "c.p.c."]], "text": "Litisconsorzio necessario (art. 102 c.p.c.).",
               "matched": ["parti da convenire: Domitilla"]},
              {"id": "negatoria", "refs": [["949", "c.c."]], "text": "Azione negatoria (art. 949 c.c.).", "matched": ["x"]}],
    "defendants": [{"name": "Domitilla", "why": "autore dell'atto: appone un cancello"}]}


def test_the_section_follows_the_template_mediation_first(monkeypatch):
    analysis = {"fact_signals": SIGNALS, "action_gate": {"949 c.c.": "azione negatoria: non al cliente"}}
    text = render_computed.render(analysis)
    titles = re.findall(r"^\*\*(.+?)\*\*$", text, re.MULTILINE)
    assert titles == ["Mediazione", "Termini", "Strumenti processuali", "Parti da convenire", "Competenza",
                      "Trascrizione della domanda giudiziale (art. 2653, n. 1, c.c.)"]
    assert "- azione possessoria: termine annuale DECORSO (artt. 1168 e 1170 c.c.)." in text
    assert ("- Provvedimento d'urgenza (art. 700 c.p.c.). Nel caso: l'ostacolo è ancora in atto." in text)
    assert "Va trascritta la domanda giudiziale (azione confessoria)." in text and "c.c.." not in text
    assert "Cause relative a diritti reali su immobili: giudice del luogo in cui si trova l'immobile (art. 21 c.p.c.)." \
        in text and "«" not in text.split("**Competenza**")[1]           # no institution labels
    assert "949" not in text                                       # the gated tool is not rendered
    gated_term = {**SIGNALS, "terms": SIGNALS["terms"] + [{"id": "azione_negatoria", "ref": "art. 949 c.c.",
                                                          "text": "azione negatoria: non soggetta a prescrizione"}]}
    assert "negatoria" not in render_computed.render({**analysis, "fact_signals": gated_term})     # nor its term
    assert "Litisconsorzio" in text and "Parti da convenire: Domitilla (autore dell'atto" in text   # no parties line
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    monkeypatch.setattr(render_computed, "_signals", lambda a: SIGNALS)
    from src.rag import fact_signals
    monkeypatch.setattr(fact_signals, "parties_line", lambda s, a=None: "Parti da convenire: Domitilla, Tizio (art. 102 c.p.c.)")
    text = render_computed.render(analysis)
    assert "Litisconsorzio" not in text and "Parti da convenire: Domitilla, Tizio (art. 102 c.p.c.)." in text


def test_nothing_computed_renders_nothing():
    assert render_computed.render({"fact_signals": {"terms": [], "tools": []}}) == ""


def test_the_heads_of_damage_are_the_other_sides_for_an_accused_client():
    signals = {"damages": [{"id": "x", "lines": ["Iure proprio, per ciascun familiare: il danno."]}]}
    assert render_computed.blocks({"fact_signals": signals}, "Tizio (imputato)")[0][0] == \
        "Voci di danno che la controparte può far valere"
    assert render_computed.blocks({"fact_signals": signals}, "")[0][0] == "Voci di danno da considerare"


def test_the_gated_article_leaves_the_remedies_for_the_objections_block():
    plan = {"parte_assistita": "Leonardo", "questioni": [
        {"titolo": "Tutela della servitù", "norme": [("1079", "c.c."), ("949", "c.c.")]},
        {"titolo": "Negatoria", "norme": [("949", "c.c.")]}], "rimedi": {"article": "art. 949 c.c.", "remedies": ["a"]}}
    analysis = {"action_gate": {"949 c.c.": "azione negatoria: spetta a chi resists_servitude, non al cliente"}}
    moved_plan, moved = render_computed.move_gated(plan, analysis)
    assert [q["norme"] for q in moved_plan["questioni"]] == [[("1079", "c.c.")], []]
    assert "rimedi" not in moved_plan and [m["issue"] for m in moved] == ["Tutela della servitù", "Negatoria"]
    assert [o["article"] for o in moved_plan["obiezioni_norme"]] == ["949 c.c."]
    assert "949 c.c." in moved_plan["norme_contrarie"]
    block = render_computed.objections_block(moved_plan)
    assert block.startswith("OBIEZIONI DELLA CONTROPARTE:\n- art. 949 c.c.: azione negatoria: spetta a chi contesta una "
                            "servitù altrui, non al cliente") and "Leonardo" in block
    text = ("Leonardo può agire in negatoria (art. 949 c.c.) per far cessare la turbativa. La controparte potrebbe "
            "invocare l'art. 949 c.c., ma la replica è che Leonardo vanta la servitù.")
    out, log = render_computed.gated_as_support("Applicazione al caso", text, moved_plan)
    assert out.startswith("La controparte potrebbe invocare") and log[0]["article"] == "949 c.c."


def test_the_rendered_section_is_restored_after_an_edit():
    analysis = {render_computed.RENDERED_KEY: "**Termini**\n\n- uno."}
    draft = "**4. Applicazione al caso**\n\nA.\n\n**5. Profili processuali e termini**\n\n- uno modificato.\n\n**6. Conclusioni**\n\nC."
    out, changed = render_computed.restore(draft, analysis)
    assert changed and out == ("**4. Applicazione al caso**\n\nA.\n\n**5. Profili processuali e termini**\n\n**Termini**\n\n- uno."
                               "\n\n**6. Conclusioni**\n\nC.")
    assert render_computed.restore(out, analysis) == (out, False)


def test_section_jobs_with_a_rendered_section_have_no_procedural_job(monkeypatch):
    monkeypatch.setenv(rule_quote.FLAG, "true")
    plan = {**PLAN, "obiezioni_norme": [{"article": "949 c.c.", "reason": "azione negatoria: non al cliente", "issue": "x"}]}
    jobs = ptw.section_jobs(plan, "fatti", "", ANALYSIS, ROWS, {}, "", rendered="**Termini**\n\n- uno.")
    titles = [j[0] for j in jobs]
    assert render_computed.SECTION_TITLE not in titles and len(titles) == 5
    for title in ("Risposte ai quesiti", "Applicazione al caso", "Conclusioni"):
        assert render_computed.READ_ONLY_TITLE in next(j[2] for j in jobs if j[0] == title)
    for title in ("Norme e giurisprudenza", "Applicazione al caso"):
        material = next(j[2] for j in jobs if j[0] == title)
        assert render_computed.OBJECTIONS_TITLE in material and rule_quote.TITLE in material
    assert render_computed.READ_ONLY_TITLE not in next(j[2] for j in jobs if j[0] == "Norme e giurisprudenza")
    plain = ptw.section_jobs(PLAN, "fatti", "", ANALYSIS, ROWS, {}, "")
    assert render_computed.SECTION_TITLE in [j[0] for j in plain]
    assert all(render_computed.READ_ONLY_TITLE not in j[2] for j in plain)


class _Model:
    def __init__(self):
        self.systems = []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        system = messages[0].content
        self.systems.append(system)
        if "PIANO, nell'interesse" in system:
            return json.dumps({"parte_assistita": "Leonardo", "posizione": "proprietario del fondo intercluso",
                               "obiettivo": "il passaggio", "questione_centrale": "la servitù",
                               "risposte": [{"quesito": "", "risposta": "Sì."}],
                               "questioni": [{"titolo": "Destinazione del padre di famiglia",
                                              "norme": ["art. 1062 c.c.", "art. 1054 c.c."], "pronunce": [],
                                              "argomento": "I fondi erano di Mevio.", "obiezione": "", "replica": ""}],
                               "subordinate": [], "punti_deboli": []})
        if "controlli il piano" in system:
            return "[]"
        title = next(t for t in ptw.SECTION_TITLES if f"SEZIONE: {t}" in system)
        if title == "Norme e giurisprudenza":
            return ("L'art. 1054 c.c. richiede che non esista alcuna altra strada percorribile con mezzi agricoli. "
                    "Nel caso il fondo è chiuso.")
        return f"Testo della sezione {title}."


def test_write_parere_renders_section_5_and_checks_the_rules(monkeypatch):
    for flag in ("PLAN_THEN_WRITE_ENABLED", rule_quote.FLAG, render_computed.FLAG):
        monkeypatch.setenv(flag, "true")
    logged = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    model = _Model()
    monkeypatch.setattr(dg, "_call_chat", model)
    analysis = {**ANALYSIS, "fact_signals": {**SIGNALS, "tools": SIGNALS["tools"][:1]}, "questioni": [],
                "frame": {"tasks": ["difesa di Leonardo"]}}
    out = ptw.write_parere("Leonardo ha un fondo intercluso.", "", analysis, ROWS, [], "NORME", "", "", "it")
    assert not any("SEZIONE: Profili processuali" in s for s in model.systems)
    section5 = out.split("**5. Profili processuali e termini**\n\n")[1].split("\n\n**6.")[0]
    assert section5 == analysis[render_computed.RENDERED_KEY] and section5.startswith("**Mediazione**")
    assert "L'art. 1054, co. 1, c.c. dispone: «Se il fondo è divenuto" in out and "Nel caso il fondo è chiuso." in out
    quote = [d for e, d in logged if e == "case_analysis_rule_quote"][-1]
    assert quote["replaced"] == 1 and quote["entries"][0]["article"] == "1054 c.c."
    assert [d for e, d in logged if e == "case_analysis_render_computed"][-1]["blocks"][0] == "Mediazione"


# --- C. TOPIC_STEP ---------------------------------------------------------------------------------------------------

def _pool(number, rubrica, rank, multi=True, code="c.c."):
    return {"_article": (number, code), "_multi_signal": multi, "_fused_rank": rank, "s": {"titolo_articolo": rubrica}}


UNITS = [("c.c.", "CAPO", "CAPO II – DELLE SERVITU’ COATTIVE", n) for n in ("1051", "1052", "1053", "1054", "1055")] + \
        [("c.c.", "SEZIONE", "SEZIONE IV – Del passaggio coattivo", n) for n in ("1051", "1052", "1053", "1054", "1055")] + \
        [("c.c.", "CAPO", "CAPO IV – DELLE SERVITU’ ACQUISTATE PER USUCAPIONE E PER DESTINAZIONE", n) for n in ("1061", "1062")]
REFERS = [("1052 c.c.", "1051 c.c."), ("1032 c.c.", "1051 c.c."), ("1051 c.c.", "1 c.p.c.")] + \
         [(f"{n} c.c.", "1 c.p.c.") for n in range(2000, 2041)]                        # 1 c.p.c.: a hub
RUBRICHE = {("1051", "c.c."): "Passaggio coattivo.", ("1062", "c.c."): "Destinazione del padre di famiglia.",
            ("1061", "c.c."): "Servitù non apparenti.", ("37", "c.c."): "Fondo comune.", ("52", "c.p."): "Difesa legittima",
            ("1079", "c.c."): "Accertamento della servitù e altri provvedimenti di tutela."}
POOL = [_pool("1051", "Passaggio coattivo.", 1), _pool("1061", "Servitù non apparenti.", 7),
        _pool("1054", "Interclusione per effetto di alienazione o di divisione.", 3), _pool("1168", "Azione di reintegrazione.", 9)]
WORK = {"what": "strada sterrata", "visible": "sì", "before_separation": ""}
TOPIC_FACTS = {"parties": FACTS["parties"], "property_history": [
    {"owner": "Mevio", "asset": "terreni di Leonardo e di Sempronio", "ended_by": "divisione"},
    {"owner": "Leonardo", "asset": "terreno"}, {"owner": "Domitilla", "asset": "terreno di Sempronio",
                                                "how_acquired": "compravendita", "from": "2003"}],
    "works_or_signs": [WORK], "events": [], "possession": []}


def _topic_analysis(fired=()):
    return {"frame": {"areas": ["diritti_reali"], "central_issue_candidates": [
                {"nome": "Passaggio coattivo", "codice": "c.c."}, {"nome": "Divisione di un fondo comune", "codice": "c.c."},
                {"nome": "Decadenza dall'azione di reintegrazione", "codice": "c.c."}]},
            "questioni": [{"titolo": "Q", "istituti": [{"nome": "Legittima difesa", "codice": "c.p."}]}],
            "fact_signals": {"facts": TOPIC_FACTS, "fired": list(fired), "terms": [], "tools": [], "areas": ["diritti_reali"]},
            "institution_refs": {"Decadenza dall'azione di reintegrazione": ["1168 c.c."]}}


def test_institution_names_match_rubriche_exactly_by_suffix_after_a_category_or_in_any_order():
    assert topic_step.name_match("passaggio coattivo", "Passaggio coattivo.") == "exact"
    assert topic_step.name_match("Legittima difesa", "Difesa legittima") == "exact"
    assert topic_step.name_match("servitù per destinazione del padre di famiglia",
                                 "Destinazione del padre di famiglia.") == "suffix"
    assert topic_step.name_match("Divisione di un fondo comune", "Fondo comune.") == ""     # the head word dropped
    assert topic_step.name_match("Passaggio", "Passaggio coattivo.") == ""


def test_topics_are_institutions_anchored_by_rubrica_with_their_sezione_links_and_capo(monkeypatch):
    links = topic_step.CsvLinks(UNITS, REFERS, RUBRICHE)
    others = []
    topics = topic_step.build(_topic_analysis(), POOL, links, "", others=others)
    by = {t["title"]: t for t in topics + others}
    # the destinazione rule fires once the road's existence at the separation (unknown) is taken as yes
    dest = by["Servitù per destinazione del padre di famiglia"]
    assert dest["anchor"] == "1062 c.c." and dest["how"] == "suffix" and dest["source"].startswith("regola da verificare")
    assert dest["articles"] == ["1062 c.c.", "1061 c.c."]                 # no Sezione, no link; 1061 two-signal in the Capo
    assert dest["facts"][0].startswith("da verificare: un'opera visibile e permanente")
    pc = by["Passaggio coattivo"]
    assert pc["anchor"] == "1051 c.c." and pc["how"] == "exact"
    assert pc["articles"][:3] == ["1051 c.c.", "1052 c.c.", "1032 c.c."]     # Sezione nearest, then the least cited link
    assert "1 c.p.c." not in pc["articles"]                                   # a hub, and outside the allowed codes
    assert by["Decadenza dall'azione di reintegrazione"]["anchor"] == "1168 c.c."      # the pool's, resolved for it
    assert by["Decadenza dall'azione di reintegrazione"]["how"] == "pool"
    assert "Divisione di un fondo comune" not in by and "Legittima difesa" not in by    # c.p. not allowed in this case
    # v54: the frame's institutions with no rule, term or regime behind them are "altri profili", never topics
    assert [t["title"] for t in topics] == ["Servitù per destinazione del padre di famiglia"]
    assert [t["title"] for t in others] == ["Passaggio coattivo", "Decadenza dall'azione di reintegrazione"]
    assert pc["articles"] == ["1051 c.c.", "1052 c.c.", "1032 c.c.", "1053 c.c.", "1054 c.c.", "1055 c.c."]


def test_a_topic_has_at_most_eight_articles_before_its_capos_two_signal_ones(monkeypatch):
    monkeypatch.setattr(topic_step, "ARTICLES_MAX", 3)
    links = topic_step.CsvLinks(UNITS, REFERS, RUBRICHE)
    others = []
    topic_step.build(_topic_analysis(), POOL, links, others=others)
    pc = others[0]
    assert pc["articles"] == ["1051 c.c.", "1052 c.c.", "1032 c.c.", "1054 c.c."]     # 3, then 1054 two-signal in Capo II


def test_a_fired_rule_comes_before_one_to_verify_and_without_unknowns_none_is_to_verify():
    fired = [{"id": "destinazione_padre_famiglia", "institutions": ["servitù per destinazione del padre di famiglia"],
              "conclusion": "Costituita (art. 1062 c.c.)."}]
    assert topic_step.rules_to_verify({"facts": TOPIC_FACTS, "fired": fired, "terms": []}, ["diritti_reali"]) == []
    known = {**TOPIC_FACTS, "works_or_signs": [{**WORK, "before_separation": "no"}]}
    assert topic_step.rules_to_verify({"facts": known, "fired": [], "terms": []}, ["diritti_reali"]) == []
    insts = topic_step.institutions(_topic_analysis(fired))
    assert insts[0] == {"name": "servitù per destinazione del padre di famiglia", "source": "regola",
                        "rule": "destinazione_padre_famiglia", "refs": [], "to_verify": [], "requires": []}


def test_topics_sharing_more_than_half_their_articles_are_merged():
    links = topic_step.CsvLinks(UNITS, REFERS, RUBRICHE)
    analysis = {"frame": {"areas": ["diritti_reali"], "central_issue_candidates": [{"nome": "Passaggio coattivo"}]},
                "questioni": [{"istituti": [{"nome": "Interclusione"}]}], "fact_signals": {},
                "institution_refs": {"Interclusione": ["1054 c.c."]}}
    others = []
    topics = topic_step.build(analysis, POOL, links, others=others)
    assert others == [] and len(topics) == 1 and topics[0]["merged"] == ["Interclusione"]   # 1054's Sezione is 1051's


def test_topic_recall_counts_the_expected_norms_in_all_topics_and_in_the_first_three():
    topics = [{"articles": ["1051 c.c."]}, {"articles": ["1062 c.c."]}, {"articles": ["949 c.c."]}, {"articles": ["1158 c.c."]}]
    got = topic_step.recall(topics, [[("1158", "cc")], [("1062", "cc")], [("2653", "cc"), ("2658", "cc")]],
                            lambda label: (label.split()[0], "cc"))
    assert got == {"expected": 3, "in_topics": 2, "in_top3": 1, "share": 0.667, "share_top3": 0.333,
                   "missing": [[("2653", "cc"), ("2658", "cc")]]}


def test_topic_step_runs_after_retrieval_only_with_its_flag(monkeypatch):
    assert not topic_step.enabled()
    monkeypatch.setenv(topic_step.FLAG, "true")
    assert topic_step.enabled()


def test_diagnose_topics_only_stops_at_the_pools_text_read(monkeypatch):
    from src.rag import article_text
    cap = {}
    calls = []
    monkeypatch.setattr(dg, "_add_regime_candidates", lambda s, d, db, analysis, picked, **k: calls.append(1) or picked)
    monkeypatch.setattr(article_text, "fill_texts", lambda session, rows, store=None: ["filled"])
    dl._stop_after_retrieval(dg, cap)
    analysis = {"frame": {}}
    dg._add_regime_candidates(None, None, "db", analysis, {"article_rows": []}, store=None)
    with pytest.raises(dl._RetrievalDone):
        article_text.fill_texts(None, [{"_article": ("1", "c.c.")}], None)
    assert cap["analysis_ref"] is analysis and cap["topic_rows"] == [{"_article": ("1", "c.c.")}]
    assert article_text.fill_texts(None, [], None) == ["filled"]        # once only


def test_diagnose_topic_recall_uses_the_evals_keys():
    class Ev:
        @staticmethod
        def mentions(text):
            m = re.match(r"art\. (\S+) c\.c\.", text)
            return [(m.group(1), "cc")] if m else []
    topics = [{"title": "T", "anchor": "1062 c.c.", "how": "suffix", "rubrica": "Destinazione del padre di famiglia.",
               "articles": ["1062 c.c."], "facts": [], "source": "questione", "merged": []}]
    got = dl.topic_recall(topics, [{"code": "cc", "article": "1062"}, {"code": "cc", "article": "1158"}], Ev)
    assert got["in_topics"] == 1 and got["missing"] == [[("1158", "cc")]]
    assert dl.topics_text(topics) == ("1. T  [questione; anchor 1062 c.c. by suffix: «Destinazione del padre di famiglia.»]"
                                      "\n   articles: 1062 c.c.")


def test_the_terms_dates_are_events_the_competenza_names_the_comune_and_names_are_capitalised():
    signals = {"facts": {"parties": [{"name": "Leonardo", "role": "cliente"}, {"name": "Domitilla", "role": "x"}],
                         "property_history": [{"owner": "Leonardo", "asset": "terreni in comune di Manciano"}]},
               "terms": [{"id": "possessoria", "ref": "artt. 1168 e 1170 c.c.",
                          "from": {"what": "Domitilla appone un cancello chiuso a", "date": "2016"},
                          "to": {"what": "Leonardo si rivolge ad un avvocato", "date": "2018"},
                          "text": "azione possessoria (spoglio o turbativa): termine annuale DECORSO (Domitilla appone "
                                  "un cancello chiuso a 2016, Leonardo si rivolge ad un avvocato 2018)"}],
               "tools": [{"id": "competenza_territoriale", "refs": [["21", "c.p.c."]], "text": "Foro.", "matched": ["x"]},
                         {"id": "provvedimento_urgenza", "refs": [["700", "c.p.c."]], "text": "Provvedimento (art. 700 "
                          "c.p.c.).", "matched": ["interclusione del fondo di leonardo"]},
                         {"id": "trascrizione_domanda", "refs": [["2653", "c.c."]], "text": "x", "matched": [
                             "trascrizione della domanda giudiziale (azione confessoria/negatoria) ai sensi dell'art. 2653"]}]}
    text = render_computed.render({"fact_signals": signals, "action_gate": {"949 c.c.": "azione negatoria"}})
    assert ("- azione possessoria (spoglio o turbativa): termine annuale DECORSO (cancello apposto nel 2016; avvocato "
            "contattato nel 2018), artt. 1168 e 1170 c.c.") in text
    assert "giudice del luogo in cui si trova l'immobile (Manciano) (art. 21 c.p.c.)." in text
    nonuse = {"id": "non_uso_servitu", "ref": "art. 1073 c.c.", "text": "estinzione per non uso: non decorso, scade nel 2036 "
              "(20 anni dall'ultimo uso, 2016)", "from": {"what": "Domitilla appone un cancello chiuso a", "date": "2016"},
              "to": {"what": "Leonardo si rivolge ad un avvocato", "date": "2018"}}
    assert "(20 anni dall'ultimo uso nel 2016), art. 1073 c.c." in render_computed.render(
        {"fact_signals": {**signals, "terms": [nonuse]}})
    assert "interclusione del fondo di Leonardo" in text
    assert "Va trascritta la domanda giudiziale (azione confessoria)." in text      # the negatoria is the other side's


def test_an_owner_whose_sale_is_in_the_facts_is_no_defendant():
    """Leonardo v52: Sempronio's sale to Domitilla (2003) only as an event and as Domitilla's acquisition."""
    from src.rag import fact_signals as fs
    facts = json.load(open(HERE / "fixtures" / "leonardo_v45d_facts.json", encoding="utf-8"))
    assert fs.real_action_defendants(facts)[1] == ["Tizio", "Caio", "Annibale"]
    # v52's shape: Sempronio a party with role proprietario, no entry of his own, the sale an event and Domitilla's
    v52 = {"parties": [{"name": "Leonardo", "role": "cliente, proprietario"}, {"name": "Tizio", "role": "proprietario"},
                       {"name": "Caio", "role": "proprietario"}, {"name": "Sempronio", "role": "proprietario"},
                       {"name": "Annibale", "role": "proprietario"}, {"name": "Mevio", "role": "privato"},
                       {"name": "Domitilla", "role": "proprietario"}],
           "property_history": [{"asset": "terreni di Leonardo e di Sempronio", "owner": "Mevio", "ended_by": "divisione",
                                 "ended_by_how": fs.SEQUENCE_DEDUCED},
                                {"asset": "terreno di Sempronio", "owner": "Domitilla", "from": "2003",
                                 "how_acquired": "compravendita"}],
           "events": [{"date": "2003", "what": "Sempronio vende il terreno a Domitilla", "who": "Sempronio", "kind": "vendita"},
                      {"date": "2016", "what": "Domitilla appone un cancello chiuso a chiave", "who": "Domitilla",
                       "kind": "chiusura/ostruzione accesso"}]}
    assert fs.real_action_defendants(v52) == (["Domitilla"], ["Tizio", "Caio", "Annibale"], "current owners")
    only_event = {**v52, "property_history": v52["property_history"][:1]}
    assert "Sempronio" not in fs.real_action_defendants(only_event)[1]
    only_entry = {**v52, "events": v52["events"][1:]}
    assert "Sempronio" not in fs.real_action_defendants(only_entry)[1]
    still = {**v52, "property_history": v52["property_history"] + [{"owner": "Sempronio", "asset": "un bosco"}]}
    assert "Sempronio" in fs.real_action_defendants(still)[1]       # sold one plot, still owns another


# --- v54: round-1 topic quality ---------------------------------------------------------------------------------------

def test_a_generic_rubrica_is_never_an_anchor_and_a_one_word_name_needs_a_computed_source():
    rubriche = {**RUBRICHE, ("2934", "c.c."): "Prescrizione.", ("1418", "c.c."): "Nullità.",
                **{(str(n), "c.c."): "Effetti tra le parti." for n in (1, 2, 3, 4)},
                ("1027", "c.c."): "Contenuto del diritto."}
    links = topic_step.CsvLinks(UNITS, REFERS, rubriche)
    assert topic_step.generic_rubrica("Prescrizione.", links) and topic_step.generic_rubrica("Nullità", links)
    assert topic_step.generic_rubrica("Effetti tra le parti", links)          # two words, four times in the codes
    assert not topic_step.generic_rubrica("Passaggio coattivo.", links)
    for name in ("Prescrizione", "Nullità del contratto"):
        assert topic_step.anchor({"name": name, "source": "frame"}, links, {"c.c."}, {}, [])[0] is None
    assert topic_step.anchor({"name": "Interclusione", "source": "frame"}, links, {"c.c."}, {}, [])[0] is None


def test_an_open_term_is_a_topic_on_its_article_and_sezione_and_a_decided_one_is_a_fact():
    units = UNITS + [("c.c.", "SEZIONE", "SEZIONE I – Dell'usucapione", n) for n in ("1158", "1159", "1160", "1161")]
    links = topic_step.CsvLinks(units, REFERS, {**RUBRICHE, ("1168", "c.c."): "Azione di reintegrazione."})
    analysis = _topic_analysis()
    analysis["fact_signals"]["terms"] = [
        {"id": "usucapione_ordinaria", "label": "usucapione ordinaria di beni immobili", "ref": "art. 1158 c.c.",
         "status": "DA VERIFICARE"},
        {"id": "possessoria", "label": "azione possessoria", "ref": "artt. 1168 e 1170 c.c.", "status": "DECORSO"}]
    others = []
    topics = topic_step.build(analysis, POOL, links, "", others=others)
    term = next(t for t in topics if t["source"].startswith("termine"))
    assert term["title"] == "Usucapione ordinaria di beni immobili" and term["anchor"] == "1158 c.c."
    assert term["articles"] == ["1158 c.c.", "1159 c.c.", "1160 c.c.", "1161 c.c."] and term["how"] == "term"
    assert all(t["anchor"] != "1168 c.c." for t in topics + others)         # possessoria DECORSO: a fact, no topic


def test_a_gated_action_leaves_the_title_and_an_all_gated_topic_is_not_offered():
    gate = {"949 c.c.": "azione negatoria: spetta a chi resists_servitude, non al cliente"}
    assert topic_step.without_gated("Azione confessoria e negatoria", gate) == "Azione confessoria"
    assert topic_step.without_gated("Azione negatoria", gate) == ""
    assert topic_step.without_gated("Passaggio coattivo", gate) == "Passaggio coattivo"


def test_with_no_supported_topic_the_frames_are_offered_and_incerto_terms_are_topics():
    links = topic_step.CsvLinks(UNITS, REFERS, RUBRICHE)
    analysis = {"frame": {"areas": ["diritti_reali"], "central_issue_candidates": [{"nome": "Passaggio coattivo"}]},
                "fact_signals": {}, "questioni": []}
    others = []
    topics = topic_step.build(analysis, POOL, links, others=others)
    assert [t["title"] for t in topics] == ["Passaggio coattivo"] and others == []
    assert topics[0]["why"] == topic_step.NO_SUPPORT_OFFERED
    for status in ("INCERTO", "NON VERIFICABILE"):
        terms = [{"id": "usucapione_ordinaria", "label": "usucapione ordinaria", "ref": "art. 1158 c.c.", "status": status}]
        assert topic_step.term_institutions({"terms": terms})[0]["refs"] == [("1158", "c.c.")]


def test_the_altri_profili_carry_their_anchors_capo():
    links = topic_step.CsvLinks(UNITS, REFERS, RUBRICHE)
    others = []
    topic_step.build(_topic_analysis(), POOL, links, "", others=others)
    pc = next(t for t in others if t["title"] == "Passaggio coattivo")
    assert pc["capo"] == "CAPO II – DELLE SERVITU’ COATTIVE"


def test_a_term_of_the_cpa_anchors_on_the_cpa_not_the_cp():
    terms = [{"id": "ricorso_tar_appalti", "label": "ricorso al TAR", "ref": "art. 120 c.p.a.", "status": "DA VERIFICARE"}]
    assert topic_step.term_institutions({"terms": terms})[0]["refs"] == [("120", "c.p.a.")]


# --- v55: topic merging, anchors by area and name, the draft tables -------------------------------------------------

def _topic(title, anchor, refs, rank=0):
    return {"title": title, "anchor": anchor, "how": "pool", "rubrica": "", "refs": [tuple(r.split(" ", 1)) for r in refs],
            "source": "frame", "facts": [], "merged": [], "questions": []}


def test_a_topic_whose_title_words_are_all_in_anothers_is_merged_into_the_stronger():
    topics = [_topic("Usucapione ordinaria di beni immobili", "1158 c.c.", ["1158 c.c.", "1159 c.c."]),
              _topic("Truffa", "640 c.p.", ["640 c.p."]),
              _topic("Usucapione", "1146 c.c.", ["1146 c.c.", "1167 c.c."]),
              _topic("Truffa mediante artifizi e raggiri", "640-bis c.p.", ["640-bis c.p."]),
              _topic("Custodia cautelare", "285 c.p.p.", ["285 c.p.p."]),
              _topic("Ordinanza di custodia cautelare in carcere", "292 c.p.p.", ["292 c.p.p."]),
              _topic("Favoreggiamento personale", "378 c.p.", ["378 c.p."]),
              _topic("Concorso esterno in favoreggiamento", "110 c.p.", ["110 c.p."])]
    kept = topic_step.same_name_merged(topics)
    by = {t["title"]: t for t in kept}
    assert list(by) == ["Usucapione ordinaria di beni immobili", "Truffa", "Custodia cautelare",
                        "Favoreggiamento personale", "Concorso esterno in favoreggiamento"]
    assert by["Usucapione ordinaria di beni immobili"]["merged"] == ["Usucapione"]
    assert ("1146", "c.c.") in by["Usucapione ordinaria di beni immobili"]["refs"]       # the merged one's anchor kept
    assert by["Truffa"]["merged"] == ["Truffa mediante artifizi e raggiri"]
    assert by["Custodia cautelare"]["merged"] == ["Ordinanza di custodia cautelare in carcere"]


def test_a_civil_anchor_must_lie_in_its_areas_book_and_a_pool_anchor_must_share_a_word_of_the_name():
    assert not topic_step.area_fits(("585", "c.c."), ["famiglia"])         # v54 l4: successione del coniuge separato
    assert topic_step.area_fits(("151", "c.c."), ["famiglia"])
    assert topic_step.area_fits(("2043", "c.c."), ["famiglia", "responsabilita_civile"])
    assert topic_step.area_fits(("2934", "c.c."), ["famiglia"])            # Libro VI: every civil area
    assert topic_step.area_fits(("270-ter", "c.p."), ["penale"]) and topic_step.area_fits(("585", "c.c."), ["altro"])
    assert topic_step.shares_name("Separazione de facto tra coniugi", "Successione del coniuge separato.")
    assert not topic_step.shares_name("Concorso esterno in favoreggiamento personale", "Assistenza agli associati")
    pool = {("270-ter", "c.p."): {"_multi_signal": True, "_fused_rank": 1, "s": {"titolo_articolo": "Assistenza agli associati"}},
            ("378", "c.p."): {"_multi_signal": True, "_fused_rank": 9, "s": {"titolo_articolo": "Favoreggiamento personale"}},
            ("585", "c.c."): {"_multi_signal": True, "_fused_rank": 1, "s": {"titolo_articolo": "Successione del coniuge separato."}},
            ("151", "c.c."): {"_multi_signal": True, "_fused_rank": 9, "s": {"titolo_articolo": "Separazione giudiziale."}}}
    inst = {"name": "Concorso esterno in favoreggiamento personale", "source": "frame"}
    assert topic_step.anchor(inst, None, None, pool, [], [("270-ter", "c.p.")]) == (None, "", "")
    assert topic_step.anchor(inst, None, None, pool, [], [("270-ter", "c.p."), ("378", "c.p.")])[0] == ("378", "c.p.")
    inst = {"name": "Separazione de facto tra coniugi", "source": "frame", "area": "famiglia"}
    assert topic_step.anchor(inst, None, None, pool, [], [("585", "c.c."), ("151", "c.c.")],
                             areas=["famiglia"])[0] == ("151", "c.c.")


class _AnyRubrica:
    """Links whose every rubrica is the name: a regime must still anchor on its own article."""
    def rubrica_matches(self, name, codes):
        return [(("2689", "c.c."), "Usucapione", "exact")]

    def rubrica_count(self, rubrica):
        return 1


def test_a_regime_anchors_on_its_own_table_article_before_any_rubrica():
    inst = {"name": "usucapione", "source": "regime", "refs": [("1146", "c.c."), ("1167", "c.c.")]}
    assert topic_step.anchor(inst, _AnyRubrica(), None, {}, [], []) == (("1146", "c.c."), "regime", "")
    inst = {"name": "usucapione dei beni immobili", "source": "frame", "refs": []}      # no table: the rubrica's
    assert topic_step.anchor(inst, _AnyRubrica(), None, {}, [], [])[0] == ("2689", "c.c.")


def test_the_draft_tables_are_read_only_with_their_flag_and_within_the_cases_family(monkeypatch):
    from src.rag import topic_tables
    analysis = {"frame": {"areas": ["penale"], "central_issue_candidates": [
        {"nome": "Tentata rapina"}, {"nome": "Furto in abitazione"}, {"nome": "Custodia cautelare in carcere"}]}}
    assert topic_step.institutions(analysis)[0]["source"] == "frame"            # the flag off: no table
    monkeypatch.setenv("TOPIC_TABLES", "true")
    found = {t["id"]: t for t in topic_tables.matches(analysis, ["penale"])}
    assert {"rapina", "tentativo", "furto", "misure_cautelari"} <= set(found)
    assert found["furto"]["refs"][0] == ("624-bis", "c.p.") and found["furto"]["questions"] == ["data_fatto"]
    assert found["misure_cautelari"]["refs"][0] == ("285", "c.p.p.") and found["misure_cautelari"]["questions"] == ["data_misura"]
    assert topic_tables.matches(analysis, ["contratti"]) == []                   # a civil case never reads them
    insts = topic_step.institutions(analysis)
    assert [i["source"] for i in insts[:4]] == ["tabella"] * 4 and insts[0]["refs"]
    for table in topic_tables.TABLES:                       # anchors and questions only: no conclusion in a table
        assert set(table) <= {"id", "family", "name", "triggers", "refs", "anchors", "questions"}
        assert all(q in __import__("src.rag.interaction_copy", fromlist=["QUESTIONS"]).QUESTIONS for q in table["questions"])


def test_table_topics_come_with_the_frames_when_no_rule_or_term_supports_a_topic(monkeypatch):
    monkeypatch.setenv("TOPIC_TABLES", "true")
    links = topic_step.CsvLinks([], [], {("52", "c.p."): "Difesa legittima.", ("640", "c.p."): "Truffa"})
    analysis = {"frame": {"areas": ["penale"], "central_issue_candidates": [{"nome": "Legittima difesa"},
                                                                           {"nome": "Truffa"}]}, "fact_signals": {}}
    others = []
    topics = topic_step.build(analysis, [], links, others=others)
    by = {t["title"]: t for t in topics}
    assert by["Truffa"]["source"] == "tabella truffa" and by["Truffa"]["questions"] == ["data_fatto"]
    assert by["Legittima difesa"]["anchor"] == "52 c.p."          # the frame's exact anchor is offered, not hidden


def test_a_fired_rules_institution_anchors_on_its_required_article_in_another_code():
    # v54 c3: morosita_conduttore (codice c.p.c.) names the resolution, whose article is 1453 c.c. (its requires)
    from src.rag import fact_signals
    facts = {"parties": [{"name": "Tizio", "role": "cliente, locatore"}, {"name": "Caio", "role": "controparte, conduttore"}],
             "contracts": [{"type": "contratto di locazione"}],
             "events": [{"date": "", "what": "Il conduttore non paga il canone da tre mesi"}],
             "requests": ["riottenere l'immobile e recuperare i canoni arretrati"]}
    fired = [r for r in fact_signals.fire_rules(facts, [], areas=["locazione"]) if r["id"] == "morosita_conduttore"]
    assert fired and ("1453", "c.c.") in fired[0]["requires"]
    links = topic_step.CsvLinks([], [], {("1453", "c.c."): "Risolubilità del contratto per inadempimento.",
                                         ("1455", "c.c."): "Importanza dell'inadempimento."})
    analysis = {"frame": {"areas": ["locazione"]}, "fact_signals": {"facts": facts, "fired": fired, "terms": []}}
    by = {t["title"]: t for t in topic_step.build(analysis, [], links)}
    topic = by["Risoluzione della locazione per inadempimento"]
    assert topic["anchor"] == "1453 c.c." and topic["how"] == "rule"
    assert "Sfratto per morosità" not in by or by["Sfratto per morosità"]["anchor"] != "1453 c.c."   # no word shared
