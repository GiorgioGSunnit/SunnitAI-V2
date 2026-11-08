"""v51, on the Leonardo v50 answer (leonardo_v50.docx; its sentences copied here, the stamps the qualification check added
taken out): NORME_SCOPE, the repetition guard, the rubrica stamp on an opening citation, the real action's defendants,
TOOL_CITES_PROTECTED, USUCAPIONE_CONDITIONAL. No model, no database."""
import json
import os

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

import pytest  # noqa: E402

from src.rag import citation_ledger, fact_signals as fs, repetition_guard as rg, text_tidy  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, *a, **k: seen.append((name, data)))
    yield seen
    citation_ledger.protect(())


# --- (1) NORME_SCOPE ---------------------------------------------------------------------------------------------------

def _row(n, code="c.c.", grade=2, **tags):
    return {"_article": (n, code), "_grade": grade, "s": {"plain_text": f"Testo {n}.", "titolo_articolo": f"Rubrica {n}"},
            **tags}


# v50's section 3: thirteen articles; graded here as a plan would hold them
V50_NORME = [("1079", 3), ("1062", 3), ("1061", 3), ("1066", 2), ("1031", 2), ("1073", 2), ("1059", 1), ("843", 1),
             ("1064", 3), ("1067", 2), ("1051", 3), ("1054", 2), ("1027", 2)]


def _plan(refs):
    return {"parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
            "questioni": [{"titolo": "Servitù di passaggio", "argomento": "x", "obiezione": "", "replica": "",
                           "norme": refs[:7], "pronunce": []},
                          {"titolo": "Passaggio coattivo", "argomento": "y", "obiezione": "", "replica": "",
                           "norme": refs[7:], "pronunce": []}],
            "subordinate": [], "punti_deboli": []}


def test_section_3_holds_the_primary_norms_only_the_others_stay_for_section_4():
    rows = [_row(n, grade=g) for n, g in V50_NORME]
    rows[5]["_regime_candidate"] = True                                   # 1073: a regime candidate's
    plan = _plan([r["_article"] for r in rows])
    scope, log = ptw.norme_scope(plan, rows, [("1027", "c.c.")])
    assert log["scope"] == ["1079 c.c.", "1062 c.c.", "1061 c.c.", "1073 c.c.", "1064 c.c.", "1051 c.c.", "1027 c.c."]
    assert log["why"]["1027 c.c."] == "named" and log["why"]["1073 c.c."] == "regime" and log["cut"] == []
    assert "843 c.c." in log["others"] and "1059 c.c." in log["others"]
    jobs = {title: (instruction, material) for title, instruction, material, _ in
            ptw.section_jobs({**plan, ptw.NORME_SCOPE_KEY: scope}, "Fatti.", "", {}, rows, {}, "")}
    instruction, material = jobs[ptw.SECTION_TITLES[2]]
    assert "Tratta solo le norme indicate nel PIANO" in instruction
    assert "Testo 1079." in material and "Testo 843." not in material and "art. 843 c.c." not in material
    assert "art. 843 c.c." in jobs[ptw.SECTION_TITLES[3]][1] and "Testo 843." in jobs[ptw.SECTION_TITLES[3]][1]


def test_at_most_eight_the_user_s_first_then_grade_3_then_the_regime_s():
    rows = [_row(str(n), grade=3) for n in range(1, 10)] + [_row("50", grade=1, _regime_candidate=True), _row("60", grade=1)]
    plan = _plan([r["_article"] for r in rows])
    scope, log = ptw.norme_scope(plan, rows, [("60", "c.c.")])
    assert len(scope) == 8 and ("60", "c.c.") in scope
    assert log["cut"] == ["8 c.c.", "9 c.c.", "50 c.c."]
    assert not ptw.norme_split({**plan, ptw.NORME_SCOPE_KEY: scope})              # 8: never halved


def test_no_primary_norm_keeps_every_plan_article_and_the_flag_is_off_by_default(monkeypatch):
    rows = [_row("1", grade=2), _row("2", grade=1)]
    scope, _ = ptw.norme_scope(_plan([r["_article"] for r in rows]), rows, [])
    assert scope == [] and ptw._norme_refs(_plan([r["_article"] for r in rows])) == [("1", "c.c."), ("2", "c.c.")]
    monkeypatch.delenv(ptw.NORME_SCOPE_FLAG, raising=False)
    assert not ptw.norme_scope_enabled()


# --- (2) the repetition guard ------------------------------------------------------------------------------------------

V50_APPLICAZIONE = (
    "La chiusura del cancello da parte di Domitilla nel 2016 costituisce un impedimento illegittimo al diritto di "
    "passaggio necessario per l’esercizio della proprietà di Leonardo. La servitù è già acquisita per destinazione del "
    "padre di famiglia e per uso continuo e pacifico, e quindi la chiusura del cancello è vietata. La chiusura del "
    "cancello impedisce l’accesso necessario e quindi è vietata. La materia rientra tra quelle soggette a mediazione "
    "obbligatoria. La competenza territoriale spetta al foro del luogo dove si trovano i beni immobili interessati. La "
    "mediazione è obbligatoria per questioni relative a diritti reali. Il foro è corretto in quanto si trova nel luogo "
    "dove si trovano i beni immobili interessati.")
V50_PROFILI = (
    "Litisconsorzio necessario: La decisione non può pronunciarsi che in confronto di tutte le parti necessarie. Ai fini "
    "del giudizio, va valutata la chiamata in causa di Tizio e Caio, proprietari dei fondi intermedi, in base al "
    "principio del litisconsorzio necessario [art. 102 c.p.c.].\n\n"
    "L’esperimento della mediazione è condizione di procedibilità della domanda. Prima di procedere con l’atto di "
    "citazione, Leonardo deve verificare se la mediazione è stata effettuata e documentarne l’esito.\n\n"
    "Verifica della chiamata in causa di Tizio e Caio: Ai fini del giudizio, va valutata la necessità di chiamare in "
    "causa Tizio e Caio, proprietari dei fondi intermedi, in base al principio del litisconsorzio necessario.\n\n"
    "Verifica dell’esperimento della mediazione obbligatoria: Prima di procedere con l’atto di citazione, va verificato "
    "se la mediazione obbligatoria è stata effettuata e documentarne l’esito.")


def test_the_v50_repeats_go_the_first_stays():
    text, log = rg.drop_repeats(V50_APPLICAZIONE, "Applicazione al caso")
    assert [e["sentence"][:20] for e in log] == ["Il foro è corretto i"] and log[0]["overlap"] >= 0.8
    assert "La competenza territoriale spetta al foro" in text and "Il foro è corretto" not in text
    assert "La chiusura del cancello impedisce" in text           # 0.67 at most with any one earlier sentence
    text, log = rg.drop_repeats(V50_PROFILI, "Profili processuali e termini")
    assert len(log) == 2 and "Verifica della chiamata" not in text and "Verifica dell’esperimento" not in text
    assert "[art. 102 c.p.c.]" in text


def test_a_repeat_citing_another_article_or_another_number_stays():
    cited = "Il proprietario deve lasciare libero il passaggio. Il proprietario deve lasciare libero il passaggio (art. 1064 c.c.)."
    assert rg.drop_repeats(cited)[1] == []
    dated = "Il termine di prescrizione decorre dal 2016 per Leonardo. Il termine di prescrizione decorre dal 2018 per Leonardo."
    assert rg.drop_repeats(dated)[1] == []
    same = ("Il termine di prescrizione decorre dal 2016 per Leonardo. Per Leonardo il termine di prescrizione decorre "
            "dal 2016.")
    assert len(rg.drop_repeats(same)[1]) == 1


def test_the_answer_is_guarded_section_by_section_and_logged(events):
    answer = ("**3. Norme e giurisprudenza**\n\nLa competenza territoriale spetta al foro del luogo dove si trovano i beni "
              "immobili interessati.\n\n**4. Applicazione al caso**\n\n" + V50_APPLICAZIONE + "\n\n**5. Profili**\n\nFine del testo qui.")
    text, log = rg.guard_answer(answer, "answer")
    assert len(log) == 1 and log[0]["section"] == "4. Applicazione al caso"       # not across sections
    assert text.count("La competenza territoriale spetta al foro") == 2 and "Il foro è corretto" not in text
    assert text.startswith("**3. Norme e giurisprudenza**\n\n") and text.endswith("**5. Profili**\n\nFine del testo qui.")
    assert [d["count"] for n, d in events if n == rg.EVENT] == [1]


# --- (3) the rubrica stamp: a sentence opening with the citation ---------------------------------------------------------

V50_NORME_PARAGRAPH = (
    "La servitù di passaggio può essere riconosciuta e tutelata in base all'art. 1079 c.c., che consente al titolare "
    "della servitù di farne riconoscere l'esistenza in giudizio e di far cessare gli impedimenti all'esercizio. La "
    "servitù può essere costituita per destinazione del padre di famiglia, come previsto dall'art. 1062 c.c., quando due "
    "fondi, un tempo appartenuti allo stesso proprietario, sono separati da un'opera visibile e permanente. L'art. 1031 "
    "c.c. indica che le servitù possono essere costituite volontariamente, per usucapione o per destinazione del padre di "
    "famiglia. L'art. 1073 c.c. prevede l'estinzione per prescrizione della servitù, quando non se ne usa per venti anni. "
    "L'art. 1059 c.c. disciplina la servitù concessa da uno dei comproprietari di un fondo indiviso. L'art. 1051 c.c. "
    "disciplina il passaggio coattivo. L'art. 1054 c.c. esenta dal pagamento dell'indennità il passaggio ottenuto in "
    "seguito a alienazione o divisione. L'art. 1027 c.c. definisce la servitù prediale.")
TEXTS = {"1073": ("Estinzione per prescrizione", "La servitù si estingue per prescrizione quando non se ne usa per venti "
                  "anni. Il termine decorre dal giorno in cui si è cessato di esercitarla."),
         "1054": ("Interclusione per effetto di alienazione o di divisione", "Se il fondo è divenuto da ogni parte chiuso "
                  "per effetto di alienazione a titolo oneroso, il proprietario ha diritto di ottenere dall'altro "
                  "contraente il passaggio senza alcuna indennità. La stessa norma si applica in caso di divisione.")}


@pytest.mark.parametrize("number", ["1073", "1054"])
def test_the_clause_after_an_opening_citation_is_what_the_text_must_cover(number):
    ref = (number, "c.c.")
    clause = dg._opening_clause(V50_NORME_PARAGRAPH, ref)
    assert clause.startswith(("prevede l'estinzione", "esenta dal pagamento"))
    heading, text = TEXTS[number]
    assert dg._text_covers(clause, ref, f"{heading} {text}")
    assert dg._citation_context(V50_NORME_PARAGRAPH, ref).startswith(f"L'art. {number} c.c.")    # v52: its sentence
    assert dg._opening_clause("Il termine decorre ai sensi dell'art. 1073 c.c. dal 2016.", ref) == ""


def test_the_v50_sentences_get_no_stamp(monkeypatch, events):
    monkeypatch.setenv("RUBRICA_TEXT_CHECK", "true")
    rows = [{"_article": (n, "c.c."), "_grade": 2, "s": {"name": f"{n}.0.0", "plain_text": t, "titolo_articolo": h}}
            for n, (h, t) in TEXTS.items()]
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: [
        {"articolo": f"{n} c.c.", "compatibile": False, "perche": "rubrica diversa"} for n in TEXTS])
    out = dg._check_qualifications(V50_NORME_PARAGRAPH, {}, rows, None, [], "")
    assert "DA VERIFICARE" not in out
    monkeypatch.delenv("RUBRICA_TEXT_CHECK")
    assert "[DA VERIFICARE: l'art. 1073 c.c. riguarda «Estinzione per prescrizione»]" in \
        dg._check_qualifications(V50_NORME_PARAGRAPH, {}, rows, None, [], "")


# --- (4) the real action's defendants: Annibale on the Leonardo facts fixture -------------------------------------------

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "leonardo_v45d_facts.json")
REAL = {"frame": {"areas": ["diritti_reali", "contratti", "altro"]},
        "questioni": [{"titolo": "Usucapione di servitù apparente", "istituti": [{"nome": "azione confessoria"}]}]}


@pytest.fixture
def leonardo(monkeypatch):
    for name in ("PARTIES_ENFORCED", "REGIME_CANDIDATES", "FACT_REGIMES_ENABLED", "FACT_SIGNALS_ENABLED"):
        monkeypatch.setenv(name, "true")
    facts = json.load(open(FIXTURE, encoding="utf-8"))
    signals = fs.signals_from(facts, "", REAL["frame"]["areas"], rules=True)
    return {**REAL, "fact_signals": signals}


def test_every_current_owner_is_a_candidate_defendant_annibale_too(leonardo):
    authors, owners, how = fs.real_action_defendants(leonardo["fact_signals"]["facts"])
    assert authors == ["Domitilla"] and owners == ["Tizio", "Caio", "Annibale"] and how == "current owners"
    assert "Annibale" in fs.parties_line(leonardo["fact_signals"], leonardo)
    tools = {t["id"]: t for t in fs.regime_tools(leonardo["fact_signals"], leonardo)}
    assert tools["litisconsorzio"]["matched"] == ["parti da convenire: Domitilla", "da valutare: Tizio, Caio, Annibale"]


# --- (5) TOOL_CITES_PROTECTED ------------------------------------------------------------------------------------------

def test_the_trascrizione_tool_s_line_carries_its_article_and_number(leonardo):
    signals = {**leonardo["fact_signals"], "tools": fs.regime_tools(leonardo["fact_signals"], leonardo)}
    line = next(l for l in fs.tools_block(signals).split("\n") if "trascrizione" in l.lower())
    assert "Nel caso: trascrizione della domanda giudiziale" in line and "art. 2653, n. 1, c.c." in line


def test_a_tool_or_term_article_s_sentence_is_protected_like_a_plan_article(monkeypatch, events):
    analysis = {"fact_signals": {"tools": [{"id": "trascrizione_domanda", "refs": [["2653", "c.c."]]},
                                           {"id": "accertamento_preventivo", "refs": [["696", "c.p.c."]]}],
                                 "terms": [{"id": "usucapione_ordinaria", "ref": "art. 1158 c.c."}],
                                 "damages": [{"id": "danni", "refs": [["2043", "c.c."]]}]}}
    assert ptw.protected_tool_refs(analysis) == []                                 # off
    monkeypatch.setenv(ptw.TOOL_CITES_PROTECTED_FLAG, "true")
    assert ptw.protected_tool_refs(analysis) == [("2653", "c.c."), ("696", "c.p.c."), ("1158", "c.c.")]
    sentence = ("L'art. 2653 c.c. impone la trascrizione della domanda giudiziale per l'opponibilità ai terzi. per "
                "l'azione confessoria proposta.")
    assert text_tidy.tidy(sentence)[0] == ""                                       # unprotected: dropped
    citation_ledger.protect(ptw.protected_tool_refs(analysis))
    text, _ = text_tidy.tidy(sentence)
    assert "2653" in text and any(d.get("kept") == citation_ledger.KEPT for n, d in events if n == citation_ledger.EVENT)


# --- (6) USUCAPIONE_CONDITIONAL -----------------------------------------------------------------------------------------

UNKNOWN = {"terms": [{"id": "usucapione_ordinaria", "start_unknown": True}]}
V50_ASSERTS = [
    "Leonardo ha diritto al riconoscimento e alla tutela della servitù di passaggio, acquisita per destinazione del padre "
    "di famiglia e per uso continuo e pacifico.",
    "La servitù è già acquisita per destinazione del padre di famiglia e per uso continuo e pacifico, e quindi la "
    "chiusura del cancello è vietata.",
    "Leonardo ha un diritto concreto al riconoscimento e alla tutela della servitù di passaggio, acquisita per "
    "destinazione del padre di famiglia e per uso continuo e pacifico, in base all’art. 1062 c.c.",
    "Il cancello fu apposto quando l'usucapione ventennale era già maturata a favore di Leonardo.",
    "Leonardo ha usato la strada per oltre venti anni, e l'usucapione è acquisita.",
]
V50_LEFT = [
    "La servitù è apparente e quindi acquistabile per usucapione ventennale.",
    "Usucapione ordinaria: Il termine ventennale per l’acquisto per usucapione non è verificabile, in quanto non si "
    "conosce la data in cui Leonardo ha iniziato a possedere il diritto di passaggio.",
    "L’usucapione dei diritti reali si acquista in virtù del possesso continuato per venti anni.",
    "L'usucapione non è maturata, perché l'inizio del possesso è ignoto.",
    "La controparte potrebbe sostenere che l'usucapione è maturata a suo favore.",
]


@pytest.mark.parametrize("sentence", V50_ASSERTS)
def test_an_asserted_usucapione_is_made_conditional(monkeypatch, sentence):
    monkeypatch.setenv(ptw.USUCAPIONE_CONDITIONAL_FLAG, "true")
    assert ptw.usucapione_asserted(sentence, UNKNOWN)
    new = ptw.conditional_form(sentence)
    assert ptw.CONDITIONAL_CLAUSE in new and "già" not in new and "oltre venti" not in new
    assert not ptw.usucapione_asserted(new, UNKNOWN) and not ptw._wrong_statement(new, UNKNOWN)
    assert "1062 c.c." in new or "1062" not in sentence


@pytest.mark.parametrize("sentence", V50_LEFT)
def test_a_possibility_an_open_term_a_denial_or_the_other_side_is_left(monkeypatch, sentence):
    monkeypatch.setenv(ptw.USUCAPIONE_CONDITIONAL_FLAG, "true")
    assert not ptw.usucapione_asserted(sentence, UNKNOWN)


def test_only_with_the_flag_and_a_start_unknown_and_in_final_checks_too(monkeypatch, events):
    sentence = V50_ASSERTS[1]
    assert ptw.usucapione_conditional("4", sentence, UNKNOWN) == (sentence, [])          # off
    monkeypatch.setenv(ptw.USUCAPIONE_CONDITIONAL_FLAG, "true")
    assert not ptw.usucapione_asserted(sentence, {"terms": [{"id": "usucapione_ordinaria"}]})
    [(title, text)] = ptw.final_checks([("Applicazione al caso", sentence)], [], UNKNOWN)
    assert text.startswith("La servitù è acquisibile per destinazione") and ptw.CONDITIONAL_CLAUSE in text
    [entry] = [e for n, d in events if n == "case_analysis_final_checks" for e in d["entries"]]
    assert entry["how"] == "made conditional"
    out = ptw.usucapione_sections([("1", V50_ASSERTS[0]), ("2", V50_LEFT[0])], UNKNOWN, "written")
    assert ptw.CONDITIONAL_CLAUSE in out[0][1] and out[1][1] == V50_LEFT[0]


def test_the_new_flags_are_off_by_default(monkeypatch):
    from src.rag import feature_flags
    for flag in (ptw.NORME_SCOPE_FLAG, ptw.TOOL_CITES_PROTECTED_FLAG, ptw.USUCAPIONE_CONDITIONAL_FLAG):
        monkeypatch.delenv(flag, raising=False)
        assert not feature_flags.enabled(flag)
