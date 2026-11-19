"""v79.12.1 -- the rulings record, the step-aside path, the names; synthetic records, rulings and passages only:

1. a criminal ruling dropped from a civil article (D4 penal_off) is filtered inside the read set and inside the rest,
   never re-sliced: an unread ruling never slides into the read head without its candidates -- no KeyError, the record
   kept with the other rulings
2. when the method steps aside the prose still prints §5 from the rulings record built for the record articles it cites,
   and Block B's grounded-rule check runs on it in the form a plain opinion allows (no other norm declared «applicabile»,
   «configurabile», «ricorre la fattispecie» with no record or passage; the rules on the case's own norms kept); a
   damages-measure article (1223-1227 c.c.) is never «l'addebito o la domanda» whose unread text makes it step aside
3. names (E2): a surname before «, a prescindere dalla titolarità» is redacted -- after an elided article, in capitals
   too -- and the nightly's reading counts it; a capitalised role after an article («il Ricorrente») is no surname
"""

import pytest

from src.rag import rulings


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))


# --- 1. the rulings record survives a dropped criminal ruling ----------------------------------------------------------------

def _ruling(i, archive="snpen", abstract="Effetti della risoluzione"):
    return {"ruling": f"X{i}", "file_name": f"{archive}@s20@a2025@n{i:05d}@tS.clean.pdf", "name": "",
            "document_number": "", "date_enacted": "2025-03-01", "linked": True, "section": f"S{i}", "abstract": abstract}


class Session:
    def __init__(self, rows, texts):
        self.rows, self.text_of, self.read = rows, texts, []

    def run(self, query, **kw):
        if query != rulings.FETCH_QUERY:
            self.read += list(kw["ids"])
        rows = self.rows if query == rulings.FETCH_QUERY else [
            {"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

        class R:
            def data(self_inner):
                return rows
        return R()


EL_1458 = {"id": "R1", "text": "la risoluzione del contratto ha effetto retroattivo tra le parti", "status": "da provare",
           "kind": "requisito"}
OFF = "Il reato di appropriazione indebita richiede la interversione del possesso con effetti definitivi."
CIVIL = "La risoluzione del contratto ha effetto retroattivo e obbliga le parti alle restituzioni."


def _build_1458(rows, texts):
    ctx = {"records": {"1458 c.c.": {"side": "weapon", "rubrica": "Effetti della risoluzione", "requisites": [EL_1458]}},
           "candidates": [{"id": "T1", "contesta": ["R1"]}], "case": "Alfa chiede la risoluzione."}
    session = Session(rows, texts)
    record = rulings.build(ctx, [{"_article": ("1458", "c.c."), "d": {"id": "CC"}, "s": {"name": "1458"}}], session,
                           call=lambda m, t: {})
    return record, session


def test_a_read_penal_off_ruling_with_unread_rulings_behind_it_yields_no_error_and_keeps_the_others(monkeypatch):
    monkeypatch.setattr(rulings, "PRESELECT_PER_ARTICLE", 1)       # X1 read, X2 and X3 past the read set
    rows = [_ruling(1), _ruling(2, archive="snciv"), _ruling(3, archive="snciv")]
    record, session = _build_1458(rows, {"S1": OFF, "S2": CIVIL, "S3": CIVIL})
    assert session.read == ["S1"]                                    # only the head was read
    assert [r["id"] for r in record["kept"]] == ["X2", "X3"]          # the unread rulings kept, the penal one gone
    assert not any(r["estremi"].startswith("Cass. pen.") for r in record["kept"])
    assert record["log"]["articles"]["1458 c.c."]["filters"]["penal_on_civil"] >= 1
    assert record["log"]["articles"]["1458 c.c."]["with_sentences"] == 0


def test_a_read_set_with_a_civil_ruling_beside_the_dropped_one_keeps_it_first(monkeypatch):
    monkeypatch.setattr(rulings, "PRESELECT_PER_ARTICLE", 2)
    rows = [_ruling(1), _ruling(2, archive="snciv"), _ruling(3, archive="snciv"), _ruling(4)]
    record, session = _build_1458(rows, {"S1": OFF, "S2": CIVIL, "S3": CIVIL, "S4": OFF})
    assert sorted(session.read) == ["S1", "S2"]
    ids = [r["id"] for r in record["kept"]]
    assert ids[0] == "X2" and "X1" not in ids and "X4" not in ids and "X3" in ids    # X4: unread penal, off too


def test_the_isolated_step_keeps_the_record(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(rulings, "PRESELECT_PER_ARTICLE", 1)
    rows = [_ruling(1), _ruling(2, archive="snciv")]
    got = flow._isolated("rulings_for_record", lambda: _build_1458(rows, {"S1": OFF, "S2": CIVIL})[0], None)
    assert got is not None and [r["id"] for r in got["kept"]] == ["X2"]


# --- 2. the step-aside path -------------------------------------------------------------------------------------------------

REC_1453 = {"rubrica": "Risolubilità del contratto per inadempimento", "side": "weapon",
            "requisites": [{"id": "R1", "comma": 1, "kind": "requisito", "status": "provato",
                            "text": "uno dei contraenti non adempie le sue obbligazioni del contratto"}]}
REC_2043 = {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
            "requisites": [{"id": "R2", "comma": 1, "kind": "requisito", "status": "da provare",
                            "text": "fatto doloso o colposo che cagiona ad altri un danno ingiusto"}]}
GROUNDED = "L'art. 1453 c.c. richiede che uno dei contraenti non adempia le obbligazioni del contratto."
UNGROUNDED = "Al caso è applicabile anche l'art. 640 c.p., che punisce gli artifici e i raggiri."
RENDERED = "Il termine di prescrizione decorre dal giorno dell'inadempimento."
PASSAGE = "La risoluzione per inadempimento presuppone un inadempimento di non scarsa importanza."


def _draft():
    return ("**1. Inquadramento e questione centrale**\n\nAlfa chiede la risoluzione del contratto.\n\n"
            f"**2. Norme e giurisprudenza**\n\n{GROUNDED} {UNGROUNDED}\n\n"
            f"**3. Profili processuali e termini**\n\n{RENDERED}\n\n"
            "**4. Conclusioni**\n\nLa domanda di risoluzione ex art. 1453 c.c. appare fondata.")


def _record(article="1453 c.c."):
    return {"kept": [{"id": "X9", "article": article, "estremi": "Cass. civ., Sez. 2, n. 101/2025, 01.03.2025",
                      "su": False, "element_id": "R1", "passage": PASSAGE, "verso": "favorevole", "pertinenza": 2,
                      "date": "2025-03-01", "element": ""}],
            "orientations": [], "landmarks": {}, "log": {}}


def test_the_step_aside_prose_prints_the_rulings_of_the_articles_it_cites(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "true")
    state = {"records": {"1453 c.c.": REC_1453, "2043 c.c.": REC_2043}, "method_refused": "x"}
    seen = {}

    def build(ctx, rows):
        seen.update(ctx)
        return _record()
    out = flow.step_aside_answer(state, _draft(), [], "Alfa chiede la risoluzione.", RENDERED, build=build)
    assert list(seen["records"]) == ["1453 c.c."]               # only the cited record article, not 2043
    assert seen["candidates"] == [flow.STEP_ASIDE_NO_CANDIDATE]  # no selection: no controverted point
    assert rulings.SECTION_TITLES["rulings"] in out and "Cass. civ., Sez. 2, n. 101/2025" in out
    norms = out.index("**2. Norme e giurisprudenza**")
    assert norms < out.index(rulings.SECTION_TITLES["rulings"]) < out.index("**3. Profili processuali e termini**")
    assert rulings.CORPUS_LINE in out and state["rulings_record"]["kept"][0]["id"] == "X9"


def test_the_step_aside_prose_runs_the_grounded_rule_check(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "true")
    state = {"records": {"1453 c.c.": REC_1453}, "method_refused": "x"}
    out = flow.step_aside_answer(state, _draft(), [], "", RENDERED, build=lambda ctx, rows: _record())
    assert UNGROUNDED not in out and "640 c.p." not in out          # another charge «applicabile», no record: removed
    assert GROUNDED in out                                         # a rule on the record's element stays
    assert RENDERED in out                                         # the rendered procedural section: code-written
    removed = [e for e in state["ledger"]["entries"] if e["kind"] == "removed"]
    assert [e["step"] for e in removed] == ["step_aside:rules"] and "640 c.p." in removed[0]["sentence"]
    assert state["ledger"]["removed_by_step"] == {"step_aside:rules": 1}
    assert out.startswith("**1. Inquadramento e questione centrale**\n\n")
    assert "\n\n**3. Profili processuali e termini**\n\n" in out     # the sections' layout kept


def test_a_rule_on_a_registry_passage_stays_in_the_step_aside_prose(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "true")
    rule = "Per l'art. 1453 c.c. la risoluzione presuppone un inadempimento di non scarsa importanza."
    draft = _draft().replace(UNGROUNDED, rule)
    state = {"records": {"1453 c.c.": {**REC_1453, "requisites": []}}, "method_refused": "x"}
    out = flow.step_aside_answer(state, draft, [], "", RENDERED, build=lambda ctx, rows: _record())
    assert rule in out and GROUNDED in out              # the case's own norm: its rules stay, element or not
    assert not [e for e in (state.get("ledger") or {}).get("entries") or [] if e["kind"] == "removed"]


STEP_ASIDE_WHY = "il testo dell'art. 1453 c.c. (l'addebito o la domanda) non è stato recuperato"
OWN_RULES = ("L'art. 1453 c.c. consente al contraente non inadempiente di chiedere la risoluzione. La risoluzione "
             "presuppone un inadempimento di non scarsa importanza (art. 1455 c.c.). La diffida ad adempiere richiede "
             "un termine non inferiore a quindici giorni.")


def _prose(norms):
    return ("**1. Inquadramento e questione centrale**\n\nAlfa chiede la risoluzione del contratto.\n\n"
            f"**2. Norme e giurisprudenza**\n\n{norms}\n\n**4. Conclusioni**\n\nLa domanda appare fondata.")


@pytest.mark.parametrize("flag", ["true", "false"])
def test_the_step_aside_prose_keeps_its_rules_on_the_unread_decisive_article(monkeypatch, flag):
    """review: the method stepped aside on an unread 1453 -- no record, no store field: the plain opinion's rules on the
    case's own norm and on norms it declares nothing applicable with stay; §2 is never left a bare heading."""
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", flag)
    state = {"records": {}, "method_refused": STEP_ASIDE_WHY}
    draft = _prose(OWN_RULES)
    out = flow.step_aside_answer(state, draft, [], "Alfa chiede la risoluzione.", "", build=lambda c, r: None)
    assert out == draft and "ledger" not in state


@pytest.mark.parametrize("sentence", [
    "Al caso è applicabile anche l'art. 640 c.p.",
    "Nel caso di specie si applica anche l'art. 640 c.p.",
    "È inoltre configurabile il reato di truffa (art. 640 c.p.).",
    "Ricorre altresì la fattispecie di cui all'art. 640 c.p.",
    "La condotta integra anche il reato di cui all'art. 640 c.p.",
    "Trova altresì applicazione l'art. 2049 c.c.",
])
def test_the_step_aside_prose_declares_no_other_norm_applicable_without_a_source(monkeypatch, sentence):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "true")
    state = {"records": {}, "method_refused": STEP_ASIDE_WHY}
    out = flow.step_aside_answer(state, _prose(f"{OWN_RULES} {sentence}"), [], "Alfa chiede la risoluzione.", "",
                                 build=lambda c, r: None)
    assert sentence not in out and OWN_RULES in out
    removed = [e["sentence"] for e in state["ledger"]["entries"] if e["kind"] == "removed"]
    assert removed == [sentence]


def test_another_norm_named_in_the_request_is_the_case_own(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "false")
    sentence = "Al caso è applicabile anche l'art. 1456 c.c."
    state = {"records": {}, "method_refused": STEP_ASIDE_WHY}
    out = flow.step_aside_answer(state, _prose(sentence), [], "Alfa chiede la risoluzione ex art. 1456 c.c.", "",
                                 build=lambda c, r: None)
    assert sentence in out and "ledger" not in state


def test_another_norm_declared_applicable_stays_on_a_registry_passage():
    from src.rag import rule_statements as RS
    src = {"elements": {}, "store": {"640 c.p.": {"pena"}},
           "passages": {"640 c.p.": [RS.stems("gli artifici e raggiri inducono in errore la persona offesa")]}}
    grounded = "Al caso è applicabile l'art. 640 c.p.: artifici e raggiri che inducono in errore."
    assert RS.check_declared(grounded, src, ["1453 c.c."]) == grounded
    penalty = "Al caso è applicabile l'art. 640 c.p., punito con la reclusione."     # a store field grounds no applicability
    log = []
    assert RS.check_declared(penalty, {**src, "passages": {}}, ["1453 c.c."], log) == "" and len(log) == 1


def test_the_step_aside_prose_citing_no_record_article_gets_no_rulings_section(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "true")
    calls = []
    state = {"records": {"2043 c.c.": REC_2043}, "method_refused": "x"}
    out = flow.step_aside_answer(state, _draft(), [], "", RENDERED, build=lambda ctx, rows: calls.append(1))
    assert not calls and rulings.SECTION_TITLES["rulings"] not in out and "rulings_record" not in state


def test_the_rulings_flag_off_prints_no_section_but_still_checks(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "false")
    state = {"records": {"1453 c.c.": REC_1453}, "method_refused": "x"}
    out = flow.step_aside_answer(state, _draft(), [], "", RENDERED, build=lambda ctx, rows: _record())
    assert rulings.SECTION_TITLES["rulings"] not in out and UNGROUNDED not in out


CIVIL_ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing", "intake_rito": "civile"}


def _claimed(article):
    return {**CIVIL_ATTORE, "prior_decisions": [{"outcome": "contestazione", "articles": [article]}]}


@pytest.mark.parametrize("article", ["1223 c.c.", "1224 c.c.", "1225 c.c.", "1226 c.c.", "1227 c.c."])
def test_a_damages_measure_article_never_makes_the_method_step_aside(article):
    from src.rag import interactive_flow as flow, lawyer_method as M
    assert M.damages_measure(article)
    state = {"analysis": {}, "text": "Alfa chiede il risarcimento.", "request": "", "intake": {}, "records": {},
             "sentences": [], "questions": []}
    flow.decompose_records(state, _claimed(article), [], set(), fetch=lambda refs: [])
    assert "method_refused" not in state


def test_an_unread_claim_article_still_makes_it_step_aside():
    from src.rag import interactive_flow as flow, lawyer_method as M
    assert not M.damages_measure("1218 c.c.") and not M.damages_measure("1228 c.c.")
    state = {"analysis": {}, "text": "Alfa chiede il risarcimento.", "request": "", "intake": {}, "records": {},
             "sentences": [], "questions": []}
    flow.decompose_records(state, _claimed("1218 c.c."), [], set(), fetch=lambda refs: [])
    assert state["method_refused"] == M.NOT_APPLIED_WHY["unread"].format(art="1218 c.c.")


# --- 3. names ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,gone", [
    ("Ai fini dell'art. 624-bis c.p. il luogo costituiva privata dimora del Bianchi, a prescindere dalla titolarità "
     "del diritto sull'immobile.", "Bianchi"),
    ("Il luogo era nella stabile disponibilità di Ferraro, a prescindere dalla titolarità di un diritto reale.",
     "Ferraro"),
    ("La privata dimora della Lombardi, a prescindere dalla titolarità del bene, è tutelata.", "Lombardi"),
    ("La condotta dell'Esposito, a prescindere dalla titolarità del bene, integra il reato.", "Esposito"),
    ("La dimora di dell'Orto, a prescindere dalla titolarità del bene.", "Orto"),
    ("La dimora di ROSSI, a prescindere dalla titolarità del bene.", "ROSSI"),
    ("La condotta dello Zanetti, a prescindere dalla titolarità del bene, integra il reato.", "Zanetti"),
    ("«Bianchi, a prescindere dalla titolarità del bene, vi dimorava.»", "Bianchi"),
])
def test_e2_a_surname_before_a_prescindere_dalla_titolarita_is_redacted(text, gone):
    out = rulings.redact_names(text)
    assert gone not in out and rulings.REDACTED in out
    assert gone in rulings.names_left(text) and rulings.names_left(out) == []


@pytest.mark.parametrize("text", [
    "Secondo il Legislatore la privata dimora è tutelata, a prescindere dalla titolarità del diritto.",
    "Il ricorso del Pm è infondato; il Gip ha deciso.",
    "Il Tribunale Amministrativo Regionale del Lazio ha deciso.",
    "Il credito del Fallimento è privilegiato; il Condominio ha agito.",
    "L'appellante Rossi ha agito.",
    "La dimora dello Stato, a prescindere dalla titolarità, non rileva.",
    "Secondo il Ricorrente, il Notaio e il Medico avevano agito.",
    "La Corte ha osservato che il Ricorrente non ha provato il danno subito dal Lavoratore, né la colpa del Datore.",
    "Il giudice ha recepito le conclusioni del Ctu sul nesso causale.",
    "La responsabilità del Medico verso il Paziente è contrattuale.",
    "Secondo il Notaio, a prescindere dalla titolarità del bene, l'atto è valido.",
    "La dimora del Conduttore, a prescindere dalla titolarità del bene, è tutelata.",
    "La sede della Lombardia, a prescindere dalla titolarità del bene.",
    "La sede dell'INPS, a prescindere dalla titolarità del bene.",
])
def test_e2_roles_offices_acts_and_regions_stay(text):
    """review: a capitalised role or office after an article is no surname -- never a role noun, a gazetteer word or an
    opener. v79.12.5 (79.12.3 item 7): a surname alone after an article («il Colombo») is now read too (rule
    «article», tests/test_v79_12_5.py)."""
    assert rulings.redact_names(text) == text and rulings.names_left(text) == []


def test_e2_the_redaction_log_names_the_new_rule(monkeypatch):
    seen = []
    monkeypatch.setattr(rulings, "_log", lambda event, data, *a: seen.append((event, data)))
    rulings.redact_names("La dimora di Ferraro, a prescindere dalla titolarità, e quella del Greco sono tutelate.")
    entry = next(d for e, d in seen if e == "redaction")
    assert entry["items"][0] == {"words": "Ferraro", "rule": "saving"}       # v79.12.5: «del Greco» is «article»


def test_e2_the_nightly_counts_a_lone_surname_left_in_a_printed_passage():
    import scripts.nightly_sim as nightly
    passage = "Il luogo costituiva privata dimora del Bianchi, a prescindere dalla titolarità del diritto."
    raw = f"- Cass. pen., Sez. 5, n. 9/2025 (data n.d.), {rulings.PASSAGE_LABEL}: «{passage}»."
    got = nightly.names_check(raw)
    assert got["ok"] is False and got["count"] == 1 and "Bianchi" in got["why"]
    printed = f"- Cass. pen., Sez. 5, n. 9/2025 (data n.d.), {rulings.PASSAGE_LABEL}: «{rulings.printed(passage)}»."
    assert nightly.names_check(printed)["count"] == 0
