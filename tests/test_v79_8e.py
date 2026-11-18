"""v79.8 block E -- confidence and roles, and the small items, on synthetic records, texts and passages only:

E1. «grado di certezza» on the main line's article alone: alto with none of its elements open, medio with one, basso
    otherwise; «dipende da» lists that article's open elements only; the output check: never «alto» with a «dipende da»
E2. several clients only on the user's explicit plural; the two parties of a dispute never both
E3. a civil case whose certain facts describe a reato: one line in §1 and §9, its procedibilità from the store
E4. no registry label left in any section of any client, §7 included
small: 582's shortlist from the rulings holding a quotable sentence; a criminal ruling on a civil record only naming its
institution; «configurabile» a rule verb; the signature blocks stripped are counted for the nightly
"""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RULINGS_PARTY_NAMES", "redact")


def _req(i, status):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": f"elemento {i}", "status": status}


# --- E1 -----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("statuses, grado", [(("provato", "provato"), "alto"), (("provato", "da provare"), "medio"),
                                             (("da provare", "da provare"), "basso")])
def test_the_certainty_is_the_main_articles_alone(statuses, grado):
    from src.rag import lawyer_method as M
    records = {"1667 c.c.": {"requisites": [_req(i, st) for i, st in enumerate(statuses, 1)]},
               "1218 c.c.": {"requisites": [_req(9, "da provare")]}}             # another article's open element
    a = M.certainty_of({"article": "1667 c.c.", "grado": "alto"}, records)
    assert a["grado"] == grado and "R9" not in a["open"]
    assert a["open"] == [f"R{i}" for i, st in enumerate(statuses, 1) if st == "da provare"]


def test_no_article_is_never_alto_and_the_check_flags_alto_with_a_dependency():
    from src.rag import lawyer_method as M
    assert M.certainty_of({"article": "", "grado": "alto"}, {})["grado"] == "medio"
    text = "**9. Conclusioni e valutazione del rischio**\n\n**Grado di certezza**: alto — dipende da: la prova di x."
    got = {c["check"]: c["ok"] for c in M.structure_checks(text)}
    assert got["certainty_depends"] is False
    assert {c["check"]: c["ok"] for c in M.structure_checks("**Grado di certezza**: medio — dipende da: x.")}[
        "certainty_depends"] is True


# --- E2 -----------------------------------------------------------------------------------------------------------------------

def test_several_clients_only_on_an_explicit_plural_and_never_the_two_sides(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "case_text", lambda s: s.get("case", ""))
    read = {"clients": [{"name": "Sempronio", "role": "attore"}, {"name": "Caio", "role": "convenuto"}]}
    assert flow.clients_of({"read": read, "case": "Sempronio ha convenuto Caio davanti al Tribunale. Assisto "
                                                  "Sempronio."}) == []
    assert flow.clients_of({"read": read, "case": "Sempronio ha convenuto Caio davanti al Tribunale. Sono il "
                                                  "difensore di entrambi."}) == []        # the two sides: never
    two = flow.clients_of({"case": "Tizio e Caio sono imputati. Sono il difensore di Tizio e di Caio."})
    assert [c["name"] for c in two] == ["Tizio", "Caio"]
    assert flow.adversaries("Beta ha sporto querela contro Alfa.", "Alfa", "Beta")
    assert not flow.adversaries("Alfa e Beta sono soci.", "Alfa", "Beta")


# --- E3 -----------------------------------------------------------------------------------------------------------------------

SENTS = [{"id": "S1", "text": "Tizio ha prestato la sua auto a Caio.", "source": "caso", "certainty": "accertato"},
         {"id": "S2", "text": "Scaduto il termine, Caio si rifiuta di restituirla.", "source": "caso",
          "certainty": "accertato"}]


def test_a_civil_case_describing_a_reato_gets_the_penal_route_line(monkeypatch):
    from src.rag import interactive_flow as flow, lawyer_method as M, procedural_facts
    civil = {"family": "civile", "rito": "civile", "role": "attore"}
    assert M.penal_route(SENTS, civil) == ("646 c.p.", "appropriazione indebita")
    assert M.penal_route(SENTS, {"family": "penale", "rito": "penale", "role": "imputato"}) is None
    alleged = [{**s, "certainty": "allegato"} for s in SENTS]
    assert M.penal_route(alleged, civil) is None                      # only certain facts
    row = {"_article": ("646", "c.p."), "_proc": {"querela": True, "procedibilita": "a querela della persona offesa"}}
    monkeypatch.setattr(procedural_facts, "procedibilita_text", lambda f: "Procedibile a querela della persona offesa.")
    line = flow.penal_route_text(("646 c.p.", "appropriazione indebita"), [row])
    assert line["summary"] == ("I fatti narrati possono integrare anche il reato di appropriazione indebita (art. 646 "
                               "c.p.), procedibile a querela della persona offesa: la via penale è distinta da questo "
                               "parere.")
    assert line["conclusion"].startswith("**Via penale**: appropriazione indebita (art. 646 c.p.), procedibile a querela")
    assert "procedibilità non indicata dalle fonti consultate" in flow.penal_route_text(
        ("612 c.p.", "minaccia"), [], fetch=lambda keys: [])["summary"]


def test_the_penal_route_line_is_in_section_1_and_9_once_each():
    from src.rag import lawyer_method as M
    line = M.penal_route_line("612 c.p.", "minaccia", "procedibile a querela della persona offesa")
    ctx = {"records": {}, "assessment": {}, "posture": {"family": "civile", "rito": "civile", "role": "attore"},
           "case": "Alfa minaccia Beta.", "penal_route": line}
    plan = {"soluzioni": [{"tesi": "Linea", "realismo": "medio", "contesta": [], "poggia_su": ["R1"], "pro": "",
                           "contro": "", "argomento": ""}], "obiezioni": [], "orientamenti": []}
    out = M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan)
    by = dict(M.answer_sections(out))
    assert line["summary"] in by[M.title("sintesi")] and line["conclusion"] in by[M.title("conclusioni")]
    assert out.count("art. 612 c.p.") == 2


# --- E4 -----------------------------------------------------------------------------------------------------------------------

RECORD = {"kept": [{"id": "D1", "estremi": "Cass. civ., Sez. 2, n. 77/2025 (data n.d.)", "article": "1168 c.c.",
                    "passage": "", "element": ""}]}


def test_no_registry_label_is_left_in_any_section_objections_included(monkeypatch):
    from src.rag import lawyer_method as M, rulings
    monkeypatch.setattr(M, "objections_section", lambda plan, fallback=(): "**Obiezione 1.** Lo nega G1.")
    ctx = {"records": {}, "rulings_record": RECORD, "rulings": rulings.citable(RECORD), "assessment": {},
           "posture": {"family": "civile", "rito": "civile", "role": "attore"}, "case": "Alfa è spogliato."}
    plan = {"soluzioni": [{"tesi": "Linea", "realismo": "medio", "contesta": [], "poggia_su": ["R1"], "pro": "",
                           "contro": "", "argomento": ""}], "obiezioni": [], "orientamenti": []}
    out = M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan)
    assert "G1" not in out and "Cass. civ., Sez. 2, n. 77/2025" in out
    two = M.compose_clients([("Norme applicate ai fatti", "Regola.")], [("Alfa", dict(ctx), plan),
                                                                         ("Beta", dict(ctx), plan)])
    assert "G1" not in two and rulings.printed_problems(two) == []


# --- small ----------------------------------------------------------------------------------------------------------------------

class Session:
    def __init__(self, rows, texts):
        self.rows, self.text_of = rows, texts

    def run(self, query, **kw):
        from src.rag import rulings
        rows = self.rows if query == rulings.FETCH_QUERY else [
            {"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

        class R:
            def data(self_inner):
                return rows
        return R()


def _ruling(i, archive="snpen", abstract="Lesione personale: malattia", when="2025-03-01"):
    return {"ruling": f"X{i}", "file_name": f"{archive}@s50@a2025@n{i:05d}@tS.clean.pdf", "name": "",
            "document_number": "", "date_enacted": when, "linked": True, "section": f"S{i}", "abstract": abstract}


LESIONE = {"id": "R1", "text": "dalla quale deriva una malattia nel corpo o nella mente", "status": "da provare",
           "kind": "requisito"}
PRINCIPLE = "La malattia nel corpo sussiste quando vi sia un'alterazione funzionale dell'organismo, anche modesta."
NOTHING = "Il ricorso è stato depositato nei termini e la Corte lo ha esaminato in udienza pubblica."


def test_the_shortlist_reads_past_the_best_abstracts_to_the_rulings_with_a_quotable_sentence():
    from src.rag import rulings
    ctx = {"records": {"582 c.p.": {"side": "threat", "rubrica": "Lesione personale", "requisites": [LESIONE]}},
           "candidates": [{"id": "T1", "contesta": ["R1"]}], "case": "Caio colpisce Tizio."}
    rows = [_ruling(i, abstract="Lesione personale: malattia nel corpo e nella mente") for i in range(1, 7)]
    rows.append(_ruling(9, abstract="Lesione personale"))                       # ranked last: the lowest overlap
    texts = {f"S{i}": NOTHING for i in range(1, 7)}
    texts["S9"] = PRINCIPLE + " Firmato Da: ROSSI MARIO Emesso Da: ARUBAPEC S.P.A. NG CA 3 Serial#: 7c5b1a9e"
    record = rulings.build(ctx, [{"_article": ("582", "c.p."), "d": {"id": "CP"}, "s": {"name": "582"}}],
                           Session(rows, texts), call=lambda m, t: {f"P{i}": {"frase": f"P{i}.1", "elemento": "R1",
                                                                              "pertinenza": 2, "verso": "a favore"}
                                                                    for i in range(1, 8)})
    art = record["log"]["articles"]["582 c.p."]
    assert art["with_sentences"] == 1 and art["passages"] == 1
    assert record["kept"][0]["id"] == "X9" and record["kept"][0]["passage"] == PRINCIPLE
    assert record["log"]["signatures_stripped"] == 3          # «Firmato Da», «Emesso Da», «Serial#»


def test_a_criminal_ruling_on_a_civil_record_only_naming_its_institution():
    from src.rag import rulings
    el = {"id": "R1", "text": "lo spoglio violento o clandestino del possesso", "status": "da provare",
          "kind": "requisito"}
    ctx = {"records": {"1168 c.c.": {"side": "weapon", "rubrica": "Azione di reintegrazione", "requisites": [el]}},
           "candidates": [{"id": "T1", "contesta": ["R1"]}], "case": "Alfa è spogliato."}
    plain = "Lo spoglio del possesso sussiste quando il possessore è privato della cosa con violenza."
    named = "Lo spoglio del possesso sussiste anche ai fini della reintegrazione quando il possessore è privato della cosa."
    rows = [_ruling(1, abstract="Spoglio e reintegrazione"), _ruling(2, abstract="Spoglio e reintegrazione")]
    texts = {"S1": plain, "S2": named}
    record = rulings.build(ctx, [{"_article": ("1168", "c.c."), "d": {"id": "CC"}, "s": {"name": "1168"}}],
                           Session(rows, texts), call=lambda m, t: {f"P{i}": {"frase": f"P{i}.1", "elemento": "R1",
                                                                              "pertinenza": 2, "verso": "a favore"}
                                                                    for i in range(1, 3)})
    quoted = [r["passage"] for r in record["kept"] if r["passage"]]
    # v79.9 (D4): the sentence dropped, and the ruling with it (never printed with its estremi alone)
    assert quoted == [named] and record["log"]["articles"]["1168 c.c."]["filters"]["penal_on_civil"] == 2
    assert [r["id"] for r in record["kept"]] == ["X2"]
    assert rulings.names_institution(named, "Azione di reintegrazione") and not rulings.names_institution(
        plain, "Azione di reintegrazione")


def test_configurabile_is_a_rule_verb():
    from src.rag import rulings
    assert rulings._RULE_RE.search("Il tentativo configurabile richiede atti idonei.")
