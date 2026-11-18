"""v79.10.2 -- Block B of the 79.10 brief, the user's request, on synthetic sentences, records and the nightly cells only:

B1. the party resisting a demand (a payment, a penalty, a decree asked of it or served on it; its «eccepire»; named as
    the debtor or the recipient) is never the claimant: the defending role (convenuto) in intake, resolve and the
    posture; its lines the contests, the matter's eccezioni and the reductions (1384 c.c. for a penalty), the forecast on
    the reduction's article when it is the main line; counters role_resisting, reductions_offered
B2. a question written as a task counts as a question («illustra», «valuta/valutando», «indica», «esamina»,
    «individua», «quale … sia», «se sia possibile», imperatives addressed to the lawyer); «Risposte ai quesiti» with one;
    counter questions_extracted; the nightly's check that each cell yields one at least
B3. several clients: the second person and the role forms («sei l'avvocato di X e Y», «difendi X e Y», «assisti X e
    Y», «avvocato di entrambi»), a first read listing two parties on the same side when the request addresses both;
    a section identical for the clients prints once with the note, never twice; counters clients_detected,
    clients_shared_sections, clients_sections_twice
"""

import importlib.util
import json
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
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _posture(family, role, stage="pre_filing"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


CONVENUTO = _posture("civile", "convenuto")

# --- B1: the party resisting a demand ----------------------------------------------------------------------------------------

PENALTY = ("Alfa S.r.l. chiede il pagamento della penale di 20.000 euro a Tizio per il ritardo nella consegna dei "
           "mobili. Assisto Tizio: come contestare la penale?")


@pytest.mark.parametrize("text, name", [
    (PENALTY, "Tizio"),
    ("La società Beta ha intimato a Caio di pagare 5.000 euro entro quindici giorni.", "Caio"),
    ("Gamma pretende da Mevio la restituzione della somma prestata.", "Mevio"),
    ("Il condominio ha notificato a Sempronio un decreto ingiuntivo.", "Sempronio"),
    ("Alfa chiede a Tizio il pagamento di 3.000 euro di canoni.", "Tizio"),
    ("Tizio ha ricevuto una diffida da Alfa per il saldo della fattura.", "Tizio"),
    ("Tizio, debitore della banca Delta, non ha pagato le ultime rate.", "Tizio"),
    ("Assisto il debitore ingiunto: quali difese?", ""),
    ("Ho ricevuto una lettera: la società mi chiede il pagamento della penale.", ""),
    ("Alfa ha venduto un'auto a Tizio. Cosa può eccepire Tizio?", "Tizio"),
])
def test_the_signals_of_a_demand_resisted(text, name):
    from src.rag import intake
    assert intake.resisting(text, name)


@pytest.mark.parametrize("text, name", [
    ("Tizio vuole chiedere a Caio il pagamento del prezzo.", "Tizio"),
    ("Tizio ha affidato ad Alfa il tetto. Alfa chiede il saldo di 18.000 euro e rifiuta di riparare. Assisto Tizio: "
     "quali azioni può intraprendere contro Alfa?", "Tizio"),
    ("Il pubblico ministero ha notificato a Tizio l'avviso di garanzia per il reato di truffa.", "Tizio"),
    ("Qualora Alfa chiedesse il pagamento della penale a Tizio, potrebbe agire.", "Tizio"),
])
def test_no_demand_resisted_without_the_client_as_its_object(text, name):
    from src.rag import intake
    assert intake.resisting(text, name) == ""


def test_the_party_resisting_a_demand_is_never_the_claimant_in_intake():
    from src.rag import intake
    got = intake.detect(PENALTY, model=False)
    assert got["role"] == "convenuto" and got["resisting"] and "non è mai l'attore" in got["why"]["role"]
    # the user stating the role still decides; a claim of the client's own stays the attore's
    assert intake.detect("Sono il legale dell'attore Tizio. " + PENALTY, model=False)["role"] == "attore"
    own = intake.detect("Tizio ha prestato 5.000 euro a Caio, che non li restituisce. Assisto Tizio: come recuperare la "
                        "somma?", model=False)
    assert own["role"] == "attore" and not own["resisting"]


def test_resolve_and_the_posture_never_make_the_resisting_party_the_claimant():
    from src.rag import intake, interactive_flow, posture
    unknown = {"rito": "civile", "role": intake.UNKNOWN, "stage": "pre_filing", "confidence": {"role": "default"},
               "client": {"name": "Tizio"}}
    analysis = {interactive_flow.CASE_KEY: PENALTY, "parte_assistita": "Tizio (vittima)"}
    out = intake.resolve(unknown, analysis)
    assert out["role"] == "convenuto" and out["resisting"]          # v79 (C5) does not undo it: a demand resisted
    model = {"rito": "civile", "role": "attore", "stage": "pre_filing", "confidence": {"role": "model"},
             "client": {"name": "Tizio"}}
    got = posture.of({"intake": model}, {interactive_flow.CASE_KEY: PENALTY})
    assert got["role"] == "convenuto" and got["resisting"]
    stated = {**model, "confidence": {"role": "rule"}}               # a rule's role (the user's word) is never changed
    assert posture.of({"intake": stated}, {interactive_flow.CASE_KEY: PENALTY})["role"] == "attore"


def _penalty_ctx(recs, case=PENALTY):
    return {"records": recs, "circumstances": {}, "posture": CONVENUTO, "case": case, "assessment": {},
            "institutions": [], "client": "Tizio"}


PENALTY_CLAIM = {"1382 c.c.": {"rubrica": "Effetti della clausola penale", "side": "threat",
                               "requisites": [_req(1, "il ritardo nell'adempimento", "da provare"),
                                              _req(2, "la clausola penale pattuita")]}}


def test_the_resisting_party_gets_the_contests_and_the_reduction_first():
    from src.rag import lawyer_method as M
    cands = M.thesis_candidates(_penalty_ctx(dict(PENALTY_CLAIM)))
    kinds = [c["kind"] for c in cands]
    assert kinds[0] == "contest_open" and "claim" not in kinds
    ecc = [c for c in cands if c["kind"] == "eccezione"]
    assert ecc[0]["reduction"] and ecc[0]["article"] == "1384 c.c." and ecc[0]["claim"] == "1382 c.c."
    assert ecc[0]["tesi"].startswith("Chiedere la riduzione della penale") and "art. 1384 c.c." in ecc[0]["tesi"]
    assert [c["articles"] for c in ecc[1:]] == ["art. 1460 c.c.", "art. 1455 c.c.", "art. 2946 c.c."]
    assert all(c.get("admitted_by") for c in cands)


def test_the_reduction_with_no_claim_record_and_never_without_a_penalty():
    from src.rag import lawyer_method as M
    cands = M.thesis_candidates(_penalty_ctx({}))
    assert [c["article"] for c in cands if c.get("reduction")] == ["1384 c.c."]
    sale = {"1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento", "side": "threat",
                          "requisites": [_req(1, "l'inadempimento di non scarsa importanza", "da provare")]}}
    plain = M.thesis_candidates(_penalty_ctx(sale, "Alfa chiede la risoluzione del contratto con Tizio e il "
                                                   "risarcimento del danno. Assisto Tizio."))
    assert not any(c.get("reduction") for c in plain)
    assert [c["articles"] for c in plain if c["kind"] == "eccezione"] == ["art. 1460 c.c.", "art. 1455 c.c.",
                                                                          "art. 2946 c.c."]   # v75 unchanged


def test_the_forecast_is_on_the_reductions_article_when_it_leads():
    from src.rag import lawyer_method as M
    ctx = _penalty_ctx(dict(PENALTY_CLAIM))
    ctx["candidates"] = M.thesis_candidates(ctx)
    red = next(c for c in ctx["candidates"] if c.get("reduction"))
    a = M.assessment(ctx["records"], CONVENUTO, client="Tizio")
    log = {}
    b = M.forecast(ctx, {"candidate": red["id"], "tesi": red["tesi"]}, a, log)
    assert b["article"] == "1384 c.c." and b["obiettivo"] == M.PARZIALE and b["esito"] == M.INCERTO
    assert "riduzione (art. 1384 c.c.)" in b["why"] and "1382 c.c." in b["why"] and log["forecast"]["reduction"]
    text = M.assessment_text(b, "Tizio")
    assert "**Esito probabile**: incerto: la pretesa (art. 1382 c.c.) resta" in text
    proven = dict(PENALTY_CLAIM, **{"1384 c.c.": {"rubrica": "Riduzione della penale", "side": "weapon",
                                                  "requisites": [_req(3, "la penale manifestamente eccessiva")]}})
    ctx2 = _penalty_ctx(proven)
    ctx2["candidates"] = M.thesis_candidates(ctx2)
    red2 = next(c for c in ctx2["candidates"] if c.get("reduction"))
    assert M.forecast(ctx2, {"candidate": red2["id"]}, a)["esito"] == M.FAVOREVOLE
    contest = next(c for c in ctx["candidates"] if c["kind"] == "contest_open")      # any other line: as before
    assert M.forecast(ctx, {"candidate": contest["id"], "contesta": ["R1"]}, a)["article"] == "1382 c.c."


def test_the_counters_of_b1():
    from src.rag import interactive_flow as flow
    out = flow.case_counters({"intake": {"resisting": "chiede il pagamento della penale a Tizio"},
                              "thesis_counts": {"reductions_offered": 1}})
    assert out["role_resisting"] == 1 and out["reductions_offered"] == 1
    assert flow.case_counters({})["role_resisting"] == 0


# --- B2: questions written as tasks ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("Assisto Tizio: illustra le azioni possibili.", ["illustra le azioni possibili"]),
    ("Valuti la fondatezza della pretesa di Alfa.", ["Valuti la fondatezza della pretesa di Alfa"]),
    ("Indica il termine per opporsi al decreto.", ["Indica il termine per opporsi al decreto"]),
    ("Esamina la clausola penale.", ["Esamina la clausola penale"]),
    ("Individua le eccezioni opponibili ad Alfa.", ["Individua le eccezioni opponibili ad Alfa"]),
    ("Redigi un parere, valutando se Tizio possa recedere.", ["valutando se Tizio possa recedere"]),
    ("Si chiede di valutare la validità del contratto.", ["valutare la validità del contratto"]),
    ("Avvocato, verifica se il recesso è legittimo.", ["verifica se il recesso è legittimo"]),
    ("Dimmi se posso recedere dal contratto.", ["Dimmi se posso recedere dal contratto"]),
    ("Quale sia il rimedio esperibile da Tizio.", ["Quale sia il rimedio esperibile da Tizio."]),
    ("Spiega se sia possibile chiedere i danni.", ["Spiega se sia possibile chiedere i danni."]),
    ("Il candidato, assunte le vesti del legale di Tizio, rediga parere motivato illustrando le questioni sottese.",
     ["illustrando le questioni sottese"]),
])
def test_a_question_written_as_a_task_is_a_question(text, want):
    from src.rag import intake
    assert intake.questions(text) == want


@pytest.mark.parametrize("text", [
    "La lettera non indica alcuna motivazione.",
    "Il giudice sta valutando le prove.",
    "Il giudice, valutando le prove, ha condannato Tizio.",
    "Difendi Bianca.",
    "Prepara un parere.",
    "Alfa ha venduto un'auto a Beta.",
])
def test_no_question_in_a_narrative_a_role_or_a_drafting_verb(text):
    from src.rag import intake
    assert intake.questions(text) == []


def test_one_task_is_enough_for_the_answers_block():
    from src.rag import lawyer_method as M
    ctx = {"asked": ["Illustra le azioni possibili contro Alfa"]}
    sols = [{"tesi": "Contestare il ritardo"}]
    block, _ = M.answers_block(ctx, [], "", sols, 0, {"esito": M.INCERTO, "why": "x"}, "Tizio", lambda x: x)
    assert block.startswith(M.ANSWERS_TITLE) and "1. «Illustra le azioni possibili contro Alfa» — " in block


def test_the_prose_writer_gets_the_tasks_too():
    from src.rag import plan_then_write as ptw
    got = ptw.explicit_questions("Assisto Tizio. Illustra le possibili azioni contro Caio. È valido il contratto?")
    assert got == ["Illustra le possibili azioni contro Caio", "È valido il contratto?"]
    assert ptw.explicit_questions("Prepara un parere.", "Bianca chiede: posso passare dal fondo vicino?") == [
        "Bianca chiede: posso passare dal fondo vicino?"]                            # v62's own: unchanged


def test_the_counter_and_the_nightly_check_of_b2():
    from src.rag import interactive_flow as flow
    assert flow.case_counters({"intake": {"questions": ["a b?", "c d?"]}})["questions_extracted"] == 2
    assert flow.case_counters({"text": "Tizio ha un contratto con Alfa. Esamina la clausola penale."})[
        "questions_extracted"] == 1
    nightly = _script("nightly_sim")
    for cell in nightly.cells():                      # each nightly cell yields one question at least
        assert nightly.questions_extracted_check(cell["text"])["ok"], cell["id"]
    assert not nightly.questions_extracted_check("Alfa ha venduto un'auto a Beta.")["ok"]
    assert nightly.questions_extracted_check("", {"questions_extracted": 2})["ok"]
    assert not nightly.questions_extracted_check("Indica i termini.", {"questions_extracted": 0})["ok"]


# --- B3: several clients -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", [
    "Tizio e Caio sono soci di Alfa. Sei l'avvocato di Tizio e Caio: quale strategia?",
    "Tizio e Caio sono imputati di furto. Difendi Tizio e Caio.",
    "Tizio e Caio hanno prestato denaro a Mevio. Assisti Tizio e Caio: come recuperarlo?",
    "Tizio e Caio sono imputati di furto. In qualità di difensore di Tizio e di Caio, indica la linea difensiva.",
])
def test_the_plural_in_the_second_person_and_the_role_forms(monkeypatch, case):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "case_text", lambda s: s.get("case", ""))
    assert [c["name"] for c in flow.clients_of({"case": case})] == ["Tizio", "Caio"]


def test_the_avvocato_di_entrambi_takes_the_reads_clients(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "case_text", lambda s: s.get("case", ""))
    read = {"clients": [{"name": "Alfa S.r.l.", "role": "attore"}, {"name": "Beta S.r.l.", "role": "attore"}]}
    for case in ("Sei l'avvocato di entrambi.", "Assisti entrambe le società.", "Sono il legale di ambedue."):
        assert [c["name"] for c in flow.clients_of({"read": read, "case": case})] == ["Alfa S.r.l.", "Beta S.r.l."]


def test_a_first_read_with_two_parties_on_one_side_when_the_request_addresses_both(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "case_text", lambda s: s.get("case", ""))
    read = {"parties": [{"name": "Tizio", "role": "imputato"}, {"name": "Caio", "role": "imputato"},
                        {"name": "Sempronio", "role": "persona offesa"}]}
    both = "Tizio e Caio sono imputati di furto ai danni di Sempronio. Quale strategia per Tizio e per Caio?"
    assert [c["name"] for c in flow.clients_of({"read": read, "case": both})] == ["Tizio", "Caio"]
    each = "Tizio e Caio sono imputati di furto ai danni di Sempronio. Quale strategia per ciascuno?"
    assert [c["name"] for c in flow.clients_of({"read": read, "case": each})] == ["Tizio", "Caio"]
    one = "Tizio e Caio sono imputati di furto ai danni di Sempronio. Assisto Tizio: quale strategia?"
    assert flow.clients_of({"read": read, "case": one}) == []                   # one client stated
    nobody = "Tizio e Caio sono imputati di furto ai danni di Sempronio. Quale strategia?"
    assert flow.clients_of({"read": read, "case": nobody}) == []               # the request addresses neither
    sides = {"clients": [{"name": "Sempronio", "role": "attore"}, {"name": "Caio", "role": "convenuto"}]}
    assert flow.clients_of({"read": sides, "case": "Sempronio ha convenuto Caio. Quale difesa per Sempronio e "
                                                   "per Caio?"}) == []          # two sides: never
    sued = {"clients": [{"name": "Alfa", "role": "attore"}, {"name": "Beta", "role": "attore"}]}
    assert flow.clients_of({"read": sued, "case": "Alfa ha citato Beta. Quali azioni per Alfa e Beta?"}) == []


def _composed(texts):
    """compose stubbed: each client's answer as given ({name: text})."""
    return lambda sections, ctx, plan, profiles=(), log=None: texts[ctx["name"]]


def _answer(summary, norms, solutions):
    return (f"Testa.\n\n{'**Risposte ai quesiti**'}\n\n1. «Quale strategia?» — La linea è la stessa.\n\n"
            f"**1. Sintesi**\n\n{summary}\n\n**2. Norme**\n\n{norms}\n\n**3. Soluzioni**\n\n{solutions}")


def test_a_section_identical_for_the_clients_prints_once_with_the_note(monkeypatch):
    from src.rag import lawyer_method as M
    texts = {"Alfa": _answer("Linea comune.", "Art. 1218 c.c.:  la regola.", "Per Alfa: X."),
             "Beta": _answer("Linea comune.", "Art. 1218 c.c.: la regola.\n", "Per Beta: Y.")}
    monkeypatch.setattr(M, "compose", _composed(texts))
    log = {}
    out = M.compose_clients([], [("Alfa", {"name": "Alfa"}, {}), ("Beta", {"name": "Beta"}, {})], (), log)
    note = M.SHARED_NOTE.format(names="Alfa e Beta")
    by = dict(M.answer_sections(out))
    assert note in by["Sintesi"] and "Nell'interesse" not in by["Sintesi"]       # §1 too (v79.6's «always» gone)
    assert note in by["Norme"] and by["Norme"].count("la regola") == 1          # spaces aside, the same text
    assert "**Nell'interesse di Alfa**" in by["Soluzioni"] and "**Nell'interesse di Beta**" in by["Soluzioni"]
    assert log["shared_sections"] == ["Sintesi", "Norme"] and out.count("La linea è la stessa") == 1
    assert M.printed_twice(out) == []


def test_three_clients_two_alike_print_that_text_once_under_both_names(monkeypatch):
    from src.rag import lawyer_method as M
    texts = {"Alfa": _answer("A.", "N.", "Linea X."), "Beta": _answer("B.", "N.", "Linea X."),
             "Gamma": _answer("C.", "N.", "Linea Z.")}
    monkeypatch.setattr(M, "compose", _composed(texts))
    log = {}
    out = M.compose_clients([], [(n, {"name": n}, {}) for n in ("Alfa", "Beta", "Gamma")], (), log)
    sol = dict(M.answer_sections(out))["Soluzioni"]
    assert "**Nell'interesse di Alfa e Beta**" in sol and sol.count("Linea X.") == 1 and "Linea Z." in sol
    assert log["grouped_sections"] == ["Soluzioni"] and M.printed_twice(out) == []
    twice = ("**1. Soluzioni**\n\n**Nell'interesse di Alfa**\n\nLinea X.\n\n**Nell'interesse di Beta**\n\nLinea  X.")
    assert M.printed_twice(twice) == ["Soluzioni"]


def test_a_section_only_a_later_client_has_is_kept(monkeypatch):
    from src.rag import lawyer_method as M
    texts = {"Alfa": "**1. Sintesi**\n\nA.", "Beta": "**1. Sintesi**\n\nB.\n\n**2. Norme**\n\nSolo di Beta."}
    monkeypatch.setattr(M, "compose", _composed(texts))
    out = M.compose_clients([], [("Alfa", {"name": "Alfa"}, {}), ("Beta", {"name": "Beta"}, {})], (), {})
    assert "Solo di Beta." in dict(M.answer_sections(out))["Norme"]


def test_the_nightly_check_of_b3():
    nightly = _script("nightly_sim")
    assert nightly.clients_once_check("**1. Sintesi**\n\nUna sola parte.")["ok"]
    bad = "**1. Norme**\n\n**Nell'interesse di Alfa**\n\nRegola.\n\n**Nell'interesse di Beta**\n\nRegola."
    got = nightly.clients_once_check(bad)
    assert got["check"] == "clients_sections_once" and not got["ok"] and got["why"] == "Norme"


def test_another_partys_eccepire_is_not_the_clients():
    from src.rag import intake
    assert intake.resisting("Tizio ha venduto un'auto ad Alfa. Assisto Tizio: cosa può eccepire Alfa?", "Tizio") == ""
    assert intake.resisting("Alfa ha venduto un'auto. Cosa possiamo eccepire?", "")


# --- the review of 79.10.2 ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Il mio cliente Mario Rossi mi ha chiesto di recuperare la somma di 5000 euro che gli deve la società Alfa. Come "
    "procedere?",
    "Il cliente mi ha chiesto di agire per ottenere 5000 euro da Alfa. Quali azioni ha?",
    "Il mio cliente vuole citare in giudizio la banca per il risarcimento del danno. Quali eccezioni potrebbe sollevare "
    "la banca?",
    "Assisto Mario Rossi, creditore di 10.000 euro verso la società Alfa. Quali eccezioni potrà opporre la controparte?",
    "Il datore di lavoro ha notificato al mio cliente il licenziamento per giusta causa. Come impugnarlo?",
    "Il Comune ha notificato al mio cliente un'ordinanza di demolizione. Come impugnarla?",
    "Il datore di lavoro ha intimato al mio cliente il licenziamento. Come impugnarlo?",
])
def test_the_claimant_the_dismissed_and_the_appellant_are_not_resisting_a_demand(text):
    from src.rag import intake
    got = intake.detect(text, model=False)
    assert got["role"] != "convenuto" and not got["resisting"]


@pytest.mark.parametrize("text, name", [
    ("Assisto Tizio. Alfa ha chiesto a Tizio il pagamento della penale. Come contestarla?", "Tizio"),
    ("Assisto Tizio. Alfa ha richiesto a Tizio il pagamento della penale.", "Tizio"),
    ("Assisto Tizio. Alfa ha preteso da Tizio il pagamento della penale.", "Tizio"),
    ("Assisto Tizio. Alfa intima il pagamento a Tizio. Come difenderlo?", "Tizio"),
    ("Assisto Tizio. Tizio è debitore di Alfa per 5.000 euro. Cosa fare?", "Tizio"),
    ("Assisto Tizio. Tizio è il destinatario della diffida di Alfa. Cosa fare?", "Tizio"),
    ("Assisto Tizio. Alfa pretende che Tizio paghi la penale.", "Tizio"),
    ("Assisto Tizio. Cosa può eccepire Tizio ad Alfa?", "Tizio"),
    ("Alfa ha venduto un'auto. Cosa possiamo eccepire ad Alfa?", ""),
    ("La società mi ha notificato un decreto ingiuntivo.", ""),
])
def test_the_past_tense_the_predicate_and_the_clients_own_eccepire_are_signals(text, name):
    from src.rag import intake
    assert intake.resisting(text, name)


def test_a_penalty_in_euro_is_a_penalty():
    from src.rag import intake
    assert intake.demand_kind("Assisto Tizio. Alfa pretende da Tizio 5.000 euro di penale. Cosa fare?") == "penale"
    assert intake.demand_kind("Alfa chiede 3.000 euro a titolo di penale.") == "penale"
    assert intake.demand_kind("Il codice di procedura penale non si applica.") == ""


@pytest.mark.parametrize("text, want", [
    ("Verifica fiscale del 2023: l'Agenzia ha contestato maggiori ricavi. Come difenderlo?", ["Come difenderlo?"]),
    ("Assisto Tizio. Precisa il cliente che la merce era difettosa. Come agire?", ["Come agire?"]),
    ("Esaminate le carte, il tribunale ha respinto la domanda di Tizio.", []),
    ("Indicate le somme dovute, Alfa ha agito in giudizio contro Tizio.", []),
    ("Precisa la lettera che il termine scade il 3 marzo.", []),
    ("Indica la norma che disciplina il recesso.", ["Indica la norma che disciplina il recesso"]),
    ("Determina se Tizio possa recedere dal contratto.", ["Determina se Tizio possa recedere dal contratto"]),
    ("Accerta se Tizio abbia diritto al risarcimento.", ["Accerta se Tizio abbia diritto al risarcimento"]),
    ("Suggerisci la strategia difensiva per Tizio.", ["Suggerisci la strategia difensiva per Tizio"]),
    ("Mi spieghi se Tizio può recedere.", ["Mi spieghi se Tizio può recedere"]),
    ("Ci illustri le azioni possibili.", ["Ci illustri le azioni possibili"]),
])
def test_a_task_head_with_its_object_and_never_a_noun_or_a_narrative(text, want):
    from src.rag import intake
    assert intake.questions(text) == want


def test_a_known_clients_counterparties_and_a_word_about_things_are_never_clients(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "case_text", lambda s: s.get("case", ""))
    read = {"clients": [{"name": "Tizio", "role": "imputato"}],
            "parties": [{"name": "Tizio", "role": "imputato"}, {"name": "Alfa", "role": "persona offesa"},
                        {"name": "Beta", "role": "persona offesa"}]}
    case = ("Tizio è accusato di lesioni per aver colpito Alfa e Beta durante una lite. Cosa rischia Tizio se Alfa e "
            "Beta si costituiscono parte civile?")
    assert flow.clients_of({"read": read, "case": case}) == []
    co = {"clients": [{"name": "Tizio", "role": "imputato"}],
          "parties": [{"name": "Tizio", "role": "imputato"}, {"name": "Caio", "role": "imputato"}]}
    case = ("Assisto Tizio, imputato in concorso con Caio di furto e ricettazione. Per ciascuno dei due reati, cosa "
            "rischia Tizio?")
    assert flow.clients_of({"read": co, "case": case}) == []
    case = "Assisto Tizio, imputato in concorso con Caio. Quale strategia per ciascuno dei due?"
    assert [c["name"] for c in flow.clients_of({"read": co, "case": case})] == ["Tizio", "Caio"]


def test_the_explicit_both_takes_the_reads_parties_and_the_courtesy_form(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "case_text", lambda s: s.get("case", ""))
    parties = [{"name": "Tizio", "role": "imputato"}, {"name": "Caio", "role": "imputato"},
               {"name": "Sempronio", "role": "persona offesa"}]
    case = "Sei l'avvocato di entrambi gli imputati. Quale strategia?"
    for read in ({"parties": parties}, {"parties": parties, "clients": [{"name": "Tizio", "role": "imputato"}]}):
        assert [c["name"] for c in flow.clients_of({"read": read, "case": case})] == ["Tizio", "Caio"]
    for case in ("Lei è l'avvocato di Tizio e Caio. Quale strategia?", "Lei assiste Tizio e Caio. Quale strategia?"):
        assert [c["name"] for c in flow.clients_of({"case": case})] == ["Tizio", "Caio"]
