"""v60q: single shot, final shape -- no «Punti da precisare» (OPEN_POINTS_MAX 0), the closing line one sentence (1);
articles as references, never quoted text: the writer told so, rule_quote never quoting, the record check removing a
reproduced comma, the article citations in the response (2); the Capo-scope gate of Libro IV Titolo III c.c., and the
mediazione / negoziazione from the case's contract type and area (3); a party's state of mind never a point (4); the
status words never the writer's ordinary words (5). The model and the graph are stubbed."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, record_check, render_computed, rule_quote, topic_groups, writer_rules  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from test_v60f import _graded_state  # noqa: E402
from test_v60g import _ctx  # noqa: E402
from test_v60k import WRITER, _event, _single  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
LEAD = flow.ANSWER_TITLES[0]
CASE = "Leonardo passa sul fondo. Domitilla ha chiuso il cancello nel 2016."
TEXT_1061 = ("Le servitù non apparenti non possono acquistarsi per usucapione o per destinazione del padre di famiglia.\n"
             "Non apparenti sono le servitù quando non si hanno opere visibili e permanenti destinate al loro esercizio.")


# --- 1. no stages: no «Punti da precisare», one closing sentence ---------------------------------------------------------

def test_no_points_block_by_default_and_every_missing_fact_stays_in_the_parere(monkeypatch):
    state, seen = _single(monkeypatch)
    text = flow.single_shot(CASE)["answer"]
    assert f"**{C.OPEN_POINTS_TITLE}**" not in text and "punti da precisare" not in text.lower()
    assert text.rstrip().endswith(C.SINGLE_SHOT_CLOSING) and C.SINGLE_SHOT_CLOSING.count(".") == 1
    points = _event(seen, "open_points")
    assert points["asked"] == [] and points["points"]
    assert any(d["reason"].startswith("nessun blocco") for d in points["dropped"])
    assert "da verificare: il caso non dice" in dict(answer_ledger.sections_of(text))[NORMS]


def test_open_points_max_reads_the_environment(monkeypatch):
    assert flow.open_points_max() == 0
    for value, expected in (("2", 2), ("x", 0), ("-1", 0), ("", 0)):
        monkeypatch.setenv(flow.OPEN_POINTS_ENV, value)
        assert flow.open_points_max() == expected, value


# --- 2. articles as references, never quoted text --------------------------------------------------------------------------

def _refs_ctx(single=True):
    return _ctx(texts={"1062 c.c.": "Quando consta che due fondi, attualmente divisi, sono stati posseduti dallo stesso "
                                    "proprietario, la servitù s'intende stabilita.", "1061 c.c.": TEXT_1061},
                glossed={"1062 c.c.", "1061 c.c."}, single_shot=single)


SECTION_1061 = ("**Art. 1061 c.c.**\n- co. 1: da provare.\n\n"
                "(co. 1) Le servitù non apparenti non possono acquistarsi per usucapione o per destinazione del padre di "
                "famiglia.\n«Non apparenti sono le servitù quando non si hanno opere visibili e permanenti destinate al "
                "loro esercizio» (co. 2).\nLa strada sterrata è un'opera visibile e permanente: la servitù è apparente.")


def test_the_record_check_removes_a_reproduced_comma_in_single_shot_only():
    out, log = record_check.check([(NORMS, SECTION_1061)], _refs_ctx())
    body = dict(out)[NORMS]
    assert "(co. 1)" not in body and "Non apparenti sono" not in body and "non possono acquistarsi" not in body
    assert "- co. 1: da provare." in body and "La strada sterrata è un'opera visibile e permanente" in body
    assert log["rules"]["testo_della_norma"] == 2
    assert {e["reason"] for e in log["removed"]} == {"riproduce il testo dell'art. 1061 c.c.",
                                                     "riporta tra virgolette il testo dell'art. 1061 c.c."}
    _, rounds = record_check.check([(NORMS, SECTION_1061)], _refs_ctx(single=False))
    assert "testo_della_norma" not in rounds["rules"]                                       # the rounds as before


RULE_1061 = {"ref": ("1061", "c.c."), "label": "1061 c.c.", "commi": [(1, TEXT_1061.split("\n")[0]),
                                                                        (2, TEXT_1061.split("\n")[1])], "selected": [1, 2]}
WRONG_RULE = "L'art. 1061 c.c. stabilisce che il passaggio si ottiene sempre pagando un'indennità al vicino."


def test_rule_quote_never_quotes_the_comma_in_single_shot():
    quoted, log, _ = rule_quote.check_text(NORMS, WRONG_RULE, [RULE_1061], [], [])
    assert "dispone: «Le servitù non apparenti" in quoted and log[0]["how"] == "quoted"          # the rounds as before
    dropped, log, _ = rule_quote.check_text(NORMS, WRONG_RULE, [RULE_1061], [], [], quote=False)
    assert dropped.strip() == "" and log[0]["how"] == "dropped"


class _Ledger:
    def __init__(self):
        self.steps, self.removed = [], []

    def proposed_by(self, norm):
        return ["rule_quote"] if "indennit" in norm else []

    def remove(self, *args, **kwargs):
        self.removed.append(args)


def test_the_check_confirms_rule_quotes_proposal_in_single_shot():
    text = f"{WRONG_RULE} La strada sterrata è un'opera visibile e permanente."
    out, log = record_check.check([(NORMS, text)], _refs_ctx(), _Ledger())
    assert dict(out)[NORMS] == "La strada sterrata è un'opera visibile e permanente."
    assert log["rules"] == {"regola_fuori_testo": 1}
    _, rounds = record_check.check([(NORMS, text)], _refs_ctx(single=False), _Ledger())
    assert "regola_fuori_testo" not in rounds["rules"]


def test_the_single_shot_writer_is_told_references_and_the_body_keeps_none(monkeypatch):
    writer = {**WRITER, NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù quando i fondi erano di un solo proprietario."
                               "\n\n(co. 1) La destinazione del padre di famiglia ha luogo quando consta, mediante "
                               "qualunque genere di prova, che due fondi, attualmente divisi, sono stati posseduti dallo "
                               "stesso proprietario."}
    state, seen = _single(monkeypatch, writer=writer)
    text = flow.single_shot(CASE)["answer"]
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "(co. 1) La destinazione" not in norms and "fa sorgere la servitù quando i fondi" in norms
    assert "**Art. 1062 c.c.**\nEsito sui fatti del caso:" in norms        # heading and (v60r) its status sentence
    job = next(j for j in seen["jobs"][-1] if j[0] == NORMS)
    assert job[1].startswith(writer_rules.NORMS_INSTRUCTION_REFERENCES) and writer_rules.RULES_TEXT_REFERENCES in job[2]
    assert "riportando il comma" not in job[1]
    assert any(e["rule"] == "testo_della_norma" for e in _event(seen, "record_check")["removed"])
    assert writer_rules.REFERENCES_RULE in writer_rules.system_rules("Leonardo", references=True)
    assert writer_rules.REFERENCES_RULE not in writer_rules.system_rules("Leonardo")


def test_the_response_carries_the_article_citations(monkeypatch):
    state, seen = _single(monkeypatch)
    out = flow.single_shot(CASE)
    articles = out["article_citations"]
    first = articles[0]
    assert first["id"] == "1062 c.c." and first["number"] == "1062" and first["code"] == "c.c."
    assert first["commi"][0]["text"].startswith("La destinazione del padre di famiglia ha luogo")
    assert first["rows"] and set(first["rows"][0]) == {"article", "comma", "requisite", "status", "supporting_sentence",
                                                       "contradicting_sentence"}           # v60r (1): the record's rows
    assert first["rows"][0]["status"] == C.STATUS_DA_PROVARE and first["rows"][0]["requisite"].startswith("consta")
    assert first["status"] == "da provare" and first["in_body"] is True
    assert len({a["id"] for a in articles}) == len(articles)
    assert _event(seen, "article_citations")["articles"] == [a["id"] for a in articles]
    from src.chatbot.api import ChatResponse
    assert "article_citations" in ChatResponse.model_fields


# --- 3. the Capo-scope gate; the mediazione and negoziazione from the case ----------------------------------------------------

LEASE = {"fact_signals": {"facts": {"contracts": [{"type": "contratto di locazione ad uso abitativo"}]}}}


def test_the_gate_admits_a_titolo_iii_capo_only_for_the_cases_contract_type():
    assert topic_groups.capo_of(("1490", "c.c.")) == ("I", "vendita") and topic_groups.capo_of(("1453", "c.c.")) is None
    assert topic_groups.capo_of(("1490", "c.p.")) is None and topic_groups.capo_of(("1615", "c.c."))[1] == "locazione"
    scope = topic_groups.contract_scope(LEASE, "Il conduttore non paga il canone.")
    assert scope["on"] and scope["admitted"] == ["VI"]
    assert topic_groups.out_of_scope(("1575", "c.c."), scope) is None                       # its own Capo
    assert topic_groups.out_of_scope(("1453", "c.c."), scope) is None                       # Titolo II: always
    assert topic_groups.out_of_scope(("1490", "c.c."), scope).startswith("Capo I – vendita")
    named = topic_groups.contract_scope(LEASE, "Si applica l'art. 1490 c.c.?")
    assert topic_groups.out_of_scope(("1490", "c.c."), named) is None                       # the user names it
    area = topic_groups.contract_scope({"frame": {"areas": ["locazione"]}, "fact_signals": {"facts": {"parties": []}}})
    assert not area["on"] and area["off"] == "no contract type"         # the area alone: no contract type extracted
    for analysis in ({"fact_signals": {}}, {"frame": {"areas": ["diritti_reali"]},
                                            "fact_signals": {"facts": {"parties": [{"name": "Leonardo"}],
                                                                       "contracts": [{"type": "", "parties": "x"}]}}}):
        off = topic_groups.contract_scope(analysis, CASE)                        # no contract type known: no gate
        assert not off["on"] and off["off"] == "no contract type" and off["contracts"] == []
        assert topic_groups.out_of_scope(("1490", "c.c."), off) is None
        assert topic_groups.gate_rows([{"_article": ("1882", "c.c.")}], off, {}) == [{"_article": ("1882", "c.c.")}]


def test_the_gate_drops_the_rows_and_the_topics_articles_and_reanchors():
    scope = topic_groups.contract_scope(LEASE, CASE)              # a contract type known, not the Capi of 1882 and 1834
    log = {}
    rows = topic_groups.gate_rows([{"_article": ("1062", "c.c.")}, {"_article": ("1882", "c.c.")}], scope, log)
    assert [r["_article"][0] for r in rows] == ["1062"] and log["rows"][0]["article"] == "1882 c.c."
    topics = [{"title": "Servitù", "anchor": "1834 c.c.", "articles": ["1834 c.c.", "1062 c.c."],
               "matched": ["1834 c.c.", "1062 c.c."], "planned": 2, "grades": {"1834 c.c.": 2, "1062 c.c.": 3}},
              {"title": "Assicurazione", "anchor": "1882 c.c.", "articles": ["1882 c.c.", "1917 c.c."]}]
    out = topic_groups.gate_topics(topics, scope, log)
    assert len(out) == 1 and out[0]["anchor"] == "1062 c.c." and out[0]["articles"] == ["1062 c.c."]
    assert out[0]["planned"] == 1 and out[0]["grades"] == {"1062 c.c.": 3} and out[0]["matched"] == ["1062 c.c."]
    assert [x["gone"] for x in log["topics"]] == [False, True]


def test_the_mediazione_follows_the_cases_contract_and_area_never_a_written_article():
    lease = {"store_offences_only": True, **LEASE, "topic_refs": [["1882", "c.c."]]}
    got = render_computed.mediation_of(lease)
    assert got["required"] and got["subject"] == "locazione" and got["matters"]["by"].startswith("contratto")
    insurance = {"store_offences_only": True, "frame": {"areas": ["responsabilita_civile"]},
                 "fact_signals": {"facts": {"events": [{"what": "sinistro stradale"}]}}, "topic_refs": [["1917", "c.c."]]}
    assert not render_computed.mediation_of(insurance)["required"]                          # an article never makes it
    servitu = {"store_offences_only": True, "frame": {"areas": ["diritti_reali"]},
               "fact_signals": {"facts": {"parties": []}}}
    assert render_computed.mediation_of(servitu)["subject"] == "diritti reali"
    labour = {"store_offences_only": True, "frame": {"areas": ["lavoro"]}, "fact_signals": {"facts": {"parties": []}}}
    assert render_computed.case_matter(labour)["never"] == "rito del lavoro"
    assert not render_computed.mediation_of(labour)["required"]


# --- 4. a party's state of mind is argued, never a point -------------------------------------------------------------------

def test_a_partys_state_of_mind_is_never_a_point():
    for point in ("se Domitilla ha chiuso il cancello deliberatamente", "se l'acquirente era in mala fede",
                  "se Tizio fosse consapevole del vizio", "il dolo di Caio", "se ha agito intenzionalmente"):
        assert flow.state_of_mind(point), point
    for point in ("la data del rogito", "se il cancello è chiuso a chiave", "chi ha chiuso il cancello"):
        assert not flow.state_of_mind(point), point
    parties = ["Leonardo", "Domitilla"]
    assert not flow.state_of_mind("se Leonardo era in buona fede", "Leonardo", parties)      # the client's own
    assert flow.state_of_mind("se Domitilla sapeva che Leonardo passava", "Leonardo", parties)
    dropped = []
    out = flow.with_read_questions([], ["se Domitilla ha chiuso il cancello deliberatamente", "la data del rogito"], {},
                                   dropped)
    assert [q["point"] for q in out] == ["la data del rogito"]
    assert dropped == [{"point": "se Domitilla ha chiuso il cancello deliberatamente", "dropped": "stato soggettivo"}]


def test_open_points_drop_it_and_the_writer_argues_it(monkeypatch):
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    state["states_of_mind"] = ["se Domitilla ha chiuso il cancello deliberatamente"]
    state["questions"] = [
        {"id": "read_1", "kind": "free", "family": "lettura", "text": "Il caso non dice se Domitilla fosse consapevole "
         "del passaggio: puoi precisarlo?", "point": "se Domitilla fosse consapevole del passaggio",
         "unlocks": ["requisito: x"]},
        {"id": "read_2", "kind": "free", "family": "lettura", "text": "Il caso non dice la data del rogito: puoi "
         "precisarlo?", "point": "la data del rogito", "unlocks": ["requisito: x"]}]
    got = flow.open_points(state, [])
    assert {"question": "read_1", "reason": "stato soggettivo"} in got["dropped"]
    assert {"point": "se Domitilla ha chiuso il cancello deliberatamente", "reason": "stato soggettivo"} in got["dropped"]
    assert [p["question"] for p in got["points"]] == ["read_2"]
    block = flow.unknown_block(state)
    assert f"{C.STATES_OF_MIND_WRITER_TITLE}:\n- se Domitilla ha chiuso il cancello deliberatamente\n" in block
    assert "consapevole" not in block and C.STATES_OF_MIND_WRITER_RULE in block
    assert writer_rules.STATE_OF_MIND_RULE in writer_rules.system_rules("Leonardo")


def test_the_check_cuts_a_state_of_mind_given_as_missing():
    text = ("Domitilla ha chiuso il cancello dopo la diffida di Leonardo, ma da verificare: il caso non dice se l'abbia "
            "fatto deliberatamente. Il caso non dice se Domitilla fosse consapevole del passaggio.")
    out, log = record_check.check([(LEAD, text)], _refs_ctx())
    assert dict(out)[LEAD] == "Domitilla ha chiuso il cancello dopo la diffida di Leonardo."
    assert log["rules"]["stato_soggettivo"] == 2 and log["rewritten"][0]["now"] == dict(out)[LEAD]
    _, rounds = record_check.check([(LEAD, text)], _refs_ctx(single=False))
    assert "stato_soggettivo" not in rounds["rules"]
    own = "Leonardo ha usato la strada per anni; da verificare: il caso non dice se Leonardo Rossi fosse in buona fede."
    out, log = record_check.check([(LEAD, own)], {**_refs_ctx(), "parties": ["Leonardo Rossi", "Domitilla"]})
    assert dict(out)[LEAD] == own and "stato_soggettivo" not in log["rules"]               # the client's own: a fact


# --- 5. the status words are the code's ----------------------------------------------------------------------------------

def test_the_writers_status_words_become_plain_words():
    log = []
    out = dict(flow.status_words_out([
        (NORMS, "[1] Non è da provare che i fondi fossero di Mevio: è provato dal rogito. Non risulta che Domitilla "
                "abbia ricevuto la diffida; da verificare: il caso non dice quando. «Il fatto non risulta provato» è "
                "la sua tesi. Resta da verificare la data."),
        (flow.ANSWER_TITLES[3], "Termine: da verificare.")], log))
    body = out[NORMS]
    assert "Non è da dimostrare che" in body and "è dimostrato dal rogito" in body and "Non emerge che Domitilla" in body
    assert "da verificare: il caso non dice quando" in body and "«Il fatto non risulta provato»" in body
    assert "Resta da accertare la data" in body and out[flow.ANSWER_TITLES[3]] == "Termine: da verificare."
    assert len(log) == 3


def test_the_status_lines_keep_the_codes_statuses_and_the_prose_has_none(monkeypatch):
    writer = {**WRITER, NORMS: "[1] L'art. 1062 c.c. vale se è provato che i fondi erano di un solo proprietario."}
    state, seen = _single(monkeypatch, writer=writer)
    text = flow.single_shot(CASE)["answer"]
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "se è dimostrato che i fondi" in norms and "è provato che" not in norms
    assert "Esito sui fatti del caso: da provare" in norms                 # the code's status (v60r: one sentence)
    assert _event(seen, "status_words")["rewritten"]
    assert writer_rules.STATUS_WORDS_RULE in writer_rules.system_rules("Leonardo")


def test_the_sim_report_names_the_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = {"on": True, "contracts": ["contratto di locazione"], "areas": ["locazione"], "admitted": ["VI"],
            "rows": [{"article": "1490 c.c."}], "topics": [{"title": "Vendita", "dropped": ["1490 c.c."], "gone": True}]}
    line = module.capo_gate_line(gate)
    assert "admitted Capi VI" in line and "rows dropped 1490 c.c." in line and "Vendita (1490 c.c.; gone)" in line
    assert module.capo_gate_line({"on": False, "off": "no contract type"}) == "Capo gate: off (no contract type)"
    area = module.area_gate_line({"families": ["penale"], "admitted": [{"article": "2043 c.c.", "why": "regola «x»"}],
                                  "altri_profili": ["1218 c.c."]})
    assert area == "Area gate: penale; admitted outside the area: 2043 c.c. (regola «x»); altri profili: 1218 c.c."
    assert module.area_gate_line({"families": [], "off": "areas unknown"}) == "Area gate: off (areas unknown)"


# --- v60q (2): coverage -- the criteria order the topics, the top five written, at least three -----------------------------

def test_the_criteria_order_every_topic_and_the_top_five_are_covered(monkeypatch):
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))

    def topic(n, grade=2, line=None):
        return {"title": f"T{n}", "articles": [f"{n} c.c."], "grades": {f"{n} c.c.": grade},
                "hypotheses": [{"line": line}] if line else []}
    state = {"topics": [topic(1), topic(2), topic(3, 3), topic(4)], "text": "Il caso.", "request": "", "analysis": {}}
    assert flow.covered_topics(state) == [3, 1, 2, 4]                 # one meets a criterion: first; then the rest
    entry = next(d for e, d in logged if e == "single_shot_topics")
    assert entry["why"]["1"] == ["nessun criterio: per ordine"] and entry["floor"] == {"min": 3, "ok": True}
    state["topics"] = [topic(n) for n in range(1, 8)]
    assert flow.covered_topics(state) == [1, 2, 3, 4, 5]               # none meets one: the first five, in order
    state["topics"] = [topic(1), topic(2)]
    assert flow.covered_topics(state) == [1, 2]                       # fewer than three exist: all of them


# --- v60q (3): the code-area gate, general -----------------------------------------------------------------------------------

PENAL = {"frame": {"areas": ["penale"]}}


def test_an_article_outside_the_frames_area_is_admitted_only_when_named():
    rows = [{"_article": ("640", "c.p.")}, {"_article": ("2043", "c.c.")}, {"_article": ("1218", "c.c.")},
            {"_article": ("1337", "c.c."), "_first_read": "responsabilità precontrattuale"},
            {"_article": ("185", "c.p.")}, {"_article": ("2059", "c.c.")}, {"_article": ("2", "L. 241/1990")}]
    rules = flow.rule_named([{"anchor": "185 c.p.", "matched": ["185 c.p.", "2059 c.c."], "source": "tabella reati"}])
    log = {}
    kept, outside = flow.area_gate(rows, PENAL, "Tizio chiede il danno ex art. 2043 c.c.", rules, log)
    assert [r["_article"][0] for r in kept] == ["640", "2043", "1337", "185", "2059", "2"]
    assert [r["_article"][0] for r in outside] == ["1218"] and log["altri_profili"] == ["1218 c.c."]
    assert log["admitted"] == [{"article": "2043 c.c.", "why": "nominato dall'utente"},
                               {"article": "1337 c.c.", "why": "istituto della prima lettura «responsabilità "
                                                                "precontrattuale»"},
                               {"article": "2059 c.c.", "why": "regola «tabella reati»"}]
    assert log["families"] == ["penale"]
    civil = flow.area_gate([{"_article": ("640", "c.p.")}, {"_article": ("405", "c.p.p.")}, {"_article": ("1", "c.c.")}],
                           {"frame": {"areas": ["contratti"]}}, "", {}, {})
    assert [r["_article"][0] for r in civil[1]] == ["640", "405"]                     # c.p. and c.p.p. in civile
    both = flow.area_gate(rows, {"frame": {"areas": ["penale", "responsabilita_civile"]}}, "", {}, {})
    assert not both[1]                                                                  # both areas: nothing outside


def test_the_outside_articles_are_altri_profili_by_capo_and_the_parere_lists_them(monkeypatch):
    unit_of = {("cc", "2043"): {"CAPO": "TITOLO IX – DEI FATTI ILLECITI"}, ("cc", "2059"): {"CAPO": "TITOLO IX – DEI "
                                                                                                    "FATTI ILLECITI"}}
    rows = [{"_article": ("2043", "c.c."), "d": {"id": "cc"}, "s": {"name": "2043"}},
            {"_article": ("2059", "c.c."), "d": {"id": "cc"}, "s": {"name": "2059"}}]
    others = topic_groups.outside_profiles(rows, unit_of, flow.AREA_OUT_WHY)
    assert len(others) == 1 and others[0]["articles"] == ["2043 c.c.", "2059 c.c."] and others[0]["why"] == \
        flow.AREA_OUT_WHY and others[0]["capo"] and others[0]["planned"] == 0
    state, seen = _single(monkeypatch)              # v60r (2): the code-area gate applies to the single-shot list too
    state["others"] = list(state.get("others") or []) + others
    assert "2043" not in flow.single_shot(CASE)["answer"].split("**5.", 1)[1]


def test_the_check_leaves_the_codes_closing_lines_alone():
    line = ("Altre norme del tema non trattate: art. 1061 c.c. (tema «Servitù per destinazione del padre di famiglia», "
            "oltre il limite di 3 articoli), art. 1052 c.c. (tema «Passaggio coattivo», oltre il limite di 3 articoli), "
            "art. 1054 c.c. (tema «Passaggio coattivo», oltre il limite di 3 articoli).")
    text = f"**Art. 1062 c.c.**\n- co. 1: da provare.\n\nLeonardo prova la separazione dei fondi.\n\n{line}"
    out, log = record_check.check([(NORMS, text)], _refs_ctx())
    assert line in dict(out)[NORMS] and not log["removed"]
