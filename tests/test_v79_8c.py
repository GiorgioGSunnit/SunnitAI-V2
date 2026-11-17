"""v79.8 block C -- grading from facts, not words, on synthetic sentences and conditions only:

C1. only a fatto certo integrates an element: an allegazione, a counterparty's statement («sostiene», «afferma»,
    «apprendeva che», «aveva negato»), the client's own claim letter leave it «da provare»; §2's «fatti certi» and §4's
    «provato» read the same predicate
C2. a decomposition fragment stating no condition (a bare subject, a connective, a saving clause) is no element:
    dropped at decomposition, its reason logged on the record
C3. a d'ufficio condition the store cut at its connective is one condition again; a condition joined by a connective is
    «provato» only when every part is
C4. the counters: uncertain_integrated (0), fragments per article, conditions on half a condition (0), rejoined
"""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


# --- C1 -----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence, how", [
    ("Alfa ha consegnato la merce a Beta il 3 marzo.", "accertato"),
    ("Beta sostiene di non aver mai ricevuto la merce.", "allegato"),
    ("Alfa afferma di aver consegnato la merce.", "allegato"),
    ("Il difensore apprendeva che Beta aveva ricevuto la merce.", "allegato"),
    ("Beta aveva negato la consegna della merce.", "allegato"),
    ("Con diffida del 5 aprile Alfa chiedeva a Beta il pagamento della merce consegnata.", "allegato"),
])
def test_the_certainty_of_each_sentence(sentence, how):
    from src.rag import issue_records
    assert issue_records.certainty(sentence) == how


def test_only_a_fatto_certo_integrates_and_section_2_reads_the_same_sentences():
    from src.rag import issue_records, lawyer_method as M
    text = ("Alfa ha consegnato la merce a Beta il 3 marzo. Con diffida del 5 aprile Alfa chiedeva a Beta il pagamento "
            "della merce. Beta sostiene che la merce era difettosa.")
    sents = issue_records.sentences(text)
    by_id = {s["id"]: s for s in sents}

    def settled(sup):
        req = {"kind": "requisito", "text": "la consegna della merce", "facts_status": "da provare",
               "link": {"supporting": list(sup), "contradicting": []}}
        issue_records.settle(req, by_id)
        return req
    assert settled(["S1"])["status"] == "provato"
    for sid in ("S2", "S3"):
        got = settled([sid])
        assert got["status"] == "da provare" and "allegazione" in got["why"]
    facts = M.facts_split(sents, {})
    assert facts["certi"] == [M._short(sents[0]["text"])]
    assert all(issue_records.certain_fact(by_id[s]) == (M._short(by_id[s]["text"]) in facts["certi"]) for s in by_id)


# --- C2 -----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, why", [
    ("il debitore", "soggetto senza condizione"), ("uno dei contraenti", "soggetto senza condizione"),
    ("con cui si conviene che", "connettivo senza condizione"),
    ("salvo in ogni caso il risarcimento del danno", "clausola di salvezza"),
    ("il debitore non esegue esattamente la prestazione dovuta", ""), ("la cosa mobile altrui", ""),
])
def test_a_fragment_is_no_element(text, why):
    from src.rag import rule_quote
    assert rule_quote.fragment_why(text) == why


def test_the_fragments_are_dropped_at_decomposition_with_their_reason(monkeypatch):
    from src.rag import interactive_flow as flow, issue_records, rule_quote
    rule = {"how": "civil", "commi": [(1, "x")], "selected": [1],
            "requisites": {1: ["uno dei contraenti", "il contratto è a prestazioni corrispettive",
                               "con cui si conviene che", "l'altro contraente è inadempiente"]}}
    monkeypatch.setattr(rule_quote, "article_rules", lambda *a, **k: [rule])
    monkeypatch.setattr(flow, "facts_status", lambda text, analysis: "da provare")
    frags = []
    reqs = issue_records._requisites({"_article": ("1453", "c.c."), "s": {"plain_text": "x"}}, {}, frags)
    assert [r["text"] for r in reqs] == ["il contratto è a prestazioni corrispettive",
                                         "l'altro contraente è inadempiente"]
    assert [f["why"] for f in frags] == ["soggetto senza condizione", "connettivo senza condizione"]


# --- C3 -----------------------------------------------------------------------------------------------------------------------

def test_a_condition_the_store_cut_at_its_connective_is_one_again():
    from src.rag import issue_records
    conds = [{"comma": "co. 4", "text": "il fatto è commesso con violenza alla persona e"},
             {"comma": "co. 4", "text": "la persona offesa è incapace per età"},
             {"comma": "co. 4", "text": "il fatto è connesso a un delitto procedibile d'ufficio"}]
    got = issue_records.rejoin_conditions(conds)
    assert [c["text"] for c in got] == ["il fatto è commesso con violenza alla persona e la persona offesa è incapace "
                                        "per età", "il fatto è connesso a un delitto procedibile d'ufficio"]
    assert got[0]["rejoined"] and not got[1].get("rejoined")
    assert not any(issue_records.dangling(c["text"]) for c in got)


def test_a_condition_is_provata_only_when_every_part_is():
    from src.rag import issue_records
    by_id = {"S1": {"id": "S1", "certainty": "accertato", "source": "caso",
                    "text": "Tizio ha colpito Caio con un bastone in strada."},
             "S2": {"id": "S2", "certainty": "accertato", "source": "caso",
                    "text": "Caio ha settantotto anni ed è invalido."}}
    cond = lambda: {"kind": "procedibilita", "text": "il fatto è commesso con armi quando la vittima è invalida",
                    "facts_status": "da provare"}
    half = cond()
    half["link"] = {"supporting": ["S1"], "contradicting": []}
    issue_records.settle(half, by_id)
    assert half["status"] == "da provare" and "solo una parte" in half["why"]
    assert list(half["parts"]) == ["il fatto è commesso con armi", "la vittima è invalida"]
    assert not all(half["parts"].values())
    whole = cond()
    whole["link"] = {"supporting": ["S1", "S2"], "contradicting": []}
    by_id["S1"]["text"] = "Tizio ha colpito Caio con armi in strada."
    issue_records.settle(whole, by_id)
    assert whole["status"] == "provato" and all(whole["parts"].values())


# --- C4 -----------------------------------------------------------------------------------------------------------------------

def test_the_grading_counters():
    from src.rag import interactive_flow as flow
    state = {"sentences": [{"id": "S1", "certainty": "accertato", "source": "caso", "text": "a"},
                           {"id": "S2", "certainty": "allegato", "source": "caso", "text": "b"}],
             "records": {"1453 c.c.": {"fragments": [{"why": "x"}, {"why": "y"}],
                                       "requisites": [{"status": "provato", "supporting": ["S1"]},
                                                      {"status": "provato", "supporting": ["S2"]}]},
                         "582 c.p.": {"requisites": [], "procedibilita": [
                             {"status": "da provare", "text": "il fatto è commesso e", "supporting": []},
                             {"status": "da provare", "text": "x y", "rejoined": True}]}}}
    assert flow.grading_counters(state) == {"uncertain_integrated": 1, "fragments_dropped": {"1453 c.c.": 2},
                                            "conditions_half": 1, "conditions_rejoined": 1}
