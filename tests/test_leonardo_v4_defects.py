"""Leonardo v4 on Qwen3: the defects, from their exact strings, as general rules on invented cases."""
import json
import os
from types import SimpleNamespace

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import ai_chat, fact_signals as fs, text_tidy  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

USUCAPIONE, SERVITU, CONFESSORIA, COATTIVO, DANNI = ("1158", "c.c."), ("1061", "c.c."), ("1079", "c.c."), ("1051", "c.c."), ("2043", "c.c.")


def issue(title, *norms, **extra):
    return {"titolo": title, "norme": list(norms), "pronunce": [], "argomento": f"arg {title}", "obiezione": "", "replica": "", **extra}


def plan_of(*issues, central=""):
    return {"questioni": list(issues), "subordinate": [], "punti_deboli": [], "risposte": [], "questione_centrale": central}


# --- 1 -------------------------------------------------------------------------------------------------------------

class _Model:
    """A context of 20,000 tokens, as vLLM refuses what exceeds it: the overflow _call_chat answers by lowering max_tokens."""

    def __init__(self):
        self.asked = []

    def bind(self, **kwargs):
        self.kwargs = kwargs
        return self

    def invoke(self, messages):
        self.asked.append(self.kwargs.get("max_tokens"))
        return SimpleNamespace(content="ok", response_metadata={"finish_reason": "stop"})


def test_every_call_logs_the_limit_it_really_had(monkeypatch):
    events = []
    monkeypatch.setattr(ai_chat, "vlog", lambda name, data=None, **k: events.append((name, data)))
    monkeypatch.setattr(ai_chat, "chat_model", _Model())
    ai_chat._call_chat([], max_tokens=6000)
    assert next(d for n, d in events if n == "llm_call")["max_tokens"] == 6000 and ai_chat.last_reply.max_tokens == 6000


def test_the_remake_prompt_carries_the_plan_s_issues_not_its_json():
    raw = {"questione_centrale": "Il passaggio coattivo", "questioni": [
        {"titolo": "Passaggio", "norme": ["art. 1051 c.c."], "argomento": "parola " * 900}]}
    brief = ptw._plan_brief(raw)
    assert "Passaggio (art. 1051 c.c.)" in brief and "parola" not in brief and len(brief) < 100


def _replies(monkeypatch, *replies):
    seen, queue = [], list(replies)

    def fake(messages, **kwargs):
        seen.append(kwargs.get("max_tokens"))
        text, finish = queue.pop(0)
        ai_chat.last_reply.finish_reason = finish
        return text
    monkeypatch.setattr(dg, "_call_chat", fake)
    return seen


FIRST = plan_of(issue("Uno", DANNI), issue("Due", COATTIVO))
GOOD = json.dumps({"questioni": [{"titolo": "Nuova", "norme": ["art. 2043 c.c."], "argomento": "a"},
                                 {"titolo": "Altra", "norme": ["art. 1051 c.c."], "argomento": "b"}]})
KNOWN = {DANNI, COATTIVO}


def remake(monkeypatch, *replies):
    seen = _replies(monkeypatch, *replies)
    return ptw._remade("s", "h", FIRST, {}, {"q": 1}, KNOWN, {}, 5)[0], seen


def test_a_remake_that_finishes_replaces_the_plan_at_the_same_limit(monkeypatch):
    plan, seen = remake(monkeypatch, (GOOD, "stop"))
    assert plan["questioni"][0]["titolo"] == "Nuova" and seen == [ptw._PLAN_TOKENS]


def test_a_remake_that_fails_keeps_the_first_plan(monkeypatch):
    plan, seen = remake(monkeypatch, ('{"a": "x', "length"), ('{"a": "x', "length"))
    assert plan is FIRST and seen == [ptw._PLAN_TOKENS] * 2


def test_a_remake_cut_to_fewer_issues_keeps_the_first_plan(monkeypatch):
    cut = '{"questioni": [{"titolo": "Sola", "norme": ["art. 2043 c.c."], "argomento": "a"}, {"titolo": "ta'
    plan, _ = remake(monkeypatch, (cut, "length"), (cut, "length"))
    assert plan is FIRST


def test_the_code_reorders_and_rewrites_the_central_question():
    plan = plan_of(issue("Passaggio coattivo", COATTIVO), issue("Acquisto della servitù", SERVITU),
                   central="Il passaggio coattivo del fondo intercluso")
    out, _ = ptw.order_for_acquired(plan, ["usucapione: uso"])
    assert [q["titolo"] for q in out["questioni"]] == ["Acquisto della servitù", "In via subordinata: Passaggio coattivo"]
    assert out["questione_centrale"] == "Acquisto della servitù"


def test_an_issue_on_a_fired_rule_s_articles_counts_as_recognition():
    plan = plan_of(issue("Passaggio", COATTIVO), issue("Danni", DANNI), issue("Regola", ("1062", "x")))
    out, _ = ptw.order_for_acquired(plan, ["diritto"], (("1062", "x"),))
    assert out["questioni"][0]["titolo"] == "Regola"


# --- 2 -------------------------------------------------------------------------------------------------------------

def test_a_rule_s_articles_are_exempt_from_the_cap_and_need_no_grade(monkeypatch):
    monkeypatch.setattr(ptw, "_keep_cap", lambda: 2)
    rows = [{"_article": (str(i), "c.c."), "_grade": 3} for i in range(5)]
    rows += [{"_article": ("1067", "c.c."), "_tier_why": "fact rule"}, {"_article": DANNI, "_grade": 0, "_tier_why": "fact rule"}]
    forced = ptw._forced(rows)
    assert ("1067", "c.c.") in forced and DANNI in forced and forced[DANNI] == "fact rule"
    assert len(forced) == 2


def test_the_rule_s_required_articles_are_fact_rule_tier(monkeypatch):
    monkeypatch.setattr(dg, "_fetch_articles", lambda session, refs: [{"_article": r} for r in refs])
    analysis = {"fact_signals": {"fired": [{"requires": [("1067", "c.c.")]}]}}
    dg._required_rows(None, analysis, set())
    assert analysis["rule_required"] == [("1067", "c.c.")]


# --- 3 -------------------------------------------------------------------------------------------------------------

def test_issues_sharing_their_main_article_merge_even_subordinate_to_itself():
    plan = plan_of(issue("Azione confessoria", CONFESSORIA, SERVITU), issue("In via subordinata: Azione confessoria", CONFESSORIA, ("1067", "c.c.")))
    out, log = ptw.one_issue_per_article(plan)
    assert len(out["questioni"]) == 1 and out["questioni"][0]["norme"] == [CONFESSORIA, SERVITU, ("1067", "c.c.")] and log["merged"]


def test_an_article_is_argued_once_a_later_issue_refers_back():
    out, log = ptw.one_issue_per_article(plan_of(issue("Acquisto", SERVITU, DANNI), issue("Danni e turbative", ("1067", "c.c."), DANNI)))
    assert out["questioni"][1]["norme"] == [("1067", "c.c.")]
    assert "si rinvia alla questione «Acquisto»" in out["questioni"][1]["argomento"] and "2043" in out["questioni"][1]["argomento"]
    assert log["moved"] == [{"article": "2043 c.c.", "stays_in": "Acquisto", "left": "Danni e turbative"}]


def test_distinct_issues_stay():
    plan = plan_of(issue("Acquisto", SERVITU), issue("Danni", DANNI))
    assert ptw.one_issue_per_article(plan)[0]["questioni"] == plan["questioni"]


SENTENCE = "Il proprietario del fondo servente deve astenersi da ogni turbativa dell'esercizio della servitù di passaggio."


def test_a_sentence_repeated_verbatim_four_times_stays_once():
    out, log = text_tidy.tidy(f"**3. Norme**\n\n{SENTENCE} Altro.\n\n**4. Applicazione**\n\n{SENTENCE}\n\n{SENTENCE} Fine.\n\n{SENTENCE}")
    assert out.count("turbativa") == 1 and sum(e["kind"] == "repeated sentence" for e in log) == 3


def test_near_verbatim_goes_different_numbers_stay():
    near = SENTENCE.replace("deve astenersi", "si deve astenersi")
    assert text_tidy.tidy(f"{SENTENCE} {near}")[0].count("turbativa") == 1
    one = "Il termine di prescrizione del diritto al risarcimento è di cinque anni ex art. 2947 c.c."
    assert text_tidy.tidy(f"{one} {one.replace('2947', '2946')}")[0].count("prescrizione") == 2


# --- 4 -------------------------------------------------------------------------------------------------------------

def test_the_exact_cut_quote_before_a_citation():
    assert text_tidy.unreadable("La destinazione del padre di famiglia ha luogo quando consta mediante (art. 1062 c.c., co. 1).") == "cut clause"


def test_a_citation_after_a_whole_clause_stays():
    assert text_tidy.unreadable("La servitù si acquista per destinazione (art. 1062 c.c., co. 1).") == ""


RULE_2043 = ("Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a "
             "risarcire il danno.")


def rows():
    return [{"_article": DANNI, "s": {"plain_text": RULE_2043}},
            {"_article": COATTIVO, "s": {"plain_text": "Il proprietario del fondo intercluso ha diritto al passaggio.\nIl passaggio si può "
                                                       "chiedere anche per ampliare un passaggio esistente."}}]


def test_the_exact_comma_opening_as_a_sentence_and_its_double_stop():
    text = "Bruno ha diritto al risarcimento. Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto (art. 2043 c.c., co. 1).."
    out, log = text_tidy.against_articles(text, rows())
    assert "Qualunque" not in out and "risarcimento" in out and log[0]["kind"] == "comma opening as a sentence"


def test_a_whole_comma_quoted_stays():
    out, _ = text_tidy.against_articles(f"{RULE_2043} (art. 2043 c.c., co. 1).", rows())
    assert "obbliga colui" in out


def test_a_double_stop_goes_an_ellipsis_stays():
    assert text_tidy.tidy("Il fondo è di Bruno (art. 2 c.c.).. Fine del testo qui.")[0].count("..") == 0
    assert "..." in text_tidy.tidy("Il fondo è di Bruno... Fine del testo qui.")[0]


# --- 5 -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("before, after", [
    ("Si applica l'art. 1051 c.c., comma 10, al fondo.", "Si applica l'art. 1051 c.c., al fondo."),
    ("Secondo l'art. 1051 c.c., co. 4 il giudice decide.", "Secondo l'art. 1051 c.c. il giudice decide."),
    ("Ai sensi del comma 10 dell'art. 1051 c.c. il giudice decide.", "Ai sensi dell'art. 1051 c.c. il giudice decide.")])
def test_a_comma_past_the_last_goes_the_article_stays(before, after):
    assert text_tidy.against_articles(before, rows())[0] == after


def test_a_real_comma_or_an_unknown_article_stays():
    for text in ("Si applica l'art. 1051 c.c., comma 2, al fondo.", "Si applica l'art. 9999 c.c., comma 40, al fondo."):
        assert text_tidy.against_articles(text, rows())[0] == text


# --- 6 -------------------------------------------------------------------------------------------------------------

def test_the_exact_condition_repeating_its_clause():
    assert text_tidy.self_condition("In tal caso il termine annuale è decorso, se il termine annuale è decorso.") \
        == "In tal caso il termine annuale è decorso."
    assert "se il" not in text_tidy.tidy("Il termine annuale è decorso, se il termine annuale è decorso. Fine del testo qui.")[0]


def test_a_real_condition_stays():
    sentence = "Il termine annuale è decorso, se lo spoglio risale al 2010."
    assert text_tidy.self_condition(sentence) == sentence


# --- 7 -------------------------------------------------------------------------------------------------------------

def test_the_rules_are_logged_fired_or_not_with_their_evidence(monkeypatch):
    events = []
    monkeypatch.setattr(fs, "vlog", lambda name, data=None, **k: events.append((name, data)))
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    facts = {"property_history": [{"asset": "terreni di Bruno e Carla", "owner": "Dario"}, {"asset": "nord", "owner": "Bruno"},
                                  {"asset": "sud", "owner": "Carla"}],
             "works_or_signs": [{"what": "strada", "visible": "sì", "before_separation": "sì"}]}
    fs.signals_from(facts, rules=True)
    rules = next(d for n, d in events if n == "fact_rules")["rules"]
    line = next(r for r in rules if "destinazione_padre_famiglia" in r)
    assert line.startswith("FIRED ") and "dedotto dalla sequenza" in line
    assert any(r.startswith("not fired ") for r in rules)


def test_the_log_prints_rules_and_the_final_plan_whole():
    from src.rag import verbose_logger
    assert {"rules", "issues"} <= verbose_logger.FULL_KEYS
