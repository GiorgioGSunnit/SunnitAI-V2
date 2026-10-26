"""RULING_FIT_ENABLED: a ruling reaches the plan and the writer only if a fact of the case meets its conditions,
led by that fact (defensive_generation._ruling_fit / _with_fit_facts; the lawyer's review, 26 Oct)."""
import json
import os

import pytest

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402

CASE = ("Tizio acquista dalla società Marex una barca con motore entrobordo. Durante la navigazione il motore "
        "segnala una serie di allarmi. Intervenuta la società venditrice garantisce la riparazione e la messa a "
        "punto. Tuttavia l'episodio si ripete in alto mare.")


def _ruling(name, passage):
    return {"d": {"id": name, "name": name}, "s": {"name": "corte_cassazione.1", "plain_text": passage},
            "_passage": passage}


RULINGS = [
    _ruling("Occultamento", "Gli interventi del venditore volti a rendere difficile la scoperta dei vizi escludono la "
                            "necessità della denuncia."),
    _ruling("Asilo", "Al minore bisognoso di protezione viene garantito il diritto d'asilo."),
    _ruling("Riparazione", "L'impegno del venditore a eliminare i vizi ne costituisce riconoscimento."),
    _ruling("Prescrizione", "Le manifestazioni extragiudiziali del compratore interrompono la prescrizione."),
]


class Model:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, messages, max_tokens=None, **kwargs):
        self.calls.append({"system": messages[0].content, "human": messages[-1].content, **kwargs})
        return self.reply


@pytest.fixture
def logged(monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: events.append((event, data)))
    return events


@pytest.fixture(autouse=True)
def flag_off(monkeypatch):
    monkeypatch.delenv(dg.RULING_FIT_FLAG, raising=False)
    monkeypatch.delenv(dg.PLAN_THEN_WRITE_FLAG, raising=False)


VERDICTS = json.dumps([
    {"id": "S1", "uso": "fatti", "fatto": "la condotta di Marex ha reso difficile la scoperta dei vizi",
     "perche": "occultamento"},
    {"id": "S2", "uso": "nessuno", "fatto": "", "perche": "riguarda la protezione internazionale"},
    {"id": "S3", "uso": "fatti", "fatto": "intervenuta la società venditrice garantisce la riparazione",
     "perche": "riconoscimento"},
])


def test_flag_off_leaves_the_rulings_as_they_are(monkeypatch):
    model = Model(VERDICTS)
    monkeypatch.setattr(dg, "_call_chat", model)
    assert dg._ruling_fit(CASE, "", RULINGS) is RULINGS and model.calls == []


def test_rulings_the_case_does_not_meet_go_and_the_kept_ones_carry_the_case_fact(monkeypatch, logged):
    monkeypatch.setenv(dg.RULING_FIT_FLAG, "true")
    model = Model(VERDICTS)
    monkeypatch.setattr(dg, "_call_chat", model)
    kept = dg._ruling_fit(CASE, "", RULINGS)
    # S1's "fact" is the ruling's, not the case's; S2 serves nothing; S4 has no verdict and stays
    assert [r["d"]["name"] for r in kept] == ["Riparazione", "Prescrizione"]
    assert kept[0]["_fit_fact"] == "intervenuta la società venditrice garantisce la riparazione"
    assert "_fit_use" not in kept[1]
    assert model.calls[0]["temperature"] == 0 and "[S4] Prescrizione" in model.calls[0]["human"]
    log = [d for e, d in logged if e == "case_analysis_ruling_fit"][-1]
    assert [d["ruling"] for d in log["dropped"]] == ["Occultamento", "Asilo"]
    assert log["dropped"][0]["why"].startswith("fact not in the case")
    led = dg._with_fit_facts(kept)
    assert led[0]["_passage"].startswith("Fatto del caso che ne integra i presupposti: intervenuta la società "
                                         "venditrice garantisce la riparazione. L'impegno del venditore")
    assert led[1]["_passage"] == RULINGS[3]["_passage"]


def test_general_rules_and_steps_stay_with_their_use(monkeypatch, logged):
    """Round 2 (26 Oct): a ruling stating a general rule, or a step the client can take, is kept."""
    monkeypatch.setenv(dg.RULING_FIT_FLAG, "true")
    monkeypatch.setattr(dg, "_call_chat", Model(json.dumps([
        {"id": "S1", "uso": "nessuno", "perche": "presuppone un occultamento che il caso non descrive"},
        {"id": "S2", "uso": "nessuno"},
        {"id": "S3", "uso": "regola", "fatto": ""},
        {"id": "S4", "uso": "passo", "fatto": "inviare a Marex una diffida scritta"}])))
    kept = dg._with_fit_facts(dg._ruling_fit(CASE, "", RULINGS))
    assert [r["d"]["name"] for r in kept] == ["Riparazione", "Prescrizione"]
    assert kept[0]["_passage"].startswith("Uso nel parere: regola generale sulla questione, da citare come tale")
    assert kept[1]["_passage"].startswith("Uso nel parere: passo che la parte assistita può compiere ora: inviare a "
                                          "Marex una diffida scritta. Le manifestazioni")
    log = [d for e, d in logged if e == "case_analysis_ruling_fit"][-1]
    assert [d["use"] for d in log["kept"]] == ["regola", "passo"]


def test_a_failed_check_keeps_every_ruling(monkeypatch):
    monkeypatch.setenv(dg.RULING_FIT_FLAG, "true")
    monkeypatch.setattr(dg, "_call_chat", Model("non so"))
    assert dg._ruling_fit(CASE, "", RULINGS) == RULINGS
    assert dg._ruling_fit(CASE, "", []) == []
