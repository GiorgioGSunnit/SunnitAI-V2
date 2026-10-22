"""r14 (21 Oct), draft quality after the held-out run: sources limited to what the
draft cites (DRAFT_CITATIONS_ONLY_ENABLED), section 3 checked for arguments that
favour the other side (CLIENT_STANCE_CHECK_ENABLED), cited articles checked by
their real heading (QUALIFICATION_CHECK_ENABLED). Offline: model replies and the
database are mocked; the draft snippets reproduce the phrasings the held-out
drafts used, never their cases' expected answers."""

import json
import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402

L241 = "Legge sul Procedimento Amministrativo 2026"
DOCS = [{"id": "l241", "name": L241}, {"id": "cons", "name": "Codice del Consumo 2026"}]


def _article(number, code, doc="Codice Civile 2026", section=None, heading=""):
    return {"d": {"id": doc, "name": doc}, "_article": (number, code),
            "s": {"name": section or f"{number}.0.0", "plain_text": "testo", "titolo_articolo": heading}}


def _ruling(name):
    return {"d": {"id": name, "name": name}, "s": {"name": "1", "plain_text": "passo"}}


# --- which articles and rulings the draft cites ------------------------------------

def test_every_way_a_draft_cites_an_article_is_read():
    known = {("628", "c.p."), ("56", "c.p."), ("593", "c.p.p."), ("2931", "c.c."), ("21-septies", L241)}
    cases = {
        "- **Art. 2048 c.c.**\n  > Il padre e la madre": [("2048", "c.c.")],
        "Secondo l'articolo 593 del codice di procedura penale, l'imputato": [("593", "c.p.p.")],
        "l'articolo 628 del codice penale, che regola la rapina, e l'articolo 56 del codice penale":
            [("628", "c.p."), ("56", "c.p.")],
        "l'azione prevista dall'art. 112 c.p.a. e art. 114 c.p.a.": [("112", "c.p.a."), ("114", "c.p.a.")],
        "artt. 112 e 114 c.p.a.": [("112", "c.p.a."), ("114", "c.p.a.")],
        "**Norma:** Art. 609 bis c.p.": [("609-bis", "c.p.")],
        "art. 2931, comma 1, del codice civile": [("2931", "c.c.")],
        "art. 21-septies della Legge sul Procedimento Amministrativo": [("21-septies", L241)],
        "l'art. 56 è centrale": [("56", "c.p.")],                      # no code: the one known 56
        "Art. 413 CORTE SUPREMA DI CASSAZIONE - PRINCIPÎ": [],          # a book's section, not a law
    }
    for text, refs in cases.items():
        assert dg._cited_articles(text, known, DOCS) == refs, text


def test_a_ruling_is_cited_by_its_name_or_its_number():
    label = "Ordinanza sul ricorso iscritto al n. 22524/2023 R.G. proposto da Luca Colangelo contro Ministero"
    draft = "- **Ordinanza sul ricorso iscritto al n. 22524/2023 R.G. proposto da Luca Colangelo contro Ministero**"
    assert dg._ruling_is_cited(label, " ".join(draft.lower().split()), dg._decision_numbers(draft))
    assert dg._ruling_is_cited("Sentenza della Corte di Cassazione n. 7230 del 2026", "",
                               dg._decision_numbers("come affermato da Cass. 7230/2026"))
    assert dg._ruling_is_cited("Ordinanza sul ricorso 4336-2025", "", dg._decision_numbers("ricorso 4336-2025"))
    assert not dg._ruling_is_cited("Ordinanza sul ricorso 4336-2025", "", dg._decision_numbers("n. 1234/2025"))


def test_only_the_cited_sources_are_kept_with_every_row_of_a_cited_article(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    rows = [_article("2048", "c.c."), _article("2048", "c.c.", section="2048.1.0"), _article("1117", "c.c."),
            _article("189", "c.d.s.", doc="Codice della strada 2026"),
            _article("21-septies", L241, doc=L241)]
    rulings = [_ruling("Ordinanza sul ricorso iscritto al n. 22524/2023 R.G."), _ruling("Ordinanza sul ricorso 7580-2020")]
    draft = ("**Norma:** Art. 2048 c.c. e art. 21-septies della Legge sul Procedimento Amministrativo.\n"
             "Giurisprudenza: Ordinanza sul ricorso iscritto al n. 22524/2023 R.G.")
    kept = dg._draft_sources(draft, rows, rulings, DOCS)
    assert [((r["s"]["name"]), r["d"]["name"]) for r in kept] == [
        ("2048.0.0", "Codice Civile 2026"), ("2048.1.0", "Codice Civile 2026"), ("21-septies.0.0", L241),
        ("1", "Ordinanza sul ricorso iscritto al n. 22524/2023 R.G.")]


# --- client stance ---------------------------------------------------------------------

DRAFT = ("### 1. Inquadramento\nfatti\n\n### 3. Argomenti a favore di genitori di Tizio\n\n"
         "#### 3.1. Dimostrazione dell'adeguatezza delle misure di sicurezza\nL'insegnante ha vigilato.\n\n"
         "#### 4. Altro argomento interno\nancora sezione 3\n\n"
         "### 4. Punti deboli e rischio\nrischio\n\n### 5. Profili processuali\n")
ANALYSIS = {"parte_assistita": "genitori di Tizio", "posizione": "danneggiati",
            "frame": {"tasks": ["rediga motivato parere richiesto dai genitori di Tizio"]}}


def test_section_3_ends_at_punti_deboli_not_at_an_argument_numbered_4():
    start, end = dg._section_3_bounds(DRAFT)
    assert DRAFT[start:end].startswith("### 3. Argomenti") and "ancora sezione 3" in DRAFT[start:end]
    assert DRAFT[end:].startswith("### 4. Punti deboli")
    assert dg._section_3_bounds("nessuna sezione") is None


def test_an_opposing_argument_rewrites_section_3_only(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages[1].content)
        return json.dumps({"parte_assistita": "genitori di Tizio", "argomenti_contrari": [
            {"argomento": "3.1. Dimostrazione dell'adeguatezza delle misure di sicurezza",
             "perche": "prova la diligenza dell'insegnante: giova alla scuola"}]})
    monkeypatch.setattr(dg, "_chat", fake_chat)
    monkeypatch.setattr(dg, "_call_chat", lambda messages, **k: (
        calls.append(messages[1].content) or "### 3. Argomenti a favore di genitori di Tizio\n\n#### 3.1. Culpa in vigilando"))
    out = dg._check_client_stance(DRAFT, "fatti", "", ANALYSIS, "SYSTEM", "HUMAN")
    assert "Culpa in vigilando" in out and "adeguatezza delle misure" not in out
    assert out.startswith("### 1. Inquadramento") and "### 4. Punti deboli e rischio\nrischio" in out
    assert "PARTE ASSISTITA: genitori di Tizio (danneggiati)" in calls[0]
    assert "giova alla scuola" in calls[1] and calls[1].startswith("HUMAN")


def test_no_opposing_argument_or_a_bad_rewrite_keeps_the_draft(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    monkeypatch.setattr(dg, "_chat", lambda *a, **k: json.dumps({"argomenti_contrari": []}))
    assert dg._check_client_stance(DRAFT, "f", "", ANALYSIS, "S", "H") == DRAFT
    monkeypatch.setattr(dg, "_chat", lambda *a, **k: json.dumps({"argomenti_contrari": [{"argomento": "x"}]}))
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: "Ecco la sezione riscritta, senza intestazione")
    assert dg._check_client_stance(DRAFT, "f", "", ANALYSIS, "S", "H") == DRAFT


# --- qualification ---------------------------------------------------------------------

def test_an_article_whose_heading_does_not_fit_its_use_is_flagged(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    draft = ("#### 3. Attenuante\n- **Norma:** Art. 609 bis c.p. [DA VERIFICARE]\n"
             "#### 1. Consapevolezza\n- **Norma:** Art. 117 c.p.\n- **Norma:** Art. 110 c.p.")
    rows = [_article("117", "c.p.", doc="Codice Penale 2026", heading="Mutamento del titolo del reato per taluno dei concorrenti")]
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs: [
        _article(n, c, doc="Codice Penale 2026", heading="Violenza sessuale") for n, c in refs])
    sent = {}

    def fake_chat(messages, **kwargs):
        sent["human"] = messages[1].content
        return json.dumps([{"articolo": "609-bis c.p.", "compatibile": False, "perche": "reato sessuale"},
                           {"articolo": "117 c.p.", "compatibile": True, "perche": "concorso"}])
    monkeypatch.setattr(dg, "_chat", fake_chat)
    analysis = {"area": "penale", "frame": {"areas": ["penale"], "posture_type": "giudizio_penale_primo_grado"}}
    out = dg._check_qualifications(draft, analysis, rows, object(), [])
    assert "Art. 609 bis c.p. [DA VERIFICARE: l'art. 609-bis c.p. riguarda «Violenza sessuale»]" in out
    assert out.count("[DA VERIFICARE") == 1                              # the plain mark was replaced
    assert "[110 c.p.]" not in sent["human"]                             # in the area's own list: trusted
    assert "Rubrica: Violenza sessuale\nFrase: - **Norma:** Art. 609 bis c.p." in sent["human"]


def test_the_check_keeps_the_draft_on_failure(monkeypatch):
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    rows = [_article("2048", "c.c.", heading="Responsabilità dei genitori")]
    monkeypatch.setattr(dg, "_chat", lambda *a, **k: "non json")
    assert dg._check_qualifications("Art. 2048 c.c.", {"area": "civile"}, rows, object(), []) == "Art. 2048 c.c."
