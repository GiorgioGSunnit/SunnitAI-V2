"""v79.12.5 -- standalone surnames, text hygiene, the two counters that moved the wrong way; synthetic texts only (no
model, no Neo4j, no network):

a. (79.12.3 item 7) a surname alone after an article or an articled preposition («il Rossi», «del Bianchi», «dal
   Verdi», «la Russo») is redacted (rule «article», logged with its word) and counted by the nightly's own reading
   (names_left); a role noun (data/redaction_stoplist.json «role_nouns»), a gazetteer word or an opener after an
   article stays; review: an office's, a body's or an act's name continued by «del» and its forms is no surname («la
   Corte dei Conti», «il Codice del Consumo»), a party after an office still is
b. (item 8) text hygiene: the 168-bis list's draft status only in the table's «status», never printed; every special
   law's document title printed as its legal citation (data/corpus_expected.json's archive names, never a guess), read
   back to its document; the consumer code's terms bar its guarantee actions through the same table; the counters
   see_point and to_verify per answer
c. (item 9) the two counters that moved the wrong way: a logged-removed sentence longer than the log's 300 characters
   is struck whole when reprinted (before: matched by its cut prefix, «contained», left in the text); one bad entry
   never voids the settle; a ruling's date read in more shapes, from its name and its own text, never from a date it
   cites -- and «data n.d.» counted against the rulings printed
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))


# --- a. standalone surnames after an article (79.12.3 item 7) ---------------------------------------------------

@pytest.fixture
def names(monkeypatch, tmp_path):
    from src.rag import rulings
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv(rulings.NAMES_ENV, raising=False)
    seen = []
    monkeypatch.setattr(rulings, "_log", lambda event, data, *a: seen.append((event, data)))
    rulings.seen = seen
    yield rulings
    del rulings.seen


@pytest.mark.parametrize("text,gone", [
    ("Il Rossi si è introdotto nell'abitazione.", "Rossi"),
    ("La condotta del Bianchi integra il reato.", "Bianchi"),
    ("Il bene fu sottratto dal Verdi durante la notte.", "Verdi"),
    ("Secondo la Corte, la Russo non aveva titolo.", "Russo"),
    ("La somma fu versata al Gallo il giorno stesso.", "Gallo"),
    ("Nella dichiarazione resa dalla Ferri si legge altro.", "Ferri"),
    ("L'accordo stipulato col Moretti era nullo.", "Moretti"),
    ("Il ricorso proposto nell'interesse del ROSSI è inammissibile.", "ROSSI"),
    ("Il ricorso del Rossi-Bianchi è infondato.", "Rossi-Bianchi"),
    ("«Il Colombo si è introdotto nell'abitazione.»", "Colombo"),
])
def test_item7_a_surname_alone_after_an_article_is_redacted_and_counted(names, text, gone):
    out = names.redact_names(text)
    assert gone not in out and names.REDACTED in out
    assert gone in names.names_left(text) and names.names_left(out) == []


@pytest.mark.parametrize("text", [
    "Il Tribunale ha rigettato la domanda; la Corte ha confermato.",
    "Il Pubblico Ministero ha impugnato la sentenza dinanzi alla Cassazione.",
    "Il Codice e la Legge lo prevedono; la Costituzione lo garantisce.",
    "Secondo il Ricorrente, il Notaio e il Giudice avevano errato.",
    "L'esame del Teste e la relazione del Perito sono concordi.",
    "Il Convenuto ha eccepito la prescrizione contro l'Attore; il Curatore è intervenuto.",
    "La difesa dell'imputato e quella dell'Imputato, e la posizione dell'Indagato.",
    "La responsabilità dell'imputato è esclusa; il ricorso dell'Imputata e quello dei Ricorrenti.",
    "La Parte Civile e la Difesa hanno concluso; la Sentenza impugnata è nulla.",
    "Il Consiglio di Stato e la Procura della Repubblica; la Sezione ha deciso.",
    "Il Sindaco del Comune e il Prefetto della Provincia.",
    "Gli atti del XVIII secolo non rilevano.",
    "MOTIVI. IL RICORSO È INFONDATO.",
])
def test_item7_an_office_a_role_or_an_act_after_an_article_stays(names, text):
    assert names.redact_names(text) == text and names.names_left(text) == []


def test_item7_the_role_list_lives_in_data_and_has_the_briefs_roles(names):
    raw = json.loads((ROOT / "data" / "redaction_stoplist.json").read_text(encoding="utf-8"))
    roles = set(raw["role_nouns"])
    for word in ("Ricorrente", "Notaio", "Giudice", "Imputato", "Teste", "Convenuto", "Attore", "Curatore", "Perito"):
        assert word in roles, word
    assert {"Tribunale", "Corte", "Pubblico", "Ministero", "Cassazione", "Codice", "Legge"} <= set(raw["words"])


def test_item7_a_word_added_to_the_role_list_stays(names, monkeypatch):
    text = "La dichiarazione del Brambilla è attendibile."
    assert names.redact_names(text) != text and names.names_left(text) == ["Brambilla"]
    stop = dict(names._stoplist())
    stop["role_nouns"] = stop["role_nouns"] | {"Brambilla"}
    monkeypatch.setattr(names, "_stoplist", lambda: stop)
    assert names.redact_names(text) == text and names.names_left(text) == []


def test_item7_each_article_redaction_is_logged_with_its_rule_and_word(names):
    names.redact_names("La dimora di Ferraro, a prescindere dalla titolarità, e quella del Greco sono tutelate.")
    entry = next(d for e, d in names.seen if e == "redaction")
    assert entry["items"] == [{"words": "Ferraro", "rule": "saving"}, {"words": "Greco", "rule": "article"}]
    assert entry["rules"] == {"saving": 1, "article": 1}


def test_item7_one_surname_is_one_span_whatever_rules_read_it(names):
    out = names.redact_names("Il sig. Rossi, il Rossi Mario e del Bianchi, a prescindere dalla titolarità del bene.")
    assert out == "Il sig. [omissis], il [omissis] e del [omissis], a prescindere dalla titolarità del bene."
    entry = next(d for e, d in names.seen if e == "redaction")
    assert [i["rule"] for i in entry["items"]] == ["titled", "pair", "saving"]


def test_item7_the_nightly_counts_a_surname_left_after_an_article(names):
    import scripts.nightly_sim as nightly
    passage = "Il bene fu sottratto dal Verdi e consegnato alla Russo."
    raw = f"- Cass. pen., Sez. 2, n. 7/2025 (data n.d.), {names.PASSAGE_LABEL}: «{passage}»."
    got = nightly.names_check(raw)
    assert got["ok"] is False and got["count"] == 2 and "Verdi" in got["why"] and "Russo" in got["why"]
    printed = f"- Cass. pen., Sez. 2, n. 7/2025 (data n.d.), {names.PASSAGE_LABEL}: «{names.printed(passage)}»."
    assert nightly.names_check(printed)["count"] == 0
    kept = f"- Cass. pen., Sez. 2, n. 7/2025 (data n.d.), {names.PASSAGE_LABEL}: «Il Ricorrente e il Tribunale.»."
    assert nightly.names_check(kept)["count"] == 0


# review (79.12.5): an office's, a body's or an act's name continued by «del» and its forms is no surname
@pytest.mark.parametrize("text", [
    "La Corte dei Conti ha condannato; il Consiglio dei Ministri ha deliberato.",
    "Ai sensi del Codice del Consumo e del Codice della Strada.",
    "Lo Statuto del Contribuente e il Garante della Privacy.",
    "L'Autorità Garante della Concorrenza e del Mercato ha sanzionato.",
    "Il Testo Unico delle Imposte sui Redditi e il Codice della Crisi d'Impresa.",
    "La Presidenza del Consiglio e il Ministero dello Sviluppo Economico.",
    "Il Ministero del Lavoro e delle Politiche Sociali; il Ministero delle Infrastrutture e dei Trasporti.",
    "Il Codice dei Contratti Pubblici e il Codice delle Assicurazioni Private.",
    "Dal Lunedì al Venerdì, nel Gennaio scorso e dopo il Natale.",
    "Il Responsabile, il Contribuente, il Conducente e il Segretario; il Lodo e il Dispositivo.",
])
def test_review_an_offices_name_after_del_stays(names, text):
    assert names.redact_names(text) == text and names.names_left(text) == []


@pytest.mark.parametrize("text,gone", [
    ("La Corte dei Conti e il Conti hanno deciso.", "il [omissis]"),
    ("La domanda proposta al Tribunale dal Rossi è infondata.", "dal [omissis]"),
    ("Il ricorso contro il Ministero della Salute e del Bianchi.", "e del [omissis]"),
])
def test_review_a_party_after_an_office_is_still_a_surname(names, text, gone):
    out = names.redact_names(text)
    assert gone in out and len(names.names_left(text)) == 1 and names.names_left(out) == []


def test_review_office_tail_reads_only_del_and_its_forms(names):
    text = "la Corte dei Conti, al Tribunale dal Rossi"
    assert names.office_tail(text, text.index("dei"))
    assert not names.office_tail(text, text.index("dal"))
    assert not names.office_tail(text, text.index("al "))


# --- b. item 8: text hygiene ------------------------------------------------------------------------------------------

@pytest.fixture
def M():
    from src.rag import lawyer_method
    return lawyer_method


def test_b_the_168_bis_draft_status_is_never_printed(M):
    raw = json.loads((ROOT / "data" / "lawyer_method.json").read_text(encoding="utf-8"))
    entry = next(r for r in raw["riti_alternativi"] if r["id"] == "messa_alla_prova")
    assert entry["status"].startswith("bozza") and "provisional" not in entry
    printed = [v for r in raw["riti_alternativi"] for k, v in r.items() if isinstance(v, str) and k != "status"]
    assert not [x for x in printed if "elenco interno" in x or "bozza" in x]
    lines = [x["line"] for pena, n in (("reclusione da due a otto anni", "648"), ("reclusione da tre a sette anni", "572"),
                                       ("", "648")) for x in M.rite_statuses(pena, n)]
    assert lines and not [x for x in lines if "bozza" in x or "elenco interno" in x or x.endswith("..")]


SYNTH = ("Si applica l'art. 73 Testo Unico sugli Stupefacenti 2026, e l'art. 18 (Riforma del Lavoro Fornero) non "
         "rileva; vedi anche Testo  Unico   sugli Stupefacenti 2026, art. 75.")


def test_b_a_special_laws_title_is_printed_as_its_citation():
    from src.rag import corpus_laws
    log = []
    out = corpus_laws.cite_titles(SYNTH, log)
    assert out == ("Si applica l'art. 73 d.P.R. 309/1990, e l'art. 18 L. 92/2012 non rileva; vedi anche d.P.R. "
                   "309/1990, art. 75.")
    assert sorted(x["citation"] for x in log) == ["L. 92/2012", "d.P.R. 309/1990", "d.P.R. 309/1990"]
    assert corpus_laws.cited_label("73 Testo Unico sugli Stupefacenti 2026 co. 5") == "73 d.P.R. 309/1990 co. 5"
    assert corpus_laws.cited_label("2043 c.c.") == "2043 c.c."


def test_b_the_table_is_the_archive_names_mapping_never_a_guess():
    from src.rag import corpus_laws
    exp = corpus_laws.expected()
    for table in (exp["laws"], exp["codes"]):
        for key, spec in table.items():
            for name in spec.get("archive_names") or []:
                assert corpus_laws.citation_of(name) == key
    # never by a number in a name, nor a title the table does not map
    assert corpus_laws.citation_of("Legge 7 agosto 1990, n. 241") == ""
    assert corpus_laws.citation_of("Testo unico enti locali 2025") == ""
    assert corpus_laws.cite_titles("art. 3 Testo unico enti locali 2025") == "art. 3 Testo unico enti locali 2025"
    # a title the table does not map is never printed by guess either (v79.19.6: the Statuto is mapped now; a name of
    # another act, numbered or not, stays as written)
    assert corpus_laws.cite_titles("Statuto delle Imprese 2026") == "Statuto delle Imprese 2026"
    assert corpus_laws.cite_titles("Legge 20 maggio 1970, n. 300") == "Legge 20 maggio 1970, n. 300"


def test_b_every_label_builder_prints_the_citation_and_the_reader_finds_the_document():
    from src.rag import article_codes, citations, issue_spotting
    assert article_codes.law_label("Testo Unico sugli Stupefacenti 2026") == "d.P.R. 309/1990"
    assert article_codes.law_label("Riforma del Lavoro Fornero 2026") == "L. 92/2012"
    assert article_codes.law_label("Testo unico enti locali 2025") == "(Testo unico enti locali)"
    docs = [{"id": "D1", "name": "Testo Unico sugli Stupefacenti 2026"}, {"id": "D2", "name": "Codice Penale 2026"}]
    assert issue_spotting.cited_document("d.P.R. 309/1990", docs)["id"] == "D1"
    assert citations.refs_in("l'art. 73 d.P.R. 309/1990", docs) == [("73", "Testo Unico sugli Stupefacenti 2026")]


def test_b_the_flow_prints_no_title_and_counts_what_it_cited_and_what_is_left():
    from src.rag import interactive_flow as flow
    state = {}
    docs = [{"name": "Testo Unico sugli Stupefacenti 2026"}, {"name": "Testo unico enti locali 2025"},
            {"name": "Codice Penale 2026"}]
    draft = SYNTH + " L'art. 50 Testo unico enti locali 2025 e l'art. 575 Codice Penale 2026."
    out = flow.law_titles_cited(state, draft, docs)
    assert "Stupefacenti 2026" not in out and "Fornero" not in out and "d.P.R. 309/1990" in out
    assert state["block_counters"] == {"law_titles_cited": 3, "law_titles_left": 1}    # the unmapped TUEL, a code aside


def test_b_the_see_point_and_to_verify_counters():
    from src.rag import interactive_flow as flow
    state = {}
    flow.text_counted(state, "Il termine è da verificare (v. punto 3). Da verificare anche la data (v. punti 2 e 4). "
                             "V. punto 1.")
    c = state["block_counters"]
    assert c["see_point"] == 3 and c["to_verify"] == 2 and c["rulings_printed"] == 0 and c["rulings_date_nd"] == 0


def test_b_the_consumer_codes_terms_bar_its_guarantee_actions_through_the_table():
    from src.rag import fact_signals as F
    modules = json.loads((ROOT / "data" / "institution_modules.json").read_text(encoding="utf-8"))["modules"]
    consumo = next(m for m in modules if m["id"] == "consumo")
    assert set(F.CONSUMER_GUARANTEE_ACTIONS) <= set(consumo["actions"])         # the module's own, never invented
    dec = [{"id": "denuncia_conformita_consumo", "status": "DECORSO"}]
    pres = [{"id": "azione_conformita_consumo", "status": "DECORSO"}]
    assert [x["kind"] for x in F.barring_terms("135-bis Codice del Consumo 2026", dec)] == ["decadenza"]
    assert [x["kind"] for x in F.barring_terms("135-ter d.lgs. 206/2005", pres)] == ["prescrizione"]
    assert F.barring_terms("135-bis Codice del Consumo 2026 co. 2", dec)
    assert F.barring_terms("1490 c.c.", dec) == [] and F.barring_terms("52 Codice del Consumo 2026", dec) == []
    assert F.barring_terms("135-bis Codice del Consumo 2026", [{"id": "denuncia_conformita_consumo",
                                                                 "status": "IN CORSO"}]) == []
    assert F.same_code("Codice del Consumo 2026", "d.lgs. 206/2005") and not F.same_code("c.c.", "d.lgs. 206/2005")


# --- c. item 9: the two counters that moved the wrong way ----------------------------------------------------------------

LONG = ("Il compratore che abbia scoperto il difetto della merce consegnata dal venditore deve denunciarlo entro il "
        "termine previsto dalla legge, a pena di decadenza dalla garanzia, salvo che il venditore abbia riconosciuto "
        "l'esistenza del vizio oppure lo abbia occultato, e in tal caso la denuncia non è necessaria e l'azione resta "
        "proponibile nei termini ordinari di prescrizione decorrenti dalla consegna del bene al compratore.")


def test_c_a_long_removed_sentence_reprinted_is_struck_whole():
    from src.rag import answer_ledger as L
    assert len(LONG) > L.SENTENCE_CHARS
    led = L.Ledger()
    led.remove("non registrato", "Valutazione", LONG, [], rule="requisito_non_registrato")
    entry = led.entries[0]
    assert len(entry["sentence"]) == L.SENTENCE_CHARS and entry["full"] == LONG      # the log cut, the whole kept
    text = f"**1. Sintesi**\n\nIl caso riguarda una vendita.\n\n**2. Valutazione**\n\n{LONG} Il resto resta.\n"
    old = {k: v for k, v in entry.items() if k != "full"}           # the entry as logged before v79.12.5
    out, report = L.settle(text, led.entries)
    assert LONG[:80] not in out and "Il resto resta." in out
    assert report["struck"] and not report["kept"] and entry.get("settled") == "struck"
    assert L.removed_present(out, led.entries) == []
    # what the cut prefix alone did: found inside the longer sentence, never struck, left in the text as «kept»
    still, rep = L.settle(text, [old])
    assert LONG[:80] in still and rep["kept"] and old["kept_why"] == "contained"


def test_c_a_steps_long_removal_is_whole_too_and_never_unaccounted():
    from src.rag import answer_ledger as L
    led = L.Ledger()
    before = [("Valutazione", f"{LONG} La domanda è fondata.")]
    after = [("Valutazione", "La domanda è fondata.")]
    led.record("unsupported_out", before, after)
    entry = next(e for e in led.entries if e["kind"] == "removed")
    assert entry["full"] == LONG and led.unaccounted(after) == []
    reprinted = f"**2. Valutazione**\n\n{LONG} La domanda è fondata."
    out, _ = L.settle(reprinted, led.entries)
    assert LONG[:80] not in out and "La domanda è fondata." in out


def test_c_one_bad_entry_never_voids_the_settle():
    from src.rag import answer_ledger as L
    from src.rag import interactive_flow as flow
    good = "La responsabilità del venditore è esclusa per il vizio palese al momento della consegna."
    entries = [{"step": "x", "kind": "removed", "sentence": ["non", "un", "testo"]}, "not an entry",
               {"step": "compose", "kind": "removed", "sentence": good}]
    text = f"**1. Sintesi**\n\n{good} Altro testo.\n"
    out, report = L.settle(text, entries)
    assert good not in out and report["struck"] == [good]
    state = {"ledger": {"entries": [dict(e) if isinstance(e, dict) else e for e in entries]}}
    assert good not in flow.ledger_settled(state, text) and state["block_counters"]["ledger_removed_present"] == 0


def test_c_a_settle_that_raises_still_counts_on_the_text_as_it_is(monkeypatch):
    from src.rag import answer_ledger as L
    from src.rag import interactive_flow as flow
    sentence = "La domanda di risoluzione del contratto è fondata sui fatti accertati."
    state = {"ledger": {"entries": [{"step": "compose", "kind": "removed", "sentence": sentence}]}}

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(L, "settle", boom)
    text = f"**1. Sintesi**\n\n{sentence}\n"
    assert flow.ledger_settled(state, text) == text
    c = state["block_counters"]
    assert c["ledger_removed_present"] == 1 and c["ledger_settle_failed"] == 1     # never a stale 0


@pytest.mark.parametrize("value,want", [
    ("2025-03-12", "2025-03-12"), ("2025-03-12T00:00:00Z", "2025-03-12"), ("12/03/2025", "2025-03-12"),
    ("2025/03/12", "2025-03-12"), ("2025.03.12", "2025-03-12"), ("20250312", "2025-03-12"),
    ("12 marzo 2025", "2025-03-12"), ("1° aprile 2025", "2025-04-01"), ("3 Aprile 2025", "2025-04-03"),
    ("2019-01-01", None), ("", None), (None, None), ("n. 1234", None), ("1741737600000", None),
])
def test_c_a_rulings_date_in_every_shape(value, want):
    from src.rag import rulings
    got = rulings.ruling_date(value, "2025")
    assert (got.isoformat() if got else None) == want


def test_c_the_rulings_own_date_never_a_date_it_cites():
    from src.rag import rulings
    assert str(rulings.own_date("Il ricorso è rigettato. Così deciso in Roma, nella camera di consiglio del "
                                "12/03/2025.", "2025")) == "2025-03-12"
    assert str(rulings.own_date("Depositata in Cancelleria il 3 aprile 2025", "2025")) == "2025-04-03"
    assert rulings.own_date("come già affermato da Cass. n. 3 del 04/05/2025, il principio è consolidato",
                            "2025") is None
    p = rulings.parse_estremi("snpen@s20@a2025@n00007@tS")
    assert rulings.dated({"date_enacted": None, "name": "Cass. pen. 12 marzo 2025"}, p).isoformat() == "2025-03-12"
    assert rulings.dated({"date_enacted": "2025-01-09", "name": "12 marzo 2025"}, p).isoformat() == "2025-01-09"
    assert rulings.dated({"date_enacted": None, "name": ""}, p) is None
    assert rulings.render_estremi(p, rulings.dated({"date_enacted": "20250312"}, p)).endswith("(12.03.2025)")


class _Session:
    def __init__(self, by_article, texts):
        self.by_article, self.text_of = by_article, texts

    def run(self, query, **kw):
        from src.rag import rulings
        rows = self.by_article.get((kw["doc"], kw["num"]), []) if query == rulings.FETCH_QUERY else \
            [{"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

        class R:
            def data(self_inner):
                return rows
        return R()


def test_c_the_record_reads_a_rulings_date_from_its_text_and_logs_the_nd():
    from src.rag import interactive_flow as flow
    from src.rag import rulings
    passage = ("In tema di omicidio preterintenzionale, l'elemento soggettivo è costituito dalla volontà di percuotere "
               "o ledere, senza che sia necessaria la prevedibilità dell'evento morte.")
    rows = [{"ruling": f"R{i}", "file_name": f"snpen@s50@a2025@n{i:05d}@tS.clean.pdf", "name": "",
             "document_number": "1", "date_enacted": when, "linked": True, "section": f"S{i}",
             "abstract": "omicidio preterintenzionale volontà percuotere"}
            for i, when in ((1, "2025-01-09"), (2, None), (3, "garbage"))]
    texts = {"S1": passage, "S2": passage + " Così deciso in Roma, il 4 febbraio 2025.", "S3": passage}
    ctx = {"records": {"584 c.p.": {"side": "threat", "rubrica": "Omicidio preterintenzionale", "requisites": [
        {"id": "R2", "text": "volontà di percuotere, senza prevedibilità della morte", "status": "da provare",
         "kind": "requisito"}]}}, "case": "Caso sintetico.", "assessment": {"article": "584 c.p."},
        "candidates": [{"id": "T1", "kind": "contest_open", "tesi": "Contestare R2", "contesta": ["R2"],
                        "poggia_su": [], "label": "sostenibile"}]}
    session = _Session({("CP", "584"): rows}, texts)
    verdicts = {f"P{i}": {"frase": f"P{i}.1", "elemento": "R2", "pertinenza": 2, "verso": "a favore"}
                for i in range(1, 4)}
    record = rulings.build(ctx, [{"_article": ("584", "c.p."), "d": {"id": "CP"}, "s": {"name": "584"}}], session,
                           call=lambda messages, max_tokens: verdicts)
    by_id = {r["id"]: r for r in record["kept"]}
    assert by_id["R1"]["estremi"].endswith("(09.01.2025)") and by_id["R2"]["estremi"].endswith("(04.02.2025)")
    assert by_id["R3"]["estremi"].endswith("(data n.d.)")
    assert record["log"]["date_nd"] == 1 and record["log"]["dates_from_text"] == 1
    printed = "\n".join(f"- {r['estremi']}." for r in record["kept"])
    assert len(rulings.estremi_printed(printed)) == 3
    state = {}
    flow.text_counted(state, printed)
    assert state["block_counters"]["rulings_printed"] == 3 and state["block_counters"]["rulings_date_nd"] == 1


# --- review fixes (79.12.5) ---------------------------------------------------------------------------------------------

# a: after an office, only a word that continues its name («office_tails», the gazetteer, the role nouns) is kept --
# any other capitalised word is a surname, redacted and counted by the nightly's names_left
@pytest.mark.parametrize("text,surname", [
    ("Secondo la Procura del Rossi il fatto sussiste.", "Rossi"),
    ("Il Consiglio del Ferrari ha deliberato.", "Ferrari"),
    ("L'Appello del Romano va respinto.", "Romano"),
    ("Il ricorso era rivolto contro il Servizio del Colombo.", "Colombo"),
    ("Le direttive erano quelle del Governo del Ricci.", "Ricci"),
    ("Il Consulente del Bianchi ha stimato il danno.", "Bianchi"),
])
def test_review_a_surname_after_an_office_and_del_is_redacted_and_counted(names, text, surname):
    out = names.redact_names(text)
    assert surname not in out and "[omissis]" in out
    assert names.names_left(text) == [surname] and names.names_left(out) == []


@pytest.mark.parametrize("text", [
    "La Corte Europea dei Diritti dell'Uomo ha condannato lo Stato.",
    "Il Ministero della Salute e il Ministero dell'Economia e delle Finanze.",
    "Il Codice dei Contratti Pubblici si applica.",
    "il Chirurgo ha operato; la Struttura ha risposto; il Primario e il Progettista; il Gestore; il Tecnico.",
    "Il Consulente Tecnico ha stimato il danno; il Tar Lazio ha respinto il ricorso.",
])
def test_review_roles_and_office_tails_stay(names, text):
    assert names.redact_names(text) == text and names.names_left(text) == []


def test_review_office_tail_needs_a_tail_word(names):
    text = "la Corte dei Conti e la Procura del Rossi"
    assert names.office_tail(text, text.index("dei")) and not names.office_tail(text, text.index("del R"))
    assert "Conti" in names.office_tails() and "Rossi" not in names.office_tails()


# b (item 8): the title printed without its year, or in another case, is cited too -- and counted if left
@pytest.mark.parametrize("text,want", [
    ("Si applica l'art. 73 del Testo Unico sugli Stupefacenti.", "Si applica l'art. 73 del d.P.R. 309/1990."),
    ("Ai sensi del Testo unico sugli stupefacenti 2026, art. 73.", "Ai sensi del d.P.R. 309/1990, art. 73."),
    ("La Riforma del Lavoro Fornero ha modificato l'art. 18.", "La L. 92/2012 ha modificato l'art. 18."),
    ("art. 73 (Testo Unico sugli Stupefacenti)", "art. 73 d.P.R. 309/1990"),
    ("il d.lgs. 206/2005 (Codice del Consumo) prevede", "il d.lgs. 206/2005 prevede"),
])
def test_review_every_printed_form_of_a_title_is_cited(text, want):
    from src.rag import interactive_flow as flow
    state = {}
    docs = [{"name": "Testo Unico sugli Stupefacenti 2026"}, {"name": "Riforma del Lavoro Fornero 2026"}]
    assert flow.law_titles_cited(state, text, docs) == want
    assert state["block_counters"]["law_titles_cited"] == 1 and state["block_counters"]["law_titles_left"] == 0


def test_review_law_titles_left_reads_the_title_without_its_year_in_any_case():
    from src.rag import corpus_laws, interactive_flow as flow
    state = {}
    docs = [{"name": "Legge Sintetica Sulle Api 2026"}]          # no mapping: never rewritten, always counted
    out = flow.law_titles_cited(state, "Lo prevede la legge sintetica sulle api, art. 2.", docs)
    assert out == "Lo prevede la legge sintetica sulle api, art. 2." and state["block_counters"]["law_titles_left"] == 1
    assert corpus_laws.title_forms("Legge Sintetica Sulle Api 2026") == ["Legge Sintetica Sulle Api 2026",
                                                                         "Legge Sintetica Sulle Api"]
    assert corpus_laws.title_forms("Legge 2026") == ["Legge 2026"]        # a one-word title: the bare form too loose


# c (item 9): a date the ruling only cites -- a filing, an appeal's deposit, a hearing, mid-sentence -- is never its own
@pytest.mark.parametrize("text,want", [
    ("Con ricorso depositato il 3 marzo 2024 il Procuratore impugnava la sentenza. Il ricorso è infondato. "
     "Così deciso il 12/03/2025.", "2025-03-12"),
    ("La sentenza di appello, depositata il 10/11/2024, è impugnata.", None),
    ("Rilevato che l'atto di appello veniva depositato in data 5 maggio 2024.", None),
    ("Il giudice, all'udienza del 12/01/2024, sentiva il teste.", None),
    ("UDIENZA PUBBLICA DEL 12/03/2025\nREPUBBLICA ITALIANA", "2025-03-12"),
    ("Rigetta il ricorso. Depositato in Cancelleria oggi 3 aprile 2025.", "2025-04-03"),
    ("Data di deposito: 3 aprile 2025", "2025-04-03"),
])
def test_review_own_date_reads_only_the_rulings_own_formulas(text, want):
    from src.rag import rulings
    got = rulings.own_date(text, "2025")
    assert (got.isoformat() if got else None) == want
    p = rulings.parse_estremi("snpen@s20@a2025@n00007@tS")
    got = rulings.dated({"date_enacted": None, "name": "x"}, p, text)
    assert (got.isoformat() if got else None) == want
