"""PROTECTED_PARAPHRASE_ENABLED (src/rag/protected_sources.py): which sources are protected, the writer's tag and
rule, the post-check on the answer, the sources shown, the measurement. Every text here is invented."""
import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from src.rag import protected_sources as ps  # noqa: E402

# An invented doctrine passage and an invented law article.
BOOK = ("Secondo l'impostazione accolta nel volume, la clausola di gradimento opera come filtro preventivo sulla "
        "circolazione delle quote, impedendo che soggetti estranei entrino nella compagine senza il consenso dei soci "
        "fondatori. Il gradimento va espresso entro trenta giorni dalla comunicazione, pena la sua inefficacia.")
LAW = "Il socio può recedere dalla società quando non consente al cambiamento dell'oggetto sociale."
COPIED = ("La clausola di gradimento opera come filtro preventivo sulla circolazione delle quote, impedendo che "
          "soggetti estranei entrino nella compagine [1].")
# One word changed: no run of ten, a 3-shingle Jaccard of 8/14 with the protected sentence.
NEAR = "Il gradimento va manifestato entro trenta giorni dalla comunicazione, pena la sua inefficacia."
PARAPHRASE = ("Il patto che subordina il trasferimento delle partecipazioni all'assenso dei soci serve a tenere "
              "fuori dalla società chi non è gradito [1].")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(ps.FLAG, "true")
    ps._NOTICE_DOCS.clear()
    yield
    ps._NOTICE_DOCS.clear()


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(ps.FLAG, raising=False)
    ps._NOTICE_DOCS.clear()
    yield
    ps._NOTICE_DOCS.clear()


def _row(doc_id, kind, name, text, section="1"):
    return {"d": {"id": doc_id, "document_type": kind, "name": name},
            "s": {"id": f"DOCUMENT_SECTION::{doc_id}::{section}", "name": section, "plain_text": text}}


# --- which sources ------------------------------------------------------------------------------------------

def test_special_and_ccnl_are_protected_primary_and_rulings_are_not(off):
    assert ps.classify("a", "special", "Commentario Immaginario", []) == "special"
    assert ps.classify("b", "ccnl", "CCNL Settore Inventato", []) == "ccnl"
    assert ps.classify("c", "primary", "Codice Civile", [LAW]) is None
    assert ps.classify("d", "interpretation", "Cass. civ. n. 1/2099", [LAW]) is None


def test_the_name_list_matches_exact_names_from_the_variable_and_the_file(off, monkeypatch, tmp_path):
    names = tmp_path / "protected.txt"
    names.write_text("# books\nPrincipî di diritto societario immaginario\n", encoding="utf-8")
    monkeypatch.setenv(ps.NAMES_ENV, "Manuale di Prova; Dispensa del Seminario Fittizio")
    monkeypatch.setenv(ps.NAMES_FILE_ENV, str(names))
    # mislabelled as primary on prod: the name protects it anyway; case, spaces and a [TAG] do not count
    assert ps.classify("a", "primary", "[DOC]  manuale di  PROVA", []) == "by name"
    assert ps.classify("b", None, "Principî di diritto societario immaginario", []) == "by name"
    assert ps.classify("c", "primary", "Dispensa del Seminario Fittizio", []) == "by name"
    # exact: a longer or a different name is not on the list
    assert ps.classify("d", "primary", "Manuale di Prova, seconda parte", []) is None
    assert ps.classify("e", "primary", "Manuale", []) is None


def test_a_reproduction_notice_protects_a_non_primary_document_and_its_other_rows(off):
    assert ps.classify("x", "interpretation", "Raccolta di massime", ["Testo. Riproduzione Riservata"]) == "notice"
    assert ps.classify("x", "interpretation", "Raccolta di massime", ["un'altra pagina, senza nota"]) == "notice"
    # primary law keeps its text: a notice on it is stripped, it does not protect it
    assert ps.classify("y", "primary", "Legge Inventata", ["Art. 1. Riproduzione Riservata"]) is None


@pytest.mark.parametrize("text", [
    "IlCCNL | S.I.A. Srl | Riproduzione Riservata",
    "È vietata la riproduzione, anche parziale, con qualsiasi mezzo effettuata.",
    "© 2099 Editore Fittizio S.p.A.",
    "Tutti i diritti riservati.",
    "Proprietà letteraria riservata",
    "Il presente materiale è di proprietà di Studio Immaginario e non può essere diffuso.",
    "Testo di proprietà di Edizioni Inventate.",
])
def test_notices_are_found_and_stripped(text):
    assert ps.has_notice(text)
    assert not ps.has_notice(ps.strip_notices(f"Prima frase. {text} Ultima frase."))


@pytest.mark.parametrize("text", [
    "Il fondo di proprietà di Tizio è gravato da una servitù di passaggio.",
    "L'immobile è di proprietà di Alfa S.r.l. dal 2090.",
    "La legge punisce la riproduzione abusiva di opere dell'ingegno.",
    "Il diritto di proprietà di Caio si estende al sottosuolo.",
])
def test_ordinary_legal_prose_is_not_a_notice(text):
    assert not ps.has_notice(text)
    assert ps.strip_notices(text) == text


class _Session:
    """A Neo4j session answering the Document lookup."""

    def __init__(self, docs):
        self.docs, self.calls = docs, 0

    def run(self, query, **params):
        self.calls += 1
        rows = [{"id": i, "type": self.docs[i][0], "name": self.docs[i][1]} for i in params["ids"] if i in self.docs]
        return type("R", (), {"data": lambda _self: rows})()


def test_mark_does_nothing_with_the_flag_off(off):
    rows = [_row("b", "special", "Commentario Immaginario", BOOK)]
    assert ps.mark(rows) is rows and ps.KEY not in rows[0]


def test_mark_flags_every_channel_s_rows_looks_up_untyped_documents_and_strips_notices(on):
    untyped = {"s": {"id": "DOCUMENT_SECTION::m::3", "name": "3", "plain_text": "Una massima. Riproduzione Riservata"}}
    sibling = {"s": {"id": "DOCUMENT_SECTION::m::4", "name": "4", "plain_text": "Un'altra massima, senza nota."}}
    law = _row("c", "primary", "Codice Immaginario", LAW + " Riproduzione Riservata")
    rows = [_row("b", "special", "Commentario Immaginario", BOOK), untyped, sibling, law]
    session = _Session({"m": ("interpretation", "Massimario Fittizio")})
    ps.mark(rows, session, "test")
    assert [r.get(ps.KEY) for r in rows] == ["special", "notice", "notice", None]
    assert session.calls == 1
    assert "Riservata" not in untyped["s"]["plain_text"] and "Riservata" not in law["s"]["plain_text"]
    assert law["s"]["plain_text"].startswith("Il socio può recedere")


def test_the_graph_node_wrapper_marks_raw_result_and_dottrina_rows(on):
    rows = [_row("b", "special", "Commentario Immaginario", BOOK)]
    dottrina = [_row("k", "ccnl", "CCNL Inventato", "Art. 5 - Ferie")]
    node = ps.node(lambda state: {"raw_result": rows, "special_dottrina_rows": dottrina}, "x")
    out = node({})
    assert out["raw_result"][0][ps.KEY] == "special" and out["special_dottrina_rows"][0][ps.KEY] == "ccnl"


# --- the writer ---------------------------------------------------------------------------------------------

def test_the_writer_gets_the_tag_and_the_rule_only_with_the_flag_and_a_protected_source(on, monkeypatch):
    citations = [{"document_name": "Commentario Immaginario", "document_type": "special", "document_id": "b",
                  "sections": [{"name": "12", "plain_text": BOOK}]},
                 {"document_name": "Codice Immaginario", "document_type": "primary", "document_id": "c",
                  "sections": [{"name": "2473", "plain_text": LAW}]}]
    lines = ps.citation_lines(citations, 300).splitlines()
    assert ps.TAG in lines[0] and ps.TAG not in lines[1]
    prompt = ps.with_rule("Fonti:\n" + "\n".join(lines))
    assert prompt.endswith(ps.WRITER_RULE) and ps.with_rule(prompt) == prompt
    assert ps.with_rule('Data: [{"s": {}, "_protected": "special"}]').endswith(ps.WRITER_RULE)
    assert ps.with_rule("Fonti: solo legge") == "Fonti: solo legge"
    monkeypatch.setenv(ps.FLAG, "false")
    assert ps.TAG not in ps.citation_lines(citations, 300) and ps.with_rule(prompt.replace(ps.WRITER_RULE, "")).strip()


def test_the_strategy_source_block_tags_a_protected_ruling_without_changing_its_label(on):
    from src.rag import defensive_generation as dg
    rows = [_row("b", "special", "Commentario Immaginario", BOOK), dict(_row("c", "primary", "Codice", LAW),
                                                                        _article=("2473", "c.c."))]
    block = dg._source_block(rows, 500)
    assert f"[Commentario Immaginario] {ps.TAG}\n" in block and "[art. 2473 c.c.]\n" in block


# --- the post-check -----------------------------------------------------------------------------------------

def test_overlaps_find_ten_word_runs_and_near_duplicates_not_paraphrases():
    found = dict((s, why) for s, why in ps.overlaps(f"{COPIED} {NEAR} {PARAPHRASE} Breve frase uguale.", [BOOK]))
    assert found == {COPIED: "verbatim", NEAR: "near-duplicate"}
    # a short figure or a short heading is data, not a reproduction
    assert ps.overlaps("Il termine è di trenta giorni dalla comunicazione.", [BOOK]) == []
    assert ps.overlaps(COPIED, []) == []


def test_with_the_flag_off_the_answer_is_unchanged_and_the_overlaps_counted(off):
    answer = f"{COPIED}\n{PARAPHRASE}"
    out, stats = ps.enforce(answer, rows=[_row("b", "special", "Commentario", BOOK)], stage="t",
                            rewrite=lambda found: pytest.fail("no rewrite with the flag off"))
    assert out == answer
    assert (stats["found"], stats["rewritten"], stats["dropped"], stats["remaining"]) == (1, 0, 0, 1)


def test_a_good_rewrite_replaces_the_sentence_keeping_its_citation_numbers(on):
    rows = [_row("b", "special", "Commentario", BOOK)]
    seen = []

    def rewrite(found):
        seen.extend(found)
        return ["Per l'autore, il patto di gradimento seleziona in anticipo chi può acquistare le quote [1]."]
    out, stats = ps.enforce(f"Premessa.\n{COPIED}\nConclusione.", rows=rows, stage="t", rewrite=rewrite)
    assert seen == [COPIED]
    assert "seleziona in anticipo" in out and COPIED not in out and "[1]" in out
    assert (stats["found"], stats["rewritten"], stats["dropped"], stats["remaining"]) == (1, 1, 0, 0)


def test_a_rewrite_losing_a_number_or_still_too_close_is_refused_and_the_sentence_dropped(on):
    rows = [_row("b", "special", "Commentario", BOOK)]
    answer = f"- {COPIED}\n- {NEAR}\n- {PARAPHRASE}"
    out, stats = ps.enforce(answer, rows=rows, stage="t",
                            rewrite=lambda found: ["Il patto seleziona chi entra.", found[1]])
    assert COPIED not in out and NEAR not in out and PARAPHRASE in out
    assert out.strip() == f"- {PARAPHRASE}"           # no empty bullet left behind
    assert (stats["found"], stats["rewritten"], stats["dropped"], stats["remaining"]) == (2, 0, 2, 0)


def test_a_failed_rewrite_call_drops_the_sentences_and_notices_are_stripped(on):
    def boom(found):
        raise RuntimeError("model down")
    out, stats = ps.enforce(f"{COPIED} {PARAPHRASE} Riproduzione Riservata.", texts=[BOOK], stage="t", rewrite=boom)
    assert out.strip() == PARAPHRASE
    assert stats["dropped"] == 1 and stats["remaining"] == 0


def test_the_post_check_compares_only_with_protected_rows(on):
    rows = [_row("c", "primary", "Codice Immaginario", BOOK)]          # the same words, but primary law
    out, stats = ps.enforce(COPIED, rows=rows, stage="t", rewrite=lambda f: pytest.fail("primary law is quotable"))
    assert out == COPIED and stats["found"] == 0


# --- the sources shown --------------------------------------------------------------------------------------

def _citations():
    return [{"document_name": "Commentario Immaginario", "document_id": "b", "document_type": "special",
             "sections": [{"name": "La clausola di gradimento opera come filtro preventivo sulla circolazione",
                           "title": "Clausole di gradimento", "plain_text": BOOK, "score": 0.9}]},
            {"document_name": "Codice Immaginario", "document_id": "c", "document_type": "primary",
             "sections": [{"name": "2473", "title": "Recesso", "plain_text": LAW + " Riproduzione Riservata"}]}]


def test_protected_citations_show_the_reference_only_and_primary_law_its_text(on):
    cites = _citations()
    out = ps.redact(cites, "t")
    book, law = out
    assert book["protected"] == "special"
    assert book["sections"][0]["plain_text"] == "" and book["sections"][0]["protected"] is True
    assert book["sections"][0]["title"] == "Clausole di gradimento"
    assert len(book["sections"][0]["name"].split()) < ps.VERBATIM_WORDS
    assert law["sections"][0]["plain_text"] == LAW
    assert cites[0]["sections"][0]["plain_text"] == BOOK              # the input is not changed
    assert ps.excerpts_shown(cites) == 1 and ps.excerpts_shown(out) == 0


def test_with_the_flag_off_the_sources_are_unchanged_but_counted(off):
    cites = _citations()
    assert ps.redact(cites, "t") == cites and ps.excerpts_shown(cites) == 1


def test_flat_pending_sections_are_redacted_for_showing(on):
    flat = [{"document_name": "CCNL Inventato", "document_type": "ccnl", "name": "Art. 5", "plain_text": "Ferie..."},
            {"document_name": "Codice", "document_type": "primary", "name": "1", "plain_text": LAW}]
    out = ps.redact_flat(flat)
    assert out[0]["plain_text"] == "" and out[1]["plain_text"] == LAW


def test_the_chat_response_and_the_saved_session_show_references_only(on):
    import src.chatbot.api as api
    response = api.ChatResponse(session_id="s", answer="a", original_query="q", resolved_query="q",
                                citations=_citations())
    assert response.citations[0]["sections"][0]["plain_text"] == ""
    assert response.citations[1]["sections"][0]["plain_text"] == LAW
    view = api._protected_session_view({"messages": [{"role": "assistant", "content": "a", "citations": _citations(),
                                                      "pending_sections": [{"document_type": "special",
                                                                            "plain_text": BOOK, "name": "1"}]}]})
    message = view["messages"][0]
    assert message["citations"][0]["sections"][0]["plain_text"] == ""
    assert message["pending_sections"][0]["plain_text"] == ""


# --- measuring ----------------------------------------------------------------------------------------------

def test_run_record_and_the_report_line(off):
    events = [("protected_paraphrase", {"found": 2, "rewritten": 1, "dropped": 1, "remaining": 0}, None),
              ("other", {}, None)]
    record = ps.run_record(events, COPIED, rows=[_row("b", "special", "Commentario", BOOK)], citations=_citations())
    # the final answer still copies one sentence: remaining is read off it
    assert (record["found"], record["rewritten"], record["dropped"], record["remaining"],
            record["excerpts_shown"]) == (2, 1, 1, 1, 1)
    total = ps.sum_records([record, None, record])
    assert ps.report_line(total) == ("verbatim overlaps with protected sources: found 4 / rewritten 2 / dropped 2 / "
                                     "remaining 2 (target 0); protected excerpts shown in sources: 2 (target 0)")


def test_the_eval_summaries_carry_the_protected_line():
    import eval_ccnl
    import eval_direct
    record = {"mentioned": 1, "expected": 2, "non_primary_hits": 0, "non_primary_after_filter": 0,
              "rulings": {"rulings": 0, "citing_answer_article": 0, "boilerplate_only": 0, "unrelated": 0},
              "id": "d1", "route": "direct", "status": "ok",
              "protected": {"found": 1, "rewritten": 1, "dropped": 0, "remaining": 0, "excerpts_shown": 0}}
    assert any(line.startswith("verbatim overlaps with protected sources: found 1 / rewritten 1")
               for line in eval_direct.summary_lines([record], "x"))
    assert eval_direct.totals([record])["protected_rewritten"] == 1
    assert eval_ccnl._protected_line([record]).endswith("protected excerpts shown in sources: 0 (target 0)")


# --- through the answer paths -------------------------------------------------------------------------------

COMMENTARY = [{"d": {"id": "LEGAL_DOC::cm", "name": "Commentario Immaginario", "document_type": "special"},
               "s": {"id": "DOCUMENT_SECTION::cm::x1", "name": "Clausole di gradimento", "plain_text": BOOK}}]
CODE = [{"d": {"id": "LEGAL_DOC::cc", "name": "Codice Civile 2099", "document_type": "primary"},
         "s": {"id": "DOCUMENT_SECTION::cc::2473", "name": "2473.0.0", "plain_text": LAW}}]


def test_the_chat_answer_paraphrases_the_doctrine_note_and_shows_its_reference_only(on, monkeypatch):
    import io
    import src.rag.nodes.synthesis as S
    seen = []

    def fake_chat(messages, max_tokens=None, stop=None, **_):
        seen.append(messages[-1].content)
        if "dottrina giuridica" in messages[0].content:
            return f"Secondo il Commentario Immaginario, {COPIED[0].lower()}{COPIED[1:]}"
        return "Il socio può recedere quando non consente al cambiamento dell'oggetto (art. 2473 c.c.)."
    monkeypatch.setattr(S, "_call_chat", fake_chat)
    monkeypatch.setattr(S, "rerank_results", lambda query, rows: rows)
    monkeypatch.setattr(S, "log_cypher_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "open", lambda *a, **k: io.StringIO(), raising=False)
    monkeypatch.setattr(ps, "_rewrite", lambda found: ["Per il Commentario Immaginario il patto di gradimento "
                                                       "seleziona in anticipo chi può acquistare le quote [1]."])
    out = S.synthesize_answer({"query": "Come funziona la clausola di gradimento?", "session_language": "it",
                               "tone": 2, "standing": 2, "response_length": 2,
                               "raw_result": [dict(r) for r in CODE], "special_dottrina_rows": COMMENTARY})
    note_prompt = next(p for p in seen if "Estratti dottrinali" in p)
    assert ps.TAG in note_prompt and ps.WRITER_RULE in note_prompt
    assert "filtro preventivo sulla circolazione delle quote" not in out["answer"]
    assert "seleziona in anticipo" in out["answer"]
    book = next(c for c in out["citations"] if c["document_name"] == "Commentario Immaginario")
    code = next(c for c in out["citations"] if c["document_name"] == "Codice Civile 2099")
    assert all(s["plain_text"] == "" for s in book["sections"]) and code["sections"][0]["plain_text"] == LAW


from test_ccnl_path import _Driver, path  # noqa: E402,F401  (the CCNL path's fakes)


def test_the_ccnl_path_rewrites_a_copied_clause_with_the_shared_post_check(on, path, monkeypatch):
    from datetime import date
    from src.rag import ai_chat
    from src.rag import ccnl_path as cp
    session, _ = path
    prompts = []

    def fake_chat(messages, max_tokens=None, **_):
        prompts.append(messages[-1].content)
        return ("**Cosa prevede il CCNL**\nIl lavoratore ha diritto a un periodo di ferie di quattro settimane "
                "(Art. 10 - Ferie).\n**Quadro normativo**\nL'art. 2109 c.c. fissa il minimo.")
    monkeypatch.setattr(ai_chat, "_call_chat", fake_chat)
    monkeypatch.setattr(ps, "_rewrite", lambda found: ["Le ferie annuali durano quattro settimane (Art. 10 - Ferie)."])
    out = cp.answer("Secondo il CCNL Metalmeccanica Industria quante ferie spettano?", _Driver(session), "neo4j",
                    today=date(2026, 10, 28))
    assert ps.WRITER_RULE in prompts[0]
    assert "ha diritto a un periodo di ferie" not in out["answer"] and "durano quattro settimane" in out["answer"]
    protected = out["ccnl"]["answer"]["protected"]
    assert (protected["found"], protected["rewritten"], protected["dropped"], protected["remaining"]) == (1, 1, 0, 0)
    assert ps.excerpts_shown(out["citations"]) == 0


def test_with_the_flag_off_the_ccnl_path_keeps_its_own_check_and_counts(off, path):
    from datetime import date
    from src.rag import ccnl_path as cp
    session, _ = path
    out = cp.answer("Secondo il CCNL Metalmeccanica Industria quante ferie spettano?", _Driver(session), "neo4j",
                    today=date(2026, 10, 28))
    assert out["ccnl"]["answer"]["protected"]["enabled"] is False
    assert out["ccnl"]["answer"]["protected"]["rewritten"] == 0
