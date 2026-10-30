"""COMMA_MATERIAL_ENABLED (src/rag/comma_material.py): an article's commi, chosen for its issue, instead of a prefix
cut. Every article here is invented."""
import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest  # noqa: E402

from src.rag import comma_material as cm  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402

# Art. 9001 c.c. (invented): four commi; the third, on the late notice, is the one the issue needs.
C1 = "Il custode immaginario risponde dei danni cagionati dalle cose che ha in custodia, salvo il caso fortuito."
C2 = "La responsabilità si estende ai beni affidati in deposito per ragioni di servizio pubblico locale."
C3 = ("La denuncia del danno deve pervenire entro sessanta giorni dalla scoperta; la denuncia tardiva estingue "
      "il diritto al risarcimento salvo che il ritardo sia dovuto a forza maggiore.")
C4 = "Restano salve le disposizioni delle leggi speciali sul trasporto di merci pericolose."
SECTIONS = {"9001.0.0": "In vigore dal 1 gennaio 2099", "9001.1.0": C1, "9001.2.0": C2, "9001.3.0": C3,
            "9001.4.0": C4, "9001": " ".join([C1, C2, C3, C4])}


def _row(**extra):
    return {"d": {"id": "LEGAL_DOC::cc", "name": "Codice Civile", "document_type": "primary"},
            "s": {"name": "9001", "plain_text": " ".join([C1, C2, C3, C4])}, "_article": ("9001", "c.c."), **extra}


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(cm.FLAG, "true")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(cm.FLAG, raising=False)


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    """Answers article_text's sections query for invented articles."""

    def __init__(self, articles):
        self.articles, self.queries = articles, 0

    def run(self, query, **params):
        assert "UNWIND $items AS it" in query
        self.queries += 1
        out = []
        for it in params["items"]:
            for name, text in self.articles.get((it["doc"], it["num"]), {}).items():
                out.append({"doc": it["doc"], "num": it["num"], "d": {"id": it["doc"]},
                            "s": {"name": name, "plain_text": text}})
        return _Result(out)


# --- the commi ----------------------------------------------------------------------------------------------

def test_commi_come_from_the_comma_sections_in_order_without_the_date_line():
    commi, how = cm.commi_from_sections("9001", SECTIONS)
    assert (commi, how) == ([C1, C2, C3, C4], "sections")


def test_without_comma_sections_the_text_is_split_by_lines_numbers_or_sentences():
    assert cm.commi_from_sections("9002", {"9002.0.0": f"{C1}\n{C3}"}) == ([C1, C3], "lines")
    numbered = f"1. {C1} 2. {C2} 3. {C3}"
    assert cm.split_commi(numbered) == ([f"1. {C1}", f"2. {C2}", f"3. {C3}"], "numbered")
    parts, how = cm.split_commi(f"{C1} {C4}")
    assert how == "sentences" and parts == [C1, C4]
    assert cm.split_commi("") == ([], "none")
    # a short fragment joins the part before it
    assert cm.split_commi(f"{C1}\na) breve\n{C4}")[0] == [f"{C1} a) breve", C4]


def test_named_commi_in_every_form():
    text = ("Si veda l'art. 9001, comma 3, c.c.; l'art. 9002 c.p.c., comma 2; il co. 1 dell'art. 9003 c.p. e il "
            "secondo comma dell'articolo 9004 c.c.; non l'art. 9005 c.c. da solo.")
    assert cm.named_commi(text) == {("9001", "c.c."): {3}, ("9002", "c.p.c."): {2}, ("9003", "c.p."): {1},
                                    ("9004", "c.c."): {2}}


def test_select_keeps_the_commi_of_the_issue_named_first_within_the_budget():
    commi = [C1, C2, C3, C4]
    focus = "denuncia tardiva del danno oltre sessanta giorni: estinzione del diritto al risarcimento"
    picked = cm.select(commi, len(C3) + 20, focus)
    assert [k for k, _ in picked] == [3]                         # not the prefix: comma 3
    picked = cm.select(commi, len(C3) + len(C1) + 40, focus)
    assert [k for k, _ in picked] == [1, 3]                      # article order; 1 shares "danni"/"risarcimento"
    named = cm.select(commi, len(C4) + 20, focus, named=[4])
    assert [k for k, _ in named] == [4]                          # a named comma always in
    cut = cm.select(commi, 60, "", named=[3])
    assert cut[0][0] == 3 and cut[0][1].endswith("[…]")          # cut, but there
    assert [k for k, _ in cm.select(commi, 10_000, "")] == [1, 2, 3, 4]


def test_passage_labels_the_commi_and_lists_the_left_out(on):
    row = _row(_commi=[C1, C2, C3, C4], _commi_from="sections", _focus="denuncia tardiva sessanta giorni")
    text = cm.passage(row, len(C3) + 30)
    assert text == f"(co. 3) {C3} [commi non riportati: 1, 2, 4]"
    parts = _row(_commi=[C1, C4], _commi_from="sentences", _focus="trasporto merci pericolose leggi speciali")
    assert cm.passage(parts, len(C4) + 20) == f"[…] {C4}"
    assert cm.passage(_row(), 500) is None


# --- through the strategy's material ------------------------------------------------------------------------

def test_flag_off_the_block_is_the_prefix_as_before(off):
    row = _row(_commi=[C1, C2, C3, C4], _commi_from="sections", _focus="denuncia tardiva")
    assert dg._source_block([row], 150) == f"[art. 9001 c.c.]\n{' '.join([C1, C2, C3, C4])[:150]}"
    assert dg._with_text_cut(row, 150)["_passage"] == " ".join([C1, C2, C3, C4])[:150]


def test_flag_on_the_block_and_the_trim_give_the_commi_of_the_issue(on):
    row = _row(_commi=[C1, C2, C3, C4], _commi_from="sections", _focus="denuncia tardiva sessanta giorni")
    block = dg._source_block([row, _row()], 1500)                # a second row of the same article adds nothing
    assert block.startswith("[art. 9001 c.c.]\n(co. 1) ") and f"(co. 3) {C3}" in block and block.count(C1) == 1
    cut = dg._with_text_cut(row, len(C3) + 30)
    assert "_passage" not in cut and dg._source_block([cut], 1500) == (
        f"[art. 9001 c.c.]\n(co. 3) {C3} [commi non riportati: 1, 2, 4]")
    # a row without commi keeps its prefix cut
    assert dg._source_block([_row()], 100) == f"[art. 9001 c.c.]\n{' '.join([C1, C2, C3, C4])[:100]}"


def test_attach_reads_every_article_in_one_query_and_does_nothing_off(on, monkeypatch):
    session = _Session({("LEGAL_DOC::cc", "9001"): SECTIONS})
    rows = [_row(), _row(), {"d": {"id": "LEGAL_DOC::x"}, "s": {}, "_article": ("1", "c.p.")}, {"d": {}, "s": {}}]
    assert cm.attach(session, rows) == 2 and session.queries == 1
    assert rows[0][cm.COMMI_KEY] == [C1, C2, C3, C4] and rows[0][cm.FROM_KEY] == "sections"
    assert cm.COMMI_KEY not in rows[2]
    monkeypatch.setenv(cm.FLAG, "false")
    fresh = [_row()]
    assert cm.attach(session, fresh) == 0 and cm.COMMI_KEY not in fresh[0]


def test_for_plan_focuses_each_article_on_its_issue_and_reads_a_named_comma_it_lacks(on):
    plan = {"questioni": [{"titolo": "Decadenza dal risarcimento", "norme": [("9001", "c.c.")],
                           "argomento": "la denuncia tardiva oltre sessanta giorni estingue il diritto",
                           "obiezione": "", "replica": ""},
                          {"titolo": "Altro", "norme": [("9009", "c.c.")], "argomento": "x", "obiezione": "",
                           "replica": "si applica l'art. 9009, comma 2, c.c."}],
            "risposte": [], "subordinate": [], "punti_deboli": []}
    first = _row(_commi=[C1, C2, C3, C4], _commi_from="sections")
    other = {"d": {"id": "LEGAL_DOC::cc"}, "s": {"name": "9009"}, "_article": ("9009", "c.c."),
             "_commi": [C1], "_commi_from": "sections"}                 # holds 1 comma, the plan names the 2nd
    session = _Session({("LEGAL_DOC::cc", "9009"): {"9009.1.0": C1, "9009.2.0": C2}})
    cm.set_focus([first, other], cm.plan_focus(plan), cm.named_commi(cm.plan_text(plan)))
    assert "sessanta giorni" in first[cm.FOCUS_KEY] and other[cm.NAMED_KEY] == [2]
    assert cm.ensure([first, other], cm.named_commi(cm.plan_text(plan)), session) == ["9009 c.c. co. 2"]
    assert other[cm.COMMI_KEY] == [C1, C2] and session.queries == 1
    assert f"(co. 2) {C2}" in dg._source_block([other], 400)                      # the named comma in
    assert dg._source_block([other], len(C2) + 20) == f"[art. 9009 c.c.]\n(co. 2) {C2} [commi non riportati: 1]"


def test_the_log_says_which_commi_each_article_was_given(on, monkeypatch):
    events = []
    monkeypatch.setattr(cm, "vlog", lambda event, data=None, **_: events.append((event, data)))
    row = _row(_commi=[C1, C2, C3, C4], _commi_from="sections", _focus="denuncia tardiva", _named_commi=[3])
    material = "TESTI DELLE NORME:\n" + dg._source_block([dg._with_text_cut(row, len(C3) + 30)], 1500)
    cm.log_material("section Applicazione", material, [row])
    assert events == [("comma_material", {"stage": "section Applicazione", "articles": [
        {"ref": "9001 c.c.", "total": 4, "from": "sections", "given": [3], "named": [3], "chars": len(
            f"(co. 3) {C3} [commi non riportati: 1, 2, 4]")}]})]


def test_fetch_audit_finds_the_supporting_comma_in_the_commi_material(on):
    """The measure the flag is for: the sentence relying on comma 3, beyond the old prefix cut, is now in."""
    import fetch_audit as fa
    sentence = ("La denuncia è tardiva perché pervenuta oltre sessanta giorni dalla scoperta, sicché il diritto al "
                "risarcimento si è estinto (art. 9001 c.c.).")
    cite = {"number": "9001", "code": "c.c.", "comma": None}
    prefix = fa.material_blocks("[art. 9001 c.c.]\n" + " ".join([C1, C2, C3, C4])[:250])[("9001", "c.c.")]
    assert fa.granularity_claim(sentence, cite, prefix, [C1, C2, C3, C4])["status"] == "beyond_truncation"
    row = _row(_commi=[C1, C2, C3, C4], _commi_from="sections", _focus="denuncia tardiva sessanta giorni")
    commi_block = fa.material_blocks(dg._source_block([dg._with_text_cut(row, 250)], 1500))[("9001", "c.c.")]
    assert fa.granularity_claim(sentence, cite, commi_block, [C1, C2, C3, C4])["status"] == "in_material"
