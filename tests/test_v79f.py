"""v79 block F (79.5) -- rulings as printed, on synthetic passages only:

F1. every registry label the writer emits («[G1]», «(G1)», «G1») is the ruling's reference in the final text (Cass.,
    section, number, year, the date when held); a label no registry ruling holds drops its sentence
F2. a quoted passage: no PDF stamp («Corte di Cassazione - copia non ufficiale», page counts), never starting inside a
    reference, the parties' names redacted (RULINGS_PARTY_NAMES=keep leaves them -- the decision is pending)
F3. the nightly's check: no «G<digit>» label, no stamp, no citation without its head
"""

import importlib.util
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


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORD = {"kept": [{"id": "D1", "estremi": "Cass. pen., Sez. 5, n. 1234/2025 (12.03.2025)", "article": "646 c.p.",
                    "passage": "L'appropriazione si configura con l'interversione del possesso.", "element": "x"},
                   {"id": "D2", "estremi": "Cass. pen., Sez. Un., n. 77/2024 (data n.d.)", "article": "646 c.p.",
                    "passage": "", "element": ""}]}


# --- F1 ------------------------------------------------------------------------------------------------------------------

def test_every_label_form_becomes_the_rulings_reference():
    from src.rag import rulings
    text = ("Il possesso basta [G1]. La tesi è sostenuta (G1). Lo afferma anche G2, e il punto è chiuso. "
            "Altro punto.")
    out = rulings.labels_out(text, RECORD)
    assert "Il possesso basta Cass. pen., Sez. 5, n. 1234/2025, 12.03.2025." in out
    assert "(Cass. pen., Sez. 5, n. 1234/2025, 12.03.2025)" in out
    assert "Lo afferma anche Cass. pen., Sez. Un., n. 77/2024, e il punto" in out          # no date held: none
    assert "G1" not in out and "G2" not in out and "Altro punto." in out


def test_a_label_no_registry_ruling_holds_drops_its_sentence():
    from src.rag import rulings
    log = []
    out = rulings.labels_out("Il possesso basta [G7]. Altro punto.", RECORD, log)
    assert out == "Altro punto." and log and log[0].startswith("G7")
    assert rulings.labels_out("Il possesso basta (G1).", None) == ""


def test_compose_prints_the_reference_and_the_check_keeps_it():
    from src.rag import lawyer_method as M, rulings
    ctx = {"records": {}, "rulings_record": RECORD, "rulings": rulings.citable(RECORD), "assessment": {},
           "posture": {"family": "penale", "role": "imputato"}, "case": "Alfa trattiene il denaro."}
    plan = {"soluzioni": [{"tesi": "Linea", "realismo": "medio", "contesta": [], "poggia_su": ["R1"],
                           "pro": "", "contro": "", "argomento": ""}],
            "obiezioni": [], "orientamenti": []}
    out = M.compose([("Norme applicate ai fatti", "L'appropriazione si configura con l'interversione del possesso "
                                                  "(G1).")], ctx, plan)
    assert "G1" not in out and "n. 1234/2025" in out
    assert rulings.printed_problems(out) == []


# --- F2 ------------------------------------------------------------------------------------------------------------------

def test_the_pdf_stamp_and_page_counts_leave_the_passage():
    from src.rag import rulings
    text = ("Il reato si configura con l'interversione del possesso. Corte di Cassazione - copia non ufficiale 4 "
            "La condotta richiede un atto di dominio. Pag. 3 di 7 Il fatto è punito.")
    out = rulings.strip_stamp(text)
    assert "copia non ufficiale" not in out and "Pag. 3" not in out
    assert "possesso. La condotta richiede un atto di dominio. Il fatto è punito." in out
    assert all("copia" not in s for s in rulings.sentences(text))


@pytest.mark.parametrize("sentence, clean", [
    ("La condotta richiede un atto di dominio sulla cosa.", True),
    ("«La condotta richiede un atto di dominio sulla cosa.»", True),
    ("n. 1234/2025, ha affermato che la condotta richiede un atto di dominio.", False),
    ("Rv. 28111, secondo cui la condotta richiede un atto di dominio.", False),
    ("Sez. 5, la condotta richiede un atto di dominio.", False),
    (", e la condotta richiede un atto di dominio.", False),
    ("che la condotta richiede un atto di dominio sulla cosa.", False),
])
def test_a_passage_never_begins_inside_a_reference(sentence, clean):
    from src.rag import rulings
    assert rulings.starts_clean(sentence) is clean


def test_a_mid_reference_sentence_is_never_a_candidate():
    from src.rag import rulings
    text = ("n. 1234/2025, la Corte ha affermato che l'appropriazione richiede l'interversione del possesso della cosa. "
            "L'appropriazione richiede l'interversione del possesso della cosa altrui.")
    got = rulings.candidate_sentences(text, [{"id": "R1", "text": "l'interversione del possesso della cosa"}],
                                      "646 c.p.")
    assert got == ["L'appropriazione richiede l'interversione del possesso della cosa altrui."]


def test_party_names_are_redacted_and_offices_are_not():
    from src.rag import rulings
    text = ("La Corte di Appello di Milano ha ritenuto che Mario Rossi, amministratore, e il sig. Bianchi abbiano "
            "agito con dolo; la Corte Costituzionale e le Sezioni Unite lo escludono, come ROSSI Mario sostiene.")
    out = rulings.redact_names(text)
    assert "Mario Rossi" not in out and "Bianchi" not in out and "ROSSI Mario" not in out
    assert out.count(rulings.REDACTED) == 3
    assert "Corte di Appello di Milano" in out and "Corte Costituzionale" in out and "Sezioni Unite" in out


def test_party_names_stay_when_told_so(monkeypatch):
    from src.rag import rulings
    monkeypatch.setenv("RULINGS_PARTY_NAMES", "keep")
    assert rulings.redact_names("Ha agito Mario Rossi.") == "Ha agito Mario Rossi."


def test_the_printed_line_and_the_registry_carry_the_clean_passage():
    from src.rag import rulings
    r = {"estremi": "Cass. pen., Sez. 2, n. 5/2025 (data n.d.)", "article": "646 c.p.", "id": "D9",
         "passage": "Il reato si configura quando Mario Rossi dispone della cosa. Corte di Cassazione - copia non "
                    "ufficiale 2"}
    line = rulings.ruling_line(r)
    assert "Mario Rossi" not in line and "copia non ufficiale" not in line and rulings.REDACTED in line
    assert "Mario Rossi" not in rulings.registry_block({"kept": [r]})
    assert "Mario Rossi" not in rulings.citable({"kept": [r]})[0]["massima"]


# --- F3 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, problem", [
    ("Come affermato in [G1], il possesso basta.", "etichetta"),
    ("Come affermato da G3, il possesso basta.", "etichetta"),
    ("- Cass. pen., Sez. 5, n. 1/2025, passaggio della motivazione: «Il reato si configura. Corte di Cassazione - "
     "copia non ufficiale 3».", "timbro"),
    ("La tesi è condivisa (Sez. 5, n. 1234/2025).", "senza la testa"),
    ("Si veda n. 1234/2025 sul punto.", "senza la testa"),
])
def test_the_output_check_finds_each_problem(text, problem):
    from src.rag import rulings
    assert any(problem in x for x in rulings.printed_problems(text))


def test_the_output_check_passes_a_clean_text():
    from src.rag import rulings
    text = ("La tesi è condivisa (Cass. pen., Sez. 5, n. 1234/2025, 12.03.2025). Si applica il d.lgs. n. 81/2008 "
            "e la legge n. 689/1981. - Cass. civ., Sez. Un., n. 9/2024 (data n.d.).")
    assert rulings.printed_problems(text) == []


def test_the_nightly_runs_the_printed_check():
    nightly = _script("nightly_sim")
    assert nightly.printed_check("Testo pulito.")["ok"] is True
    bad = nightly.printed_check("Come in [G1].")
    assert bad["ok"] is False and bad["check"] == "rulings_printed"
