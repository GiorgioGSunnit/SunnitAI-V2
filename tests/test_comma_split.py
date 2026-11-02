"""Leonardo v41: a line of the PDF is no comma. Texts in the corpus' shape, wrapped as the PDFs wrap them."""
import importlib.util
import os
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import anchored_writing as aw, comma_material as cm, fact_signals as fs, text_tidy  # noqa: E402

A1067 = ("Il proprietario del fondo dominante non può fare innovazioni che rendano\npiù gravosa la condizione del fondo servente.\n"
         "Il proprietario del fondo servente non può compiere alcuna cosa che tenda a\ndiminuire l'esercizio della servitù o a renderlo più incomodo.")
A1051 = ("Il proprietario, il cui fondo è circondato da fondi altrui, e che non ha uscita\nsulla via pubblica né può procurarsela senza eccessivo dispendio o disagio, ha\n"
         "diritto di ottenere il passaggio sul fondo vicino per la coltivazione e il\nconveniente uso del proprio fondo.\n"
         "Il passaggio si deve stabilire in quella parte per cui l'accesso alla via\npubblica è più breve e riesce di minore danno al fondo sul quale è consentito.\n"
         "Esso può essere stabilito anche mediante sottopassaggio, qualora ciò sia\nconveniente.\n"
         "Le stesse disposizioni si applicano nel caso in cui taluno, avendo un passaggio\nsul fondo altrui, abbia bisogno di ampliarlo.")
A1062 = ("La destinazione del padre di famiglia ha luogo quando consta, mediante\nqualunque genere di prova, che due fondi, attualmente divisi, sono stati posseduti\n"
         "dallo stesso proprietario, e che questi ha posto o lasciato le cose nello stato\ndal quale risulta la servitù.\n"
         "Se i due fondi cessarono di appartenere allo stesso proprietario, senza alcuna\ndisposizione relativa alla servitù, questa s'intende stabilita.")
A1079 = ("Il titolare della servitù può farne riconoscere in giudizio l'esistenza contro\nchi ne contesta l'esercizio e può far cessare gli eventuali impedimenti e\n"
         "turbative. Può anche chiedere la rimessione delle cose in pristino, oltre il\nrisarcimento dei danni.")


def test_1067_has_two_commi():
    commi, how = cm.split_commi(A1067)
    assert how == "lines" and len(commi) == 2
    assert commi[0].endswith("condizione del fondo servente.") and commi[1].startswith("Il proprietario del fondo servente")
    assert cm.comma_numbers(commi, how) == [1, 2]


def test_1051_has_its_four_commi():
    commi, how = cm.split_commi(A1051)
    assert len(commi) == 4 and cm.comma_numbers(commi, how) == [1, 2, 3, 4] and commi[2].startswith("Esso può essere")


def test_1062_has_two():
    assert len(cm.split_commi(A1062)[0]) == 2


def test_1079_is_one_comma_of_two_sentences_with_no_number():
    commi, how = cm.split_commi(A1079)
    assert how == "sentences" and cm.comma_numbers(commi, how) == [None, None]


def test_a_wrap_after_an_abbreviation_is_no_comma():
    assert cm.unwrap("Si applica l'art.\n1051 c.c. e il c.c.\nsi applica ancora.") == ["Si applica l'art. 1051 c.c. e il c.c. si applica ancora."]


def test_a_note_marker_a_number_or_a_letter_opens_a_paragraph():
    assert len(cm.unwrap("Il primo comma senza il suo punto (1)\nIl secondo comma.")) == 2
    assert len(cm.unwrap("1. Il primo comma, che va\n2. Il secondo comma.")) == 2
    assert len(cm.unwrap("Sono previsti i seguenti casi\na) il primo;\nb) il secondo.")) == 3


def test_a_word_broken_across_lines_is_mended():
    assert cm.unwrap("La condi-\nzione del fondo.") == ["La condizione del fondo."]


def test_a_repealed_comma_ends_its_paragraph():
    assert len(cm.unwrap("(abrogato)\nIl secondo comma.")) == 2


def test_the_numbers_checks_read_the_same_split():
    rows = [{"_article": ("1067", "c.c."), "s": {"plain_text": A1067}}]
    out, log = text_tidy.against_articles("Leonardo ha torto (art. 1067 c.c., co. 4).", rows)
    assert "co. 4" not in out and log[0]["kind"] == "comma past the article's last"
    assert "co. 2" in text_tidy.against_articles("Leonardo ha torto (art. 1067 c.c., co. 2).", rows)[0]


def row(number, text):
    return {"_article": (number, "c.c."), "_grade": 3, "s": {"plain_text": text}}


def test_a_short_article_s_anchor_holds_every_comma():
    cat = aw.Catalogue()
    pid = aw._article_passage(cat, row("1067", A1067), None, "innovazioni")
    passage = cat.passages[pid]
    assert passage["label"] == "art. 1067 c.c."
    assert "(co. 1) Il proprietario del fondo dominante" in passage["text"] and "(co. 2) Il proprietario del fondo servente" in passage["text"]
    assert "più gravosa la condizione del fondo servente." in passage["text"] and "più incomodo." in passage["text"]


def test_a_long_article_s_anchor_is_one_whole_comma():
    first = "Il primo comma riguarda il titolare e il suo diritto " * 30
    second = "Il secondo comma riguarda il vicino e la sua pretesa " * 30
    cat = aw.Catalogue()
    pid = aw._article_passage(cat, row("9", f"{first.strip()}.\n{second.strip()}."), None, "vicino pretesa")
    passage = cat.passages[pid]
    assert passage["label"] == "art. 9 c.c., co. 2" and len(passage["text"]) > aw.PASSAGE_CHARS and not passage["text"].endswith("[…]")


def test_the_writer_gets_all_of_a_short_article(monkeypatch):
    monkeypatch.setattr(cm, "enabled", lambda: True)
    commi, how = cm.split_commi(A1051)
    out = cm.passage({cm.COMMI_KEY: commi, cm.FROM_KEY: how, cm.FOCUS_KEY: "passaggio"}, 300)
    assert "commi non riportati" not in out and "(co. 4)" in out and "ampliarlo" in out
    assert cm.trim_row(A1067, 100, {}) == A1067


def test_a_row_the_prompt_shrank_stays_shrunk(monkeypatch):
    monkeypatch.setattr(cm, "enabled", lambda: True)
    commi, how = cm.split_commi(A1051)
    assert "commi non riportati" in cm.passage({cm.COMMI_KEY: commi, cm.FROM_KEY: how, "_comma_chars": 300}, 300)


# --- the non-use term ------------------------------------------------------------------------------------------------

def facts(**kw):
    base = {k: ([] if isinstance(v, list) else "") for k, v in fs.SCHEMA.items()}
    base.update(kw)
    return base


GATE = {"what": "Domitilla chiude il cancello", "date": "2016", "who": "Domitilla", "kind": "chiusura/ostruzione accesso"}
TEXT = "estinzione per non uso: 2036 (20 anni dall'ultimo uso, 2016)"


def test_the_term_from_the_closure_with_no_possession_at_all():
    f = facts(events=[GATE], works_or_signs=[{"what": "strada carrabile", "visible": "sì"}])
    term = next(t for t in fs.compute_terms(f) if t["id"] == "non_uso_servitu")
    assert term["text"].startswith(TEXT) and term["expires"] == "2036"


def test_the_term_from_a_possession_of_a_way_whatever_its_use_is_called():
    use = [{"who": "Leonardo", "asset": "strada", "use": "transito con mezzi agricoli", "until": "2016"}]
    assert next(t for t in fs.compute_terms(facts(possession=use)) if t["id"] == "non_uso_servitu")["text"].startswith(TEXT)


def test_the_latest_use_counts():
    use = [{"asset": "strada", "use": "passaggio", "until": "2010"}]
    assert "2036" in next(t for t in fs.compute_terms(facts(possession=use, events=[GATE])) if t["id"] == "non_uso_servitu")["text"]


def test_a_fired_servitude_rule_brings_it_whatever_the_facts_call_the_way():
    fired = [{"id": "x", "acquired_right": True, "institutions": ["servitù per destinazione"]}]
    f = facts(events=[{"what": "il fondo viene sbarrato", "date": "2016", "kind": "chiusura/ostruzione accesso"}])
    assert fs._non_use_for_a_servitude(f, [], fired)[0]["text"].startswith(TEXT)


def test_a_term_already_past_says_so(monkeypatch):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-11-01")
    use = [{"asset": "strada", "use": "passaggio", "until": "1990", "since": "1950"}]
    text = next(t for t in fs.compute_terms(facts(possession=use)) if t["id"] == "non_uso_servitu")["text"]
    assert "2010" in text and "già decorso" in text


def test_no_way_no_term():
    f = facts(events=[{"what": "incidente sulla strada provinciale", "date": "2020", "kind": ""}])
    assert not any(t["id"] == "non_uso_servitu" for t in fs.compute_terms(f))


# --- the check script ------------------------------------------------------------------------------------------------

def _script():
    spec = importlib.util.spec_from_file_location("comma_check", Path(__file__).parent.parent / "scripts" / "comma_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_reports_before_after_and_what_is_still_a_line():
    script = _script()
    assert len(script.lines_of(A1067)) == 4 and len(cm.split_commi(A1067)[0]) == 2
    assert script.suspects(["Il proprietario del fondo dominante che rendano"]) == ["Il proprietario del fondo dominante che rendano"]
    assert script.suspects(["Breve comma.", "(abrogato)"]) == []
    assert script.line("1067", "c.c.", 9, [1, 2, 3, 4], ["a.", "b."]).startswith("| art. 1067 c.c. | 9 | 4 | 2 |")
