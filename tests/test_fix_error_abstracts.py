"""fix_error_abstracts.py: the rebuilt abstract follows flatten.py's rule; the description follows NEXT."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fix_error_abstracts", HERE.parent / "scripts" / "fix_error_abstracts.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

ERR = fx.ERROR
TEXT = "La Corte rigetta il ricorso perché il motivo è generico e non si confronta con la motivazione impugnata."


def test_pure_error_becomes_the_first_200_words_with_the_title_in_front():
    new, pure = fx.rebuild_abstract(f"Motivi della decisione - {ERR}", TEXT, "Motivi della decisione")
    assert pure and new == f"Motivi della decisione - {TEXT}"


def test_several_failed_chunks_are_still_pure():
    new, pure = fx.rebuild_abstract(f"Titolo - {ERR}; {ERR}; {ERR}", TEXT, "Titolo")
    assert pure and new == f"Titolo - {TEXT}"


def test_mixed_abstract_keeps_the_good_fragments_and_the_core_text():
    old = f"Titolo - Il motivo è generico; {ERR}; La pena è congrua Testo significativo: motivo generico"
    new, pure = fx.rebuild_abstract(old, TEXT, "Titolo")
    assert not pure
    assert new == "Titolo - Il motivo è generico; La pena è congrua Testo significativo: motivo generico"


def test_core_text_alone_is_kept_over_the_text():
    old = f"Titolo - {ERR} Testo significativo: la frase centrale"
    new, pure = fx.rebuild_abstract(old, TEXT, "Titolo")
    assert not pure and new == f"Titolo - {TEXT} Testo significativo: la frase centrale"


def test_no_title_and_empty_text():
    assert fx.rebuild_abstract(ERR, TEXT, "") == (TEXT, True)
    assert fx.rebuild_abstract(ERR, "   ", "") == ("(sezione senza testo)", True)


def test_title_already_in_the_text_is_not_added_twice():
    new, _ = fx.rebuild_abstract(f"Fatto - {ERR}", "Fatto e diritto: il ricorso è inammissibile.", "Fatto")
    assert new == "Fatto e diritto: il ricorso è inammissibile."


def test_long_text_is_trimmed_to_200_words():
    text = " ".join(f"w{i}" for i in range(300))
    new, _ = fx.rebuild_abstract(f"T - {ERR}", text, "T")
    assert len(new.split()) == 200 and new.startswith("T - w0 ")


def test_content_filter_fragment_counts_as_error():
    new, pure = fx.rebuild_abstract("[Content filter: comma non analizzabile]", TEXT, "")
    assert pure and new == TEXT


def test_description_follows_next_and_leaves_split_pieces_out():
    rows = [
        {"eid": "b", "name": "2", "abstract": "second", "next": "c"},
        {"eid": "a", "name": "1", "abstract": "first", "next": "b"},
        {"eid": "c", "name": "3", "abstract": "third", "next": None},
        {"eid": "x", "name": "2_1", "abstract": "split piece", "next": None},
    ]
    assert fx.rebuild_description(rows) == "first second third"


def test_description_of_a_single_section_document():
    assert fx.rebuild_description([{"eid": "a", "name": "1", "abstract": "only", "next": None}]) == "only"


def test_description_is_trimmed_to_500_words():
    rows = [{"eid": "a", "name": "1", "abstract": " ".join(["w"] * 400), "next": "b"},
            {"eid": "b", "name": "2", "abstract": " ".join(["v"] * 400), "next": None}]
    assert len(fx.rebuild_description(rows).split()) == 500
