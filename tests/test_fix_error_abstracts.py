"""fix_error_abstracts.py: the error fragments go wherever they sit (the stored abstracts are post-processed, so the
"Testo significativo:" marker is gone and a title may differ from the node's); the description follows NEXT."""
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


def test_title_mangled_by_the_post_processing_is_still_pure_and_gets_the_node_title():
    # flatten wrote "Motivo 10 - [ERR]"; post-processing's replace("0 ", "") left "Motivo 1- [ERR]"
    new, pure = fx.rebuild_abstract(f"Motivo 1- {ERR}", TEXT, "Motivo 10")
    assert pure and new == f"Motivo 10 - {TEXT}"


def test_mixed_abstract_keeps_the_good_fragments_and_the_core_text_without_marker():
    # as stored after post-processing: no "Testo significativo:", the core text follows the error string
    old = f"Titolo - Il motivo è generico; {ERR} motivo generico e inammissibile"
    new, pure = fx.rebuild_abstract(old, TEXT, "Titolo")
    assert not pure and new == "Titolo - Il motivo è generico motivo generico e inammissibile"


def test_error_in_the_middle_keeps_the_joiner_between_the_good_fragments():
    new, pure = fx.rebuild_abstract(f"T - primo; {ERR}; secondo", TEXT, "T")
    assert not pure and new == "T - primo; secondo"


def test_error_first_then_good_fragment():
    new, pure = fx.rebuild_abstract(f"T - {ERR}; secondo", TEXT, "T")
    assert not pure and new == "T - secondo"


def test_flatten_shape_with_marker_still_works():
    old = f"Titolo - Il motivo è generico; {ERR} Testo significativo: motivo generico"
    new, pure = fx.rebuild_abstract(old, TEXT, "Titolo")
    assert not pure and new == "Titolo - Il motivo è generico Testo significativo: motivo generico"


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


def test_result_never_carries_the_string():
    for old in (ERR, f"T - {ERR}", f"{ERR}; {ERR}", f"a; {ERR} b", f"{ERR} b", f"T - a; {ERR}"):
        new, _ = fx.rebuild_abstract(old, TEXT, "T")
        assert not fx.ERR_RE.search(new)


def test_shape_masks_every_word():
    s = fx.shape(f"Motivi della decisione - Il ricorso di Mario Rossi è inammissibile; {ERR} testo")
    assert s == "W3 - W7; § w" and "Rossi" not in s


def test_description_follows_next_and_leaves_split_pieces_out():
    rows = [
        {"eid": "b", "name": "2", "abstract": "second", "next": "c"},
        {"eid": "a", "name": "1", "abstract": "first", "next": "b"},
        {"eid": "c", "name": "3", "abstract": "third", "next": None},
        {"eid": "x", "name": "2_1", "abstract": "split piece", "next": None},
    ]
    assert fx.rebuild_description(rows) == "first second third"
    assert fx.chain_health(rows) == "partial"
    assert fx.chain_health(rows[:3]) == "single"


def test_description_of_a_single_section_document():
    rows = [{"eid": "a", "name": "1", "abstract": "only", "next": None}]
    assert fx.rebuild_description(rows) == "only" and fx.chain_health(rows) == "no_next"


def test_description_is_trimmed_to_500_words():
    rows = [{"eid": "a", "name": "1", "abstract": " ".join(["w"] * 400), "next": "b"},
            {"eid": "b", "name": "2", "abstract": " ".join(["v"] * 400), "next": None}]
    assert len(fx.rebuild_description(rows).split()) == 500
