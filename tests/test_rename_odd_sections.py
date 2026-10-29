"""scripts/rename_odd_sections.py: "tutti_cittadini.3.0.0" -> "3.0.0", so lookups by number find the article (29 Oct)."""
import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rename_odd_sections", HERE.parent / "scripts" / "rename_odd_sections.py")
ro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ro)


def test_new_name():
    assert ro.new_name("tutti_cittadini.3.0.0") == ("3.0.0", None)
    assert ro.new_name("lordinamento_giuridico_italiano.10") == ("10", None)
    assert ro.new_name("tutte_confessioni.8_3") == ("8_3", None)
    assert ro.new_name("classificazione_delle.2.3.0_2") == ("2.3.0_2", None)
    assert ro.new_name("informazioni_fornire.13.1_6") == ("13.1_6", None)
    assert ro.new_name("misure.21-bis.0.0") == ("21-bis.0.0", None)
    # the number may not be the article's
    assert ro.new_name("capitolo_1_governo.1") == (None, "digits in the word")
    assert ro.new_name("definizioni32.1.1_7") == (None, "digits in the word")
    assert ro.new_name("art_4_ulteriori.012.0.0_11") == (None, "digits in the word")
    assert ro.new_name("parte.5.2.ALTRI ELEMENTI DA DEDURRE.0") == (None, "not an article's name")
    assert ro.new_name("parte.5..1del TUB (sottovoce 59713.02).0") == (None, "not an article's name")
    # not odd
    assert ro.new_name("3.0.0") == (None, None)
    assert ro.new_name("21-bis") == (None, None)
    assert ro.new_name(None) == (None, None)


def test_plan_skips_collisions():
    odd = ["tutti_cittadini.3.0.0", "stato.7.0.0", "repubblica_riconosce.2.0.0", "altro_nome.2.0.0",
           "litalia_una.1.0.0", "capitolo_1_x.4", "tutti_cittadini.3.0.0"]
    renames, left = ro.plan(odd, existing={"7.0.0", "13.0.0"})
    assert renames == [("litalia_una.1.0.0", "1.0.0"), ("tutti_cittadini.3.0.0", "3.0.0")]
    assert sorted(left) == [("altro_nome.2.0.0", "two old names give it"),
                            ("capitolo_1_x.4", "digits in the word"),
                            ("repubblica_riconosce.2.0.0", "two old names give it"),
                            ("stato.7.0.0", "the new name is already in the document")]
    assert ro._articles(renames) == "1, 3"


def test_the_cypher_pattern_matches_what_the_python_one_does():
    pattern = re.compile(ro.ODD_CYPHER)
    for name in ["tutti_cittadini.3.0.0", "capitolo_1_governo.1", "parte.5.2.X.0", "3.0.0", "21-bis", "preambolo"]:
        assert bool(pattern.match(name)) == bool(ro._ODD.match(name)), name
