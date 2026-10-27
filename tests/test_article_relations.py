"""Relations agent v0 (28 Oct): src/rag/article_relations.py (the app's reader, behind ARTICLE_RELATIONS_ENABLED) and
scripts/maint_relations.py (the builder of its file; reads the graph, never writes it)."""
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from src.rag import article_relations as ar  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("maint_relations", HERE.parent / "scripts" / "maint_relations.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)

DATA = {
    "related": {"1490": [["1492", 0.03], ["1495", 0.029], ["1453", 0.02]], "1218": [["1223", 0.03], ["1176", 0.02]],
                "2051": [["2043", 0.03], ["1227", 0.02]]},
    "via": {"1490": {"1492": "applied,cited,text", "1495": "applied,cited", "1453": "cited"}},
    "profiles": {"rulings": ar.profile_vectors({"1490": Counter(ar.stems("vizi cosa venduta garanzia compratore")),
                                                "2051": Counter(ar.stems("custodia cosa danno caso fortuito")),
                                                "1495": Counter(ar.stems("denuncia vizi otto giorni decadenza")),
                                                "1497": Counter(ar.stems("qualità promesse essenziali uso"))})},
}


def test_stems_and_profiles():
    assert ar.stems("La garanzia per i vizi della cosa venduta") == ["garanz", "vizi", "vendut"]
    ranked = ar.rank_profiles(DATA["profiles"]["rulings"], "la barca venduta ha vizi, il compratore chiede la garanzia")
    assert ranked[0][0] == "1490"


def test_neighbours_fused_from_the_seeds_ties_to_the_earlier_seed():
    rel = ar.Relations(DATA)
    found = rel.neighbours(["1490", "1218"], exclude={"1490", "1218", "1495"})
    assert [a for a, _ in found] == ["1492", "1223", "1453", "1176"]
    assert dict(found)["1492"] == "1490" and dict(found)["1176"] == "1218"


def test_candidates_cc_only_civil_profiles_and_why():
    rel = ar.Relations(DATA)
    added, why = ar.candidates([("1490", "c.c."), ("640", "c.p.")], "la barca non ha le qualità promesse",
                               have={("1492", "c.c."), ("640", "c.p.")}, civil=True, relations=rel)
    assert added[:2] == [("1495", "c.c."), ("1453", "c.c.")] and all(c == "c.c." for _, c in added)
    assert why[("1495", "c.c.")] == "related to 1490 c.c. (applied,cited)"
    assert [ref for ref, w in why.items() if w.startswith("rulings' profile")] == [("1497", "c.c.")]
    added, why = ar.candidates([("640", "c.p.")], "la barca non ha le qualità promesse", set(), civil=False,
                               relations=rel)
    assert added == [] and why == {}                                                  # criminal seeds, not civil


def test_store_reads_the_file_again_when_it_changes(tmp_path, monkeypatch):
    path = tmp_path / "relations_cc.json"
    monkeypatch.setenv(ar.FILE_ENV, str(path))
    assert ar.store() is None                                                         # no file yet
    path.write_text(json.dumps(DATA), encoding="utf-8")
    assert len(ar.store().related) == 3
    path.write_text(json.dumps({**DATA, "related": {"1490": [["1492", 0.1]]}}), encoding="utf-8")
    os.utime(path, (1, 1))
    assert list(ar.store().related) == ["1490"]
    assert ar.candidates([("1490", "c.c.")], "", set(), civil=False) == (
        [("1492", "c.c.")], {("1492", "c.c."): "related to 1490 c.c. (applied,cited,text)"})
    monkeypatch.setenv(ar.FILE_ENV, str(tmp_path / "missing.json"))
    assert ar.candidates([("1490", "c.c.")], "x", set(), True) == ([], {})
    monkeypatch.delenv(ar.ARTICLE_RELATIONS_FLAG, raising=False)
    assert not ar.enabled()


def test_builder_makes_the_file_and_writes_it_whole(tmp_path):
    cards = [{"articoli_applicati": ["1490 c.c.", "1495 c.c.", "360 c.p.c."], "argomento_decisivo": ["vizi della cosa"]},
             {"articoli_applicati": ["1490 c.c.", "1495 c.c.", "1492 c.c."], "parole_chiave_questioni": ["denuncia"]},
             {"articoli_applicati": ["2697 c.c.", "1490 c.c."]}]
    cited = {"r1": {"1490", "1492"}, "r2": {"1490", "1492", "1497"}, "r3": {"2051", "2043"}, "r4": {"2051", "2043"}}
    texts = {"1490": "Il venditore è tenuto a garantire.", "1492": "Nei casi indicati dall'articolo 1490 il compratore",
             "1495": "Il compratore decade.", "2051": "Custodia.", "2043": "Fatto illecito.", "1497": "Qualità."}
    data = mr.build(cards, cited, texts, {"1490": "Garanzia per i vizi della cosa venduta"}, generic_share=1.0)
    assert data["sources"] == {"cards": 3, "cards_with_cc": 3, "rulings_citing_cc": 4, "articles_with_text": 6}
    assert "2697" in data["generic"] and "2697" not in data["related"]
    assert {b for b, _ in data["related"]["1490"]} >= {"1492", "1495"}
    assert data["via"]["1490"]["1492"] == "cited,text" and data["via"]["1490"]["1495"] == "applied"
    assert set(data["profiles"]) == {"rulings", "text"} and "1490" in data["profiles"]["text"]["vectors"]
    out = tmp_path / "rel" / "relations_cc.json"
    mr.write_atomically(data, str(out))
    assert not (tmp_path / "rel" / "relations_cc.json.tmp").exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert ar.Relations(written).neighbours(["2051"], set())[0][0] == "2043"
