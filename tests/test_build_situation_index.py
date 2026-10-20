"""scripts/build_situation_index.py's pure helpers, offline: which Section
is an article's main one, which documents the default list resolves to,
what the model's reply is cut down to, and the index statement copied from
section_embeddings_v2's own config. The script itself needs a live Neo4j,
chat model and embedding endpoint."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_situation_index",
                                               HERE.parent / "scripts" / "build_situation_index.py")
bsi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsi)


def test_base_article_reads_main_articles_only():
    assert bsi._base_article("640") == "640"
    assert bsi._base_article("640-ter.1.0") == "640-ter"
    assert bsi._base_article("640_2") == "640"
    assert bsi._base_article("452-terdecies") == "452-terdecies"
    assert bsi._base_article("all2-1") is None          # annex article, not a main one
    assert bsi._base_article("decreto-1.1.0") is None
    assert bsi._base_article("Premessa") is None


def test_main_sections_prefers_the_exact_article_section_then_the_longest_text():
    rows = [
        {"name": "112.0.0", "text": "In vigore dal 8 dicembre 2011"},
        {"name": "112", "text": "1. I provvedimenti ..."},
        {"name": "112.1.0", "text": "I provvedimenti del giudice amministrativo devono essere eseguiti ..."},
        {"name": "114.1.0", "text": "breve"},
        {"name": "114.2.0", "text": "", "abstract": "un abstract più lungo del testo breve"},
        {"name": "115.1.0", "text": ""},                   # no text at all: not indexed
    ]
    main = bsi._main_sections(rows)
    assert main["112"]["name"] == "112"
    assert main["114"]["name"] == "114.2.0"
    assert "115" not in main


def test_resolve_documents_prefers_exact_names_and_ignores_tags_and_case():
    docs = [{"id": "1", "name": "[PENALE] Codice Penale 2026"},
            {"id": "2", "name": "Codice Penale 2026 - Commentato"},
            {"id": "3", "name": "Legge 7 agosto 1990, n. 241 (L. 241/1990)"}]
    resolved = bsi._resolve_documents(["codice penale 2026", "L. 241/1990", "Codice Civile 2026"], docs)
    assert [d["id"] for d in resolved["codice penale 2026"]] == ["1"]
    assert [d["id"] for d in resolved["L. 241/1990"]] == ["3"]
    assert resolved["Codice Civile 2026"] == []


def test_parse_situations_keeps_3_to_5_distinct_plain_strings():
    reply = ["Un uomo prende il portafoglio dalla borsa di una collega.",
             "un uomo prende il portafoglio  dalla borsa di una collega.",   # duplicate, different spacing/case
             "Applicazione dell'art. 624 c.p. al furto",                     # cites the law
             "Il codice prevede la pena",                                    # names a code
             42, "",
             "Una ragazza esce da un negozio con un vestito senza pagarlo.",
             "Qualcuno ruba la bicicletta legata davanti a casa.",
             "Un dipendente porta a casa materiale dell'ufficio.",
             "Un turista sfila il telefono dalla tasca di un passante.",
             "Uno in più."]
    out = bsi._parse_situations(reply)
    assert len(out) == 5
    assert out[0] == "Un uomo prende il portafoglio dalla borsa di una collega."
    assert not any("624" in s or "codice" in s.lower() for s in out)
    assert bsi._parse_situations({"situazioni": []}) == []


def test_index_statement_copies_the_reference_config_verbatim():
    config = {"vector.dimensions": 4096, "vector.similarity_function": "COSINE",
              "vector.quantization.enabled": True, "vector.hnsw.m": 16}
    statement = bsi._index_statement(config)
    assert statement.startswith("CREATE VECTOR INDEX situation_embeddings_v2 IF NOT EXISTS "
                                "FOR (x:Situation) ON (x.embedding)")
    for fragment in ("`vector.dimensions`: 4096", "`vector.similarity_function`: 'COSINE'",
                     "`vector.quantization.enabled`: true", "`vector.hnsw.m`: 16"):
        assert fragment in statement


def test_situation_id_is_stable_and_distinct():
    assert bsi._situation_id("d", "640", 1) == bsi._situation_id("d", "640", 1)
    assert bsi._situation_id("d", "640", 1) != bsi._situation_id("d", "640", 2)
