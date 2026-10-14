"""V2 no longer writes Section, CommentatoSection or FiscalitaSection vectors.

BE writes them on upload and its backfill fills gaps. V2's writers embedded a
different text (title + text_en + abstract + plain_text) into `embedding`, the
old vectors a rollback depends on, and never wrote `embedding_v2`: the
background job (embed_missing) is gone, and the command line refuses or skips
those labels.
"""

import logging
import os
import pathlib
import sys

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import neo4j  # noqa: E402

import src.preprocessing.generate_embeddings as gen  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Session:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.log.append(query)
        return []


class _Driver:
    def __init__(self):
        self.queries = []

    def session(self, **config):
        return _Session(self.queries)

    def close(self):
        pass


@pytest.mark.parametrize("label", ["Section", "CommentatoSection", "FiscalitaSection"])
def test_asking_for_a_section_label_is_refused_before_connecting(label, monkeypatch, capsys):
    def no_connection(*a, **k):
        raise AssertionError("must not connect")

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", no_connection)
    monkeypatch.setattr(gen, "_init_embedding_model", no_connection)
    monkeypatch.setattr(sys, "argv", ["generate_embeddings", "--label", label])
    with pytest.raises(SystemExit) as exit_info:
        gen.main()
    assert exit_info.value.code != 0
    assert "backfill_embeddings_v2.py" in capsys.readouterr().err


def test_a_full_run_skips_the_section_labels(monkeypatch, caplog):
    driver = _Driver()
    processed = []
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: driver)
    monkeypatch.setattr(gen, "_init_embedding_model", lambda: object())
    monkeypatch.setattr(gen, "_process_label", lambda drv, db, cfg, model, size: processed.append(cfg["label"]) or 0)
    monkeypatch.setattr(sys, "argv", ["generate_embeddings", "--create-indexes"])
    with caplog.at_level(logging.INFO):
        gen.main()
    assert processed and not set(processed) & set(gen.BE_OWNED_LABELS)
    assert "Document" in processed and "Institution" in processed       # other labels as before
    assert not [q for q in driver.queries if "section_embeddings" in q]  # no section index either
    assert "backfill_embeddings_v2.py" in caplog.text


def test_other_labels_can_still_be_asked_for(monkeypatch):
    processed = []
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: _Driver())
    monkeypatch.setattr(gen, "_init_embedding_model", lambda: object())
    monkeypatch.setattr(gen, "_process_label", lambda drv, db, cfg, model, size: processed.append(cfg["label"]) or 0)
    monkeypatch.setattr(sys, "argv", ["generate_embeddings", "--label", "Article"])
    gen.main()
    assert processed == ["Article"]


def test_the_background_writer_is_gone():
    assert not hasattr(gen, "embed_missing")
    api = (ROOT / "src" / "chatbot" / "api.py").read_text(encoding="utf-8")
    assert "_background_embedding_job" not in api and "embed_missing" not in api
