"""scripts/article_text_compare.py: the old rule is today's master, kept here only as the shape it had."""
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")

from src.rag import article_text, comma_material  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("article_text_compare", ROOT / "scripts" / "article_text_compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class _Old:
    """As the old reader: N.0.0 alone."""
    @staticmethod
    def article_rows(number, rows):
        return [r for r in rows if r["s"]["name"] == f"{number}.0.0"][:1]


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    SECTIONS = {"61.0.0": ("Aggravano il reato le circostanze seguenti:", "Sottocomma"),
                "61": ("Aggravano il reato le circostanze seguenti: " + "1) i motivi abietti della lunga lista; " * 60, "Articolo"),
                "62.0.0": ("Attenuano il reato le cose dette qui:", "Sottocomma")}

    def run(self, query, **p):
        if "d.name STARTS WITH" in query:
            return _Rows([{"id": "cp", "name": "Codice Penale 2026"}] if p["prefix"] == "Codice Penale" else [])
        return _Rows([{"s": {"name": k, "type": t, "plain_text": x}, "next": None} for k, (x, t) in self.SECTIONS.items()])


def test_the_report_counts_the_changes_the_budget_and_what_to_look_at():
    text = "\n".join(compare.compare(_Session(), _Old, article_text, comma_material.trim, comma_material.article_parts))
    assert "| Codice Penale 2026 | 2 | 1 | 1 | 0 | 2 → 1 |" in text
    assert "1 articles. In a prompt" in text and "art. 61 c.p.: 2," in text
    assert "### art. 61 c.p." in text and "art. 62 c.p.: still ends at a colon" in text
    assert "<=1,500: 2" in text and "1,501-4,000: 1" in text
