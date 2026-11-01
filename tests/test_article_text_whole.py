"""The whole article, from the Section shapes of scripts/article_sections.py on prod (1 Nov)."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")

from src.rag import article_text, comma_material  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402

AT = article_text


def choose(number, sections, following=None):
    texts = {name: text for name, (text, _) in sections.items()}
    return AT.choose_text(number, texts, {name: kind for name, (_, kind) in sections.items()}, following)


def fill(prefix, size=3990):
    return prefix + " parola" * ((size - len(prefix)) // 7)


OPENING_61 = "Aggravano il reato quando non ne sono elementi costitutivi le circostanze seguenti:"
LIST_61 = " 1) l'avere agito per motivi abietti o futili; 2) l'aver commesso il reato per eseguirne un altro;"


def test_61_cp_chunks_joined_the_list_in_and_its_notes_out():
    first = fill(OPENING_61 + LIST_61)
    sections = {"61": (first, "Articolo"), "61.0.0": (OPENING_61, "Sottocomma"),
                "61_1": ("l'impiego di sistemi di intelligenza artificiale. (1) Numero aggiunto dall'art. 1, L. 132/2025.",
                         "Articolo"),
                "61.1.0": ("l'avere agito per motivi abietti o futili;", "Sottocomma"),
                "61.11": ("132, a decorrere dal 10 ottobre 2025.", "Sottocomma")}
    text = choose("61", sections)
    assert text.startswith(OPENING_61) and "motivi abietti" in text
    assert text.endswith("l'impiego di sistemi di intelligenza artificiale.")


def test_177_cc_the_articolo_under_a_suffix_beats_a_bare_section_holding_item_a():
    whole = ("Costituiscono oggetto della comunione: a) gli acquisti compiuti dai due coniugi; b) i frutti dei beni propri; "
             "c) i proventi dell'attività separata; d) le aziende gestite da entrambi i coniugi.")
    sections = {"177": ("gli acquisti compiuti dai due coniugi;", "Sottocomma"),
                "177.0.0": ("Costituiscono oggetto della comunione:", "Sottocomma"),
                "177_3": ("i frutti dei beni propri;", "Sottocomma"), "177_9": (whole, "Articolo")}
    assert choose("177", sections) == whole
    rows = [{"d": {"id": "cc"}, "s": {"name": n, "plain_text": t, "type": k}} for n, (t, k) in sections.items()]
    row = AT.article_rows("177", rows)[0]
    assert row["s"]["name"] == "177.0.0" and row["s"]["plain_text"] == whole


def test_590_cp_a_bare_sottocomma_is_not_the_article():
    opening = "Chiunque cagiona ad altri per colpa una lesione personale è punito con la reclusione"
    whole = opening + " fino a tre mesi. Se i fatti di cui al secondo comma sono commessi nell'esercizio abusivo."
    assert choose("590", {"590": ("Se i fatti di cui al secondo comma sono commessi", "Sottocomma"),
                          "590.0.0": (opening, "Sottocomma"), "590_5": (whole, "Articolo")}) == whole


def test_582_cp_624_bis_the_record_that_opens_like_the_article_and_a_rubrica_tail_dropped():
    opening = "Chiunque si impossessa della cosa mobile altrui, sottraendola a chi la detiene"
    tail = fill(f"casi particolari (1) (6) {opening}, è punito.")
    sections = {"624-bis": ("c), n. 2), D.L. 24 febbraio 2026, n. 23", "Sottocomma"), "624-bis.0.0": (opening, "Sottocomma"),
                "624-bis_5": (tail, "Articolo"), "624-bis_5_1": ("2026.", "Articolo")}
    text = choose("624-bis", sections)
    assert text.startswith(opening) and text.endswith("2026.")


def test_records_are_not_chained_unless_the_one_before_was_full():
    opening = "Il debitore che non esegue esattamente la prestazione dovuta"
    sections = {"9.0.0": (opening, "Sottocomma"), "9_21": (f"{opening} è tenuto al risarcimento.", "Articolo"),
                "9_22": ("Altra cosa, un altro testo del tutto.", "Articolo")}
    assert choose("9", sections) == f"{opening} è tenuto al risarcimento."


def test_593_cpp_no_n00_a_comma_number_kept_and_quoted_commi_go_with_their_note():
    body = ("1. Salvo quanto previsto dagli articoli 443, l'imputato può appellare (3). 2. Il pubblico ministero non può "
            "appellare (4). 3. Sono in ogni caso inappellabili le sentenze (5).")
    note = (" (1) Articolo prima modificato dall'art. 18, L. 468/1999. L'art. 10 della citata legge così dispone: «Art. 10 - "
            "1. La presente legge si applica. 4. Nei termini per la presentazione dei motivi.» (3) Comma così modificato.")
    sections = {"593": (fill(body + note), "Articolo"), "593.1.0": ("Salvo quanto previsto dagli articoli 443", "Sottocomma"),
                "593_1": ("punite con la sola ammenda. (5) Comma aggiunto dall'art. 2.", "Articolo")}
    assert choose("593", sections) == body


def test_chunks_joined_keep_an_overlap_once():
    assert AT._join("alfa beta gamma delta epsilon", "gamma delta epsilon zeta") == "alfa beta gamma delta epsilon zeta"
    assert AT._join("alfa", "beta") == "alfa beta"


def test_without_an_articolo_section_n00_then_the_items_in_next_order():
    sections = {"5.0.0": ("Costituiscono oggetto:", "Sottocomma"), "5": ("le prime cose;", "Sottocomma"),
                "5_3": ("le seconde cose;", "Sottocomma"), "5.1.0": ("le ultime cose.", "Sottocomma"),
                "5.2": ("n. 41, a decorrere dal 2016.", "Sottocomma")}
    assert choose("5", sections) == "Costituiscono oggetto:\nle prime cose;\nle seconde cose;\nle ultime cose."
    following = {"5.0.0": "5.1.0", "5.1.0": "5", "5": "5_3"}
    assert choose("5", sections, following) == "Costituiscono oggetto:\nle ultime cose.\nle prime cose;\nle seconde cose;"


def test_notes_are_blocks_real_text_after_one_survives():
    assert AT.strip_notes("Testo (2). (2) Le parole: «dieci» sono state sostituite. (3) La Corte costituzionale ha detto.") \
        == "Testo (2)."
    assert AT.strip_notes("Primo. (1) Comma aggiunto dall'art. 1. Secondo vero comma lungo qui.",
                          ["Secondo vero comma lungo qui"]) == "Primo. Secondo vero comma lungo qui."
    assert AT.strip_notes("1. Primo. (1) Comma così modificato dall'art. 3. 2. Secondo vero.") == "1. Primo. 2. Secondo vero."


def test_a_marker_followed_by_real_text_is_no_note_nor_is_a_recital():
    text = "Soggetto a stupefacenti.] (2) Nel caso di morte di più persone si applica la pena (3)."
    assert AT.strip_notes(text) == text
    assert AT.strip_notes("(1) Considerando che gli Stati membri devono cooperare") \
        == "(1) Considerando che gli Stati membri devono cooperare"
    assert AT.strip_notes("(2) Il comma 1 non si applica ai minori.") == "(2) Il comma 1 non si applica ai minori."


def test_abrogato_markers_stay():
    assert AT.strip_notes("Abrogato.") == "Abrogato."
    assert AT.strip_notes("Primo (abrogato). Secondo.") == "Primo (abrogato). Secondo."
    assert AT.strip_notes("[Numero vecchio.] (2) Altro. (2) Numero abrogato dall'art. 1.") == "[Numero vecchio.] (abrogato) Altro."
    assert AT.strip_notes("(1) Articolo abrogato dalla L. 19 maggio 1975, n. 151.") == "(abrogato)"
    assert AT.strip_notes("(1) Articolo inserito dall'art. 1.") == "(1) Articolo inserito dall'art. 1."     # never empty


def test_every_lookup_sees_the_chunks():
    class Session:
        queries = []

        def run(self, query, **params):
            self.queries.append(query)
            return type("R", (), {"data": lambda s: []})()
    AT.fetch_article(Session(), "doc", "61")
    dg._fetch_articles(Session(), [("61", "c.p.")])
    assert all("+ '_'" in q and "NEXT" in q for q in Session.queries)
    retrieval = open(os.path.join(os.path.dirname(__file__), "..", "src", "rag", "nodes", "retrieval.py"), encoding="utf-8").read()
    assert "STARTS WITH $num + '_'" in retrieval and "STARTS WITH $article_number + '_'" in retrieval
    assert "RETURN d, s LIMIT 15\",\n                    article_ref" not in retrieval.split("Fallback: prefixed")[0]


def test_the_router_keeps_the_whole_article_on_one_row():
    rows = [{"d": {"id": "cp"}, "s": {"name": n, "plain_text": t, "type": k}} for n, t, k in
            (("61", "Aggravano il reato le circostanze seguenti: 1) i motivi;", "Articolo"), ("61.1.0", "i motivi;", "Sottocomma"))]
    AT.fill_from_rows("61", rows)
    assert [r["s"]["name"] for r in AT.canonical_only("61", rows)] == ["61"]


def test_over_budget_keeps_the_opening_and_the_items_of_the_issue_never_the_prefix():
    items = " ".join(f"{i}) voce generica numero {i} del lungo elenco di circostanze;" for i in range(1, 12))
    text = f"Aggravano il reato le circostanze seguenti: {items} 12) l'avere commesso il fatto con crudeltà verso le persone."
    out = comma_material.trim(text, 220, "crudeltà verso le persone")
    assert out.startswith("Aggravano il reato") and "crudeltà" in out and "[…]" in out and len(out) <= 240
    assert comma_material.trim("corto", 50) == "corto"
    assert "(co." not in out
    assert "crudeltà" in dg._source_block([{"_article": ("61", "c.p."), "_focus": "crudeltà verso le persone",
                                            "s": {"plain_text": text}}], 220)


def test_more_note_openings_are_notes():
    for note in ("Il presente numero è stato modificato dall'art. 1.", "Intestazione così modificata dall'art. 2.",
                 "Denominazione così sostituita dall'art. 3.", "La punteggiatura è stata così modificata dall'art. 4.",
                 "I Capi da IV a VI sono stati aggiunti dall'art. 5.", "Le originarie parole «dieci» sono state sostituite.",
                 "Successivamente, il presente comma è stato così modificato.", "Il previgente Capo V è stato abrogato."):
        assert AT.strip_notes(f"Il giudice provvede (2). (1) {note}") == "Il giudice provvede (2)."
    assert AT.strip_notes("Successivamente, il giudice provvede.") == "Successivamente, il giudice provvede."
    assert AT.strip_notes("(2) I Capi delle sezioni decidono.") == "(2) I Capi delle sezioni decidono."


def test_marker_only_results():
    assert AT.strip_notes("(…) (1) (1) L'articolo che recitava: “Testo vecchio.” è stato abrogato dall'art. 1, L. 219/2012.") == "(abrogato)"
    assert choose("284", {"284": ("(…) (1) (1) L'articolo che recitava: “La legittimazione.” è stato abrogato dall'art. 1.",
                                  "Articolo"), "284.0.0": ("(…) (1)", "Sottocomma")}) == "(abrogato)"
    previous = "(1) (2) Articolo inserito dall'art. 1, L. 3/2000."
    assert choose("280", {"280": (previous, "Articolo")}) == previous           # no repeal: the text as it stood
    assert choose("281", {"281": ("(1)", "Articolo"), "281.0.0": ("Articolo inserito dall'art. 2.", "Sottocomma")}) \
        == "Articolo inserito dall'art. 2."


def test_a_leading_marker_goes_with_its_note_a_recital_keeps_its_own():
    assert AT.strip_notes("(1) Il coniuge provvede. (1) Comma aggiunto dall'art. 1.") == "Il coniuge provvede."
    assert AT.strip_notes("(1) Gli Stati provvedono.") == "(1) Gli Stati provvedono."


def test_commi_come_from_the_whole_text_and_co_labels_only_real_commi(monkeypatch):
    monkeypatch.setattr(comma_material, "enabled", lambda: True)
    whole = ("Aggravano il reato quando non ne sono elementi costitutivi le circostanze seguenti: 1) i motivi abietti; "
             "2) la crudeltà verso le persone.")
    parts, how = comma_material.commi_from_sections("61", {"61": (whole), "61.1.0": "i motivi abietti;"}, {"61": "Articolo"})
    assert len(parts) == 3 and parts[0].endswith("seguenti:") and how == "sentences"
    assert comma_material.comma_numbers(parts, how) == [None, None, None]
    numbered = "1. Il giudice provvede quando ricorrono le condizioni seguenti: a) la prima condizione; b) la seconda.\n2. Altro comma che dispone diversamente."
    parts, how = comma_material.article_parts(numbered)
    assert how == "lines" and comma_material.comma_numbers(parts, how) == [1, 1, 1, 2]
    row = {comma_material.COMMI_KEY: parts, comma_material.FROM_KEY: how, comma_material.FOCUS_KEY: "seconda condizione"}
    out = comma_material.passage(row, 140)
    assert out.startswith("(co. 1) Il giudice") and "(co. a" not in out and "(co. 3)" not in out and "[commi non riportati: 2]" in out
    row[comma_material.NAMED_KEY] = [2]
    assert "(co. 2) Altro comma" in comma_material.passage(row, 400)


def test_the_grader_reads_the_opening_and_the_items_of_the_issue():
    items = " ".join(f"{i}) voce generica numero {i} del lungo elenco di circostanze;" for i in range(1, 16))
    text = f"Aggravano il reato le circostanze seguenti: {items} 16) l'avere agito con crudeltà verso le persone."
    rows = {("61", "c.p."): [{"_article": ("61", "c.p."), "s": {"plain_text": text}}]}
    focus = dg._graded_focus(rows, "Tizio agisce con crudeltà verso le persone", "", None)
    got = dg._graded_texts({"61 c.p.": text[:1500]}, rows, None, focus)["61 c.p."]
    assert "Testo: Aggravano il reato" in got and "crudeltà" in got and "[…]" in got
    assert len(got.split("Testo: ")[1]) <= dg._GRADED_TEXT_CHARS + 20
