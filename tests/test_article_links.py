"""scripts/article_links.py offline: the heading markers along the NEXT chain, the cross-references by pattern, the CSVs,
and that it only reads. The c.c. texts are the corpus' own (h3's dumps, Leonardo v46), the chain shortened."""
import csv
import importlib.util
import pathlib
import re

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("article_links", HERE.parent / "scripts" / "article_links.py")
al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(al)

TEXTS = {
    "1030": "Il proprietario del fondo servente non è tenuto a compiere alcun atto per rendere possibile l'esercizio.",
    "1031": "Le servitù prediali possono essere costituite coattivamente o volontariamente. Possono anche essere costituite "
            "per usucapione o per destinazione del padre di famiglia. CAPO II – DELLE SERVITU’ COATTIVE",
    "1032": "Quando, in forza di legge, il proprietario di un fondo ha diritto di ottenere la costituzione di una servitù, "
            "questa, in mancanza di contratto, è costituita con sentenza.",
    "1051": "Art. 1051 Il proprietario, il cui fondo è circondato da fondi altrui, ha diritto di ottenere il passaggio.",
    "1052": "Le disposizioni dell'articolo precedente si possono applicare anche se il proprietario ha un accesso inadatto.",
    "1053": "Nei casi previsti dai due articoli precedenti è dovuta un'indennità proporzionata al danno.",
    "1079": "Il titolare della servitù può farne riconoscere in giudizio l'esistenza. CAPO VIII – DI ALCUNE SERVITU’ IN "
            "MATERIA DI ACQUA SEZIONE I – Delle servitù di presa o di derivazione di acque",
    "1080": "Il proprietario che ha diritto di derivare acque ai sensi degli articoli 1032 e 1051 del presente codice, "
            "nonché dell'art. 700 del codice di procedura civile, e dell'art. 12 della legge n. 10 del 1990, e "
            "dell'art. 9999.",
    "1081": "LIBRO QUARTO – DELLE OBBLIGAZIONI TITOLO I – DELLE OBBLIGAZIONI IN GENERALE Art. 1173 Le obbligazioni derivano "
            "da contratto.",
}


def _sections():
    order = list(TEXTS)
    out = [{"id": f"s{k}", "name": f"{k}.0.0", "text": TEXTS[k], "next": f"s{order[i + 1]}" if i + 1 < len(order) else None}
           for i, k in enumerate(order)]
    out.append({"id": "lost", "name": "2000.0.0", "text": "Fuori dalla catena.", "next": None})
    out.append({"id": "odd", "name": "preambolo", "text": "", "next": "s1030"})
    return out


class Session:
    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append(query)

        class Result:
            def __init__(self, rows):
                self.rows = rows

            def data(self):
                return self.rows
        if "document_type" in query:
            return Result([{"id": "dcpc", "name": "Codice di procedura civile 2026"}, {"id": "dx", "name": "Altra legge"},
                           {"id": "dcc", "name": "Codice Civile 2025"}])
        return Result({"dcc": _sections(), "dx": [],
                       "dcpc": [{"id": "p700", "name": "700.0.0", "text": "Provvedimenti d'urgenza.", "next": None}]}
                      [params["doc"]])


def _results():
    session = Session()
    docs = al.ordered(session.run(al.QUERY_DOCS).data())
    return al.analyse([al.read_document(session, d) for d in docs]), session, docs


def test_the_six_codes_first_and_only_reads():
    results, session, docs = _results()
    assert [d["name"] for d in docs] == ["Codice Civile 2025", "Codice di procedura civile 2026", "Altra legge"]
    assert [d["name"] for d in al.ordered(docs, ["codice civile"])] == ["Codice Civile 2025"]
    for query in session.queries + [al.QUERY_DOCS, al.QUERY_SECTIONS]:
        assert not re.search(r"\b(?:CREATE|MERGE|SET|DELETE|REMOVE|DETACH|CALL)\b", query, re.IGNORECASE)


def test_a_trailing_marker_opens_its_unit_from_the_next_article_a_leading_one_from_its_own():
    cc = _results()[0][0]
    units = cc["units"]
    # the chain starts mid-code: 1030-1031 in the opening units (LIBRO QUARTO and CAPO II seen later), no heading-only
    # Section to name them; TITOLO I comes after LIBRO QUARTO: no opening TITOLO
    opening = {"LIBRO": "LIBRO TERZO (intestazione non nel testo)", "CAPO": "CAPO I (intestazione non nel testo)"}
    assert units["1030"] == units["1031"] == opening
    assert units["1032"] == {"LIBRO": opening["LIBRO"], "CAPO": "CAPO II – DELLE SERVITU’ COATTIVE"}
    assert units["1079"]["CAPO"].startswith("CAPO II")
    assert units["1080"] == {"LIBRO": opening["LIBRO"], "CAPO": "CAPO VIII – DI ALCUNE SERVITU’ IN MATERIA DI ACQUA",
                             "SEZIONE": "SEZIONE I – Delle servitù di presa o di derivazione di acque"}
    # a leading LIBRO closes the CAPO and the SEZIONE; its name stops at the article's own "Art." heading
    assert units["1081"] == {"LIBRO": "LIBRO QUARTO – DELLE OBBLIGAZIONI",
                             "TITOLO": "TITOLO I – DELLE OBBLIGAZIONI IN GENERALE"}
    why = dict(cc["unplaced"])
    assert "1030" not in why and "1031" not in why
    assert units["2000"]["LIBRO"].startswith("LIBRO QUARTO")         # off the chain: placed by its number
    assert why["preambolo"] == "Section name with no article number"


def test_references_by_pattern_and_their_resolution():
    cc = _results()[0][0]
    refs = [(r["from"], r["to_article"], r["to_code"], r["pattern"], r["resolved"], r["this_code"]) for r in cc["references"]]
    assert ("1052", "1051", "c.c.", "relative", True, False) in refs
    assert {r[1] for r in refs if r[0] == "1053"} == {"1052", "1051"}                 # "dai due articoli precedenti"
    assert ("1080", "1032", "c.c.", "list", True, True) in refs and ("1080", "1051", "c.c.", "list", True, True) in refs
    assert ("1080", "700", "c.p.c.", "other code", True, False) in refs
    assert ("1080", "12", "other act", "other code", False, False) in refs
    assert ("1080", "9999", "c.c.", "abbr", False, False) in refs
    assert not [r for r in refs if r[0] == "1051"]                                    # its own "Art. 1051" heading
    assert not [r for r in refs if r[1] == "1173"]                    # the next article's heading in 1081's text
    lines = al.census_lines(cc)
    assert "- articles: 10" in lines and any(ln.startswith("| relative | 3 | 3 |") for ln in lines)
    assert "| of them «del presente codice» | 2 | 2 |" in lines


def test_the_dry_run_files(tmp_path):
    results = _results()[0]
    paths = al.write_csvs(results, str(tmp_path))
    units = list(csv.reader(open(paths["units.csv"], encoding="utf-8")))
    assert units[0] == ["code", "level", "unit name", "article"] and ["c.c.", "SEZIONE",
                                                                     "SEZIONE I – Delle servitù di presa o di derivazione di acque", "1080"] in units
    refers = list(csv.reader(open(paths["refers_to.csv"], encoding="utf-8")))
    assert refers[0] == ["from article", "to article", "pattern", "matched text"]
    assert ["1080 c.c.", "700 c.p.c.", "other code", "art. 700"] in refers
    unresolved = list(csv.reader(open(paths["unresolved_sample.csv"], encoding="utf-8")))
    assert {row[1] for row in unresolved[1:]} == {"12 other act", "9999 c.c."}


# --- v48: a second numbering run, the graph writes ---------------------------------------------------------------------

CPC = [("1", "LIBRO PRIMO – DISPOSIZIONI GENERALI TITOLO I – DEGLI ORGANI GIUDIZIARI CAPO I – DEL GIUDICE La giurisdizione "
              "civile è esercitata dai giudici ordinari."),
       ("2", "La giurisdizione italiana non può essere derogata."),
       ("73", "Il pubblico ministero esercita presso il giudice le funzioni."),
       ("77", "Il procuratore generale può stare in giudizio. DISPOSIZIONI PER L'ATTUAZIONE DEL CODICE DI PROCEDURA "
              "CIVILE TITOLO I – DISPOSIZIONI GENERALI"),
       ("att-1", "Il pubblico ministero ai sensi dell'art. 73 del codice interviene."),
       ("att-2", "Le disposizioni dell'articolo 1 si applicano anche al procuratore dell'art. 77.")]


def _cpc_doc():
    sections = [{"id": f"e{i}", "name": f"{n}.0.0", "text": t, "next": f"e{i + 1}" if i + 1 < len(CPC) else None}
                for i, (n, t) in enumerate(CPC)]
    walk, off, runs, names = al.arrange(sections)
    return {"doc": {"id": "dcpc", "name": "Codice di procedura civile 2026"}, "code": "c.p.c.", "sections": sections,
            "walk": walk, "off": off, "runs": runs, "run_names": names}


def test_a_second_numbering_is_its_own_tree_and_sections_are_known_by_element_id():
    doc = _cpc_doc()
    assert doc["runs"] == [0, 0, 0, 0, 2, 2] and doc["run_names"][2] == "disposizioni di attuazione"
    result = al.analyse([doc])[0]
    assert result["units"]["2"]["CAPO"] == "CAPO I – DEL GIUDICE"
    assert result["units"]["1 [disposizioni di attuazione]"] == {"TITOLO": "TITOLO I – DISPOSIZIONI GENERALI"}
    assert result["pieces"]["1"] == ["e0"] and result["pieces"]["1 [disposizioni di attuazione]"] == ["e4"]
    by = {(r["from_id"], r["to_article"]): r for r in result["references"]}
    assert by[("e4", "73")]["to_id"] == "e2"              # not in its own run: the code's own art. 73
    assert by[("e5", "1")]["to_id"] == "e4"               # in its own run: disp. att. art. 1, never the code's
    assert by[("e5", "77")]["to_id"] == "e3"


def test_the_csvs_tell_the_two_runs_apart(tmp_path):
    paths = al.write_csvs(al.analyse([_cpc_doc()]), str(tmp_path))
    refers = list(csv.reader(open(paths["refers_to.csv"], encoding="utf-8")))
    assert ["2 c.p.c. [disposizioni di attuazione]", "1 c.p.c. [disposizioni di attuazione]", "word",
            "articolo 1"] in refers
    assert ["1 c.p.c. [disposizioni di attuazione]", "73 c.p.c.", "abbr", "art. 73"] in refers
    assert b"\r\n" not in open(paths["refers_to.csv"], "rb").read()                   # lineterminator "\n"


class WriteSession:
    """Answers MERGE batches like a server would: a row is created the first time its key is seen."""
    def __init__(self):
        self.seen, self.batches = set(), []

    def run(self, query, rows=(), **params):
        self.batches.append((query, len(rows)))
        created = 0
        for row in rows:
            key = (query, tuple(sorted(row.items())))
            if key not in self.seen:
                self.seen.add(key)
                created += 1
        is_node = query.startswith("UNWIND $rows AS r MERGE (u:Unit")

        class Counters:
            nodes_created = created if is_node else 0
            relationships_created = 0 if is_node else created

        class Result:
            def consume(self):
                class Summary:
                    counters = Counters()
                return Summary()
        return Result()


def test_write_merges_in_batches_and_counts_what_was_there(monkeypatch):
    monkeypatch.setattr(al, "BATCH", 2)
    rows = al.graph_rows(al.analyse([_cpc_doc()]))
    assert {"code": "c.p.c.", "level": "TITOLO", "path": "disposizioni di attuazione > TITOLO I – DISPOSIZIONI GENERALI",
            "name": "TITOLO I – DISPOSIZIONI GENERALI"} in rows["units"]
    assert {"code": "c.p.c.", "level": "CAPO", "path": "LIBRO PRIMO – DISPOSIZIONI GENERALI > TITOLO I – DEGLI ORGANI "
            "GIUDIZIARI > CAPO I – DEL GIUDICE", "parent_level": "TITOLO",
            "parent_path": "LIBRO PRIMO – DISPOSIZIONI GENERALI > TITOLO I – DEGLI ORGANI GIUDIZIARI"} in rows["child_of"]
    assert {"from": "e5", "to": "e4", "pattern": "word", "text": "articolo 1", "resolution": "same_code"} in rows["refers_to"]
    session = WriteSession()
    first = al.write_graph(session, rows, "2026-11-04T00:00:00+00:00")
    assert first["units"]["created"] == len(rows["units"]) and first["refers_to"]["present"] == 0
    assert all(n <= 2 for q, n in session.batches)                      # sequential batches of BATCH
    again = al.write_graph(session, rows, "2026-11-05T00:00:00+00:00")
    assert all(v["created"] == 0 and v["present"] == v["rows"] for v in again.values())     # idempotent
    for query in (al.WRITE_UNITS, al.WRITE_CHILD_OF, al.WRITE_HAS_ARTICLE, al.WRITE_REFERS_TO):
        assert "MERGE" in query and "CREATE (" not in query and "DELETE" not in query
    assert "source: $source" in al.WRITE_REFERS_TO and "ON CREATE SET e.computed_at" in al.WRITE_REFERS_TO


def test_codes_filter():
    docs = [{"id": "a", "name": "Codice Civile 2025"}, {"id": "b", "name": "Codice di procedura civile 2026"},
            {"id": "c", "name": "Codice del Consumo 2026"}]
    assert [d["id"] for d in al.select_codes(docs, ["c.p.c.", "codice del consumo"])] == ["b", "c"]
    assert al.select_codes(docs, []) == docs


# c.p.c. as the live graph holds it: each article's fragments and its whole-article Section on one chain
WHOLE = [("1.0.0", "LIBRO PRIMO – DISPOSIZIONI GENERALI TITOLO I – DEGLI ORGANI GIUDIZIARI La giurisdizione civile è "
                   "esercitata dai giudici ordinari."),
         ("1", "La giurisdizione civile è esercitata dai giudici ordinari."),
         ("2.0.0", "La giurisdizione italiana non può essere derogata."), ("2", "La giurisdizione italiana non può essere derogata."),
         ("3.0.0", "Abrogato."), ("3", "Abrogato."),
         ("4.0.0", "Lo straniero può essere convenuto"), ("4.1.0", "1) se è residente"), ("4.2.0", "2) se ha accettato"),
         ("4.3.0", "3) se la domanda"), ("4.4.0", "4) se si tratta"), ("4", "Lo straniero può essere convenuto ... 4) se si tratta"),
         ("5.0.0", "La giurisdizione e la competenza si determinano con riguardo alla legge vigente; ai sensi dell'art. 4."),
         ("5", "La giurisdizione e la competenza si determinano con riguardo alla legge vigente; ai sensi dell'art. 4."),
         ("6.0.0", "La competenza non può essere derogata per accordo delle parti. CAPO I – DEL GIUDICE"),
         ("6", "La competenza non può essere derogata per accordo delle parti.")]


def test_whole_article_sections_and_fragments_are_one_article_and_no_second_run(tmp_path):
    sections = [{"id": f"w{i}", "name": n, "text": t, "next": f"w{i + 1}" if i + 1 < len(WHOLE) else None}
                for i, (n, t) in enumerate(WHOLE)]
    walk, off, runs, names = al.arrange(sections)
    assert set(runs) == {0} and [w["name"] for w in walk][:3] == ["1", "1.0.0", "2"]        # the whole node first
    doc = {"doc": {"id": "dcpc", "name": "Codice di procedura civile 2026"}, "code": "c.p.c.", "sections": sections,
           "walk": walk, "off": off, "runs": runs, "run_names": names}
    result = al.analyse([doc])[0]
    assert result["articles"] == 6 and result["pieces"]["4"] == ["w11", "w6", "w7", "w8", "w9", "w10"]
    assert sorted(result["units"]) == ["1", "2", "3", "4", "5", "6"] and not result["unplaced"]
    units = list(csv.reader(open(al.write_csvs([result], str(tmp_path))["units.csv"], encoding="utf-8")))
    titolo = [row[3] for row in units[1:] if row[1] == "TITOLO"]
    assert titolo == ["1", "2", "3", "4", "5", "6"]                     # c.p.c. 1-5 once each
    assert {(r["from_id"], r["to_id"]) for r in result["references"]} == {("w12", "w11"), ("w13", "w11")}


def test_a_late_piece_places_an_article_its_first_piece_could_not():
    sections = [{"id": "a", "name": "1.0.0", "text": "Testo. TITOLO I – DISPOSIZIONI", "next": "b"},
                {"id": "b", "name": "1", "text": "Testo.", "next": None}]
    units, unplaced = al.place(sections)
    assert units == {"1": {"TITOLO": "TITOLO I – DISPOSIZIONI"}} and unplaced == []


# --- v48: the opening units (c.p.c. 1-5 on the live graph), the heading-only Sections, the preleggi ---------------------

def _cpc_live(headings=True):
    """c.p.c. as the live graph holds it: the chain starts at 1.0.0, the article itself; the first marker in any text is
    SEZIONE II at the end of art. 5; CAPO II, TITOLO II and LIBRO SECONDO later; the opening headings only as Sections
    with no article number (off the chain)."""
    chain_ = []
    for n, text in (("1", "La giurisdizione civile, salvo speciali disposizioni di legge, è esercitata dai giudici "
                          "ordinari secondo le norme del presente codice."),
                    ("2", "La giurisdizione italiana non può essere derogata per convenzione."),
                    ("3", "Abrogato."), ("4", "Lo straniero può essere convenuto davanti ai giudici della Repubblica."),
                    ("5", "La giurisdizione e la competenza si determinano con riguardo alla legge vigente. SEZIONE II – "
                          "Della competenza per materia e per valore"),
                    ("6", "La competenza non può essere derogata per accordo delle parti."),
                    ("56", "Il giudice ricusato. CAPO II – DEL CANCELLIERE E DELL'UFFICIALE GIUDIZIARIO"),
                    ("57", "Il cancelliere documenta le attività. TITOLO II – DEL PUBBLICO MINISTERO"),
                    ("69", "Il pubblico ministero esercita l'azione civile. LIBRO SECONDO – DEL PROCESSO DI COGNIZIONE"),
                    ("163", "La domanda si propone mediante citazione.")):
        chain_ += [(f"{n}.0.0", text), (n, text)]
    sections = [{"id": f"c{i}", "name": name, "text": text, "next": f"c{i + 1}" if i + 1 < len(chain_) else None}
                for i, (name, text) in enumerate(chain_)]
    if headings:
        sections += [{"id": f"h{i}", "name": name, "text": text, "next": None} for i, (name, text) in enumerate((
            ("intestazione libro", "LIBRO PRIMO – DISPOSIZIONI GENERALI"),
            ("intestazione titolo", "TITOLO I – DEGLI ORGANI GIUDIZIARI"),
            ("intestazione capo", "CAPO I – DEL GIUDICE"),
            ("intestazione sezione", "SEZIONE I – Della giurisdizione e della competenza in generale"),
            ("intestazione titolo 2", "TITOLO II – DEL PUBBLICO MINISTERO")))]
    return sections


def _doc(sections, name="Codice di procedura civile 2026", code="c.p.c."):
    walk, off, runs, names = al.arrange(sections)
    return {"doc": {"id": "d", "name": name}, "code": code, "sections": sections, "walk": walk, "off": off,
            "runs": runs, "run_names": names}


def test_cpc_1_to_5_in_the_opening_units_named_from_the_heading_only_sections(tmp_path):
    result = al.analyse([_doc(_cpc_live())])[0]
    units = result["units"]
    opening = {"LIBRO": "LIBRO PRIMO – DISPOSIZIONI GENERALI", "TITOLO": "TITOLO I – DEGLI ORGANI GIUDIZIARI",
               "CAPO": "CAPO I – DEL GIUDICE", "SEZIONE": "SEZIONE I – Della giurisdizione e della competenza in generale"}
    assert all(units[n] == opening for n in ("1", "2", "3", "4", "5"))
    assert units["6"] == {**opening, "SEZIONE": "SEZIONE II – Della competenza per materia e per valore"}
    assert units["163"] == {"LIBRO": "LIBRO SECONDO – DEL PROCESSO DI COGNIZIONE"}
    assert {w for _, w in result["unplaced"]} == {"heading-only Section"}
    assert result["headings"]["sections"] == 5 and result["headings"]["named_opening"] == 4
    assert result["headings"]["matched"] == 5                   # TITOLO II matches the unit art. 69 is in
    rows = list(csv.reader(open(al.write_csvs([result], str(tmp_path))["units.csv"], encoding="utf-8")))
    assert [r[3] for r in rows[1:] if r[1] == "SEZIONE" and r[2].startswith("SEZIONE I –")] == ["1", "2", "3", "4", "5"]
    lines = al.census_lines(result)
    assert any(ln.startswith("- heading-only Sections (no article number, a marker line): 5, matched to a unit") for ln in lines)


def test_without_heading_only_sections_the_opening_units_say_so():
    units = al.analyse([_doc(_cpc_live(headings=False))])[0]["units"]
    assert units["1"] == {"LIBRO": "LIBRO PRIMO (intestazione non nel testo)",
                          "TITOLO": "TITOLO I (intestazione non nel testo)",
                          "CAPO": "CAPO I (intestazione non nel testo)",
                          "SEZIONE": "SEZIONE I (intestazione non nel testo)"}


def test_ordinals():
    assert [al.ordinal(n) for n in ("II", "IV", "XIV", "SECONDO", "PRIMA", "UNICO", "3", "PRELIMINARE", "II-BIS")] == \
        [2, 4, 14, 2, 1, 1, 3, None, None]
    assert al.ordinal_text(1, "SECONDO") == "PRIMO" and al.ordinal_text(3, "IV") == "III"


# c.c. with the preleggi as prel- Sections: their own run, never merged with the code's articles 1-10
PRELEGGI = [("prel-1", "DISPOSIZIONI SULLA LEGGE IN GENERALE CAPO I – DELLE FONTI DEL DIRITTO Sono fonti del diritto: 1) "
                       "le leggi; 2) i regolamenti; 3) gli usi."),
            ("prel-2", "La formazione delle leggi è regolata da leggi di carattere costituzionale."),
            ("prel-9", "Gli usi pubblicati nelle raccolte si presumono esistenti. CAPO II – DELL'APPLICAZIONE DELLA "
                       "LEGGE IN GENERALE"),
            ("prel-10", "Le leggi divengono obbligatorie nel decimoquinto giorno successivo alla pubblicazione."),
            ("prel-31", "Le leggi e gli atti di uno Stato estero non possono avere effetto se contrari all'ordine "
                        "pubblico."),
            ("1", "LIBRO PRIMO – DELLE PERSONE E DELLA FAMIGLIA TITOLO I – DELLE PERSONE FISICHE La capacità giuridica "
                  "si acquista dal momento della nascita.")]
PRELEGGI += [(str(n), f"Testo dell'articolo {n}.") for n in range(2, 11)]
PRELEGGI[-1] = ("10", "Testo dell'articolo 10, ai sensi dell'art. 9. TITOLO II – DELLE PERSONE GIURIDICHE CAPO I – "
                      "DISPOSIZIONI GENERALI")
PRELEGGI += [("11", "Le province e i comuni sono persone giuridiche.")]


def _preleggi_sections():
    """In a scrambled NEXT order (the code before the preleggi): the names decide."""
    order = PRELEGGI[5:] + PRELEGGI[:5]
    return [{"id": f"p{PRELEGGI.index(x)}", "name": f"{x[0]}.0.0" if not x[0].startswith("prel") else x[0], "text": x[1],
             "next": f"p{PRELEGGI.index(order[i + 1])}" if i + 1 < len(order) else None} for i, x in enumerate(order)]


def test_the_preleggi_are_their_own_run_and_cc_1_to_10_the_code_s_own():
    doc = _doc(_preleggi_sections(), "Codice Civile 2025", "c.c.")
    assert doc["runs"][:6] == [1, 1, 1, 1, 1, 0] and doc["run_names"][1] == "disposizioni sulla legge in generale"
    result = al.analyse([doc])[0]
    units = result["units"]
    code = {"LIBRO": "LIBRO PRIMO – DELLE PERSONE E DELLA FAMIGLIA", "TITOLO": "TITOLO I – DELLE PERSONE FISICHE"}
    assert all(units[str(n)] == code for n in range(1, 11))
    assert units["11"]["CAPO"] == "CAPO I – DISPOSIZIONI GENERALI"
    assert units["1 [disposizioni sulla legge in generale]"] == {"CAPO": "CAPO I – DELLE FONTI DEL DIRITTO"}
    assert units["10 [disposizioni sulla legge in generale]"] == {"CAPO": "CAPO II – DELL'APPLICAZIONE DELLA LEGGE IN "
                                                                          "GENERALE"}
    by = {(r["from_id"], r["to_article"]): r for r in result["references"]}
    assert by[("p14", "9")]["to_id"] == "p13"           # c.c. 10's "art. 9": the code's own, not the preleggi's


def test_order_by_number_suffix_whole_node_and_next_only_for_ties():
    names = ["1159-ter", "1159.1.0", "1159", "1159-bis", "1160", "1159bis.1.0", "1158", "att-3", "prel-2", "1159.0.0",
             "intestazione"]
    sections = [{"id": n, "name": n, "text": "", "next": None} for n in names]
    sections += [{"id": "tie-b", "name": "1161", "text": "", "next": None},
                 {"id": "tie-a", "name": "1161", "text": "", "next": "tie-b"}]       # on the chain: tie-a first
    sections[-4]["next"] = "intestazione"                                              # after 1159.0.0 on the chain
    walk, off, runs, names_ = al.arrange(sections)
    assert [w["id"] for w in walk] == ["prel-2", "1158", "1159", "1159.0.0", "intestazione", "1159.1.0", "1159-bis",
                                       "1159bis.1.0", "1159-ter", "1160", "tie-a", "tie-b", "att-3"]
    assert runs[0] == 1 and runs[-1] == 2 and set(runs[1:-1]) == {0} and not off
    assert al.article_of("1159bis.1.0") == "1159-bis" and al.article_of("att-38_1") == "38"


def test_the_duplicate_names_are_an_ingestion_defect():
    sections = [{"id": n, "name": n, "text": "Testo.", "next": None} for n in ("att-38", "att-38_1", "5", "5_1", "7_1")]
    doc = _doc(sections, "Codice Civile 2025", "c.c.")
    result = al.analyse([doc])[0]
    assert result["duplicates"] == [("5", "5_1"), ("att-38", "att-38_1")]              # 7_1 has no original
    assert result["pieces"]["38 [disposizioni di attuazione]"] == ["att-38", "att-38_1"]
    assert "- ingestion defect: Section names duplicated with a _N suffix: 2 (5 / 5_1, att-38 / att-38_1)" in \
        al.census_lines(result)


# --- v53: the comma clause before the code, the resolution, the stale edges ---------------------------------------------

# 2044 c.c. as the corpus holds it (the live graph had it pointing to c.c. 52, 55 and 7)
ART_2044 = ("Non è responsabile chi cagiona il danno per legittima difesa di sé o di altri. Nei casi di cui all'articolo 52, "
            "commi secondo, terzo e quarto, del codice penale, la responsabilità di chi ha compiuto il fatto è esclusa. Nel "
            "caso di cui all'articolo 55, secondo comma, del codice penale, al danneggiato è dovuta una indennità la cui "
            "misura è rimessa all'equo apprezzamento del giudice, tenuto altresì conto della gravità, delle modalità "
            "realizzative e del contributo causale della condotta posta in essere dal danneggiato. Si applica, in quanto "
            "compatibile, l'art. 7, comma 1, L. 26 aprile 2019, n. 36.")


def test_the_comma_clause_is_skipped_before_the_code_or_act_2044_cc():
    refs = [(r["matched"], r["to_code"], r["pattern"], r["resolution"]) for r in al.references(ART_2044, "c.c.")]
    assert refs == [("articolo 52", "c.p.", "other code", "explicit_code"),
                    ("articolo 55", "c.p.", "other code", "explicit_code"),
                    ("art. 7", "other act", "other code", "explicit_code")]
    broken = ART_2044.replace("articolo 52, commi", "articolo 52,  \n  commi").replace(", del codice penale, la", ",\n  del "
                                                                                       "codice penale, la")
    assert al.references(broken, "c.c.")[0]["to_code"] == "c.p."                    # a line break, double spaces


def test_2044_resolves_52_and_55_to_the_penal_code_and_7_nowhere():
    def doc(doc_id, name, code, texts):
        sections = [{"id": f"{code}{n}", "name": f"{n}.0.0", "text": t, "next": None} for n, t in texts]
        walk, off, runs, names = al.arrange(sections)
        return {"doc": {"id": doc_id, "name": name}, "code": code, "sections": sections, "walk": walk, "off": off,
                "runs": runs, "run_names": names}
    cc = doc("dcc", "Codice Civile 2025", "c.c.", [("7", "Ogni persona ha diritto al nome."), ("52", "Testo."),
                                                    ("55", "Testo."), ("2044", ART_2044)])
    cp = doc("dcp", "Codice Penale 2026", "c.p.", [("52", "Difesa legittima."), ("55", "Eccesso colposo.")])
    results = al.analyse([cc, cp])
    refs = [(r["to_article"], r["to_code"], r["resolved"], r.get("to_id"), r["resolution"])
            for r in results[0]["references"] if r["from"] == "2044"]
    assert refs == [("52", "c.p.", True, "c.p.52", "explicit_code"), ("55", "c.p.", True, "c.p.55", "explicit_code"),
                    ("7", "other act", False, None, "explicit_code")]           # never 7 c.c.
    edges = al.graph_rows(results)["refers_to"]
    assert {(e["from"], e["to"], e["resolution"]) for e in edges} == {("c.c.2044", "c.p.52", "explicit_code"),
                                                                      ("c.c.2044", "c.p.55", "explicit_code")}


def test_resolution_labels_and_a_clause_without_its_keyword_is_not_skipped():
    def one(text, own="c.c."):
        r = al.references(text, own)[0]
        return r["to_code"], r["resolution"]
    assert one("ai sensi dell'articolo 1032 del presente codice") == ("c.c.", "explicit_code")
    assert one("ai sensi dell'articolo 1032 c.c.") == ("c.c.", "explicit_code")
    assert one("ai sensi dell'articolo 1032, primo comma, e poi") == ("c.c.", "same_code")
    assert one("dell'art. 3, lettera b), del d.lgs. 28/2010") == ("other act", "explicit_code")
    assert one("le disposizioni dell'articolo precedente") == ("c.c.", "relative")
    assert al.skip_clause(", e poi del codice penale") == ", e poi del codice penale"      # no keyword: nothing skipped


class StaleSession(WriteSession):
    """WriteSession that also holds the graph's edges of this source and deletes by id."""
    def __init__(self, edges):
        super().__init__()
        self.edges, self.deleted = list(edges), []

    def run(self, query, rows=(), **params):
        if query == al.FIND_STALE:
            class Found:
                def __init__(self, data):
                    self._data = data

                def data(self):
                    return self._data
            return Found([e for e in self.edges if e["doc"] in params["docs"]])
        if query == al.DELETE_STALE:
            gone = [e for e in self.edges if e["edge"] in params["edges"]]
            self.deleted += [e["edge"] for e in gone]
            self.edges = [e for e in self.edges if e not in gone]

            class Counters:
                relationships_deleted = len(gone)

            class Result:
                def consume(self):
                    class Summary:
                        counters = Counters()
                    return Summary()
            return Result()
        return super().run(query, rows, **params)


def test_the_stale_edges_are_found_read_only_and_deleted_by_id_in_batches(monkeypatch):
    monkeypatch.setattr(al, "BATCH", 1)
    rows = {"refers_to": [{"from": "c.c.2044", "to": "c.p.52"}]}
    graph = [{"edge": "x1", "from": "c.c.2044", "to": "c.c.52", "pattern": "word", "text": "articolo 52", "doc": "dcc"},
             {"edge": "x2", "from": "c.c.2044", "to": "c.c.7", "pattern": "abbr", "text": "art. 7", "doc": "dcc"},
             {"edge": "x3", "from": "c.c.2044", "to": "c.p.52", "pattern": "other code", "text": "articolo 52", "doc": "dcc"},
             {"edge": "x4", "from": "p1", "to": "p2", "pattern": "word", "text": "articolo 2", "doc": "dcpc"}]
    session = StaleSession(graph)
    stale = al.stale_edges(session, ["dcc"], rows)
    assert [e["edge"] for e in stale] == ["x1", "x2"]           # kept: the new pair; another document's never looked at
    assert "DELETE" not in al.FIND_STALE and "MATCH" in al.FIND_STALE
    assert al.delete_stale(session, stale) == 2 and session.deleted == ["x1", "x2"]
    assert "source: $source" in al.DELETE_STALE and "e.resolution = r.resolution" in al.WRITE_REFERS_TO
    # v60h: additive -- an existing edge's resolution is set only where it has none, never overwritten
    assert "ON MATCH SET e.resolution = coalesce(e.resolution, r.resolution)" in al.WRITE_REFERS_TO


def test_compare_reports_the_targets_that_changed_and_the_ones_no_longer_resolved(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("from article,to article,pattern,matched text\n2044 c.c.,52 c.c.,word,articolo 52\n"
                   "2044 c.c.,7 c.c.,abbr,art. 7\n1052 c.c.,1051 c.c.,relative,articolo precedente\n"
                   "5 c.p.,1 c.p.,abbr,art. 1\n", encoding="utf-8")            # a code not read now: left out
    results = [{"code": "c.c.", "references": [
        {"from": "2044", "matched": "articolo 52", "resolved": True, "to_article": "52", "to_code": "c.p.",
         "resolution": "explicit_code"},
        {"from": "2044", "matched": "art. 7", "resolved": False, "to_article": "7", "to_code": "other act",
         "resolution": "explicit_code"},
        {"from": "1052", "matched": "articolo precedente", "resolved": True, "to_article": "1051", "to_code": "c.c.",
         "resolution": "relative"}]}]
    diff = al.compare(results, str(old), str(tmp_path))
    assert (diff["changed"], diff["unresolved"], diff["new"]) == (1, 1, 1)
    assert [(r["kind"], r["old"], r["new"]) for r in diff["samples"]] == [("changed target", "52 c.c.", "52 c.p."),
                                                                         ("now unresolved", "7 c.c.", "-")]
    assert list(csv.reader(open(diff["path"], encoding="utf-8")))[0][0] == "kind"


# --- v60h: a document whose headings are in title case (Codice del processo amministrativo 2025), any type, by its id ----

CPA = {
    "1": "Libro I Disposizioni generali Titolo I Principi e organi della giurisdizione amministrativa Capo I Principi "
         "generali Art. 1 Effettività La giurisdizione amministrativa assicura una tutela piena ed effettiva.",
    "2": "Il processo amministrativo attua i principi della parità delle parti. Capo II Organi della giurisdizione "
         "amministrativa",
    "4": "Sono organi della giurisdizione amministrativa i tribunali amministrativi regionali e il Consiglio di Stato.",
    "29": "Titolo III Azioni di cognizione Capo I Azione di annullamento 1. L'azione di annullamento si propone nel termine "
          "di decadenza di sessanta giorni, salvo quanto previsto dal Capo II del presente Titolo.",
    "30": "Si applicano le disposizioni di cui al Titolo IV e l'articolo 29.",
}


def _cpa_doc():
    order = list(CPA)
    sections = [{"id": f"a{k}", "name": k, "text": CPA[k], "next": f"a{order[i + 1]}" if i + 1 < len(order) else None}
                for i, k in enumerate(order)]
    walk, off, runs, names = al.arrange(sections)
    return {"doc": {"id": "LEGAL_DOC::cba1e3", "name": "Codice del processo amministrativo 2025"}, "code": "c.p.a.",
            "sections": sections, "walk": walk, "off": off, "runs": runs, "run_names": names,
            "markers": al.marker_style(CPA.values())}


def test_title_case_headings_are_read_where_a_heading_stands_never_in_a_reference():
    assert al.marker_style(CPA.values()) == "title case" and al.marker_style(TEXTS.values()) == "capitals"
    found = al.markers(CPA["1"], title=True)
    assert [(m["level"], m["number"], m["name"], m["lead"]) for m in found] == [
        ("LIBRO", "I", "Disposizioni generali", True),
        ("TITOLO", "I", "Principi e organi della giurisdizione amministrativa", True),
        ("CAPO", "I", "Principi generali", True)]
    assert [(m["level"], m["name"], m["lead"]) for m in al.markers(CPA["2"], title=True)] == [
        ("CAPO", "Organi della giurisdizione amministrativa", False)]               # at the end: opens the next article
    assert [m["name"] for m in al.markers(CPA["29"], title=True)] == ["Azioni di cognizione", "Azione di annullamento"]
    assert al.markers(CPA["30"], title=True) == [] and al.markers(CPA["30"]) == []  # «di cui al Titolo IV»: a reference
    assert al.markers(CPA["1"]) == []                                               # never read in capitals mode


def test_the_cpa_is_placed_and_its_rows_counted_for_the_write():
    result = al.analyse([_cpa_doc()])[0]
    assert result["markers"] == "title case"
    assert result["units"]["1"] == {"LIBRO": "LIBRO I – Disposizioni generali",
                                    "TITOLO": "TITOLO I – Principi e organi della giurisdizione amministrativa",
                                    "CAPO": "CAPO I – Principi generali"}
    assert result["units"]["4"]["CAPO"] == "CAPO II – Organi della giurisdizione amministrativa"
    assert result["units"]["29"]["TITOLO"] == "TITOLO III – Azioni di cognizione" and result["units"]["30"]["CAPO"] == \
        "CAPO I – Azione di annullamento"
    rows = al.graph_rows([result])
    lines = "\n".join(al.rows_census(rows))
    assert "units 6 (LIBRO 1, TITOLO 2, CAPO 3)" in lines and f"HAS_ARTICLE {len(rows['has_article'])}" in lines
    assert "REFERS_TO 1 (explicit_code 0, same_code 1, relative 0)" in lines           # 30 -> 29
    assert "- markers: title case" in "\n".join(al.census_lines(result))


def test_the_write_is_additive_and_stamped():
    for query in (al.WRITE_UNITS, al.WRITE_CHILD_OF, al.WRITE_HAS_ARTICLE, al.WRITE_REFERS_TO):
        assert "MERGE" in query and "DELETE" not in query and "ON CREATE SET" in query
        assert "$computed_at" in query and "$source" in query
        rest = query.replace("ON CREATE SET", "").replace("ON MATCH SET e.resolution = coalesce", "")
        assert "SET" not in rest                                  # nothing set on what exists, but a missing resolution
    rows = al.graph_rows(al.analyse([_cpa_doc()]))
    session = WriteSession()
    first = al.write_graph(session, rows, "2026-11-10T00:00:00+00:00")
    assert first["units"]["created"] == 6 and all(n["present"] == 0 for n in first.values())


def test_documents_are_read_by_id_or_type_and_the_unitless_listed_read_only():
    for query in (al.QUERY_DOCS_TYPES, al.QUERY_DOC_IDS, al.QUERY_UNITLESS):
        assert not re.search(r"\b(?:CREATE|MERGE|SET|DELETE|REMOVE|DETACH|CALL)\b", query, re.IGNORECASE)

    class Session:
        def run(self, query, **params):
            assert query == al.QUERY_UNITLESS and params["types"] == ["primary", "special"]
            rows = [{"id": "LEGAL_DOC::cba1e3", "name": "Codice del processo amministrativo 2025", "type": "special",
                     "articles": 137},
                    {"id": "x", "name": "appunti_rossi", "type": "special", "articles": 3}]
            return type("R", (), {"data": lambda self: rows})()
    found = al.unitless(Session(), ["primary", "special"])
    assert [d["id"] for d in found] == ["LEGAL_DOC::cba1e3"]                      # a normative name only
