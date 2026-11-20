"""v79.15.1 build B (brief item 4) -- the Q&A, §4's prose and §7's written objections reviewed sentence by sentence
(src/rag/analysis_step.py: ProseReview, prose_scope, outside, anaphoric, orphans, prose_counters; lawyer_method:
orphans_out, answers_block, compose's wiring; interactive_flow.legacy_plan_logged's sums). Synthetic views only
(invented names; statutory article wording); no model -- every model path and every outbound socket refused
(tests/test_v79_15_0_analysis.py's sealing, re-exported).

1. a rule sentence with no verified twin is replaced by the first verified affermazione on its article, verbatim
   (qa_sentences_replaced); with none it is dropped (qa_sentences_dropped / norms_ / objections_) and so is the next
   sentence when it opens with one of the brief's anaphoric connectives (qa_sentences_dropped_dependent)
2. a sentence citing an article the binding excluded or the fit removed is dropped (prose_excluded_article_dropped)
3. a sentence citing an article outside the fitted set that the user did not name is dropped
   (prose_unfitted_article_dropped); a procedural code's article never counts as outside
4. a sentence the answer-wide checks removed takes its anaphoric follower with it (orphans)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_v79_15_0_analysis as T  # noqa: E402  (its autouse sealing fixture is re-exported below)
from src.rag import analysis_step as A  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402

_sealed = T._sealed

ST_1385 = {"testo": "Se inadempiente è la parte che ha ricevuto la caparra, l'altra può recedere dal contratto ed "
                    "esigere il doppio della caparra.", "articoli": ["1385 c.c."], "cita_norma": ["A1385cc.2"]}
ST_1385_B = {"testo": "L'art. 1385 c.c. consente di ritenere la caparra.", "articoli": ["1385 c.c."]}
LINE = {"articolo": "1385 c.c.", "elemento_in_gioco": "inadempiente è invece la parte che l'ha ricevuta",
        "tesi": "Anna Corsi può esigere il doppio della caparra.", "stato": A.VERIFIED, "cita_fatti": ["S2"]}
VIEW = {"status": "ok", "analysis": {"linea_principale": LINE, "linea_subordinata": None, "elementi": {},
                                     "frame_charge": "1385 c.c."},
        "sentences": {"S1": "Anna Corsi ha versato a Ettore Bassi una caparra.",
                      "S2": "Anna Corsi vuole esigere il doppio della caparra."},
        "rubriche": {"1385 c.c.": "Caparra confirmatoria", "1453 c.c.": "Risolubilità del contratto per "
                                                                         "inadempimento"},
        "remedies": ["1453 c.c."], "statements": [ST_1385, ST_1385_B]}

RULE_1385 = ("L'art. 1385 c.c. prevede che, se la parte non inadempiente preferisce domandare l'esecuzione, il "
             "risarcimento del danno è regolato dalle norme generali.")
RULE_1453 = "L'art. 1453 c.c. prevede che il contraente possa chiedere la risoluzione del contratto."
TWIN_1385 = ("L'art. 1385 c.c. consente alla parte non inadempiente di recedere dal contratto ed esigere il doppio "
             "della "
             "caparra ricevuta dall'altra parte.")


def _review(site=A.SITE_QA, excluded=(), named=(), view=None):
    v = view or VIEW
    events = []
    return A.ProseReview(v, A.prose_scope(v, excluded, named), site, events), events


def _run(sentences, **kw):
    review, events = _review(**kw)
    return [x for x in (review(s) for s in sentences) if x], events


# --- 1. replaced, not deleted; the anaphoric follower -----------------------------------------------------------------

def test_a_rule_with_no_twin_is_replaced_by_the_first_verified_statement_verbatim():
    out, events = _run([RULE_1385, "Ai sensi del primo comma, la caparra si imputa alla prestazione."])
    assert out == [ST_1385["testo"][:-1] + " (art. 1385 c.c.).",            # the first statement, its article added
                   "Ai sensi del primo comma, la caparra si imputa alla prestazione."]   # a replacement keeps it
    assert [e["esito"] for e in events] == [A.REPLACED] and events[0]["con"] == ST_1385["testo"]
    assert A.prose_counters(events)["qa_sentences_replaced"] == 1


def test_a_statement_naming_its_article_is_printed_without_a_second_label():
    view = dict(VIEW, statements=[ST_1385_B])
    out, _ = _run([RULE_1385], view=view)
    assert out == ["L'art. 1385 c.c. consente di ritenere la caparra."]


def test_a_sentence_with_a_verified_twin_is_kept_as_written():
    out, events = _run([TWIN_1385])
    assert out == [TWIN_1385] and events == []


def test_a_rule_with_no_statement_for_its_article_is_dropped_with_its_anaphoric_followers():
    follow = ["Inoltre, il risarcimento spetta sempre.", "Ai sensi del medesimo articolo, la domanda preclude "
              "l'adempimento.", "Anna Corsi ha firmato il 3 marzo.", "Tuttavia, il termine è breve."]
    out, events = _run([RULE_1453] + follow)
    assert out == ["Anna Corsi ha firmato il 3 marzo.", "Tuttavia, il termine è breve."]
    assert [e["esito"] for e in events] == [A.DROPPED, A.DEPENDENT, A.DEPENDENT]
    c = A.prose_counters(events)
    assert c["qa_sentences_dropped"] == 1 and c["qa_sentences_dropped_dependent"] == 2


def test_the_brief_s_anaphoric_openers_and_only_them():
    for x in ("Inoltre, x.", "Pertanto x.", "Ai sensi del primo comma, x.", "Ai sensi del secondo comma x.",
              "Ai sensi del medesimo articolo x.", "Tale norma x.", "Questo x.", "Questa x.", "Essa x.", "Esso x.",
              "In particolare, x.", "Infatti x.", "Tuttavia x.", "«Inoltre, x.»", "- Pertanto x.", "**Infatti** x."):
        assert A.anaphoric(x), x
    for x in ("Inoltrare la domanda.", "Talento.", "Questore.", "Ai sensi dell'art. 1385 c.c. x.", "Anna Corsi x.",
              "Essere x.", "In parte x."):
        assert not A.anaphoric(x), x


def test_one_statement_replaces_two_sentences_once():
    out, events = _run([RULE_1385, "L'art. 1385 c.c. stabilisce che la caparra è sempre dovuta."])
    assert len(out) == 1 and [e["esito"] for e in events] == [A.REPLACED, A.REPLACED]


def test_the_site_decides_the_dropped_counter():
    for site, key in ((A.SITE_QA, "qa_sentences_dropped"), (A.SITE_NORMS, "norms_sentences_dropped"),
                      (A.SITE_OBJECTIONS, "objections_sentences_dropped")):
        _, events = _run([RULE_1453], site=site)
        c = A.prose_counters(events)
        assert c[key] == 1 and sum(c.values()) == 1


# --- 2. excluded; 3. unfitted -----------------------------------------------------------------------------------------

def test_a_sentence_on_an_excluded_article_is_dropped_with_its_follower():
    out, events = _run(["Il preliminare è annullabile per dolo ai sensi dell'art. 1439 c.c.",
                        "Pertanto, Anna Corsi può chiedere l'annullamento.", "Anna Corsi ha pagato."],
                       excluded=["1439 c.c."], named=["Il dolo dell'art. 1439 c.c."])
    assert out == ["Anna Corsi ha pagato."]          # excluded wins over named
    assert [e["esito"] for e in events] == [A.ARTICLE_EXCLUDED, A.DEPENDENT]
    assert A.prose_counters(events)["prose_excluded_article_dropped"] == 1


def test_a_sentence_on_an_unfitted_article_is_dropped_unless_the_user_named_it():
    s = "La responsabilità del debitore segue l'art. 1218 c.c."
    out, events = _run([s])
    assert out == [] and events[0]["esito"] == A.ARTICLE_UNFITTED and events[0]["articoli"] == ["1218 c.c."]
    out, events = _run([s], named=["Anna Corsi invoca l'art. 1218 c.c."])
    assert out == [s] and events == []


def test_the_fitted_set_is_the_inputs_norms_and_the_named_remedies():
    scope = A.prose_scope(VIEW, (), (), ["62-bis c.p.", "640 c.p."])
    assert scope["fitted"] == ["1385 c.c.", "1453 c.c.", "62-bis c.p."]      # 640 c.p. is no remedy: never added
    assert "1385 c.c." in scope["named"]                                        # the frame's verified charge
    assert A.outside(["1453 c.c."], scope) == ("", [])
    assert A.outside(["409 c.p.p."], scope) == ("", [])                         # a procedural code: never outside
    assert A.outside(["2043 c.c."], scope) == (A.ARTICLE_UNFITTED, ["2043 c.c."])


def test_no_input_no_outside_check():
    view = A.view(None, None)
    scope = A.prose_scope(view, ["1439 c.c."])
    assert scope["fitted"] is None
    assert A.outside(["2043 c.c."], scope) == ("", [])
    assert A.outside(["1439 c.c."], scope) == (A.ARTICLE_EXCLUDED, ["1439 c.c."])


def test_a_norms_block_on_an_unfitted_article_goes_whole_and_a_numberless_rule_reads_its_heading():
    text = ("**Art. 1218 c.c. — Responsabilità del debitore**\n\nIl debitore risponde del danno. La norma prevede "
            "il risarcimento.\n\n**Art. 1385 c.c. — Caparra confirmatoria**\n\nLa norma prevede che la caparra sia "
            "sempre trattenuta.")
    review, events = _review(site=A.SITE_NORMS)
    out = M._by_sentence(text, review)
    assert "1218" not in out and "Il debitore risponde" not in out
    assert out.endswith("**Art. 1385 c.c. — Caparra confirmatoria**\n\n" + ST_1385["testo"][:-1] +
                        " (art. 1385 c.c.).")
    assert [e["esito"] for e in events] == [A.ARTICLE_UNFITTED] * 3 + [A.REPLACED]


# --- 4. a sentence the checks removed: its follower goes too ----------------------------------------------------------

def test_a_sentence_the_checks_removed_takes_its_anaphoric_follower():
    before = ["A x.", "B prevede y.", "Inoltre z.", "In particolare w.", "C v."]
    after = ["A x.", "Inoltre z.", "In particolare w.", "C v."]
    assert A.orphans(before, after) == {1, 2}
    assert A.orphans(before, ["A x.", "B prevede (riscritto) y.", "Inoltre z.", "In particolare w.", "C v."]) == set()
    events = []
    out = M.orphans_out("A x. B prevede y. Inoltre z. C v.", "A x. Inoltre z. C v.", events, A.SITE_NORMS)
    assert out == "A x. C v." and events[0]["esito"] == A.DEPENDENT and events[0]["dopo"] == "controlli"


# --- the Q&A path end to end (answers_block) and the flow's sums ------------------------------------------------------

def test_answers_block_reviews_the_writers_answer_before_the_checks():
    q = "Anna Corsi può chiedere il risarcimento?"
    written = f"{RULE_1453} Inoltre, il risarcimento spetta sempre. {RULE_1385} {TWIN_1385}"
    ctx = {"asked": [q], "case": "Anna Corsi vuole recedere ai sensi dell'art. 1385 c.c."}
    seen = []

    def checked(text):
        seen.append(text)
        return text
    block, _ = M.answers_block(ctx, [(q, written)], "", [], 0, {}, "Anna Corsi", checked, {}, verified=VIEW)
    assert RULE_1453 not in seen[0] and RULE_1385 not in seen[0]       # reviewed first: the checks see the result
    answer = next(x for x in block.splitlines() if x.startswith("1. "))
    assert answer.endswith("— " + ST_1385["testo"][:-1] + " (art. 1385 c.c.). " + TWIN_1385)
    assert ctx["qa_sentences_dropped"] == 1 and ctx["qa_sentences_dropped_dependent"] == 1
    assert ctx["qa_sentences_replaced"] == 1


def test_answers_block_drops_the_follower_of_a_sentence_the_checks_removed():
    q = "Anna Corsi può chiedere il risarcimento?"
    written = f"Anna Corsi ha un diritto inventato. Inoltre, lo può cedere. {TWIN_1385}"
    ctx = {"asked": [q]}
    block, _ = M.answers_block(ctx, [(q, written)], "", [], 0, {}, "Anna Corsi",
                               lambda t: t.replace("Anna Corsi ha un diritto inventato. ", ""), {}, verified=VIEW)
    answer = next(x for x in block.splitlines() if x.startswith("1. "))
    assert answer.endswith(f"— {TWIN_1385}") and "Inoltre" not in answer
    assert ctx["qa_sentences_dropped_dependent"] == 1


def test_the_flow_sums_every_prose_counter_over_the_clients():
    from src.rag import interactive_flow as flow
    state = {}
    ctxs = [{"qa_sentences_replaced": 1, "prose_unfitted_article_dropped": 2, "qa_sentences_dropped": 1},
            {"qa_sentences_replaced": 2, "objections_sentences_dropped": 1, "norms_sentences_dropped": 3}]
    flow.legacy_plan_logged(state, ctxs)
    c = state["block_counters"]
    assert c["qa_sentences_replaced"] == 3 and c["prose_unfitted_article_dropped"] == 2
    assert c["objections_sentences_dropped"] == 1 and c["qa_sentences_dropped"] == 1
    assert c["norms_sentences_dropped"] == 3 and c["qa_sentences_dropped_dependent"] == 0
    assert set(A.PROSE_COUNTERS) <= set(A.RENDER_COUNTERS)


# --- review fixes: an article cited by number alone; a repeated statement's follower; §7's objection never replaced --

def test_an_article_cited_by_number_alone_is_read_against_the_scope():
    # excluded by number alone; an unknown number outside the fitted set; a fitted one by number alone kept
    out, events = _run(["Ai sensi dell'art. 1439 il contratto è annullabile per dolo.",
                        "Pertanto, Anna Corsi può chiedere l'annullamento.",
                        "Gli artt. 1218 e 1223 regolano il danno.",
                        "Anna Corsi invoca l'art. 1385."], excluded=["1439 c.c."])
    assert out == ["Anna Corsi invoca l'art. 1385."]
    assert [(e["esito"], e["articoli"]) for e in events] == [
        (A.ARTICLE_EXCLUDED, ["1439 c.c."]), (A.DEPENDENT, []), (A.ARTICLE_UNFITTED, ["art. 1218", "art. 1223"])]


def test_a_number_tied_to_a_named_source_is_not_a_bare_citation():
    for s in ("L'art. 5 della legge 104/1992 riconosce i permessi.", "L'art. 2 Cost. tutela la persona.",
              "L'art. 18 dello Statuto dei lavoratori è citato."):
        assert A.bare_cited(s, A._cited(s), A.prose_scope(VIEW)) == [], s
    # a comma clause between the number and its code: the article read whole
    assert A._cited("Ai sensi dell'art. 1385 secondo comma c.c. la parte può recedere.") == ["1385 c.c."]
    assert A.bare_cited("Ai sensi dell'art. 1385, secondo comma, la parte recede.", [], A.prose_scope(VIEW)) == \
        ["1385 c.c."]


def test_a_rule_on_an_article_cited_by_number_alone_is_replaced():
    out, events = _run(["L'art. 1385 prevede che la caparra vada sempre restituita."])
    assert out == [ST_1385["testo"][:-1] + " (art. 1385 c.c.)."]
    assert [(e["esito"], e["articoli"]) for e in events] == [(A.REPLACED, ["1385 c.c."])]


def test_a_repeated_statement_prints_nothing_and_takes_its_anaphoric_follower():
    out, events = _run([RULE_1385, "L'art. 1385 c.c. stabilisce che la caparra è sempre dovuta.",
                        "Inoltre, il giudice può ridurla.", "Anna Corsi ha pagato."])
    assert out == [ST_1385["testo"][:-1] + " (art. 1385 c.c.).", "Anna Corsi ha pagato."]
    assert [e["esito"] for e in events] == [A.REPLACED, A.REPLACED, A.DEPENDENT]
    assert events[1].get("duplicato") is True


def test_the_objection_itself_is_never_replaced_by_a_statement_of_law():
    events = []
    review = A.ProseReview(VIEW, A.prose_scope(VIEW), A.SITE_OBJECTIONS, events, replace=False)
    assert review(RULE_1385) == ""
    assert [e["esito"] for e in events] == [A.DROPPED]
    assert A.prose_counters(events)["objections_sentences_dropped"] == 1
    # the answer (replace=True, the default) is replaced as before
    events = []
    review = A.ProseReview(VIEW, A.prose_scope(VIEW), A.SITE_OBJECTIONS, events)
    assert review(RULE_1385).startswith("Se inadempiente") and events[0]["esito"] == A.REPLACED
