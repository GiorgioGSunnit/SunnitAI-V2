"""Lever 6 on be2 (prepared 13 Nov 2026): the chunk answers that never end are stopped by the model server.

What 13 Nov showed (runbook 3f): about one chunk answer in twenty never ends. Its content is complete, then the
model writes white space inside the list of references until the worker gives up (40 s, up to three times).
That waiting is about 30% of a worker's time, and a quarter of what the shared model server writes.

ANALISI_STOP_REPETITION=1 in a worker's environment file adds one key to that worker's chunk requests, vLLM's
`repetition_detection` (a pattern of up to 20 tokens repeated 10 times ends the answer; an answer that does not
repeat is never touched). The endless answer then comes back in the time of a normal one, cut inside its JSON,
and the library cannot read it. The worker asks again, as it does today after a timeout, so a section still gets
the full answer whenever a fresh request gives one. Only when no attempt ends by itself does it keep the content
of the cut answer (summary, quote, search text, tag and the references written whole) where today it falls back
to the plain-text analysis.

Without the setting nothing changes: every edit is guarded by it.

One file, lex_package/analisi_parallel.py, checked against the checksum of be2's copy after lever 5.

    python3 patch_ingest_lever6.py --selftest                   # the inserted functions; touches no file
    sudo python3 /home/sunnit/patch_ingest_lever6.py --check    # checksum only, changes nothing
    sudo python3 /home/sunnit/patch_ingest_lever6.py            # apply (prints what changed)
    sudo python3 /home/sunnit/patch_ingest_lever6.py --undo     # restore the newest backup

The rehearsal on be2, before applying (read-only): probe_runaway.py --lever6, with this file beside it.
Nothing else in the file is touched; no service is restarted by this script.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import datetime as dt

PATH = "/opt/sunnitai-be/src/be/src/lex_package/analisi_parallel.py"
MD5 = "c2a57da447fc46b7349ec6023fcbe958"          # after lever 5 (11 Nov 2026)
SETTING = "ANALISI_STOP_REPETITION"
HELPERS = r'''# --- Lever 6 (Nov 2026): the stop on repetition -------------------------------------------------------------
# About one chunk answer in twenty never ends: its content is complete, then the model writes white space
# inside the list of references until the worker gives up. ANALISI_STOP_REPETITION=1 in a worker's environment
# file asks the model server to end an answer when a short pattern repeats (vLLM's repetition_detection; an
# answer that does not repeat is never touched). That answer arrives cut inside its JSON and the library cannot
# read it: the worker asks again, as after a timeout, and keeps the cut answer's content only when no attempt
# ends by itself. Without the setting nothing changes.
_REPETITION_STOP = {"max_pattern_size": 20, "min_count": 10}    # the values measured on 13 Nov 2026
_CUT_CONTENT_DONE = re.compile(r'"pattern_type"\s*:\s*(null|"[^"]*")')    # the content fields come before it


def _stop_on_repetition() -> bool:
    return os.environ.get("ANALISI_STOP_REPETITION", "").strip().lower() in ("1", "true", "yes")


def _with_repetition_stop(llm_base):
    """The model object whose requests carry the stop, when the setting is on: a copy, the factory's object
    stays as it was built."""
    if not _stop_on_repetition():
        return llm_base
    extra = {**(getattr(llm_base, "extra_body", None) or {}), "repetition_detection": _REPETITION_STOP}
    return llm_base.model_copy(update={"extra_body": extra})


def _cut_answer_text(exc):
    """The text of an answer the library could not read as JSON (its validation error carries it), else None."""
    try:
        first = exc.errors()[0]
        text = first["input"] if first["type"] == "json_invalid" else None
    except Exception:
        return None
    return text if isinstance(text, str) else None


def _salvage_cut_answer(text):
    """An Analisi_Paragrafo from an answer cut after its content: the fields up to pattern_type and the
    references written whole. None when the cut came before the content was complete."""
    done = _CUT_CONTENT_DONE.search(text)
    if not done:
        return None
    try:
        data = json.loads(text[:done.end()] + "}")
        if not all(k in data for k in ("requirement", "core_text", "search_text")):
            return None
        refs, tail = [], text[done.end():]
        pos = tail.find("[") + 1                        # 0: the list was never opened
        while pos:
            while pos < len(tail) and tail[pos] in " \t\r\n,":
                pos += 1
            try:
                ref, pos = json.JSONDecoder().raw_decode(tail, pos)
            except ValueError:
                break                                   # white space to the end, or an entry cut half way
            if ref not in refs:
                refs.append(ref)
        data["riferimenti"] = refs or None
        return Analisi_Paragrafo.model_validate(data)
    except Exception:
        return None
'''
EDITS = [
    ("import logging\nimport os\nimport time\n", "import json\nimport logging\nimport os\nimport re\nimport time\n"),
    ("    return llm_base.with_structured_output(Analisi_Paragrafo)\n",
     "    return _with_repetition_stop(llm_base).with_structured_output(Analisi_Paragrafo)\n\n\n" + HELPERS),
    ("                for attempt, timeout_secs in enumerate(chunk_timeouts):\n",
     "                salvaged = None      # lever 6: the content of an answer the repetition stop cut\n"
     "                for attempt, timeout_secs in enumerate(chunk_timeouts):\n"),
    ("                    except Exception as e:\n"
     "                        logger.warning(\n"
     "                            \"Chunk %d/%d structured attempt %d/2 failed (%s)\",\n",
     "                    except Exception as e:\n"
     "                        cut = _cut_answer_text(e) if _stop_on_repetition() else None\n"
     "                        if cut is not None:\n"
     "                            # lever 6: the server's repetition stop ended this answer inside its JSON. Ask\n"
     "                            # again, as after a timeout; the content is kept for when no attempt ends by itself\n"
     "                            kept = _salvage_cut_answer(cut)\n"
     "                            if salvaged is None:\n"
     "                                salvaged = kept\n"
     "                            logger.warning(\n"
     "                                \"Chunk %d/%d structured attempt %d/2 cut by the repetition stop (content %s)\",\n"
     "                                i + 1, total_chunks, attempt + 1, \"complete\" if kept is not None else \"incomplete\",\n"
     "                            )\n"
     "                            continue\n"
     "                        logger.warning(\n"
     "                            \"Chunk %d/%d structured attempt %d/2 failed (%s)\",\n"),
    ("                # Attempt 2: plain text prompt, 30s timeout on fallback model\n",
     "                if salvaged is not None:\n"
     "                    # lever 6: no attempt ended by itself; the cut answer's content instead of the fallback\n"
     "                    call_times.append(time.time() - t_call)\n"
     "                    logger.info(\"Chunk %d/%d: content of the cut answer kept\", i + 1, total_chunks)\n"
     "                    return salvaged\n"
     "\n"
     "                # Attempt 2: plain text prompt, 30s timeout on fallback model\n"),
]
SUFFIX = ".bak_lever6_" + dt.date.today().strftime("%d%b").lower()


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def check(path=PATH):
    have = md5(path)
    same = have == MD5
    print(f"{'OK  ' if same else 'DIFF'} {path} md5 {have}" + ("" if same else f" (expected {MD5})"))
    print("ready to apply" if same else "STOP: the file is not the expected copy; nothing would be changed")
    return 0 if same else 2


def apply(path=PATH):
    have = md5(path)
    if have != MD5:
        sys.exit(f"STOP: {path} is not the expected copy (md5 {have}); nothing changed")
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    for old, _ in EDITS:
        if s.count(old) != 1:
            sys.exit(f"STOP: expected text not found exactly once in {path}: {old[:60]!r}; nothing changed")
    backup = path + SUFFIX
    shutil.copy2(path, backup)
    for old, new in EDITS:
        s = s.replace(old, new)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)
    print(f"patched {path} ({len(EDITS)} edits); backup {backup}")
    print(f"done: {SETTING}=1 in .env.worker2 + a restart of worker 2 on its empty queue turns it on")


def undo(path=PATH):
    backups = sorted(glob.glob(path + ".bak_lever6_*"), key=os.path.getmtime)
    if not backups:
        print(f"no backup {path}.bak_lever6_*: nothing restored")
        return
    shutil.copy2(backups[-1], path)
    print(f"restored {path} from {backups[-1]}")


def helpers(model):
    """The inserted functions, run on their own with `model` in the place of Analisi_Paragrafo."""
    scope = {"json": json, "os": os, "re": re, "Analisi_Paragrafo": model}
    exec(HELPERS, scope)
    return scope


def selftest():
    class Model:                                        # in the place of Analisi_Paragrafo: hands the data back
        model_validate = staticmethod(lambda data: data)

    class Unreadable(Exception):                        # what the library raises for an answer cut inside its JSON
        def __init__(self, text, kind="json_invalid"):
            super().__init__("unreadable")
            self.first = {"type": kind, "input": text}

        def errors(self):
            return [self.first]

    class Llm:                                          # in the place of the chat model object
        extra_body = {"other": 1}

        def model_copy(self, update):
            new = Llm()
            new.extra_body = update["extra_body"]
            return new

    f = helpers(Model)
    is_on, with_stop = f["_stop_on_repetition"], f["_with_repetition_stop"]
    cut_text, salvage = f["_cut_answer_text"], f["_salvage_cut_answer"]

    kept_env = os.environ.pop(SETTING, None)
    try:
        llm = Llm()
        for value, expected in {None: False, "": False, "0": False, "no": False, "1": True, " TRUE ": True, "yes": True}.items():
            os.environ.pop(SETTING, None)
            if value is not None:
                os.environ[SETTING] = value
            assert is_on() is expected, value
            got = with_stop(llm)
            if expected:                                # a copy that carries the stop; the model object as it was
                assert got is not llm and llm.extra_body == {"other": 1}
                assert got.extra_body == {"other": 1, "repetition_detection": {"max_pattern_size": 20, "min_count": 10}}
            else:
                assert got is llm
    finally:
        os.environ.pop(SETTING, None)
        if kept_env is not None:
            os.environ[SETTING] = kept_env

    assert cut_text(Unreadable("abc")) == "abc"
    assert cut_text(Unreadable("abc", "missing")) is None           # a validation error of another kind
    assert cut_text(Unreadable(b"abc")) is None and cut_text(ValueError("x")) is None

    head = ('{\n  "requirement": "r",\n  "core_text": "a \\"quoted\\" [word], \\"pattern_type\\": no",\n'
            '  "search_text": "s",\n  "pattern_type": "altro"')
    want = {"requirement": "r", "core_text": 'a "quoted" [word], "pattern_type": no', "search_text": "s", "pattern_type": "altro"}
    one = {"n_articolo": "12", "n_paragrafo": "", "nome_documento": "codice penale", "codice_documento": "cp"}
    two = {**one, "n_articolo": "13"}
    ref, ref2 = json.dumps(one), json.dumps(two)
    blank = "\n    " * 10                               # what the stop leaves at the end
    whole = {
        head + ',\n  "riferimenti": [\n    ' + ref + "," + blank: [one],             # the usual cut: after an entry
        head + ',\n  "riferimenti": [\n    ' + ref + blank: [one],
        head + ',\n  "riferimenti": [' + ref + ", " + ref2 + ", " + ref + "," + blank: [one, two],   # a repeated entry once
        head + ',\n  "riferimenti": [' + ref + ', {"n_articolo": "13", "n_par': [one],                # half an entry dropped
        head + ',\n  "riferimenti": [' + ref + "]" + blank: [one],                   # only the last brace missing
        head + ',\n  "riferimenti": [' + blank: None,
        head + ',\n  "riferimenti": null' + blank: None,
        head + "," + blank: None,                                                    # cut before the list
        head + blank: None,
    }
    for text, refs in whole.items():
        assert salvage(text) == {**want, "riferimenti": refs}, (text[-60:], salvage(text))
    assert salvage(head.replace('"altro"', "null") + "," + blank) == {**want, "pattern_type": None, "riferimenti": None}
    refused = ("", blank, '{"requirement": "r", "core_text": "q q q q q q',       # cut inside the quote
               head[:head.rindex('"altro"')] + blank,                                # the tag not written yet
               '{"pattern_type": "altro",' + blank)                                  # the content did not come first
    for text in refused:
        assert salvage(text) is None, text[-60:]

    class Strict:                                       # a model that refuses the data: nothing is kept
        @staticmethod
        def model_validate(data):
            raise ValueError("refused")
    assert helpers(Strict)["_salvage_cut_answer"](head + blank) is None
    print(f"selftest OK: the setting, the request's extra key, {len(whole) + 1} cut answers kept, {len(refused) + 1} refused")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        selftest()
    elif "--check" in args:
        sys.exit(check())
    else:
        undo() if "--undo" in args else apply()
