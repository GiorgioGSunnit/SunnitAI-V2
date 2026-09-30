"""Step 2 of 3 - the model calls only. Standard library only: runs on the server.

Reads enrich_inputs.json (from prepare_catalog_inputs.py) and asks the model, per
template: description, categoria, sottocategoria, fields (up to 25). Appends one
JSON line per template to the results file as soon as it has it, so it can be
stopped at any time and resumes where it left off.

    python3 enrich_run.py --inputs enrich_inputs.json --out results.jsonl \\
        --env /opt/chatbot/.env [--limit 20] [--window 20-7]

--window 20-7 : only work between 20:00 and 07:00 (server time); sleeps otherwise.
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

CAP = 25


def load_env(path):
    env = dict(os.environ)
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                env.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    return env


def act_text(sections, limit=6000):
    return "\n\n".join("%s\n%s" % (s["heading"], s["content"]) for s in sections)[:limit]


def system_prompt(codice, known):
    rules = (
        "Sei un avvocato italiano che cataloga modelli di atti giuridici. "
        "Rispondi SOLO con un oggetto JSON valido, senza markdown, con le chiavi:\n"
        '  "description": una frase che dice cosa fa l\'atto e chi lo presenta, max 25 parole;\n'
        '  "categoria": la categoria che raggruppa l\'atto;\n'
        '  "sottocategoria": una sottocategoria più specifica;\n'
        '  "fields": fino a {cap} nomi di campo in snake_case, uno per ogni dato DISTINTO che '
        "l'utente deve fornire, dedotti dai segnaposto [DA COMPILARE] (lista vuota se il "
        "modello non ne ha).\n"
        "Non superare MAI {cap} campi: se i dati sono di più, raggruppa quelli di dettaglio "
        "(es. un solo campo indirizzo_ricorrente invece di via, numero civico e CAP separati; "
        "un solo campo luogo_data).\n"
        "Regole per i campi: se lo stesso tipo di dato si ripete per soggetti diversi, "
        "distinguilo con un suffisso (nome_ricorrente e nome_resistente; nome_primo_coniuge e "
        "nome_secondo_coniuge), senza ometterne nessuno; raggruppa i segnaposto ripetuti dello "
        "stesso dato in un solo campo; nomi in italiano, solo lettere minuscole, cifre e "
        "underscore; nessun campo duplicato.\n"
    ).replace("{cap}", str(CAP))
    if known:
        rules += ("Per categoria usa ESATTAMENTE una di quelle già in uso, se adatta:\n"
                  + "\n".join("  - " + c for c in known)
                  + "\nAltrimenti proponine una nuova, breve e nello stesso stile.")
    else:
        rules += ("La categoria deve raggruppare atti simili per fase o oggetto "
                  "(es. \"Assemblee e verbali\", \"Atti introduttivi\"): NON usare come "
                  "categoria il nome dell'area (\"%s\")." % codice)
    return rules


GROUP_PROMPT = (
    "Ti viene dato un elenco JSON di nomi di campo di un modello di atto giuridico. "
    "Riducilo a MASSIMO %d campi raggruppando i dati di dettaglio dello stesso soggetto "
    "(es. via, numero civico, CAP e città -> un solo indirizzo_<soggetto>; luogo e data -> "
    "luogo_data). Non eliminare nessun soggetto e nessun dato che non si possa raggruppare. "
    "Nomi in italiano, snake_case. Rispondi SOLO con JSON: {\"fields\": [...]}" % CAP
)


def call(env, system, user, max_tokens=900):
    url = env["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    body = json.dumps({"model": env.get("LLM_MODEL"), "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(m.group(0) if m else raw)


def in_window(window):
    if not window:
        return True
    start, end = (int(x) for x in window.split("-"))
    h = datetime.datetime.now().hour
    return (start <= h or h < end) if start > end else (start <= h < end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", default="")
    args = ap.parse_args()
    env = load_env(args.env)
    if not env.get("LLM_BASE_URL"):
        sys.exit("LLM_BASE_URL not set (pass --env /opt/chatbot/.env)")

    data = json.load(open(args.inputs, encoding="utf-8"))
    entries, categories = data["entries"], {k: set(v) for k, v in data["categories"].items()}
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ok"):
                done.add(r["id"])
                cat = (r.get("data") or {}).get("categoria")
                if cat:                      # categories proposed so far feed later calls
                    categories.setdefault(r["codice"], set()).add(str(cat).strip())
    todo = [e for e in entries if e["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("%s  %d templates, %d already done, %d to do" % (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), len(entries), len(done), len(todo)), flush=True)

    t0, failures_in_row = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as out:
        for n, e in enumerate(todo, 1):
            while not in_window(args.window):
                time.sleep(300)
            known = sorted(categories.get(e["codice"], []))
            user = "Area: %s\nTitolo del modello: %s\n\nTesto del modello:\n%s" % (
                e["codice"], e["label"], act_text(e["sections"]))
            rec = {"id": e["id"], "codice": e["codice"], "label": e["label"], "ok": False}
            started = time.time()
            for attempt, wait in enumerate((0, 20, 120, 600)):
                time.sleep(wait)
                try:
                    raw = call(env, system_prompt(e["codice"], known), user)
                    parsed = parse(raw)
                    cat = str(parsed.get("categoria", "")).strip()
                    if not cat or cat.lower() == e["codice"].lower():      # one stricter retry
                        raw = call(env, system_prompt(e["codice"], []), user)
                        parsed = parse(raw)
                    fields = parsed.get("fields") or []
                    # Grouped by the model rather than cut: a cut drops the tail,
                    # which is where date and place usually are. Up to two passes;
                    # any shorter list is kept, the last few over the cap are
                    # trimmed at assembly (by then the tail is signature lines).
                    for _ in range(2):
                        if len(fields) <= CAP:
                            break
                        try:
                            g = parse(call(env, GROUP_PROMPT, json.dumps(fields, ensure_ascii=False), 700))
                        except (ValueError, KeyError):
                            break
                        grouped = g.get("fields") or []
                        if not 0 < len(grouped) < len(fields):
                            break
                        rec.setdefault("fields_before_grouping", parsed.get("fields"))
                        fields = grouped
                    parsed["fields"] = fields
                    rec.update(ok=True, data=parsed, raw=raw)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    rec["error"] = "network: %s" % str(exc)[:200]         # retried after a pause
                except (ValueError, KeyError) as exc:
                    rec["error"] = "unreadable answer: %s" % str(exc)[:200]
                    if attempt >= 1:
                        break
            rec["secs"] = round(time.time() - started, 1)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if rec["ok"]:
                failures_in_row = 0
                cat = str(rec["data"].get("categoria", "")).strip()
                if cat:
                    categories.setdefault(e["codice"], set()).add(cat)
            else:
                failures_in_row += 1
                if failures_in_row >= 5:     # model server down: wait instead of burning the list
                    print("  5 failures in a row - pausing 15 min", flush=True)
                    time.sleep(900)
                    failures_in_row = 0
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            print("%s  %d/%d  %s  %-60s %4.0fs  eta %.1fh" % (
                datetime.datetime.now().strftime("%H:%M"), n, len(todo), "ok " if rec["ok"] else "ERR",
                e["label"][:60], rec["secs"], eta / 3600), flush=True)


if __name__ == "__main__":
    main()
