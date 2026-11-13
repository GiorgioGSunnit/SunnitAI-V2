"""Read-only probe: what does a chunk answer that never ends consist of? (runbook 3f)

13 Nov 2026: the model server writes about a quarter of its tokens for requests that never finish (its counter of
generated tokens grows 30% faster than the tokens of finished requests, and the prompts of unfinished requests are
5% of all prompts: the share of chunk calls the ingest gives up on). So a "hung" chunk call is an answer that does
not end, dropped when the ingest hangs up. The fix depends on what that answer is made of: white space the JSON
shape allows, a list of references that never closes, a text field repeating itself. This probe looks.

It sends the sections of a few rulings already in the graph to the model exactly as the ingest's chunk analysis
does: the ingest's own model object builds the request (taken by pointing it once at a local port that records
the request and refuses it), with the ingest's two prompt files. The only differences: the answer is streamed and
capped (--cap tokens), so an endless one costs under a minute. One ruling's sections go out together, as in the
ingest. Nothing is written anywhere.

It prints numbers and field names only, never a word of a ruling or of an answer.

    cd /opt/sunnitai-be && sudo venv/bin/python3 ~/probe_runaway.py                # 12 rulings, about 150 calls
    python probe_runaway.py --selftest                                              # no server needed

Load: about what one ingest worker sends in six minutes.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

INGEST = "/opt/sunnitai-be"
PACKAGES = f"{INGEST}/src/be/src"                       # what batch_ingest.py puts on sys.path
PROMPTS = f"{PACKAGES}/lex_package/utils/prompts"
FIELDS = ("requirement", "core_text", "search_text", "pattern_type", "riferimenti")
CHUNK_WORDS = 250                                       # analisi_parallel.MAX_CHUNK_WORDS


def read_env(path: str) -> dict:
    env = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def period(tail: str, most: int = 600) -> int:
    """The shortest length the tail repeats with, three times in a row at its end; 0 when it does not."""
    for p in range(1, min(most, len(tail) // 3) + 1):
        if tail[-p:] == tail[-2 * p:-p] == tail[-3 * p:-2 * p]:
            return p
    return 0


def shape(text: str) -> dict:
    """An answer in numbers: the field being written at its end, the references it opened, and its last 2,000
    characters: the share of white space, the distinct characters, the repeat length, how small it packs."""
    tail = text[-2000:]
    found = {f: text.rfind(f'"{f}"') for f in FIELDS}
    last = max(found, key=found.get)
    raw = tail.encode("utf-8")
    return {"chars": len(text),
            "open_field": last if found[last] >= 0 else "(none)",
            "open_field_chars": len(text) - found[last] if found[last] >= 0 else len(text),
            "references": text.count('"n_articolo"'),
            "tail_space": sum(c.isspace() for c in tail) / len(tail) if tail else 0.0,
            "tail_distinct": len(set(tail)),
            "tail_period": period(tail),
            "tail_packs_to": len(zlib.compress(raw)) / len(raw) if raw else 1.0}


def ask(url: str, key: str, body: dict, most_seconds: float = 120.0) -> dict:
    """One streamed answer: its text (kept in memory only), how it finished, tokens, seconds."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    out = {"text": "", "finish": None, "tokens": None, "pieces": 0, "thinking_chars": 0, "error": ""}
    parts, t0 = [], time.time()
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    out["tokens"] = event["usage"].get("completion_tokens")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + "".join(
                        ((call.get("function") or {}).get("arguments") or "") for call in delta.get("tool_calls") or [])
                    if piece:
                        parts.append(piece)
                        out["pieces"] += 1
                    out["thinking_chars"] += len(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    out["finish"] = choice.get("finish_reason") or out["finish"]
                if time.time() - t0 > most_seconds:
                    out["finish"] = out["finish"] or "cut by the probe"
                    break
    except Exception as exc:                            # the type only: a message could quote the request
        out["error"] = type(exc).__name__
    out["text"], out["seconds"] = "".join(parts), time.time() - t0
    if out["tokens"] is None:
        out["tokens"] = out["pieces"]                   # a streamed piece is about one token
    return out


def captured_request() -> dict:
    """The request LangChain builds for a chunk on this machine: the ingest's model object, pointed at a local
    port that records one request and refuses it. {"path", "body"}; empty when nothing arrived."""
    seen: dict = {}

    class Recorder(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "probe", "type": "invalid_request_error"}}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sys.path.insert(0, PACKAGES)
        from langchain_core.messages import HumanMessage, SystemMessage
        from lex_package.llm.factory import build_chat_model
        from lex_package.t.analisi_articolo import Analisi_Paragrafo
        llm = build_chat_model(target="primary", temperature=0,
                               overrides={"base_url": f"http://127.0.0.1:{server.server_port}/v1"})
        try:
            llm.with_structured_output(Analisi_Paragrafo).invoke([SystemMessage(content="S"), HumanMessage(content="U")])
        except Exception:
            pass                                        # the refusal is the plan: the request is what we came for
    finally:
        server.shutdown()
    return seen


def request_shape(body: dict) -> str:
    """How the answer's shape is asked for, in the request's own key names."""
    keys = ", ".join(sorted(k for k in body if k != "messages"))
    fmt = body.get("response_format")
    if isinstance(fmt, dict):
        how = (f"response_format {fmt.get('type')}"
               + (f" (strict {fmt['json_schema'].get('strict')})" if isinstance(fmt.get("json_schema"), dict) else ""))
    elif body.get("tools"):
        choice = body.get("tool_choice")
        how = f"tools x{len(body['tools'])}, tool_choice {choice.get('type') if isinstance(choice, dict) else choice}"
    else:
        how = "no shape asked"
    cap = next((f"{k} {body[k]}" for k in ("max_tokens", "max_completion_tokens") if body.get(k)), "no length cap")
    return f"keys [{keys}]; {how}; {cap}"


def rulings_sections(env: dict, rulings: int) -> list:
    """[[(title, text), ...] per ruling]: the sections of `rulings` random 2025 rulings (read-only session)."""
    from neo4j import GraphDatabase, READ_ACCESS
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    by_doc: dict = {}
    with driver.session(database=env.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as session:
        rows = session.run(
            "MATCH (d:Document) WHERE d.document_type = 'interpretation' "
            "AND (d.file_name CONTAINS 'a2025' OR d.name CONTAINS 'a2025') "
            "WITH d, rand() AS r ORDER BY r LIMIT $n "
            "MATCH (d)-[:CONTAINS]->(s:Section) WHERE size(coalesce(s.plain_text, '')) >= 200 "
            "RETURN d.id AS doc, coalesce(s.title, '') AS title, s.plain_text AS text", n=rulings)
        for row in rows:
            by_doc.setdefault(row["doc"], []).append((row["title"], row["text"]))
    driver.close()
    return list(by_doc.values())


def messages_for(system: str, user: str, title: str, text: str) -> list:
    """The two messages of analisi_parallel's comma branch, for one chunk (the section's first 250 words)."""
    chunk = " ".join(text.split()[:CHUNK_WORDS])
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{user} titolo articolo cui appartiene il comma:'{title}';"
                                        f"contenuto del comma: '{chunk}';identificativo del comma: '1';"
                                        "flag del comma: 'False'"}]


def report(results: list, cap: int) -> None:
    errors = Counter(r["error"] for r in results if r["error"])
    done = [r for r in results if not r["error"] and r["finish"] not in ("length", "cut by the probe")]
    endless = [r for r in results if not r["error"] and r["finish"] in ("length", "cut by the probe")]
    if done:
        tokens = sorted(r["tokens"] for r in done)
        print(f"answers that ended: {len(done)} -- tokens median {statistics.median(tokens):.0f}, "
              f"95th percentile {tokens[int(0.95 * (len(tokens) - 1))]}, longest {tokens[-1]}; "
              f"seconds median {statistics.median(r['seconds'] for r in done):.1f}; "
              f"with references {sum(1 for r in done if shape(r['text'])['references'])}; "
              f"finish {dict(Counter(r['finish'] for r in done))}")
    share = f" ({100 * len(endless) / len(results):.1f}% of {len(results)})" if results else ""
    print(f"answers that did not end within {cap} tokens: {len(endless)}{share}")
    for i, r in enumerate(endless, 1):
        s = shape(r["text"])
        print(f"  {i}: {r['tokens']} tokens in {r['seconds']:.0f} s | field being written: {s['open_field']} "
              f"(for {s['open_field_chars']} of {s['chars']} characters), references opened {s['references']} | "
              f"last 2,000 characters: {100 * s['tail_space']:.0f}% white space, {s['tail_distinct']} distinct "
              f"characters, " + (f"repeat every {s['tail_period']}" if s["tail_period"] else "no exact repeat")
              + f", pack to {100 * s['tail_packs_to']:.0f}%"
              + (f" | thinking text {r['thinking_chars']} characters" if r["thinking_chars"] else ""))
    if endless:
        print("  by field: " + ", ".join(f"{f} {n}" for f, n in Counter(shape(r["text"])["open_field"] for r in endless).most_common()))
    if any(r["thinking_chars"] for r in results):
        print(f"answers with thinking text: {sum(1 for r in results if r['thinking_chars'])}")
    if errors:
        print("errors (type, count): " + ", ".join(f"{t} {n}" for t, n in errors.items()))


def selftest() -> None:
    assert period("ab" * 50) == 2 and period(" " * 30) == 1 and period("the quick brown fox jumps over") == 0
    ref = '{"n_articolo": "12", "n_paragrafo": "", "nome_documento": "codice penale", "codice_documento": "cp"}, '
    s = shape('{"requirement": "x", "riferimenti": [' + ref * 40)
    assert s["open_field"] == "riferimenti" and s["references"] == 40 and s["tail_period"] == len(ref) and s["tail_packs_to"] < 0.1
    s = shape('{"requirement": "x",' + "\n" * 3000)
    assert s["open_field"] == "requirement" and s["tail_space"] == 1.0 and s["tail_distinct"] == 1 and s["tail_period"] == 1
    assert shape("")["open_field"] == "(none)"

    class Model(BaseHTTPRequestHandler):                # a stand-in model server: streams, honours max_tokens
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            endless = "ENDLESS" in body["messages"][-1]["content"]
            pieces = ['{"requirement": "x",'] + ["\n"] * (body["max_tokens"] - 1) if endless else ['{"requirement":', ' "x"}']
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for n, piece in enumerate(pieces, 1):
                finish = ("length" if endless else "stop") if n == len(pieces) else None
                event = {"choices": [{"delta": {"content": piece}, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            usage = {"choices": [], "usage": {"completion_tokens": len(pieces)}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    try:
        good = ask(url, "EMPTY", {"messages": [{"role": "user", "content": "a section"}], "max_tokens": 50, "stream": True})
        bad = ask(url, "EMPTY", {"messages": [{"role": "user", "content": "ENDLESS"}], "max_tokens": 50, "stream": True})
        gone = ask("http://127.0.0.1:9/v1/chat/completions", "EMPTY", {"messages": []})
    finally:
        server.shutdown()
    assert good["finish"] == "stop" and good["tokens"] == 2 and good["text"] == '{"requirement": "x"}' and not good["error"]
    assert bad["finish"] == "length" and bad["tokens"] == 50 and shape(bad["text"])["tail_space"] > 0.5
    assert gone["error"] and not gone["text"]
    assert "response_format json_schema (strict True)" in request_shape(
        {"model": "m", "messages": [], "response_format": {"type": "json_schema", "json_schema": {"strict": True}}})
    assert "tool_choice function" in request_shape({"tools": [{}], "tool_choice": {"type": "function"}, "max_tokens": 9})
    m = messages_for("S", "U", "T", "w " * 300)
    assert m[0] == {"role": "system", "content": "S"} and len(m[1]["content"].split("'")[3].split()) == CHUNK_WORDS
    report([good, bad, gone], 50)
    print("selftest ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rulings", type=int, default=12, help="rulings to take sections from (about 13 calls each)")
    p.add_argument("--cap", type=int, default=3000, help="tokens after which an answer counts as never-ending")
    p.add_argument("--env", default=f"{INGEST}/.env", help="the ingest's settings (model, address, key)")
    p.add_argument("--graph-env", default="/opt/v2tools/health.env", help="NEO4J_* settings for the read")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return 0
    for k, v in read_env(args.env).items():
        os.environ.setdefault(k, v)
    try:
        seen = captured_request()
    except Exception as exc:
        print(f"the ingest's model object could not be built here ({type(exc).__name__}): nothing sent to the model")
        return 1
    if not seen.get("body"):
        print("no request captured from the ingest's model object: nothing sent to the model")
        return 1
    body = seen["body"]
    print("the chunk request as the ingest builds it: " + request_shape(body))
    base = (os.environ.get("LLM_BASE_URL") or "").rstrip("/")
    url = base + (seen["path"][len("/v1"):] if seen["path"].startswith("/v1") else seen["path"])
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    with open(f"{PROMPTS}/system.txt", encoding="utf-8", errors="replace") as fh:
        system = fh.read()
    with open(f"{PROMPTS}/analisi.txt", encoding="utf-8", errors="replace") as fh:
        user = fh.read()
    try:
        rulings = rulings_sections(read_env(args.graph_env), args.rulings)
    except Exception as exc:
        print(f"the sections could not be read from the graph ({type(exc).__name__}): nothing sent to the model")
        return 1
    print(f"rulings {len(rulings)}, sections {sum(len(r) for r in rulings)}")
    if not rulings:
        return 1
    for k in ("max_completion_tokens", "stream"):
        body.pop(k, None)
    results = []
    for sections in rulings:                            # one ruling's chunks go out together, as in the ingest
        bodies = [{**body, "messages": messages_for(system, user, title, text), "stream": True,
                   "stream_options": {"include_usage": True}, "max_tokens": args.cap} for title, text in sections]
        with ThreadPoolExecutor(max_workers=max(1, len(bodies))) as pool:
            results += list(pool.map(lambda b: ask(url, key, b), bodies))
    report(results, args.cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
