"""Copy the criminal rulings that exist only in be1's Neo4j into be2's (prod's).

The list is /tmp/only_in_be1.txt on be2, one document per line: id|type|file_name|name.
Documents whose file is still in /opt/sunnitai-be/document_upload are left out: the
queue now writes to be2 itself.

For each document:
  - its Document node and every node it CONTAINS (the Sections), with every property
    (embedding, embedding_v2, plain_text, titolo_articolo...) and every label, MERGEd
    by id;
  - the relationships between those nodes (CONTAINS, NEXT, PART_OF...), as they are;
  - relationships to nodes that are not a Section or Document (LEGAL_CONCEPT, DATE,
    EDITOR, LEGAL_SOURCE...): ingestion derives those ids from content, so the node is
    MERGEd by id (created with its properties only if be2 lacks it) and the edge recreated;
  - relationships to another document's Section or Document (a ruling -> a code
    article): the codes were re-ingested on prod with other ids, so the target is looked
    up again on be2 by (document name, section name), then by article number ("584.0.0",
    "584", the first "584.x"). Edges no lookup resolves go to --log. An edge to another
    document of this same copy is left to that document's turn (counted "deferred"); its
    Sections keep their ids, so the edge is made then.

First it prints the labels, property keys and relationship types these documents use on
be1. Then, per document, what it would copy. Nothing is written without --write.

Small transactions (--tx-nodes sections, --tx-edges relationships each), --batch
documents at a time with --pause seconds between batches: be2 runs prod on 15 GB.
The Document node and its own edges are written last, in one transaction, so a
Document on be2 means its copy finished: a rerun skips it and redoes any half-copied one.

Run on be2, from the app folder (for .env: NEO4J_URI/USER/PASSWORD are be2's):
    cd /opt/chatbot && venv/bin/python /tmp/copy_rulings_be1_to_be2.py              # dry run
    ... --write                                                                      # copy
be1's credentials: SRC_NEO4J_USER / SRC_NEO4J_PASSWORD, else the same as be2's.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.getcwd(), ".env"))
except ImportError:
    pass

import neo4j
from neo4j import GraphDatabase

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# "584.0.0" -> 584, "21-novies.0.0" -> 21-novies, "584_1" (a chunk) -> 584, "raccolte_usi.9.0.0" -> 9
_ARTICLE_RE = re.compile(r"^(?:[a-z][a-z0-9_]*\.)?(\d+(?:-[a-z]+)?)(?:[._]|$)", re.IGNORECASE)


def q(name: str) -> str:
    """A label or relationship type, backquoted, after checking it is a plain identifier."""
    if not _IDENT.match(name or ""):
        raise ValueError(f"unsafe identifier {name!r}")
    return f"`{name}`"


def primary_label(labels) -> str:
    """The label a node is MERGEd on (the one its id index is on)."""
    for label in ("Document", "Section"):
        if label in labels:
            return label
    return sorted(labels)[0] if labels else None


def article_of(section_name: str):
    m = _ARTICLE_RE.match((section_name or "").strip())
    return m.group(1) if m else None


class Profiler:
    """With --profile: the time of each be2 query and Python step in plan, by name,
    and the first time a query runs, its EXPLAIN plan's operators (NodeIndexSeek
    uses an index; NodeByLabelScan / AllNodesScan read every node of the label)."""

    def __init__(self):
        self.enabled = False
        self.seconds, self.calls, self.explained = Counter(), Counter(), set()
        self.id_indexes = None    # labels with an ONLINE index on id; None = could not list them
        self.warned = set()

    def run(self, session, name, query, **params):
        if self.enabled and name not in self.explained:
            self.explained.add(name)
            plan = session.run("EXPLAIN " + query, **params).consume().plan
            ops = []

            def walk(p):
                p = p if isinstance(p, dict) else getattr(p, "__dict__", {})
                ops.append(str(p.get("operatorType") or p.get("operator_type") or "?").split("@")[0])
                for c in p.get("children") or []:
                    walk(c)
            walk(plan or {})
            scan = [o for o in ops if o in ("NodeByLabelScan", "AllNodesScan")]
            print(f"  [explain] {name}: {' <- '.join(ops)}" + ("   <-- SCAN, no index used" if scan else ""))
        t = time.time()
        rows = list(session.run(query, **params))
        self.seconds[name] += time.time() - t
        self.calls[name] += 1
        return rows

    def step(self, name, started):
        self.seconds[name] += time.time() - started
        self.calls[name] += 1

    def check_index(self, label):
        """Warn once when be2 has no index on :label(id): every lookup by id scans the label."""
        if self.id_indexes is not None and label and label not in self.id_indexes and label not in self.warned:
            self.warned.add(label)
            print(f"  WARNING: no ONLINE index on :{label}(id) on be2 -- every lookup or MERGE by id scans all "
                  f":{label} nodes. CREATE INDEX {label.lower()}_id IF NOT EXISTS FOR (n:{q(label)}) ON (n.id)")

    def report(self, documents):
        if self.enabled and self.seconds:
            print(f"  [profile] per document over {documents}: " + ", ".join(
                f"{k} {v / documents:.3f}s ({self.calls[k] / documents:.1f} calls)"
                for k, v in self.seconds.most_common()))


PROFILE = Profiler()


def load_id_indexes(session):
    """The labels with an ONLINE single-property index on id (a uniqueness constraint
    makes one too)."""
    try:
        rows = session.run("SHOW INDEXES YIELD labelsOrTypes, properties, state, entityType "
                           "WHERE entityType = 'NODE' AND properties = ['id'] AND state = 'ONLINE' "
                           "RETURN labelsOrTypes AS labels").data()
    except Exception as exc:   # Neo4j < 4.2 has no SHOW INDEXES
        print(f"  (could not list indexes: {exc})")
        return None
    return {label for r in rows for label in (r["labels"] or [])}


def run_write(session, fn, *args):
    return (getattr(session, "execute_write", None) or session.write_transaction)(fn, *args)


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# --- The list -------------------------------------------------------------------

def load_list(path: str) -> list:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 3)
            if len(parts) < 4 or not parts[0].strip() or parts[0].strip() in ("id", "d.id"):
                continue
            docs.append({"id": parts[0].strip(), "type": parts[1].strip(),
                         "file_name": parts[2].strip(), "name": parts[3].strip()})
    return docs


def queued_files(path: str) -> set:
    """Basenames and stems (lower case) of every file under the upload folder."""
    names = set()
    for _root, _dirs, files in os.walk(path):
        for fn in files:
            names.add(fn.lower())
            names.add(os.path.splitext(fn)[0].lower())
    return names


def is_queued(file_name: str, queued: set) -> bool:
    base = os.path.basename(file_name or "").lower()
    return bool(base) and (base in queued or os.path.splitext(base)[0] in queued)


# --- Discovery on be1 ------------------------------------------------------------

_DISCOVER_NODES = """
UNWIND $ids AS id
MATCH (d:Document {id: id})
OPTIONAL MATCH (d)-[:CONTAINS]->(s)
WITH d, collect(DISTINCT s) AS secs
UNWIND [d] + secs AS n
RETURN labels(n) AS labels, keys(n) AS keys, count(*) AS c
"""

_DISCOVER_RELS = """
UNWIND $ids AS id
MATCH (d:Document {id: id})
OPTIONAL MATCH (d)-[:CONTAINS]->(s)
WITH d, collect(DISTINCT s) AS secs
WITH [d] + secs AS owned
UNWIND owned AS n
MATCH (n)-[r]-(o)
WITH n, r, o, o IN owned AS intra
WHERE NOT intra OR startNode(r) = n
RETURN labels(n) AS n_labels, type(r) AS type, startNode(r) = n AS outgoing, labels(o) AS o_labels, intra,
       CASE WHEN o:Section AND NOT intra
            THEN head([(od:Document)-[:CONTAINS]->(o) | od.document_type]) END AS o_doc_type,
       count(*) AS c
"""


def discover(src, db, ids, size):
    nodes, keys, rels = Counter(), defaultdict(Counter), Counter()
    with src.session(database=db, default_access_mode=neo4j.READ_ACCESS) as s:
        for i, part in enumerate(chunks(ids, size)):
            for r in s.run(_DISCOVER_NODES, ids=part):
                labels = ":".join(sorted(r["labels"]))
                nodes[labels] += r["c"]
                for k in r["keys"]:
                    keys[labels][k] += r["c"]
            for r in s.run(_DISCOVER_RELS, ids=part):
                n_l, o_l = ":".join(sorted(r["n_labels"])), ":".join(sorted(r["o_labels"])) or "(no label)"
                if r["intra"]:
                    rels[("inside", f"({n_l})-[{r['type']}]->({o_l})")] += r["c"]
                else:
                    arrow = f"-[{r['type']}]->" if r["outgoing"] else f"<-[{r['type']}]-"
                    other = f"({o_l}" + (f" of a '{r['o_doc_type']}' document" if r["o_doc_type"] else "") + ")"
                    rels[("leaving", f"({n_l}){arrow}{other}")] += r["c"]
            print(f"  discovery: {min((i + 1) * size, len(ids))}/{len(ids)} documents", file=sys.stderr)

    print("\n== On be1, these documents use ==")
    print("Nodes (the Document and what it CONTAINS):")
    for labels, c in nodes.most_common():
        print(f"  {c:>9,}  :{labels}")
        print("             " + ", ".join(f"{k} ({n:,})" for k, n in sorted(keys[labels].items())))
    for where, title in (("inside", "Relationships inside a document"),
                         ("leaving", "Relationships to nodes outside the document")):
        print(f"{title}:")
        rows = [(k[1], c) for k, c in rels.items() if k[0] == where]
        for desc, c in sorted(rows, key=lambda x: -x[1]):
            print(f"  {c:>9,}  {desc}")
        if not rows:
            print("  (none)")
    print()


# --- One document from be1 --------------------------------------------------------

# One round trip: the Document, its sections and every edge, each edge once (an edge
# between two of the document's nodes is kept from its start node only).
_FETCH = """
MATCH (d:Document {id: $id})
OPTIONAL MATCH (d)-[:CONTAINS]->(s)
WITH d, collect(DISTINCT s) AS secs
WITH d, secs, [d] + secs AS owned
CALL {
  WITH owned
  UNWIND owned AS n
  MATCH (n)-[r]-(o)
  WITH n, r, o, o IN owned AS inside
  WHERE NOT inside OR startNode(r) = n
  RETURN collect({type: type(r), props: properties(r), n_id: n.id, outgoing: startNode(r) = n,
                  o_labels: labels(o), o_id: o.id,
                  o_name: CASE WHEN o:Section OR o:Document THEN o.name END,
                  o_doc_type: CASE WHEN o:Document THEN o.document_type END,
                  o_doc: CASE WHEN o:Section AND NOT inside THEN head([(od:Document)-[:CONTAINS]->(o) |
                         {id: od.id, name: od.name, type: od.document_type}]) END,
                  o_props: CASE WHEN inside OR o:Section OR o:Document THEN null ELSE properties(o) END}) AS edges
}
RETURN d, secs, edges
"""


def fetch(src_session, doc_id):
    row = src_session.run(_FETCH, id=doc_id).single()
    if not row or row["d"] is None:
        return None
    owned = {}
    for node in [row["d"]] + list(row["secs"]):
        props = dict(node)
        owned[props.get("id")] = {"id": props.get("id"), "labels": sorted(node.labels), "props": props}
    edges = [{"type": e["type"], "props": e["props"], "n_id": e["n_id"], "outgoing": e["outgoing"],
              "o": {"labels": sorted(e["o_labels"]), "id": e["o_id"], "name": e["o_name"],
                    "doc_type": e["o_doc_type"], "doc": e["o_doc"], "props": e["o_props"]}}
             for e in row["edges"]]
    return {"doc": owned.pop(row["d"].get("id")), "sections": list(owned.values()), "edges": edges}


# --- Finding targets on be2 --------------------------------------------------------

class Resolver:
    """Finds on be2 the node a cross-document edge points to. Answers are cached:
    many rulings cite the same articles."""

    def __init__(self, dst_session, copy_ids):
        self.s = dst_session
        self.copy_ids = copy_ids
        self.docs_on_be2 = {}      # Document id -> bool; the only ids cached (see known_ids)
        self.present = set()       # (label, id) on be2 among the current document's neighbours
        self.docs_named = {}       # name -> [(id, type)]
        self.sections = {}         # (doc name, doc type, section name) -> (id, method) | (None, reason)

    def known_ids(self, label, ids):
        """The ids among `ids` that be2 has under `label`, one query per 500. Only
        Documents are cached: satellite ids belong to one document each, and a
        million of them would cost be2 memory it does not have."""
        ids = list(dict.fromkeys(i for i in ids if i))
        cache = label == "Document"
        found = {i for i in ids if cache and self.docs_on_be2.get(i)}
        todo = [i for i in ids if not (cache and i in self.docs_on_be2)]
        PROFILE.check_index(label)
        for part in chunks(todo, 500):
            hits = {r["id"] for r in PROFILE.run(self.s, f"ids:{label}",
                                                 f"UNWIND $ids AS id MATCH (n:{q(label)} {{id: id}}) RETURN n.id AS id",
                                                 ids=part)}
            found |= hits
            if cache:
                self.docs_on_be2.update({i: i in hits for i in part})
        return found

    def prefetch(self, others):
        """Which of a document's neighbours be2 already has: one query per label,
        instead of one per edge."""
        by_label = defaultdict(list)
        for o in others:
            label = primary_label(o["labels"])
            if label and o["id"]:
                by_label[label].append(o["id"])
        self.present = {(label, i) for label, ids in by_label.items() for i in self.known_ids(label, ids)}

    def mark_written(self, doc_id):
        self.docs_on_be2[doc_id] = True

    def _documents(self, name, doc_type):
        if name not in self.docs_named:
            self.docs_named[name] = [(r["id"], r["type"]) for r in PROFILE.run(
                self.s, "document_by_name", "MATCH (d:Document {name: $name}) RETURN d.id AS id, d.document_type AS type", name=name)]
        docs = self.docs_named[name]
        same_type = [d for d in docs if d[1] == doc_type]
        return same_type or docs

    def document(self, name, doc_type):
        if not name:
            return None, "document_without_name"
        docs = self._documents(name, doc_type)
        if not docs:
            return None, "document_not_on_be2"
        if len(docs) > 1:
            return None, "ambiguous_document"
        return docs[0][0], "by_name"

    def section(self, doc_name, doc_type, section_name):
        key = (doc_name, doc_type, section_name)
        if key not in self.sections:
            self.sections[key] = self._section(doc_name, doc_type, section_name)
        return self.sections[key]

    def _section(self, doc_name, doc_type, section_name):
        doc_id, how = self.document(doc_name, doc_type)
        if not doc_id:
            return None, how
        if not section_name:
            return None, "section_without_name"
        art = article_of(section_name)
        names = list(dict.fromkeys([section_name] + ([f"{art}.0.0", art] if art else [])))
        rows = [r.data() for r in PROFILE.run(
            self.s, "section_by_name", "MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.name IN $names "
            "RETURN s.id AS id, s.name AS name", doc=doc_id, names=names)]
        if rows:
            rows.sort(key=lambda r: names.index(r["name"]))
            return rows[0]["id"], "same_name" if rows[0]["name"] == section_name else "article"
        if art:
            rows = PROFILE.run(self.s, "section_fragment", "MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) "
                               "WHERE s.name STARTS WITH $p RETURN s.id AS id ORDER BY s.name LIMIT 1",
                               doc=doc_id, p=art + ".")
            if rows:
                return rows[0]["id"], "article_fragment"
        return None, "article_not_on_be2"

    def target(self, o):
        """(be2 id, label, how); id None when unresolved (how = the reason) or deferred."""
        labels, oid = o["labels"], o["id"]
        label = primary_label(labels)
        if not label or not oid:
            return None, label, "target_without_id_or_label"
        if (label, oid) in self.present:
            return oid, label, "same_id"
        if label == "Section":
            doc = o["doc"]
            if not doc:
                return None, label, "section_of_no_document"
            if doc["id"] in self.copy_ids:
                return None, label, "deferred"
            sid, how = self.section(doc["name"], doc["type"], o["name"])
            return sid, label, how
        if label == "Document":
            if oid in self.copy_ids:
                return None, label, "deferred"
            did, how = self.document(o["name"], o["doc_type"])
            return did, label, how
        return oid, label, "satellite_new"


# --- Writing to be2 --------------------------------------------------------------

def _merge_nodes(tx, rows, create_only=False):
    by_labels = defaultdict(list)
    for r in rows:
        by_labels[tuple(r["labels"])].append({"id": r["id"], "props": r["props"]})
    for labels, items in by_labels.items():
        prim = primary_label(labels)
        extra = "".join(f":{q(l)}" for l in labels if l != prim)
        set_ = "ON CREATE SET" if create_only else "SET"
        tx.run(f"UNWIND $rows AS row MERGE (n:{q(prim)} {{id: row.id}}) {set_} n += row.props"
               + (f" {set_} n{extra}" if extra else ""), rows=items).consume()


def _merge_edges(tx, rows):
    written = 0
    groups = defaultdict(list)
    for r in rows:
        groups[(r["type"], r["a_label"], r["b_label"])].append({"a": r["a"], "b": r["b"], "props": r["props"]})
    for (typ, a_label, b_label), items in groups.items():
        rec = tx.run(f"UNWIND $rows AS row MATCH (a:{q(a_label)} {{id: row.a}}) MATCH (b:{q(b_label)} {{id: row.b}}) "
                     f"MERGE (a)-[r:{q(typ)}]->(b) SET r += row.props RETURN count(r) AS c", rows=items).single()
        written += rec["c"]
    return written


def _final(tx, doc, edges):
    _merge_nodes(tx, [doc])
    return _merge_edges(tx, edges)


def write_document(dst_session, plan, args):
    """Sections, new satellites, then the edges not touching the Document, then the
    Document with its own edges in one transaction. Returns edges written."""
    written = 0
    for part in chunks(plan["sections"], args.tx_nodes):
        run_write(dst_session, _merge_nodes, part)
    for part in chunks(plan["satellites"], args.tx_nodes):
        run_write(dst_session, _merge_nodes, part, True)
    for part in chunks(plan["edges_rest"], args.tx_edges):
        written += run_write(dst_session, _merge_edges, part)
    written += run_write(dst_session, _final, plan["doc"], plan["edges_doc"])
    return written


# --- Planning one document ---------------------------------------------------------

def plan_document(data, resolver, dst_session, hashes, stats, log):
    doc, doc_id = data["doc"], data["doc"]["id"]
    owned = {doc_id: doc, **{s["id"]: s for s in data["sections"]}}
    counts = Counter(sections=len(data["sections"]))

    if None in owned:
        return None, "node_without_id", counts
    other = hashes.get(doc["props"].get("hash")) or ""
    if other and other != doc_id:
        return None, f"same_hash_on_be2:{other}", counts
    by_label = defaultdict(list)
    for s in data["sections"]:
        by_label[primary_label(s["labels"])].append(s["id"])
    for label, ids in by_label.items():
        PROFILE.check_index(label)
        for part in chunks(ids, 500):
            rows = PROFILE.run(
                dst_session, f"section_collision:{label}",
                f"UNWIND $ids AS id MATCH (s:{q(label)} {{id: id}}) "
                "MATCH (od:Document)-[:CONTAINS]->(s) WHERE od.id <> $doc RETURN od.id AS od LIMIT 1",
                ids=part, doc=doc_id)
            if rows:
                return None, f"section_id_in_be2_document:{rows[0]['od']}", counts

    edges_doc, edges_rest, satellites = [], [], {}
    resolver.prefetch([e["o"] for e in data["edges"] if e["o"]["id"] not in owned])
    started = time.time()
    for e in data["edges"]:
        o = e["o"]
        if o["id"] is not None and o["id"] in owned:
            other_id, other_label, how = o["id"], primary_label(owned[o["id"]]["labels"]), "inside"
            counts["inside"] += 1
        else:
            other_id, other_label, how = resolver.target(o)
            stats["how:" + how] += 1
            if how == "deferred":
                counts["deferred"] += 1
                continue
            if other_id is None:
                counts["unresolved"] += 1
                target_doc = (o["doc"] or {}).get("name") if o["doc"] else o.get("name")
                stats_key = f"{how} | {e['type']} | {target_doc or ''}"
                stats["unresolved_by_target"][stats_key] += 1
                log.write(json.dumps({"document": doc_id, "rel": e["type"],
                                      "direction": "out" if e["outgoing"] else "in", "reason": how,
                                      "target_labels": o["labels"], "target_id_be1": o["id"],
                                      "target_document": target_doc, "target_section": o["name"]
                                      if "Section" in o["labels"] else None}, ensure_ascii=False) + "\n")
                continue
            if how == "satellite_new":
                satellites[o["id"]] = {"id": o["id"], "labels": o["labels"], "props": o["props"] or {}}
                counts["satellite_new"] += 1
            elif "Section" in o["labels"] or "Document" in o["labels"]:
                counts["cross"] += 1
            else:
                counts["satellite_same_id"] += 1
        n_label = primary_label(owned[e["n_id"]]["labels"])
        a, a_label, b, b_label = ((e["n_id"], n_label, other_id, other_label) if e["outgoing"]
                                  else (other_id, other_label, e["n_id"], n_label))
        row = {"type": e["type"], "a": a, "a_label": a_label, "b": b, "b_label": b_label, "props": e["props"]}
        (edges_doc if doc_id in (a, b) else edges_rest).append(row)

    counts["edges"] = len(edges_doc) + len(edges_rest)
    PROFILE.step("python:edges", started)
    return {"doc": doc, "sections": data["sections"], "satellites": list(satellites.values()),
            "edges_doc": edges_doc, "edges_rest": edges_rest}, "ok", counts


# --- Main --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", default="/tmp/only_in_be1.txt")
    p.add_argument("--upload-dir", default="/opt/sunnitai-be/document_upload")
    p.add_argument("--src-uri", default=os.getenv("SRC_NEO4J_URI", "bolt://10.15.214.1:7687"))
    p.add_argument("--src-db", default=os.getenv("SRC_NEO4J_DATABASE", os.getenv("NEO4J_DATABASE", "neo4j")))
    p.add_argument("--dst-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--dst-db", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    p.add_argument("--write", action="store_true", help="copy (default: dry run, nothing written)")
    p.add_argument("--batch", type=int, default=200, help="documents per batch")
    p.add_argument("--pause", type=float, default=3.0, help="seconds between batches")
    p.add_argument("--tx-nodes", type=int, default=50, help="sections per transaction")
    p.add_argument("--tx-edges", type=int, default=500, help="relationships per transaction")
    p.add_argument("--limit", type=int, default=0, help="only the first N documents still to copy")
    p.add_argument("--id", action="append", default=[], help="only this document id (repeatable)")
    p.add_argument("--skip-discovery", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="time each be2 query and step in plan, and print each query's EXPLAIN operators once")
    p.add_argument("--log", default="/tmp/copy_rulings_unresolved.jsonl", help="one line per unresolved edge")
    args = p.parse_args()
    PROFILE.enabled = args.profile

    if args.src_uri.rstrip("/") == args.dst_uri.rstrip("/"):
        sys.exit("source and destination are the same URI")
    src = GraphDatabase.driver(args.src_uri, auth=(os.getenv("SRC_NEO4J_USER", os.getenv("NEO4J_USER", "neo4j")),
                                                   os.getenv("SRC_NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", ""))))
    dst = GraphDatabase.driver(args.dst_uri, auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")))
    print(f"from be1 {args.src_uri} ({args.src_db}) to be2 {args.dst_uri} ({args.dst_db}) -- "
          + ("WRITING" if args.write else "dry run, nothing is written"))

    docs = load_list(args.list)
    print(f"{len(docs):,} documents in {args.list}")
    not_interp = [d for d in docs if d["type"] != "interpretation"]
    if not_interp:
        print(f"  {len(not_interp):,} have a type other than 'interpretation': left out")
        docs = [d for d in docs if d["type"] == "interpretation"]
    if os.path.isdir(args.upload_dir):
        queued = queued_files(args.upload_dir)
        before = len(docs)
        docs = [d for d in docs if not is_queued(d["file_name"], queued)]
        print(f"  {before - len(docs):,} still in {args.upload_dir}: left out")
    else:
        print(f"  WARNING: {args.upload_dir} not found, nothing left out for the queue")
    if args.id:
        docs = [d for d in docs if d["id"] in set(args.id)]
    copy_ids = {d["id"] for d in docs}

    dst_mode = neo4j.WRITE_ACCESS if args.write else neo4j.READ_ACCESS
    with dst.session(database=args.dst_db, default_access_mode=neo4j.READ_ACCESS) as s:
        PROFILE.id_indexes = load_id_indexes(s)
        if PROFILE.id_indexes is not None:
            print("  be2 labels with an ONLINE index on id: " + (", ".join(sorted(PROFILE.id_indexes)) or "none"))
        resolver = Resolver(s, copy_ids)
        done = resolver.known_ids("Document", [d["id"] for d in docs])
        # Every be2 Document's hash, once (Document.hash may have no index): a file already
        # ingested on be2 under another id is not copied again.
        hashes = {r["h"]: r["id"] for r in s.run("MATCH (d:Document) WHERE d.hash IS NOT NULL "
                                                 "RETURN d.hash AS h, d.id AS id")}
    todo = [d for d in docs if d["id"] not in done]
    print(f"  {len(done):,} already on be2: skipped. {len(todo):,} to copy")
    if args.limit:
        todo = todo[:args.limit]
        print(f"  --limit: the first {len(todo):,}")

    if not args.skip_discovery and todo:
        discover(src, args.src_db, [d["id"] for d in todo], args.batch)

    stats = Counter()
    stats["unresolved_by_target"] = Counter()
    totals, statuses, seconds = Counter(), Counter(), Counter()
    started = time.time()
    with open(args.log, "w", encoding="utf-8") as log, \
            src.session(database=args.src_db, default_access_mode=neo4j.READ_ACCESS) as ss, \
            dst.session(database=args.dst_db, default_access_mode=dst_mode) as ds:
        resolver = Resolver(ds, copy_ids)
        n = 0
        for b, batch in enumerate(chunks(todo, args.batch)):
            if b:
                time.sleep(args.pause)
            for d in batch:
                n += 1
                t0 = time.time()
                data = fetch(ss, d["id"])
                t1 = time.time()
                if data is None:
                    status, counts, plan = "missing_on_be1", Counter(), None
                else:
                    plan, status, counts = plan_document(data, resolver, ds, hashes, stats, log)
                t2 = time.time()
                if plan and args.write:
                    try:
                        written = write_document(ds, plan, args)
                        resolver.mark_written(d["id"])
                        if written != counts["edges"]:
                            status = f"edges_written_{written}_of_{counts['edges']}"
                    except Exception as exc:   # one document's failure stops nothing else
                        status = f"error:{type(exc).__name__}:{exc}"[:200]
                t3 = time.time()
                seconds.update(fetch=t1 - t0, plan=t2 - t1, write=t3 - t2)
                totals.update(counts)
                statuses[status.split(":")[0]] += 1
                print(f"[{n}/{len(todo)}] {d['id']}  sections={counts['sections']} inside={counts['inside']} "
                      f"cross={counts['cross']} satellites={counts['satellite_same_id']}+{counts['satellite_new']}new "
                      f"deferred={counts['deferred']} unresolved={counts['unresolved']}  {status}  "
                      f"be1 {t1 - t0:.1f}s plan {t2 - t1:.1f}s write {t3 - t2:.1f}s", flush=True)
            print(f"-- batch {b + 1} done, {n}/{len(todo)} documents, {time.time() - started:.0f}s; per document "
                  + ", ".join(f"{k} {v / n:.2f}s" for k, v in seconds.items()), flush=True)
            PROFILE.report(n)

    by_target = stats.pop("unresolved_by_target")
    print("\n== Summary ==" + ("" if args.write else " (dry run: nothing written)"))
    print("Documents: " + ", ".join(f"{k} {v:,}" for k, v in statuses.most_common()))
    print(f"Sections {totals['sections']:,}; relationships {totals['edges']:,}: inside the documents "
          f"{totals['inside']:,}, to other documents re-resolved {totals['cross']:,}, to other nodes "
          f"{totals['satellite_same_id']:,} existing + {totals['satellite_new']:,} new on be2")
    print(f"Deferred to another document of this copy: {totals['deferred']:,}; unresolved: {totals['unresolved']:,}")
    print("How edges leaving a document were resolved:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {v:>9,}  {k[4:]}")
    if by_target:
        print("Unresolved, by reason | relationship | target document (top 25):")
        for k, v in by_target.most_common(25):
            print(f"  {v:>9,}  {k}")
        print(f"Each one: {args.log}")
    src.close()
    dst.close()


if __name__ == "__main__":
    main()
