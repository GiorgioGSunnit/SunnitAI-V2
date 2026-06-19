"""
One-time bulk title generation for all Section nodes in Neo4j.
Safe to run while the chatbot is live — only writes s.title,
never touches abstract, plain_text, embedding, or any other field.
Safe to interrupt and resume — skips sections that already have a title.
"""

import os
import re
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv('/opt/chatbot/.env')

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://5331f28a.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "5331f28a")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "LFnXy6xYKE-cSuIP73Pr7cvw-kKENnh46YQOzoFnhAw")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "5331f28a")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

WORKERS = 10
BATCH_SIZE = 500

def generate_title(plain_text: str, section_name: str, existing_abstract: str = "") -> str:
    if not LLM_BASE_URL:
        return ""
    title_hint = ""
    abstract_hint = f"Contesto/abstract esistente: {existing_abstract[:300]}\n" if existing_abstract else ""
    prompt = (
        f"Sei un esperto di diritto italiano. "
        f"Genera un titolo brevissimo (massimo 8 parole) che descriva "
        f"l'argomento di questa sezione di un testo legale italiano. "
        f"Il titolo deve essere il nome dell'istituto giuridico o reato "
        f"trattato (es. 'Resistenza a un pubblico ufficiale', 'Omicidio colposo'). "
        f"ATTENZIONE: il testo della sezione potrebbe essere un frammento "
        f"incompleto (es. un rinvio a circostanze elencate altrove). In tal "
        f"caso usa il contesto/abstract esistente, se fornito, per capire "
        f"l'argomento reale; non inventare dettagli non presenti nel testo o "
        f"nel contesto. "
        f"Rispondi SOLO con il titolo, senza virgolette, senza punteggiatura finale, "
        f"senza prefissi come 'Titolo:'.\n\n"
        f"Articolo: {section_name}\n"
        f"{abstract_hint}"
        f"Testo: {plain_text[:300]}"
    )
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 30,
            },
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=20,
        )
        resp.raise_for_status()
        title = resp.json()["choices"][0]["message"]["content"].strip()
        return title.strip('"\'.,')
    except Exception as e:
        return ""

def process_batch(rows: list, driver) -> tuple[int, int]:
    """Process a batch of sections. Returns (success_count, skip_count)."""
    success = 0
    skipped = 0
    for row in rows:
        eid = row["eid"]
        name = row["name"] or ""
        plain_text = row["plain_text"] or ""
        abstract = row["abstract"] or ""

        # Skip if dash-pattern title already extractable from abstract
        if re.match(r"^-\s*(.+?)\s*-", abstract):
            # Extract and write it directly without LLM call
            match = re.match(r"^-\s*(.+?)\s*-", abstract)
            title = match.group(1)
        else:
            if not plain_text.strip() and not abstract.strip():
                skipped += 1
                continue
            title = generate_title(plain_text, name, abstract)
            if not title:
                skipped += 1
                continue

        try:
            with driver.session(database=NEO4J_DATABASE) as session:
                session.run(
                    "MATCH (s) WHERE elementId(s) = $eid SET s.title = $title",
                    eid=eid, title=title,
                )
            success += 1
        except Exception as e:
            skipped += 1
    return success, skipped

def fetch_batch(driver, skip: int, limit: int) -> list:
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("""
            MATCH (s:Section)
            WHERE (s.title IS NULL OR s.title = '')
            RETURN elementId(s) AS eid, s.name AS name,
                   s.plain_text AS plain_text, s.abstract AS abstract
            SKIP $skip LIMIT $limit
        """, skip=skip, limit=limit)
        return [dict(r) for r in result]

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Get total count
    with driver.session(database=NEO4J_DATABASE) as session:
        total = session.run("""
            MATCH (s:Section)
            WHERE (s.title IS NULL OR s.title = '')
            RETURN count(s) AS n
        """).single()["n"]

    print(f"Total sections needing titles: {total}")
    print(f"Workers: {WORKERS}, Batch size per worker: {BATCH_SIZE}")
    print(f"Estimated time at ~2s/section with {WORKERS} workers: "
          f"~{int(total * 2 / WORKERS / 60)} minutes")
    print("Starting... (safe to Ctrl+C and resume — already-titled sections are skipped)\n")

    total_success = 0
    total_skipped = 0
    t_start = time.time()
    offset = 0

    while True:
        # Fetch next batch of WORKERS * BATCH_SIZE sections
        chunk_size = WORKERS * BATCH_SIZE
        rows = fetch_batch(driver, offset, chunk_size)
        if not rows:
            break

        # Split into per-worker sub-batches
        sub_batches = [
            rows[i:i + BATCH_SIZE]
            for i in range(0, len(rows), BATCH_SIZE)
        ]

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(process_batch, batch, driver) for batch in sub_batches]
            for future in as_completed(futures):
                s, sk = future.result()
                total_success += s
                total_skipped += sk

        offset += chunk_size
        elapsed = time.time() - t_start
        rate = total_success / elapsed if elapsed > 0 else 0
        remaining = (total - total_success - total_skipped)
        eta_mins = int(remaining / (rate * 60)) if rate > 0 else "?"
        print(f"  Progress: {total_success + total_skipped}/{total} "
              f"| Success: {total_success} | Skipped: {total_skipped} "
              f"| Rate: {rate:.1f}/s | ETA: {eta_mins} min")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} minutes.")
    print(f"Titled: {total_success} | Skipped/failed: {total_skipped}")
    driver.close()

if __name__ == "__main__":
    main()
