"""The fact-signal rules (src/rag/fact_signal_rules.yaml) as a table in Italian, for the lawyer to review:
id, materia, quando (the conditions in words), istituto, dove si cerca, perché, stato. Then the legal
terms the code computes from the case's dates (fact_signals.TERMS), with the article each belongs to.

    python scripts/export_fact_rules.py                    # Markdown to the screen
    python scripts/export_fact_rules.py --out regole.md    # or to a file (.csv for a spreadsheet)
"""

import argparse
import csv
import io
import os
import sys

AREAS = {"diritti_reali": "Diritti reali e possesso", "contratti": "Obbligazioni e contratti",
         "responsabilita_civile": "Responsabilità civile", "consumo": "Consumo", "penale": "Penale",
         "amministrativo": "Amministrativo", "famiglia": "Famiglia", "successioni": "Successioni",
         "locazione": "Locazione", "lavoro": "Lavoro"}
STATUS = {"draft": "bozza", "experiment": "in sperimentazione (approvata per l'esperimento)", "approved": "approvata"}
HEAD = ["id", "materia", "quando", "istituto", "dove si cerca", "perché", "stato"]


def rows(fs) -> list:
    out = []
    for rule in fs.load_rules():
        institutions = rule["institution"] if isinstance(rule["institution"], list) else [rule["institution"]]
        out.append([rule["id"], AREAS.get(rule["area"], rule["area"]), fs.describe(rule["when"]),
                    "; ".join(institutions), rule["codice"], rule["why"], STATUS.get(rule["status"], rule["status"])])
    return out


def markdown(fs) -> str:
    table = rows(fs)
    cell = lambda v: str(v).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = ["# Regole fatti → istituti (FACT_SIGNALS_ENABLED)", "",
             f"{len(table)} regole. Ogni regola indica istituti, mai numeri di articolo: gli articoli sono trovati "
             "nella banca dati dal nome dell'istituto. \"Quando\": tutte le condizioni devono valere, lette dai fatti "
             "estratti dal caso.", "",
             "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in table]
    lines += ["", "## Termini calcolati dalle date del caso", "",
              "Calcolati nel codice, mai dal modello: dalla data del fatto a quella dell'atto che lo interrompe (o a oggi).",
              "", "| termine | durata | riferimento |", "|---|---|---|"]
    lines += [f"| {t.label} | {t.length()} | {t.ref} |" for t in fs.TERMS]
    lines.append("| impugnazione della sentenza penale | 15, 30 o 45 giorni dal deposito della motivazione | "
                 f"{fs._APPEAL_REF} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="a .md or .csv file (default: Markdown to the screen)")
    args = parser.parse_args()
    sys.path.insert(0, os.getcwd())
    from src.rag import fact_signals as fs
    if args.out and args.out.endswith(".csv"):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(HEAD)
        writer.writerows(rows(fs))
        text = buffer.getvalue()
    else:
        text = markdown(fs)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Written to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
