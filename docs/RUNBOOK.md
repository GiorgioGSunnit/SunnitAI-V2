# Runbook: release, deploy, promote, measure

From v77 on, every deploy is a release. A release has a version, every answer it serves names that version, and its
quality is judged in two ways: the five-cell nightly for structure and regressions, and the lawyer's scores on the
held-out set for substance. This runbook covers the whole loop. The be2 ingestion workers have their own runbook,
`ops/be2/RUNBOOK_cassazione_batches.md`.

## 1. Where things live

| where | what |
|---|---|
| the Mac, this repository | the code; `scripts/release.sh` and `scripts/deploy.sh` run here |
| `/root/v38` (server) | the test copy: `.env.v45`, `scripts/serve_test.sh` on ports 8030-8049 (never 8010) |
| `/opt/chatbot` (server) | the live copy: systemd service `chatbot`, `/opt/chatbot/.env`, `venv/` |
| `/opt/chatbot/state/` | runtime state: `sessions.json`, `tenant_logos/`, `system_templates/`, `article_cards.jsonl`, `logs/` (`src/state_paths.py`) |
| `/opt/chatbot/heldout/` | the held-out set: one `<date>_<id>.json` per case-first answer; `scores.jsonl`, `exports.jsonl` (`src/chatbot/heldout.py`) |
| `/root/snapshots/<time>_<version>/` | one per promote: `/opt/chatbot`'s src, scripts, docs, tests, data, state, VERSION, .env |

`VERSION` (repository root) holds one line, `<semver>+<git short sha>`, for example `77.0.0+1a2b3c4`. The sha is the code
commit the release names. The release commit itself changes only `VERSION` and is tagged `v<semver>`. `GET /api/version`
returns the version, and every case-first answer ends with the line `SunnitAI <version>`.

## 2. The loop, one deploy

```bash
# on the Mac, on a clean tree with the work committed
scripts/release.sh                 # next patch; or 77.1.0 | minor | major
git push origin master v77.0.1     # the command release.sh prints
SSH_HOST=<server> scripts/deploy.sh          # DRY_RUN=1 first to list what changes

# on the server
cd /root/v38 && scripts/serve_test.sh        # hand test; scripts/serve_test.sh stop when done
bash /root/v38/scripts/promote.sh            # to live
```

**`deploy.sh`** (Mac to `/root/v38`) syncs one directory at a time:

- `src/ scripts/ docs/ tests/` are mirrored (`--delete`): a file removed from the repo is removed from the copy.
- `data/` is added to and updated, never `--delete`: files that exist only on the server stay.
- `VERSION` is copied.

`scripts/deploy_exclude.txt` lists what is never copied and therefore never deleted. It starts with the PO's list:
`.env*`, `__pycache__`, `*.pyc`, `logs/`, `diagnose_reports/`, `.git/`, `state/`, `data/private_documents/`,
`data/institutions_*.json` (built on the server) and `article_cards.jsonl`. It then adds the install metadata
(`*.egg-info/`), venvs and caches, `heldout/`, and the runtime files that `data/` held before v77. Each directory is its
own rsync, so a `data/` entry is written without `data/` (`/private_documents/`). The script refuses a dirty tree
(`ALLOW_DIRTY=1` overrides) and warns when HEAD is not a release commit. It then runs `scripts/import_check.py --env .env.v45`
on the server and prints the server's `VERSION`. It never touches `/opt/chatbot`.

**`promote.sh`** (on the server, `/root/v38` to `/opt/chatbot`) runs these steps in order:

1. **Snapshot** of `/opt/chatbot`'s `src scripts docs tests data state VERSION .env` (plus the old `logs/`) to `/root/snapshots/<UTC time>_<old version>/`.
2. **Rsync**, by directory as `deploy.sh` does: code mirrored, `data/` never `--delete`, the same exclude list. A failed rsync restores the snapshot and stops.
3. **Flags**: every line of `/root/v38/.env.v45` that matches `KEY=true|false|yes|no|on|off` (quotes allowed), with KEY not starting `LLM_`, `EMBEDDING_`, `NEO4J_` or `STRIPE_`, is set in `/opt/chatbot/.env`. Each change is printed (`+` added, `~` changed). No other line of `.env` is touched; LLM settings never are.
4. **Import check**: `venv/bin/python scripts/import_check.py --env .env` in `/opt/chatbot`. If it fails, the snapshot is restored and the service is **not** restarted.
5. **Restart**: `systemctl stop chatbot`, then the one-time state move (below), then `systemctl start chatbot`.
6. **Report**: `VERSION`, `systemctl is-active chatbot`, and `GET http://127.0.0.1:8000/api/version` (waits up to 90 s). The last line prints the rollback command.

`SRC`, `DST`, `SERVICE`, `CHATBOT_URL` and `SNAPSHOTS` override the defaults.

**Rollback**: `bash /root/v38/scripts/promote.sh --rollback /root/snapshots/<dir>` puts the snapshot back (code mirrored,
data and state added over, VERSION and .env replaced) and restarts.

## 3. First promote of v77: the state move

v77 moves the runtime state out of `data/` and `logs/` into `/opt/chatbot/state/`. The code reads it from there
(`src/state_paths.py`: `STATE_DIR`, else `/opt/chatbot/state` when `/opt/chatbot` exists, else `<checkout>/state`).
Step 5 of `promote.sh` moves it while the service is stopped, so no session write is lost:

| before | after |
|---|---|
| `data/sessions.json` | `state/sessions.json` |
| `data/tenant_logos/` | `state/tenant_logos/` |
| `data/system_templates/` | `state/system_templates/` |
| `data/article_cards.jsonl` | `state/article_cards.jsonl` |
| `data/logs/`, `logs/` | `state/logs/` (merged; the old folders are left empty) |

Each item moves only when the old one exists and the new one does not. When both exist, the script prints a warning
and moves nothing. It also warns when `.env` still points `SESSIONS_FILE`, `SYSTEM_TEMPLATES_CATALOG_PATH`,
`ARTICLE_CARDS_FILE`, `CYPHER_LOG_PATH` or `THESIS_LOG_DIR` at an old location. Those overrides still win over
`state/`, so remove them.

Before the first v77 promote, check two things by hand:

- `systemctl cat chatbot | grep -i environment` must not set any of those variables either.
- `du -sh /opt/chatbot/data /opt/chatbot/state` gives the snapshot's size. `data/private_documents/` (user uploads) stays in `data/` and is copied into every snapshot.

The test copy has no state of its own. With no `STATE_DIR` in `.env.v45`, it reads `/opt/chatbot/state`, exactly as
it used to read `/opt/chatbot/data`.

The template scripts (`scripts/add_new_templates.py`, `replace_templates.py`, `regenerate_*.py`,
`register_system_templates.py`, …) now default to `/opt/chatbot/state/system_templates`.

## 4. Acceptance: what a batch report contains

A batch is accepted on:

- **(a) the five-cell nightly** (`scripts/nightly_sim.py`): structure and regressions. One row per posture cell, one column per check, including `version_footer` from v77. Any check that passed in the baseline and fails now is listed under «Regressions».
- **(b) the held-out set scored by the lawyer**: substance. The nightly's report ends with his latest mean per version.

**When you report a batch, give the nightly table and the version, nothing else as a score.** The 31 eval cases
(`tests/eval_narrative_cases.json`, `TUNING_IDS` / `HELDOUT_IDS` in `scripts/diagnose_long.py`) are retired as a
measure:

- Report no "topic recall vs v59-all", no MENTIONED, extras or misused counts, and no score on the 31.
- Validate no new rule against them.
- They stay only as fixtures for the unit tests that already use them.
- `diagnose_long.py` runs no default batch, so you name the cases with `--only` for a diagnosis. Its `--interactive-sim` report carries no score.
- `scripts/intake_check.py` checks the grid by default; `--only sim` is for diagnosis only.

Run the nightly on the live copy after each promote (and from cron at 02:15 UTC):

```bash
cd /opt/chatbot && cp -f /var/log/maint/nightly.json /var/log/maint/nightly.prev.json; \
  venv/bin/python scripts/nightly_sim.py --env .env --out /var/log/maint/nightly.md \
  --json-out /var/log/maint/nightly.json --baseline /var/log/maint/nightly.prev.json
```

The report's first line after the title is `Version: <VERSION>`. The JSON carries `"version"`. Exit 1 means at least
one failure.

## 5. The held-out loop

Every case-first answer that the live copy or the test copy serves is saved to `/opt/chatbot/heldout/` by
`api._interactive_reply`: the first parere (`kind` «answer») and the pareri written again (`regenerate`). Each record
holds the prompt, the uploaded case text when the case was a file, the intake, the answer with its footer,
`article_citations`, the version, the timings and `source` (`live` or `test`). A save that fails is logged and never
affects the answer. These are client cases: they stay on the server, and the pack goes only to the lawyer.

```bash
cd /opt/chatbot
venv/bin/python scripts/heldout_export.py -n 10 --out /root/pack_$(date +%F).md      # --version V, --source live
# the lawyer fills «voto» (0-5) and «note» for his seven sections and returns the file
venv/bin/python scripts/heldout_score.py /root/pack_2026-11-20_filled.md             # --dry-run to check first
venv/bin/python scripts/heldout_score.py --summary                                   # the per-version table
```

- The pack holds the newest unscored first pareri. Cases already sent in a pack that hasn't been scored are skipped (`exports.jsonl`; `--include-pending` lists them). `--with-updates` adds the regenerated pareri.
- The seven sections come from his critique: tesi centrale, errori di categoria, contraddizioni e incongruenze interne, profili processuali, cosa manca, pregi, strategia. Each takes a score from 0 to 5; a blank means not applicable. The labels are his own headings and still need his confirmation. If he renames one, change only the label in `heldout.SECTIONS`; the key stays, so scores already recorded remain comparable.
- `heldout_score.py` stops before writing anything if any score is out of range or not a number. A case with no score at all is listed as not scored yet. A case scored again counts once, with its later score.
- A version's mean is the mean of its case means. Compare versions only when each has several scored cases.

## 6. Checks before and after

- Before `release.sh`: the full test suite in both modes, plain and all flags on (`tests/conftest.py`). It pins `STATE_DIR` and `HELDOUT_DIR` to temporary folders, so a test run never writes into `/opt/chatbot`.
- After `deploy.sh`: the import check line and the server's VERSION. Then a hand test on `serve_test.sh`.
- After `promote.sh`: `is-active` is `active`, `/api/version` is the new version, and the nightly passes with the new version in its report.
