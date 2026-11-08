# Cassazione batches: our part of the pipeline (feeder, ingest workers, linker, cards window, health)

Written 6 Nov 2026 for the first ~35k batch (Monday 10 Nov) and the next one (~17 Nov). The database question
(size, vector version, backups, where it lives at 400k) is a separate discussion and is not covered here.

Rules that hold for every step: every file replaced on a server gets a dated backup copy first; the user runs
every server command; nothing goes live before the step above it is verified; prod's git is frozen, deploys are
file copies; never delete data (the feeder and the check set files aside, they never remove them).

Measured on the 2025 batch (5,307 files, closed 6 Nov): ingest 61 files/h at one worker with the full model
analysis (24 days per 35k); cards 6 workers ~2,800/h with the GPU idle; linker full pass 17 min; a second ingest
worker is expected, not measured, to add 50-100%.

## 0. What is already built (repo, local commits 6 Nov, 35 tests)

| piece | file | what it does |
|---|---|---|
| feeder | `scripts/maint_feed.py` | `feed`: tops each queue folder up to `--target` every run, pause file, manifest line per file, already-fed files set aside. `check`: the batch report before feeding (`--graph`, `--set-aside`). |
| linker | `scripts/link_ruling_citations.py` | `--write` stamps `Document.cites_linked_at`; `--new-only` reads only unstamped rulings; `--undo` clears stamps. |
| health | `scripts/maint_health.py` | several `--queue-dir` folders, `--units` journals, feeder line, disk free / store size, flags: chunk fallbacks, stalled ingest, disk floor. |
| worker 2 | `ops/be2/sunnitai-ingest-2.service`, `ops/be2/env.worker2` | second ingest instance on `/opt/sunnitai-be/document_upload_2`. |
| crontab | `ops/be2/crontab_lines.txt` | feeder every 10 min, pause/resume around the cards, health args, prod cards `--stop-at 22:30`. |

## 1. Linker: incremental nights (this week, ~25 min, any time the cards are not running)

```
# PC -> be2 (then on be2 as root)
scp scripts/link_ruling_citations.py sunnit@185.199.24.31:~/link_ruling_citations.py
LINKER=<path printed by: grep -o "[^ ]*link_ruling_citations[^ ]*" /opt/v2tools/link_nightly.sh>
sudo cp "$LINKER" "$LINKER.bak_$(date +%d%b)"
sudo cp ~/link_ruling_citations.py "$LINKER"
md5sum "$LINKER" ~/link_ruling_citations.py                      # equal
# one full pass, stamps every ruling (17 min at 45k): run it the way link_nightly.sh does, with --write
sudo bash -c 'grep -n "link_ruling" /opt/v2tools/link_nightly.sh'  # the exact python line and its options
...that line, by hand, with --write   (output ends with "CITES edges written (MERGE): N; rulings stamped cites_linked_at: ~45,700")
# then the nightly line gets --new-only
sudo cp /opt/v2tools/link_nightly.sh /opt/v2tools/link_nightly.sh.bak_$(date +%d%b)
sudo sed -i 's/--write/--write --new-only/' /opt/v2tools/link_nightly.sh
grep -n "new-only" /opt/v2tools/link_nightly.sh                    # exactly one line
```
Verify next morning: `/var/log/link_rulings_nightly.log` says "Reading N rulings without cites_linked_at" with N
= the rulings ingested since, and the pass took seconds, not 17 minutes.
Rollback: restore the two `.bak` files. The stamps are harmless to the old code.

## 2. Feeder and health files (this week, 5 min; nothing runs until the crontab lines change)

```
scp scripts/maint_feed.py scripts/maint_health.py sunnit@185.199.24.31:~/
sudo cp /opt/v2tools/maint_health.py /opt/v2tools/maint_health.py.bak_$(date +%d%b)
sudo cp ~/maint_feed.py ~/maint_health.py /opt/v2tools/
md5sum /opt/v2tools/maint_feed.py ~/maint_feed.py /opt/v2tools/maint_health.py ~/maint_health.py
# the health report's crontab line (05:30 CET) gets the second queue and the staging folder, see crontab_lines.txt
```
Verify: the next 05:30 report has a "disk:" line and "service sunnitai-ingest active" in the night-jobs section.
Done 7 Nov (checksums equal; test run to /tmp/maint_test fine: both queues, both services, disk 175 GB free).
Rollback: restore the health `.bak`; the feeder file is inert until its crontab line exists.

## 3a. Lever 3: three edits in the ingest (6 Nov, `ops/be2/patch_ingest_lever3.py`)

The script checks both files against the 6 Nov copy's checksums, backs them up with a dated suffix, and refuses
if anything differs. Edits: first chunk attempt 60 s instead of 15 (the retry 120), the per-document chunk burst
becomes the setting `ANALISI_CHUNK_CONCURRENCY` (default 50, unchanged today), the 5 s sleep per file becomes 0.5 s.
```
scp ops/be2/patch_ingest_lever3.py sunnit@185.199.24.31:~/
# be2: queue must be empty (ls /opt/sunnitai-be/document_upload | grep -c pdf -> 0)
sudo python3 ~/patch_ingest_lever3.py
sudo /opt/sunnitai-be/venv/bin/python3 -m py_compile /opt/sunnitai-be/batch_ingest.py /opt/sunnitai-be/src/be/src/lex_package/analisi_parallel.py && echo compiles
sudo systemctl restart sunnitai-ingest && sleep 3 && systemctl is-active sunnitai-ingest && journalctl -u sunnitai-ingest -n 2 --no-pager
```
Verify: "active" and the journal's last line says "Batch ingest daemon starting — watching /opt/sunnitai-be/document_upload".
On Monday's first files: no "timed out after 15s" lines any more (`journalctl -u sunnitai-ingest --since "1 hour ago" | grep -c "timed out"`).
Rollback: `sudo python3 ~/patch_ingest_lever3.py --undo && sudo systemctl restart sunnitai-ingest`.

## 3b. Lever 2: second ingest worker (it idles until Monday)

The unit is derived from the existing one, so User, WorkingDirectory and Restart are copied, not guessed; the
only differences are the description and a second EnvironmentFile read after .env.
```
scp ops/be2/env.worker2 sunnit@185.199.24.31:~/
# be2
sudo mkdir -p /opt/sunnitai-be/document_upload_2 && sudo chown --reference=/opt/sunnitai-be/document_upload /opt/sunnitai-be/document_upload_2 && sudo chmod --reference=/opt/sunnitai-be/document_upload /opt/sunnitai-be/document_upload_2
sudo cp ~/env.worker2 /opt/sunnitai-be/.env.worker2 && sudo chown --reference=/opt/sunnitai-be/.env /opt/sunnitai-be/.env.worker2 && sudo chmod 600 /opt/sunnitai-be/.env.worker2
sudo touch /var/log/sunnitai-ingest-2.log && sudo chown --reference=/var/log/sunnitai-ingest.log /var/log/sunnitai-ingest-2.log
UNIT=$(systemctl show -p FragmentPath --value sunnitai-ingest); echo "$UNIT"
sudo cp "$UNIT" /etc/systemd/system/sunnitai-ingest-2.service
sudo sed -i -e 's/^Description=.*/Description=SunnitAI batch ingest, worker 2 (queue document_upload_2)/' -e '/^EnvironmentFile=\/opt\/sunnitai-be\/\.env$/a EnvironmentFile=/opt/sunnitai-be/.env.worker2' /etc/systemd/system/sunnitai-ingest-2.service
systemctl cat sunnitai-ingest-2 | grep -vE "^#"                    # two EnvironmentFile lines, .env first
sudo systemctl daemon-reload && sudo systemctl enable --now sunnitai-ingest-2 && sleep 3
systemctl is-active sunnitai-ingest-2; journalctl -u sunnitai-ingest-2 -n 2 --no-pager
```
Verify: the journal's last line says "watching /opt/sunnitai-be/document_upload_2" (not the first queue).
If it says the first queue, stop it at once (`sudo systemctl disable --now sunnitai-ingest-2`): the override file
was not read. Both workers now idle, polling empty folders every 30 s, until the feeder fills them.

## 3c. Lever 4 (skip the model analysis): measured 7 Nov, not built

Self-retrieval on the 2025 batch (`scripts/self_retrieval_check.py`, app query, k 30, index section_embeddings_v2),
480 rulings per group, same court section in both groups (all section 7, the only section the analysis ran on):

| query (from the ruling's card) | with analysis | without | 
|---|---|---|
| facts + issues, hit@10 / hit@30 | 69% / 77% | 71% / 78% |
| decisive argument | 15% / 25% | 20% / 25% |
| verbatim quote | 23% / 30% | 27% / 34% |

Equal within noise (a 200-ruling run had shown an 8-point gap the other way: sampling noise). Sections per ruling
8.5 in both groups; 1,009 vs 830 chars per section. The app reads nothing the analysis produces (no reference to
`document_number` or the analysis fields in `src/`); the cards and the linker ran on both groups. Limit: only
section-7 orders could be compared; long sentences of the other sections have no no-analysis sample in the graph.
Status 7 Nov: linker step 1 done (full pass 10:00 CET, 453,070 edges, 45,646 stamped, `--new-only` on the nightly line).

### 3c.1 The switch (prepared 7 Nov, `ops/be2/patch_ingest_lever4.py`, rehearsed on the post-lever-3 copy; NOT applied)

`INGEST_SKIP_ANALYSIS=1` in a worker's environment file turns off the four model calls per file: the chunk analysis
(~50 calls), the title/number/date extraction (1), and post-processing's short-section abstract and section title
(up to one per section). The file is still parsed, split, embedded and typed 'interpretation' from its sidecar name.
What the cheap path leaves out, for the PM: the model's one-sentence "requirement" per section (the section
abstract becomes the first 200 words of the text instead), the model's 3-8 word section titles that the answer's
source list shows, and the ruling number on the Document node (nothing in the chatbot reads it). Set it in
`.env.worker2` only: worker 2 runs cheap, worker 1 full, on the same batch.
```
scp ops/be2/patch_ingest_lever4.py sunnit@185.199.24.31:~/
sudo python3 ~/patch_ingest_lever4.py --check                     # three OK lines, "ready to apply"; else stop
sudo python3 ~/patch_ingest_lever4.py                             # backups *.bak_lever4_<date>
sudo /opt/sunnitai-be/venv/bin/python3 -m py_compile /opt/sunnitai-be/src/be/src/lex_package/analisi_parallel.py /opt/sunnitai-be/src/be/src/lex_package/utils/flatten.py /opt/sunnitai-be/src/be/src/lex_package/utils/post_process.py && echo compiles
echo 'INGEST_SKIP_ANALYSIS=1' | sudo tee -a /opt/sunnitai-be/.env.worker2
sudo systemctl restart sunnitai-ingest-2 && sleep 3 && systemctl is-active sunnitai-ingest-2
```
Worker 1 is not restarted: without the variable the edits are inert, so it may be restarted any time later.
Verify on worker 2's first file: `journalctl -u sunnitai-ingest-2 --since "1 hour ago" | grep -c "SKIP ANALYSIS"`
gives 2 per file (chunks + metadata), and no "structured attempt" lines at all for that unit. The morning summary
then reads "ingest 24 h: N written (M without analysis)".
Off: delete the line from `.env.worker2`, restart worker 2. Rollback of the code: `sudo python3 ~/patch_ingest_lever4.py --undo`
(restart both workers afterwards).

### 3c.2 The comparison on the batch (the morning after the first cards)

Both workers take files from the same feeder, so the two halves have the same mix. After the night's cards:
```
cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/self_retrieval_check.py --batch cass_batch1 --cards /home/sunnit/cards_pen_v3.jsonl,/home/sunnit/cards_civ_v1.jsonl --per-group 400
```
(`scripts/self_retrieval_check.py` of 7 Nov; the card files copied from prod's /root/ruling_cards that morning.)
The groups are the batch's rulings with and without a ruling number, i.e. worker 1 and worker 2. Same reading
rule as before: within 3 points on facts+issues the cheap path stays on; 5 or more, turn it off (above).

### 3c.3 Found while building it: error strings as abstracts (counted 8 Nov: 37,510 sections, 5,993 rulings)

A chunk whose model call fails twice (structured, then plain text: the 404 of 2-5 Nov, a chunk timeout, any
error) gets the module's last-resort "requirement", the literal "[Errore analisi comma]". flatten.py joins the
chunk requirements of a section with "; " into the abstract, so the string is there alone when every chunk
failed (the 2025 no-analysis rulings) or mixed with real summaries when one chunk of several failed (rulings of
other batches too, hence 5,993 rulings, not 4,700). The post-processing only replaces abstracts under 20
characters; this one is 22. The document description (first 500 words of the section abstracts) carries it as
well. The chatbot returns the abstract in the section view, searches abstracts and descriptions for keywords,
and the nightly job embeds abstract + plain_text.

`scripts/fix_error_abstracts.py` counts (default: sections, rulings, descriptions) and with `--write` rebuilds
each abstract by flatten's own rule: the good fragments joined with "; " (first 200 words of the text when none
is left), the core text kept, the title in front; then rebuilds the descriptions along NEXT. Old values stay as
`abstract_before_fix` / `description_before_fix` (`--undo`; `--purge-backups` once checked). `--reembed` also
parks `embedding_v2` so the 02:00 backfill re-embeds the fixed sections; until then they are out of vector
search, so evening only, or start the backfill by hand. Not needed for retrieval (the measured no-analysis
group carried the string in its vectors and retrieved like the other). The write is the user's decision; it
is a plain Neo4j write, so not while the cards run (22:00 CET) and not during a backfill.

First write, 8 Nov afternoon: stopped by its own guard after 1,000 sections (11 fixed, 989 rewritten with the
string still inside) and 3,500 descriptions. Cause: the post-processing strips "Testo significativo:" and "0 "
from every abstract after flatten builds it, so the stored shape is "<title> - <good>; [Errore analisi comma]
<core text>" and the title may no longer match titolo_articolo; the first rule split on "; " and kept the
error-plus-core-text fragment as good. The rule is now shape-independent (error fragments removed wherever they
sit; title-only leftovers -> first 200 words). `--shapes` is the read-only check to run before `--write`: masked
shapes (W<n> = n words, § = the string) of a 3,000-section sample, what the fix would do with each, and the
NEXT-chain health of the documents already rebuilt. The first run's backups are kept (coalesce), so a rerun
leaves the original as the restore point.

## 4. Monday: the batch lands

1. The batch goes to `/home/sunnit/cass_batch1`, flat (PDF + `.name` sidecar per file, as the 2025 batch). Names
   must carry the Italgiure code (`snpen`/`snciv`, year, number): the labels and the duplicate check read it.
2. The check, before any file moves (the graph part uses the app's venv for the Neo4j driver):
   ```
   cd /opt/chatbot && sudo venv/bin/python -I /opt/v2tools/maint_feed.py check --src /home/sunnit/cass_batch1 --batch cass_batch1 --graph /opt/v2tools/health.env
   ```
   Read the counts. "already handed" and "already in the graph" are expected for the 2025 leftovers: rerun with
   `--set-aside` and they move to `_already_fed` / `_already_in_graph` inside the staging folder. Duplicate
   names inside the batch or repeated display names are a stop: the ingest deletes the earlier ruling when a
   later file carries the same name. Files without the code get no branch label: set them aside by hand.
3. Two-worker measurement, by hand, before the crontab: feed 60 files to each queue once
   (`sudo python3 /opt/v2tools/maint_feed.py feed --src ... --batch cass_batch1 --queue .../document_upload --queue .../document_upload_2 --target 60`),
   wait an hour, then:
   ```
   sudo python3 -c "import sys; sys.path.insert(0,'/opt/v2tools'); import maint_health as m; print(m.ingest_journal('70 minutes ago', units=('sunnitai-ingest','sunnitai-ingest-2')))"
   ```
   Baseline at one worker: ~60 written, 0 chunk_fallbacks. Two workers pass when `written` is clearly above 60
   and `chunk_fallbacks` stays 0. Fallbacks above 0 = the GPU is oversubscribed: stop worker 2
   (`sudo systemctl disable --now sunnitai-ingest-2`, move its queue's files back to staging), continue at one
   worker, and give Flavio the numbers (his code: Semaphore(50) per document and the 15 s / 60 s timeouts in
   `lex_package/analisi_parallel.py`).
4. The crontab: the install block at the end of `crontab_lines.txt` (dump, rewrite the copy, diff, install;
   rehearsed 7 Nov): the feeder replaces the hourly feed_queue.sh line, the health line gets both queues and the
   staging folder, the pause and resume lines are appended. On prod the cards line gets `--stop-at 22:30`.
5. Day one, afternoon: trace three new rulings (`scripts/trace_ruling.py`, names masked): number and date
   present, sections by the ruling's structure, one CITES set after the next linker pass.

## 5. Every morning during a batch (the 05:30 report, `/var/log/maint/summary_<date>.txt`)

| line | fine | act |
|---|---|---|
| ingest 24 h: N written | ~1,400 at one worker, ~2,500 at two | 0 while files wait = stalled: `systemctl status`, journal |
| chunk fallbacks | 0 | > 0: stop worker 2 (above) |
| LLM model errors | 0 | > 0: the ingest .env names a model the server no longer serves |
| disk free | > 40 GB | the page flags it: `touch /var/log/maint/feed.pause` and the database discussion |
| feeder: N left | falling by the day's written | PAUSED outside 21:00-23:35 CET = the pause file was left behind |
| suspected duplicates | 0 | > 0: trace them before feeding more |
| rulings without card | growing by day, cleared by the night | the cards window is too short: `--stop-at` later |

## 6. Rollback, in one place

- Worker 2: `sudo systemctl disable --now sunnitai-ingest-2`; `sudo mv /opt/sunnitai-be/document_upload_2/*.pdf /opt/sunnitai-be/document_upload_2/*.name /home/sunnit/cass_batch1/`.
- Feeder: comment its crontab line; the old `feed_queue.sh` line can return (same manifest, same queue).
- Linker: restore the `.bak` of the script and of `link_nightly.sh`; the stamps stay and hurt nothing.
- Health: restore the `.bak`; the old report ignores the new arguments' files.
- Cards: `--stop-at 00:45` back on prod's line.
