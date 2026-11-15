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

DONE 8 Nov, second write: 37,499 section abstracts (every one had nothing left of the analysis; the heading the
abstract carried was kept in front) and 5,993 descriptions rebuilt in about two minutes; every count 0 after.
Backups stay on the nodes (abstract_before_fix 37,510, description_before_fix 5,993) until `--purge-backups`.
Vectors were not re-embedded. maint_health now counts "error abstracts" every night (summary part + page line),
so a recurrence from Monday's batch (a chunk failing twice under the 60 s attempt) is visible; rerun the fix
script when the count grows. Seen on the way, for Flavio: the rulings' NEXT chains are not single chains (a
section with two NEXT in 244 of 300 sampled documents, typically 3-5 heads), and the split pieces name_i are a
quarter of the rulings' sections; the app's reading-order fallback (name order) already covers it.

Recurrence 14 Nov: the model server restarted at 06:32 CET (its own `process_start_time_seconds`; why is not
known, it is not our machine). For a few minutes every call of the three workers failed with "Connection
error". The ingest treats that like a bad chunk (no retry, then plain text, which fails too), so 216 chunks got
the string: "error abstracts 107 (+106)" in the 15 Nov report and 1.5% fallbacks over that day. No error in any
hour since. The ingest has no guard for a model server that is down, so the next restart of that server does
the same.

Repaired 15 Nov with the script above (`--shapes`, then `--write`, three seconds, workers running): 107 sections
in 15 rulings. 104 had nothing of the analysis left and got the first 200 words of their text with the heading
in front; 3 kept their good parts; 15 descriptions rebuilt; every count 0 after. Vectors not re-embedded. These
15 rulings stay without model summaries in most sections: only feeding them again would give them one, and
there is no tested procedure for that. Mind `--undo`: it restores every node holding a `*_before_fix` backup,
and the 8 Nov backups are still on the nodes (37,617 sections now), so it would put the error text back into
all of them. It is an undo of the last write only after `--purge-backups` has removed the older backups.

## 3d. Lever 5: the chunk waits as a setting (prepared 11 Nov, `ops/be2/patch_ingest_lever5.py`)

Why: the journal of 10-11 Nov (763 rulings, both workers). A chunk call answers within 30 s or never. 4.5% of the
chunks hang on the first request, and a fresh request for the same chunk answers in about 6 s half the time. The
waits are 60 s, then 120 s (inside it the client's own 90 s limit and a third request), so the waiting is about
40% of all worker time and 1% of the chunks end on the plain-text analysis. The model server's counters (vLLM
`/metrics`) show no answer over 10,000 tokens and no call over 120 s: nothing runs on after the ingest hangs up.
Why a call hangs is not known.

The patch makes the waits a setting, `ANALISI_CHUNK_TIMEOUTS` (default "60,120"; a value under 30 s or unreadable
falls back to it). The test: worker 2 at `40,40,40` (three requests, as today) beside worker 1 on the default.
**Applied on be2 on 11 Nov 2026 at 12:54 CET** (the comparison below starts from that time).

Result after 21 hours (12 Nov 09:53). The night before the change the two workers were equal (422 and 429 rulings,
plain text 1.14% and 1.16% of chunks). After it: worker 1 wrote 722 rulings from 8,980 chunks, worker 2 wrote 867
from 10,436 (+20% rulings, +16% chunks). Plain text 1.10% against 1.20%, inside chance at this size (0.15 points).
First-attempt timeouts 5.1% and 5.2% of chunks, so 40 s cuts no normal call; a hung chunk ends on plain text 23%
of the time on both. The quality guard holds: the setting stays.

The gain is worker 2's own, not taken from worker 1 (the "[TIMING] Parallel LLM analysis complete" lines, same
periods). For like rulings (8-16 chunks, no timeout) worker 1's analysis took 16.3 s before and 16.1-16.2 s after,
average call 7.3 s throughout: worker 2 asking again sooner does not slow worker 1's calls. What changed is the
ruling with a hung chunk, 4 rulings in 10 on both workers: 116 s of analysis on worker 1, 77 s on worker 2. Per
ruling, all phases: 91 s against 72 s. Chunks an hour is not a measure of capacity here: worker 1's fell (528 ->
485) only because its rulings got shorter, while its rulings an hour rose (37 -> 39.5).

**Worker 1 has the setting too since 12 Nov 2026, 11:24 CET**: the same line appended to `/opt/sunnitai-be/.env`
(backup `.env.bak_12nov`), worker 1 restarted on its empty queue. Both workers now wait 40,40,40.
```
scp ops/be2/patch_ingest_lever5.py sunnit@185.199.24.31:~/
# be2. Both queues must be empty: by day pause the feeder and let them drain (~75 min), or use the evening pause.
sudo touch /var/log/maint/feed.pause
find /opt/sunnitai-be/document_upload /opt/sunnitai-be/document_upload_2 -maxdepth 1 -name '*.pdf' | wc -l   # 0
python3 ~/patch_ingest_lever5.py --selftest && sudo python3 ~/patch_ingest_lever5.py --check
sudo python3 ~/patch_ingest_lever5.py
sudo /opt/sunnitai-be/venv/bin/python3 -m py_compile /opt/sunnitai-be/src/be/src/lex_package/analisi_parallel.py && echo compiles
echo 'ANALISI_CHUNK_TIMEOUTS=40,40,40' | sudo tee -a /opt/sunnitai-be/.env.worker2
sudo systemctl restart sunnitai-ingest sunnitai-ingest-2 && sleep 3 && systemctl is-active sunnitai-ingest sunnitai-ingest-2
sudo rm -f /var/log/maint/feed.pause     # by day only; in the evening window the 23:35 crontab line removes it
```
Verify after ~20 minutes: `sudo journalctl -u sunnitai-ingest-2 --since "30 minutes ago" | grep -c "timed out after 40s"`
is above 0, and the same count for `-u sunnitai-ingest` is 0 (worker 1 still says "after 60s"). With three waits the
journal's third attempt reads "attempt 3/2".

Compare after some hours, per worker, from the restart time:
```
for u in sunnitai-ingest sunnitai-ingest-2; do sudo journalctl -u $u --since "<restart time>" --no-pager -o short-iso | awk -v u=$u '/\[4\/4\] done/{w++} /total_chunks=[0-9]+/{match($0,/total_chunks=[0-9]+/); c+=substr($0,RSTART+13,RLENGTH-13)} /structured attempt/&&/timed out/{t++} /trying plain text fallback/{f++} END{printf "%s: written %d, chunks %d, timeouts %d, plain text %d (%.2f%% of chunks)\n", u, w, c, t, f, (c ? 100*f/c : 0)}'; done
```
Keep it, and set it for worker 1 too (the same line in `/opt/sunnitai-be/.env`, restart on an empty queue), only
when worker 2 wrote clearly more rulings AND its plain-text share is not above worker 1's (about 1%; at ~4,000
chunks per worker a difference under 0.3 points is noise). The plain-text share is the quality guard: a higher
share means more sections with the simpler analysis.
Rollback: delete the line from `.env.worker2` and restart worker 2 (the default again; the patched file behaves
as before). Full: `sudo python3 ~/patch_ingest_lever5.py --undo`, restart both workers on empty queues.
Lever 4's patch expects the file as it was before lever 5: give it the new checksum (c2a57da4...) before applying it.

## 3e. Third ingest worker, off-hours only (prepared 13 Nov)

Rule since 13 Nov: nothing goes live that may change the chatbot's answer quality or its answer times. By day a
third worker would send half as many calls again to the model server the chatbot shares, and nobody has measured
what that does to an answer. So the third worker works only while nobody uses the chatbot: 23:00-06:00 Italian
time (phase 1) and, once a night's reading is clean, Saturday and Sunday too (phase 2, `--weekends`). Production's
log for 6-13 Nov has no answer on the weekend of 7-8 Nov and one at night (02:25 on 12 Nov).

The feeder decides by the clock at every run (`maint_feed.py feed ... --off-hours-queue ... --off-hours-unit ...`):
off-hours it tops up the third queue and starts the unit; by day it gives that queue nothing and stops the unit as
soon as the queue holds no PDF (a queue file is deleted only when its ruling is fully written). The last files
are handed at 05:50 and are done before 07:00. A missed cron line cannot leave this worker running by day.

On 11-12 Nov worker 2 sent 16% more calls and worker 1's normal calls took the same time, and the model server
showed no request waiting, so the server is not the limit at two workers. be2 has cores to spare (8, load 0.45)
but little memory: on 13 Nov the database holds its full 10.5 GB (heap fixed at 6 GB plus its data cache) and
2.6 GB is available; a worker takes about 0.5 GB while it runs and nothing once stopped. The "5 GB available" of
12 Nov was misleading: up to 2.9 GB of the database sat in swap then (systemd's "memory swap peak" for the run that
ended on 13 Nov at 06:30, when an automatic system update restarted every service). What is not known: how much a
third worker adds (estimate: about 46 rulings an hour like the others, 340 a night, 1,000 a weekend day).

Before: the morning reading with two workers (section 3d's comparison and timing commands) is the baseline:
13 Nov, rulings of 8-16 chunks without a timeout 16.0-16.5 s of analysis, average call 7.2-7.5 s, plain text 1.1%.
```
# from the PC
scp scripts/maint_feed.py ops/be2/env.worker3 sunnit@185.199.24.31:~/
# be2: the queue, the settings, the log, the unit (derived from worker 2's installed unit)
sudo mkdir -p /opt/sunnitai-be/document_upload_3 && sudo chown --reference=/opt/sunnitai-be/document_upload /opt/sunnitai-be/document_upload_3 && sudo chmod --reference=/opt/sunnitai-be/document_upload /opt/sunnitai-be/document_upload_3
sudo cp ~/env.worker3 /opt/sunnitai-be/.env.worker3 && sudo chown --reference=/opt/sunnitai-be/.env /opt/sunnitai-be/.env.worker3 && sudo chmod 600 /opt/sunnitai-be/.env.worker3
sudo touch /var/log/sunnitai-ingest-3.log && sudo chown --reference=/var/log/sunnitai-ingest.log /var/log/sunnitai-ingest-3.log
sudo sed -e 's/worker2/worker3/g' -e 's/worker 2/worker 3/' -e 's/document_upload_2/document_upload_3/' /etc/systemd/system/sunnitai-ingest-2.service | sudo tee /etc/systemd/system/sunnitai-ingest-3.service > /dev/null
sudo systemctl daemon-reload && systemctl cat sunnitai-ingest-3 | grep -vE "^#"      # EnvironmentFile .env, then .env.worker3
# started only if the unit reads .env.worker3 and names nothing of worker 2: two daemons on one queue would race
F=/etc/systemd/system/sunnitai-ingest-3.service; if grep -q "^EnvironmentFile=/opt/sunnitai-be/.env.worker3" $F && ! grep -qE "worker2|document_upload_2" $F; then sudo systemctl start sunnitai-ingest-3 && sleep 5; else echo "NOT STARTED: the unit file is not as expected"; fi
systemctl is-active sunnitai-ingest-3; sudo cat /proc/$(systemctl show -p MainPID --value sunnitai-ingest-3)/environ | tr '\0' '\n' | grep -c '^BATCH_UPLOAD_DIR=/opt/sunnitai-be/document_upload_3$'
```
Verify: "active" and 1 (the running process watches `document_upload_3`). Anything else: stop it at once
(`sudo systemctl stop sunnitai-ingest-3`). The unit is not enabled: the feeder starts it. Its queue is empty, so it
sits idle and calls no model. Then the feeder (the old crontab line works unchanged with the new file):
```
md5sum ~/maint_feed.py                                             # the same as `md5sum scripts/maint_feed.py` on the PC
sudo cp /opt/v2tools/maint_feed.py /opt/v2tools/maint_feed.py.bak_13nov && sudo cp ~/maint_feed.py /opt/v2tools/maint_feed.py && md5sum /opt/v2tools/maint_feed.py
```
The switch: the feeder line gains the off-hours queue and unit, the health report the third queue and unit.
```
sudo crontab -l > ~/crontab.bak_worker3
sed -e 's#--queue /opt/sunnitai-be/document_upload_2 --target#--queue /opt/sunnitai-be/document_upload_2 --off-hours-queue /opt/sunnitai-be/document_upload_3 --off-hours-unit sunnitai-ingest-3 --target#' -e 's#document_upload_2 --staging#document_upload_2,/opt/sunnitai-be/document_upload_3 --units sunnitai-ingest,sunnitai-ingest-2,sunnitai-ingest-3 --staging#' ~/crontab.bak_worker3 > ~/crontab.now
diff ~/crontab.bak_worker3 ~/crontab.now                           # two lines change: the feeder and the health report
sudo crontab ~/crontab.now
```
Verify by day, within ten minutes: `tail -n 1 /var/log/maint/feed_queue.log` lists `document_upload_3 0` and ends
with "off-hours worker stopped", and `systemctl is-active sunnitai-ingest-3` says inactive: the day rule works.
That evening the 23:00 line ends with "off-hours worker started" and the 23:40 line feeds three queues.

Guard, the morning after (the commands of 3d with `sunnitai-ingest-3` added to the loop, from 23:40 to 06:45):
- normal calls on workers 1 and 2: more than 10% above the baseline means the model server is shared too thinly
  with three workers, even at night: stop.
- plain text stays near 1.1% of chunks on all three workers, first-attempt timeouts near 5%.
- `grep "off-hours worker" /var/log/maint/feed_queue.log | tail -n 4 | awk '{print $1, $(NF-2), $(NF-1), $NF}'`:
  started near 22:00Z, stopped near 06:00Z (the log's times are UTC). "FAILED" there: the feeder is not root.
- memory (`free -m | awk 'NR==2{print "available",$7} NR==3{print "swap used",$3}'`): before the third worker,
  13 Nov, available 2,600 MB and swap 33 MB. Swap above 3,000 MB (of 4,095): stop. The cost of tight memory is
  the database pushed into swap, slower for the ingest and for the chatbot.
- prod's cards line: "still to card 0".

**Read 15 Nov, the first two nights (23:40-06:25): the speed guard is clean.** Written / chunks / plain text /
analysis of a ruling of 8-16 chunks without a timeout:

| night | worker 1 | worker 2 | worker 3 |
|---|---|---|---|
| 12-13, two workers | 301 / 3,847 / 1.09% / 16.1 s | 316 / 4,036 / 1.14% / 16.6 s | - |
| 13-14 | 312 / 4,312 / 0.83% / 15.8 s | 350 / 4,379 / 0.55% / 17.5 s | 295 / 4,152 / 1.11% / 16.7 s |
| 14-15 | 260 / 4,784 / 0.88% / 16.1 s | 282 / 4,235 / 0.90% / 16.3 s | 279 / 3,980 / 0.90% / 16.5 s |

Three workers do not slow each other: workers 1 and 2 together wrote more chunks with the third beside them
(7,883, then 8,691 and 9,019). The third adds 280-295 rulings a night, not the 340 estimated; the rulings are
getting longer. It started at 22:00Z and stopped at 05:50Z both nights.

Memory: swap used 3,260 MB on 15 Nov, above the line written above. That line was mis-set: it counts the whole
machine, and the number it came from was the database's own. The database's swap peak since the 13 Nov restart
is 2.9 GB (`systemctl show neo4j -p MemorySwapCurrent -p MemorySwapPeak`), the same as with two workers; the
other 1.1 GB is be2's other services parked, and `vmstat 5 4` shows nothing pushed out by day. So the third
worker is not shown to push the database deeper. Proposed lines, not yet agreed: stop when the database's peak
passes 3.2 GB or total swap 3,700 MB. Standing fact: the database always has 2-3 GB in swap on be2; what that
costs a chatbot search is not measured.

Prod's cards on 14 Nov: 2,335 in 80 of their 90 minutes. The window has to grow (prod's `--stop-at`, be2's
resume time) before anything raises the daily output further.

Stop for good: `sudo crontab ~/crontab.bak_worker3` (its queue gets no more files), wait until `document_upload_3`
holds no PDF (under an hour), then `sudo systemctl stop sunnitai-ingest-3`.

Phase 2, the weekends, after a clean night:
```
sudo crontab -l > ~/crontab.bak_weekends
sed -e 's#--off-hours-unit sunnitai-ingest-3 --target#--off-hours-unit sunnitai-ingest-3 --weekends --target#' ~/crontab.bak_weekends > ~/crontab.now
diff ~/crontab.bak_weekends ~/crontab.now                          # one line changes: the feeder
sudo crontab ~/crontab.now
```
Three workers all Saturday and Sunday write about 3,000 rulings a day and the cards do about 32 a minute, 2,900
in their 90 minutes: a small "still to card" on Sunday and Monday morning is expected, and Monday night catches up.

## 3f. The chunk calls that never end (found 13 Nov; cause of the endless answer not known yet)

About 5% of chunk calls get no answer within the wait, and waiting for them is about 30% of a worker's time. What
such a call is: on 13 Nov, 12:16-12:26 CET, two workers ingesting, the model server's own counters over the ten
minutes (`<LLM_BASE_URL without /v1>/metrics`):

| | all | of requests that finished | gap |
|---|---|---|---|
| generated tokens | +172,653 | +132,098 | 40,555 = 23.5% of what the server wrote |
| prompt tokens | +370,839 | +352,213 | 18,626 = 5.0% |
| requests | | +289 (stop 281, length 8, abort 0) | about 15 unfinished |

So the unfinished requests are the 5%, and each got some 2,650 tokens written for it, forty seconds of writing
(a finished answer has about 450). Since the server started: 20.9 of 85.3 million generated tokens (24.5%). A
"hung" chunk call is an answer that does not end, dropped when the ingest hangs up; the server counts it neither
as finished nor as aborted. The reading of 11 Nov ("no long answers, so no runaway") looked only at finished
requests. A quarter of the shared model server's writing is this waste.

The request (analisi_parallel.py, llm/factory.py; be2 has langchain-openai 0.3.4, openai 1.59.9; the server is vLLM
0.28.0): `ChatOpenAI(temperature=0, request_timeout=90).with_structured_output(Analisi_Paragrafo)`, no length cap.
The answer's shape has three free text fields and an open list of references.

What the endless answer is made of: `ops/be2/probe_runaway.py` sends the sections of 12 rulings from the graph
exactly as the ingest does (the ingest's own model object builds the request), streamed and capped at 3,000
tokens, and prints numbers and field names only. Read-only; about what one worker sends in six minutes. Its run
of 13 Nov: the request is `response_format` json_schema (strict), no length cap; 117 sections, 113 answers ended
(median 330 tokens, longest 1,253, every one with at least one reference although the ingest sends "flag False"),
4 did not end (3.4%). All four are the same: the content complete in the first 1,100-1,800 characters, then,
inside the list of references and after its first entry, white space only, repeating every 4 characters, to the
cap. The JSON rule the server enforces allows white space anywhere between two elements; at temperature 0 a model
that has started to repeat it never leaves.

The server has a remedy of its own, JSON "always compact without any whitespace" (vLLM's
`disable_any_whitespace`). As a server setting (`--structured-outputs-config`, backend xgrammar or guidance) it
would change every caller's answers, the chatbot's too, and needs a restart of the model server: the proper fix,
and Flavio's and the PM's to decide. Carried by the request it does NOT work on this server (probe's second run,
13 Nov, 147 sections, 10 did not end: 8 in the references list, 2 just before it): next to `response_format` it
is refused (HTTP 400); alone with `structured_outputs.json` it is accepted and ignored (56 of 56 answers still
with line breaks; 27 of 50 did not end against 29 of 50 as today). The same run measured the chance: asked
again, a section that did not end fails 58% of the time; one that ended, 4 times in 60. Two runs of the same
request at temperature 0 give the same tag 49 times in 56 and the same whole answer 22 times in 56.

Remedies a request can carry (probe's third run, 13 Nov: 116 sections, 10 did not end, all in the references
list; then 40 more answers on those ten sections under each remedy, and one each on 40 sections that end):

| remedy | of 40, without end | on 40 sections that end |
|---|---|---|
| as today (asked again) | 10 | 39 ended; same tag as the first answer 36, same whole answer 9, quote word for word 34 |
| frequency penalty 0.1 | 0, all valid JSON | 40 ended; same tag 38, same whole answer 9, quote word for word 34, same sizes |
| frequency penalty 0.3 | 5 | 38 ended; same whole answer 0, quote shorter (534 against 634 characters) |
| stop on repetition (`repetition_detection` 20 / 10) | 0; 7 cut, content complete in all | 40 ended, 1 cut with content complete; nothing else changes |
| white-space pattern `[ ]?` | 5 | no effect |

A frequency penalty makes a piece less likely each time it is repeated, so the endless white space breaks and
the JSON closes by itself: a valid answer, with no change to the ingest's code beyond one request setting. It
has a dose: 0.3 changes the content and leaves answers without end of its own. The stop on repetition is the
gentlest (it cannot touch an answer that does not repeat) but hands the ingest a cut answer, which today's code
would send to the plain-text fallback: it would need code that keeps the complete content.

Next, read-only (about thirty minutes of one worker's load): the dose. 0.05 and 0.1 on 100 sections that end,
0.3 only on the ones that do not, to see what its failures are made of.
```
scp ops/be2/probe_runaway.py sunnit@185.199.24.31:~/
cd /opt/sunnitai-be && sudo venv/bin/python3 ~/probe_runaway.py --remedies --rulings 10 --repeats 4 --normal 100
```
The dose run (13 Nov, 95 sections, 1 did not end; then 94 sections that end, once per form) changed the choice:

| | ended, of 94 | identical to the first answer | same tag | quote word for word | quote, characters |
|---|---|---|---|---|---|
| as today (asked again) | 92 | 34 (37%) | 88 | 81 | 446 |
| frequency penalty 0.05 | 93 | 22 (24%) | 86 | 83 | 507 |
| frequency penalty 0.1 | 93 | 14 (15%) | 90 | 84 | 478 |

The penalty does change the answers, more the stronger it is (fewer identical ones), though the tag and the
word-for-word quote do not suffer; and it does not end every answer: one white-space answer without end under
each strength (over the two runs, on sections that normally end: 3 of 134 as today, 1 of 134 at 0.1; on the ones
that do not: 11 of 44 and 0 of 44). Over the four first passes 25 of 475 answers did not end (5.3%).

So the remedy to build is the stop on repetition (`repetition_detection`, in the request's extra body): it
watches for a short pattern repeated ten times and ends the answer there, and by construction leaves every other
answer as it is. In the third run it ended all 80 answers, and the 8 it cut had their four content fields
complete. The ingest must then keep that content instead of treating the cut answer as a failure (today a cut
answer would go to the plain-text fallback): a small change in analisi_parallel.py, behind a setting that is off
by default, to prepare like levers 3 and 5 (backup, undo, self-test), rehearse read-only on be2 (the same
closing of the cut answer run on real cut answers; how the library reports a cut answer) and try on worker 2
only, not before Monday 16 Nov: the third worker's first nights come first, one change at a time. Guards: the
timeouts per chunk (5% today, about 0 expected), the plain-text share (1.1% today, lower expected), worker 1's
normal calls unchanged. Server load: the writing wasted today (23.5%) pays for about 30% more answers, so even
40% more rulings an hour write about 7% more tokens than now. The server setting stays the proper fix for
everybody, and Flavio's and the PM's to decide.

Seen on the way, not looked into: in the answers that end, the summary field (`requirement`) has a median of 7-8
characters, the quote 550 and the search text 40-50. The request shows the model the two prompt files, which ask
for a tag only; the fields' descriptions sit in the schema, which shapes the answer. Whether the model sees them
is not verified, nor what the stored abstracts look like.

Read in the code while writing lever 6, not verified on stored data: the "plain-text fallback" sends its plain
prompt to the structured model object (`_get_structured_llm_fallback`) and then looks for labelled lines
(`REQUIREMENT:` ...) in the text form of the object that comes back, which has none. If so, a chunk on that
path keeps empty fields and the tag "altro", like a chunk without analysis, not a simpler analysis.

### 3f.1 Lever 6: the stop on repetition (`ops/be2/patch_ingest_lever6.py`; ON on worker 2 since 15 Nov 12:08 CET)

`ANALISI_STOP_REPETITION=1` in a worker's environment file adds `repetition_detection` (20 / 10, the values
measured) to that worker's chunk requests. An endless answer then comes back in the time of a normal one, cut
inside its JSON, and the library cannot read it (a pydantic ValidationError that carries the text). The worker:

- asks again, as it does today after a timeout (the same number of requests, without the 40 s wait), so a
  section gets the full answer whenever a fresh request gives one, exactly as today;
- only when no attempt ends by itself keeps the content of the cut answer (summary, quote, search text, tag,
  the references written whole), where today the chunk goes to the plain-text fallback;
- never keeps an answer cut before its content was complete: asked again, then the fallback, as today.

Without the setting every edit is inert. One file, 5 edits, 83 lines added, checked against be2's checksum
after lever 5 (c2a57da4...); after the patch the file reads aa293a42... Rehearsed 13 Nov on a copy rebuilt to
that checksum: it applies, compiles, refuses a second run, and the undo restores the copy. The library path
(request, cut answer, content kept) ran on the PC's library versions, which are newer than be2's (openai 2.54
against 1.59.9): be2's own are what part 1 of the rehearsal checks.

Rehearsal on be2, read-only, before anything is applied (not between 22:00 and 23:30 CET, the cards' window):
```
scp ops/be2/patch_ingest_lever6.py ops/be2/probe_runaway.py sunnit@185.199.24.31:~/
python3 ~/patch_ingest_lever6.py --selftest && sudo python3 ~/patch_ingest_lever6.py --check
cd /opt/sunnitai-be && sudo venv/bin/python3 ~/probe_runaway.py --lever6 --rulings 20
```
Part 1 sends nothing to the model server: three lines, each must say "yes". A "NO" stops the run: the patch
has to change first. Part 2 sends about 250 sections with the stop through the ingest's own call (ten minutes
of one worker's load). Expected: about 5% cut, every one with the content complete, and no call left without an
answer in 40 s (5% today). "content incomplete" on more than one or two, or `core_text` as the field being
written, means the stop cuts answers that repeat for a good reason: do not apply, the numbers come back first.

**Rehearsal run on be2, 13 Nov: passed.** Checksums as uploaded, the file "ready to apply". Part 1: the request
carries the stop next to `response_format`; a cut answer reaches the ingest as a ValidationError with its text;
the content is kept. Part 2, 203 sections of 19 rulings: 194 ended by themselves (6.0 s median, quote found word
for word in 172), 9 were cut (4.4%, back in 4.0 s median, 43 characters of white space at the end), none was left
without an answer in 40 s (about 10 expected today) and none was cut inside its content. All 9 cut answers had
summary, quote and search text filled and the quote word for word in the section. None had a whole reference
to keep: the white space starts before the first reference is complete (the earlier "after its first entry"
counted a reference as soon as it was opened). So a kept cut answer carries no references, like today's
fallback; a section gets its references from the fresh request, which is why the worker asks again first.
Limit of the sample: 0 answers cut inside their content in 203 leaves room for a rate under about 1.5%; the
plain-text share on worker 2 is the guard for it.

Apply, one worker at a time, on its empty queue (by day the pause empties the queues in about an hour; or use
the evening pause). The patch itself is applied once; the lines below are worker 2's, 15 Nov:
```
sudo touch /var/log/maint/feed.pause
find /opt/sunnitai-be/document_upload_2 -maxdepth 1 -name '*.pdf' | wc -l                                    # 0
sudo python3 ~/patch_ingest_lever6.py --check && sudo python3 ~/patch_ingest_lever6.py
sudo /opt/sunnitai-be/venv/bin/python3 -m py_compile /opt/sunnitai-be/src/be/src/lex_package/analisi_parallel.py && echo compiles
echo 'ANALISI_STOP_REPETITION=1' | sudo tee -a /opt/sunnitai-be/.env.worker2
sudo systemctl restart sunnitai-ingest-2 && sleep 3 && systemctl is-active sunnitai-ingest-2
sudo cat /proc/$(systemctl show -p MainPID --value sunnitai-ingest-2)/environ | tr '\0' '\n' | grep -cE '^(ANALISI_STOP_REPETITION=1|ANALISI_CHUNK_TIMEOUTS=40,40,40)$'   # 2
sudo rm -f /var/log/maint/feed.pause     # by day only; in the evening window the 23:35 crontab line removes it
```
The count of 2 says the running process has both settings (a line glued to the one before it would give less).
Worker 1 and the third worker are not restarted: they read the patched file at their next start, without the
setting, and behave as before. Verify after half an hour: on worker 2 "cut by the repetition stop" is above 0,
"timed out after 40s" about 0 and "/2 failed" 0 (a model server that refuses the stop would show there); on
worker 1 "cut" is 0.
```
for u in sunnitai-ingest sunnitai-ingest-2; do for w in "[4/4] done" "cut by the repetition stop" "content of the cut answer kept" "timed out after 40s" "/2 failed" "trying plain text fallback"; do echo "$u, $w: $(sudo journalctl -u $u --since '30 minutes ago' --no-pager | grep -cF "$w")"; done; done
```
**Applied 15 Nov, worker 2 only.** The model server had restarted on 14 Nov (3c.3), after the rehearsal, so the
rehearsal ran again first (`--rulings 10`): same version (vLLM 0.28.0), part 1 three "yes", 72 sections of 9
rulings: 67 ended by themselves, 5 were cut, all 5 with the content complete, none left without an answer in
40 s. Then the patch (5 edits; backup `analisi_parallel.py.bak_lever6_15nov` beside the file), "compiles", the
line in `.env.worker2`, the restart ("active", count 2) and the pause removed at 12:08 CET: the comparison with
worker 1 starts there (`--since "2026-11-15 12:08"`). Nothing was in progress on worker 2 at the restart (its
last ruling had finished at 12:04:41).

Half-hour verify, about 12:25-12:55: worker 2 wrote 30 rulings with 39 answers cut, 0 timeouts, 0 failed and
0 plain text; worker 1 wrote 21 with 17 timeouts, 1 plain text and 0 cut. The stop works. One number to watch:
10 chunks on worker 2 ended as "content of the cut answer kept", so all three of their attempts were cut. The
old path's rate (plain text about 1% of chunks) would give about 4 in as many chunks. A kept answer has the
content but no list of references, like the plain-text fallback (`riferimenti=None` there too). If a longer
reading confirms it, asking again a few seconds after a cut repeats the endless answer more often than asking
again after a 40 s wait. So the comparison counts the attempts one by one.

Compare after some hours, per worker, from the restart time:
```
for u in sunnitai-ingest sunnitai-ingest-2; do sudo journalctl -u $u --since "<restart time>" --no-pager -o short-iso | awk -v u=$u '/\[4\/4\] done/{w++} /total_chunks=[0-9]+/{match($0,/total_chunks=[0-9]+/); c+=substr($0,RSTART+13,RLENGTH-13)} match($0,/structured attempt [123]\/2 (timed out|cut by the repetition stop)/){a=substr($0,RSTART+19,1); if ($0 ~ /timed out/) t[a]++; else k[a]++} /cut by the repetition stop \(content incomplete\)/{n++} /content of the cut answer kept/{s++} /trying plain text fallback/{f++} END{printf "%s: written %d, chunks %d | timed out 1st %d, 2nd %d, 3rd %d | cut 1st %d, 2nd %d, 3rd %d (incomplete %d) | cut answers kept %d (%.2f%% of chunks), plain text %d (%.2f%%)\n", u, w, c, t[1], t[2], t[3], k[1], k[2], k[3], n, s, (c?100*s/c:0), f, (c?100*f/c:0)}'; done
```
Keep it only when worker 2 wrote clearly more rulings than worker 1 over the same hours, its timeouts are near
0, AND its chunks that end without a full answer (kept plus plain text) are not above worker 1's plain-text
share (about 1.1%; under 0.3 points at ~4,000 chunks per worker is noise). "incomplete" should be a handful at
most: every such answer is one the stop cut inside its content.
The other workers come after that reading, one at a time: `.env.worker3` for the third; worker 1's line would
go into `/opt/sunnitai-be/.env`, so check first what else on be2 reads that file.
Rollback: delete the line from `.env.worker2` and restart worker 2 on its empty queue (the patched file then
behaves as before). Full: `sudo python3 ~/patch_ingest_lever6.py --undo`, then restart on an empty queue every
worker that started after the patch. Lever 4's patch would need the checksum after lever 6 (aa293a42...).

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
   Baseline at one worker: ~60 written. Two workers pass when `written` is clearly above 60 and
   `chunk_fallbacks` stays near 1% of `chunks` (the rate of 10-11 Nov, the same in light and heavy hours; the
   counter read 0 until 11 Nov because it missed the double-timeout case). Well above 1% = the GPU is oversubscribed: stop worker 2
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
| ingest 24 h: N written | ~1,700 at two workers on Section I (10-13 chunks a ruling); about 1,000 chunks an hour | 0 while files wait = stalled: `systemctl status`, journal |
| chunk fallbacks | about 1% of chunks | well above 1%: stop worker 2 (above) |
| LLM model errors | 0 | > 0: the ingest .env names a model the server no longer serves |
| disk free | > 40 GB | the page flags it: `touch /var/log/maint/feed.pause` and the database discussion |
| feeder: N left | falling by the day's written | PAUSED outside 20:30-23:35 CET = the pause file was left behind |
| suspected duplicates | 0 | > 0: trace them before feeding more |
| rulings without card | growing by day, cleared by the night | the cards window is too short: `--stop-at` later |

## 6. Rollback, in one place

- Worker 2: `sudo systemctl disable --now sunnitai-ingest-2`; `sudo mv /opt/sunnitai-be/document_upload_2/*.pdf /opt/sunnitai-be/document_upload_2/*.name /home/sunnit/cass_batch1/`.
- Feeder: comment its crontab line; the old `feed_queue.sh` line can return (same manifest, same queue).
- Linker: restore the `.bak` of the script and of `link_nightly.sh`; the stamps stay and hurt nothing.
- Lever 5: delete `ANALISI_CHUNK_TIMEOUTS` from `.env.worker2` (and `.env`), restart; `patch_ingest_lever5.py --undo` restores the file.
- Lever 6 (on worker 2 since 15 Nov): `sudo sed -i '/^ANALISI_STOP_REPETITION=/d' /opt/sunnitai-be/.env.worker2 && sudo systemctl restart sunnitai-ingest-2`; `patch_ingest_lever6.py --undo` restores the file.
- Health: restore the `.bak`; the old report ignores the new arguments' files.
- Cards: `--stop-at 00:45` back on prod's line.
