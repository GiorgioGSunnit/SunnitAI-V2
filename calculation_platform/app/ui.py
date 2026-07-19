"""Browser dev UI for manually exercising the calculation platform.

Served by the same FastAPI app as the real API (no separate server) —
mounted at GET / by app/main.py. Talks to the app's own /calculators and
/calculate endpoints via client-side JS, so it always reflects whatever
the API actually does.

Also exposes /simulate/chat, a dev-only wrapper around the simulation
package's SimulatedConversation, so the full scripted-LLM conversation
loop (recognition -> calculation / clarification / ambiguity) can be
driven from the browser. Dev tooling only — a single in-memory
conversation, no sessions, never a production path.
"""

from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .api.routes import calculate_and_persist
from .schemas.calculation_request import CalculationRequest

ui_router = APIRouter()
_engine = None
_conversation = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def _import_simulation():
    """The simulation package sits beside `app` and imports `app.*` as a
    top-level package, so calculation_platform/ must be on sys.path — the
    tests' conftest and the demo script already arrange this; when served
    by uvicorn from the repo root we add it here (same shim)."""
    try:
        import simulation  # noqa: F401
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_conversation():
    global _conversation
    if _conversation is None:
        _import_simulation()
        from simulation.conversation import SimulatedConversation

        _conversation = SimulatedConversation(_engine)
    return _conversation


class ChatMessage(BaseModel):
    message: str


class PlanRequest(BaseModel):
    sentence: str


@ui_router.post("/plan")
def plan(payload: PlanRequest) -> Dict[str, Any]:
    """Dev endpoint: expose the hardcoded planner's structured routing
    result (ready_to_calculate / needs_clarification / ambiguous /
    no_match) for a free-text sentence. Not a production route."""
    _import_simulation()
    from simulation.planner import plan_sentence

    return plan_sentence(payload.sentence, _engine.registry.definitions()).model_dump()


@ui_router.post("/simulate/chat")
def simulate_chat(payload: ChatMessage) -> Dict[str, Any]:
    reply = _get_conversation().send(payload.message)
    calculation = reply.calculation
    if reply.calculation and reply.tool_call:
        request = CalculationRequest(
            calculator_id=reply.tool_call.calculator_id,
            inputs={k: v for k, v in reply.tool_call.inputs.items()},
            tax_year=reply.tool_call.tax_year,
            period=reply.tool_call.period,
        )
        calculation = calculate_and_persist(request)
    return {
        "kind": reply.kind,
        "text": reply.text,
        "tool_call": asdict(reply.tool_call) if reply.tool_call else None,
        "calculation": calculation.model_dump() if calculation else None,
        "plan": reply.plan.model_dump() if reply.plan else None,
    }


@ui_router.post("/simulate/reset")
def simulate_reset() -> Dict[str, str]:
    global _conversation
    _conversation = None
    return {"status": "reset"}


@ui_router.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # no-store: the page template changes between dev sessions and a stale
    # cached copy makes the UI appear broken.
    return HTMLResponse(content=_PAGE, headers={"Cache-Control": "no-store"})


_PAGE = """
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Calculation Platform — dev UI</title>
<style>
  :root {
    --border: #e3e6ea; --muted: #667085; --accent: #1f6feb;
    --answer: #0a7d32; --answer-bg: #e6f4ea;
    --question: #b26a00; --question-bg: #fff4e5;
    --ambiguous: #7a5d00; --ambiguous-bg: #fff8db;
    --no_match: #b00020; --no_match-bg: #fdecea;
  }
  body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 780px;
         margin: 28px auto 60px; padding: 0 16px; color: #1b1f24; }
  h2 { margin-bottom: 2px; }
  .subtitle { color: var(--muted); margin-top: 0; font-size: 14px; }
  .card { border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
          margin-top: 18px; box-shadow: 0 1px 3px rgba(16,24,40,.06); }
  .hint { color: var(--muted); font-size: 13px; }
  /* chat */
  #chat_log { min-height: 140px; max-height: 420px; overflow-y: auto; padding: 8px 10px;
              background: #fafbfc; border: 1px solid var(--border); border-radius: 10px; }
  .msg { max-width: 88%; padding: 10px 14px; border-radius: 14px; margin: 8px 0;
         font-size: 14px; line-height: 1.45; }
  .msg pre { margin: 0; white-space: pre-wrap; font-family: inherit; }
  .msg.user { margin-left: auto; background: var(--accent); color: #fff;
              border-bottom-right-radius: 4px; }
  .msg.assistant { background: #f2f4f7; border-bottom-left-radius: 4px; }
  .badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .4px;
           text-transform: uppercase; border-radius: 20px; padding: 2px 10px; margin-bottom: 6px; }
  .badge.answer    { color: var(--answer);    background: var(--answer-bg); }
  .badge.question  { color: var(--question);  background: var(--question-bg); }
  .badge.ambiguous { color: var(--ambiguous); background: var(--ambiguous-bg); }
  .badge.no_match  { color: var(--no_match);  background: var(--no_match-bg); }
  .msg details { margin-top: 8px; }
  .msg details summary { cursor: pointer; font-size: 12px; color: var(--muted); }
  .msg details pre { background: #fff; border: 1px solid var(--border); border-radius: 8px;
                     padding: 8px; font-family: ui-monospace, monospace; font-size: 11.5px;
                     max-height: 260px; overflow: auto; }
  /* chips */
  #chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }
  .chip { border: 1px solid var(--border); background: #fff; border-radius: 20px;
          padding: 6px 12px; font-size: 12.5px; cursor: pointer; color: #344054; }
  .chip:hover { border-color: var(--accent); color: var(--accent); }
  /* input row */
  .input-row { display: flex; gap: 8px; margin-top: 12px; }
  .input-row input { flex: 1; font-size: 15px; padding: 10px 12px; border: 1px solid var(--border);
                     border-radius: 10px; }
  button { font-size: 14px; padding: 9px 18px; border-radius: 10px; border: 1px solid var(--border);
           background: #fff; cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:hover { filter: brightness(.97); }
  #typing { color: var(--muted); font-size: 13px; display: none; padding: 4px 2px; }
  /* advanced sections */
  details.section { border: 1px solid var(--border); border-radius: 12px; margin-top: 14px;
                    padding: 12px 16px; }
  details.section summary { cursor: pointer; font-weight: 600; }
  details.section .inner { margin-top: 10px; }
  select, .inner input[type=text], .inner input[type=number], .inner input[type=date], textarea {
    font-size: 14px; padding: 8px 10px; margin: 4px 0; width: 100%; box-sizing: border-box;
    border: 1px solid var(--border); border-radius: 8px; }
  textarea { font-family: ui-monospace, monospace; font-size: 13px; height: 80px; }
  label { display: block; margin-top: 12px; font-weight: 600; font-size: 13.5px; }
  .row { display: flex; gap: 12px; }
  .row > div { flex: 1; }
  fieldset { margin-top: 14px; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
  legend { font-weight: 600; font-size: 13px; padding: 0 4px; }
  pre.out { background: #f8f9fb; border: 1px solid var(--border); padding: 12px; border-radius: 8px;
            white-space: pre-wrap; font-size: 12.5px; }
  .error { color: var(--no_match); }
  .warning { color: var(--question); }
  .citation { color: #1a4d8f; }
  .report-link { display: inline-block; margin-top: 8px; font-size: 12.5px; color: var(--accent); }
  .history-actions { display: flex; gap: 8px; align-items: center; }
  .history-actions a { color: var(--accent); font-size: 12.5px; }
  .history-actions button { padding: 4px 8px; border-radius: 8px; font-size: 12px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  td, th { border: 1px solid var(--border); padding: 4px 8px; font-size: 13px; text-align: left; }
</style>
</head>
<body>
<h2>Calculation Platform</h2>
<p class="subtitle">Motore di calcolo deterministico + router simulato (nessun LLM reale).
   Scrivi una frase come farebbe un utente, o clicca un esempio.</p>

<div class="card">
  <h3 style="margin:0 0 4px">&#128172; Conversazione — scrivi QUI la tua frase</h3>
  <p class="hint" style="margin-top:0">Questo è l'unico punto dove incollare frasi in linguaggio naturale.
     Le sezioni sotto sono strumenti tecnici (diagnostica del routing e inserimento manuale dei valori).</p>
  <div id="chips"></div>
  <div id="chat_log"></div>
  <div id="typing">sto calcolando…</div>
  <div class="input-row">
    <input type="text" id="chat_input" placeholder="es. quanto pago di irpef su 42000 euro nel 2026?">
    <button class="primary" onclick="sendChat()">Invia</button>
    <button onclick="resetChat()" title="Abbandona la conversazione in corso">Reset</button>
  </div>
</div>

<details class="section">
  <summary>&#128269; Analisi frase — perché il router sceglie un calcolo (diagnostica)</summary>
  <div class="inner">
    <p class="hint">Mostra i candidati con punteggio e termini riconosciuti, senza avviare la conversazione.</p>
    <input type="text" id="match_query" placeholder="quanto pago di tasse sul reddito...">
    <button onclick="runMatch()" style="margin-top:8px">Analizza</button>
    <div id="match_results"></div>
  </div>
</details>

<details class="section">
  <summary>&#129518; Calcolo manuale — input espliciti, senza frase</summary>
  <div class="inner">
    <label for="calculator">Calcolatore</label>
    <select id="calculator"></select>
    <p class="hint" id="description"></p>
    <div id="inputs"></div>

    <fieldset>
      <legend>Selezione regime (opzionale)</legend>
      <div class="row">
        <div>
          <label>tax_year <span class="hint">(scaglioni/tassi per anno)</span></label>
          <input type="number" id="tax_year" placeholder="es. 2026">
        </div>
        <div>
          <label>as_of_date <span class="hint">(alternativa a tax_year)</span></label>
          <input type="date" id="as_of_date">
        </div>
      </div>
    </fieldset>

    <fieldset id="period-fieldset" style="display:none">
      <legend>Periodo (per i calcoli di interessi)</legend>
      <div class="row">
        <div><label>start_date</label><input type="date" id="period_start"></div>
        <div><label>end_date</label><input type="date" id="period_end"></div>
      </div>
    </fieldset>

    <fieldset>
      <legend>Avanzato: caller_supplied_values (JSON, sovrascrive i parametri)</legend>
      <textarea id="caller_supplied_values" placeholder='{"legal_interest_rate": 0.03}'></textarea>
    </fieldset>

    <button class="primary" onclick="runCalculation()" style="margin-top:14px">Calcola</button>
    <h3>Risultato</h3>
    <div id="output">&mdash;</div>
  </div>
</details>

<details class="section" id="history_section">
  <summary>&#128196; Storico calcoli</summary>
  <div class="inner">
    <p class="hint">Ultimi calcoli salvati, con report stampabile e replay deterministico.</p>
    <button onclick="loadHistory()">Aggiorna</button>
    <div id="history_results"></div>
    <div id="history_replay"></div>
  </div>
</details>

<script>
let definitions = {};

const KIND_LABELS = {
  answer: 'risposta', question: 'domanda', ambiguous: 'ambiguo', no_match: 'nessun calcolo'
};

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

function reportLink(requestId) {
  if (!requestId) return '';
  const encoded = encodeURIComponent(requestId);
  return `<a class="report-link" href="/calculations/${encoded}/report" target="_blank" rel="noopener">Apri report</a>`;
}

async function loadCalculators() {
  const res = await fetch('/calculators');
  const list = await res.json();
  const select = document.getElementById('calculator');
  select.innerHTML = list.map(c => `<option value="${c.id}">${c.id} — ${c.name}</option>`).join('');
  select.addEventListener('change', loadDefinition);
  await loadDefinition();

  // suggestion chips: one example sentence per calculator, from its own metadata
  const chips = document.getElementById('chips');
  chips.innerHTML = list
    .map(c => (c.aliases && c.aliases[0]) || (c.keywords && c.keywords[0]) || null)
    .filter(Boolean)
    .map(text => `<span class="chip" onclick="useChip(this)">${text}</span>`)
    .join('');
}

function useChip(el) {
  document.getElementById('chat_input').value = el.textContent;
  sendChat();
}

async function loadDefinition() {
  const id = document.getElementById('calculator').value;
  const res = await fetch(`/calculators/${id}`);
  const def = await res.json();
  definitions[id] = def;

  document.getElementById('description').textContent = def.description || '';
  document.getElementById('period-fieldset').style.display =
    def.requires_period ? 'block' : 'none';

  const container = document.getElementById('inputs');
  container.innerHTML = def.inputs.map(inp => {
    const req = inp.required ? '' : ' (opzionale)';
    const hint = inp.description ? `<span class="hint"> — ${inp.description}</span>` : '';
    if (inp.type === 'boolean') {
      return `<label><input type="checkbox" id="input_${inp.name}" style="width:auto"> ${inp.name}${req}${hint}</label>`;
    }
    const htmlType = inp.type === 'date' ? 'date' : (inp.type === 'string' ? 'text' : 'number');
    const placeholder = inp.default !== null && inp.default !== undefined ? inp.default : '';
    return `<label>${inp.name}${req}${hint}</label>
            <input type="${htmlType}" id="input_${inp.name}" placeholder="${placeholder}" step="any">`;
  }).join('');
}

function appendChat(who, text, payload, kind) {
  const log = document.getElementById('chat_log');
  let html = `<div class="msg ${who}">`;
  if (who === 'assistant' && kind) {
    html += `<span class="badge ${kind}">${KIND_LABELS[kind] || kind}</span>`;
  }
  html += `<pre>${text}</pre>`;
  if (payload && payload.plan) {
    html += `<details><summary>[0] Piano del router simulato</summary>
             <pre>${JSON.stringify(payload.plan, null, 2)}</pre></details>`;
  }
  if (payload && payload.tool_call) {
    html += `<details><summary>[1] Chiamata LLM &rarr; piattaforma (POST /calculate)</summary>
             <pre>${JSON.stringify(payload.tool_call, null, 2)}</pre></details>`;
  }
  if (payload && payload.calculation) {
    html += reportLink(payload.calculation.request_id);
    html += `<details><summary>[2] Payload piattaforma &rarr; LLM</summary>
             <pre>${JSON.stringify(payload.calculation, null, 2)}</pre></details>`;
  }
  html += '</div>';
  log.innerHTML += html;
  log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById('chat_input');
  const message = input.value.trim();
  if (!message) return;
  appendChat('user', message, null, null);
  input.value = '';
  const typing = document.getElementById('typing');
  typing.style.display = 'block';
  try {
    const res = await fetch('/simulate/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const body = await res.json();
    appendChat('assistant', body.text, body, body.kind);
  } catch (e) {
    appendChat('assistant', 'Errore di comunicazione con il server: ' + e, null, 'no_match');
  } finally {
    typing.style.display = 'none';
  }
}

function showWelcome() {
  appendChat('assistant',
    'Ciao! Scrivi una frase come farebbe un cliente, oppure clicca uno degli esempi qui sopra.\\n' +
    'Se manca qualche dato ti farò una DOMANDA (badge arancione): rispondi in questo stesso campo.\\n' +
    'Esempio completo in un colpo solo: "pena per furto con 2 aggravanti e 0 attenuanti".',
    null, null);
}

async function resetChat() {
  await fetch('/simulate/reset', { method: 'POST' });
  document.getElementById('chat_log').innerHTML = '';
  showWelcome();
}

async function runMatch() {
  const query = document.getElementById('match_query').value.trim();
  const container = document.getElementById('match_results');
  if (!query) { container.innerHTML = ''; return; }

  const res = await fetch('/match', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  const body = await res.json();

  if (body.status === 'no_match') {
    container.innerHTML = '<p class="error">Nessun calcolatore corrisponde: la piattaforma non tenta di indovinare.</p>';
    return;
  }
  const statusLine = body.status === 'ambiguous'
    ? '<p class="warning">Ambiguo — il router chiederebbe di scegliere tra i candidati alla pari:</p>'
    : '<p>Corrispondenza chiara — il router proporrebbe:</p>';

  container.innerHTML = statusLine + body.candidates.map(c => {
    const required = c.required_inputs.map(i => `${i.name} (${i.type}${i.unit ? ', ' + i.unit : ''})`).join(', ');
    const extras = [];
    if (c.requires_period) extras.push('un periodo inizio/fine');
    if (c.supports_tax_year) extras.push('opzionalmente un tax_year');
    return `<div style="border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin:8px 0">
      <strong>${c.calculator_id}</strong> — punteggio ${c.score}<br>
      <span class="hint">riconosciuto da: ${c.matched_terms.join(', ')}</span><br>
      <span class="hint">mancherebbero: ${required || 'niente'}${extras.length ? '; più ' + extras.join(' e ') : ''}</span><br>
      <button onclick="useCandidate('${c.calculator_id}')" style="margin-top:6px">Apri nel calcolo manuale</button>
    </div>`;
  }).join('');
}

async function useCandidate(id) {
  document.querySelectorAll('details.section')[1].open = true;
  const select = document.getElementById('calculator');
  select.value = id;
  await loadDefinition();
  select.scrollIntoView({ behavior: 'smooth' });
}

function renderResult(body, ok) {
  const el = document.getElementById('output');
  if (!ok || body.status === 'error') {
    el.innerHTML = `<p class="error">Errore: ${(body.errors || [body.detail || 'errore sconosciuto']).map(e => e.message || e).join('; ')}</p>
                     ${reportLink(body.request_id)}
                     <pre class="out">${JSON.stringify(body, null, 2)}</pre>`;
    return;
  }
  let html = reportLink(body.request_id);
  html += '<table>' + Object.entries(body.result).map(
    ([k, v]) => `<tr><th>${k}</th><td>${typeof v === 'object' ? JSON.stringify(v) : v}</td></tr>`
  ).join('') + '</table>';
  if (body.steps && body.steps.length) {
    html += '<h4>Passaggi</h4><pre class="out">' + JSON.stringify(body.steps, null, 2) + '</pre>';
  }
  if (body.citations && body.citations.length) {
    html += '<h4>Fonti</h4><ul>' + body.citations.map(
      c => `<li class="citation">${c.reference}${c.source_name ? ' — ' + c.source_name : ''}</li>`
    ).join('') + '</ul>';
  }
  if (body.warnings && body.warnings.length) {
    html += '<h4>Avvertenze</h4><ul>' + body.warnings.map(
      w => `<li class="warning">${w.message}</li>`
    ).join('') + '</ul>';
  }
  html += '<h4>Risposta completa</h4><pre class="out">' + JSON.stringify(body, null, 2) + '</pre>';
  el.innerHTML = html;
}

async function runCalculation() {
  const id = document.getElementById('calculator').value;
  const def = definitions[id];

  const inputs = {};
  for (const inp of def.inputs) {
    const el = document.getElementById(`input_${inp.name}`);
    if (inp.type === 'boolean') {
      inputs[inp.name] = el.checked;
    } else if (el.value !== '') {
      inputs[inp.name] = (inp.type === 'decimal' || inp.type === 'integer') ? Number(el.value) : el.value;
    }
  }
  const request = { calculator_id: id, inputs };

  const taxYear = document.getElementById('tax_year').value;
  if (taxYear !== '') request.tax_year = Number(taxYear);
  const asOfDate = document.getElementById('as_of_date').value;
  if (asOfDate !== '') request.as_of_date = asOfDate;

  const periodStart = document.getElementById('period_start').value;
  const periodEnd = document.getElementById('period_end').value;
  if (periodStart && periodEnd) request.period = { start_date: periodStart, end_date: periodEnd };

  const rawCallerValues = document.getElementById('caller_supplied_values').value.trim();
  if (rawCallerValues) {
    try {
      request.caller_supplied_values = JSON.parse(rawCallerValues);
    } catch (e) {
      document.getElementById('output').innerHTML = `<p class="error">JSON non valido in caller_supplied_values: ${e}</p>`;
      return;
    }
  }

  const res = await fetch('/calculate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  const body = await res.json();
  renderResult(body, res.ok);
}

async function loadHistory() {
  const container = document.getElementById('history_results');
  container.innerHTML = '<p class="hint">Caricamento...</p>';
  try {
    const res = await fetch('/calculations?limit=50');
    const rows = await res.json();
    if (!rows.length) {
      container.innerHTML = '<p class="hint">Nessun calcolo salvato.</p>';
      return;
    }
    container.innerHTML = '<table><thead><tr>' +
      '<th>created_at</th><th>calculator_id</th><th>status</th><th>request_id</th><th>azioni</th>' +
      '</tr></thead><tbody>' + rows.map(row => {
        const encoded = encodeURIComponent(row.request_id);
        return `<tr>
          <td>${escapeHtml(row.created_at)}</td>
          <td>${escapeHtml(row.calculator_id)}</td>
          <td>${escapeHtml(row.status)}</td>
          <td>${escapeHtml(row.request_id)}</td>
          <td><span class="history-actions">
            <a href="/calculations/${encoded}/report" target="_blank" rel="noopener">report</a>
            <button onclick="replayHistory('${encoded}')">replay</button>
          </span></td>
        </tr>`;
      }).join('') + '</tbody></table>';
  } catch (e) {
    container.innerHTML = `<p class="error">Errore caricando lo storico: ${escapeHtml(e)}</p>`;
  }
}

async function replayHistory(encodedRequestId) {
  const output = document.getElementById('history_replay');
  output.innerHTML = '<p class="hint">Replay in corso...</p>';
  try {
    const res = await fetch(`/calculations/${encodedRequestId}/replay`, { method: 'POST' });
    const body = await res.json();
    output.innerHTML = `<h4>Replay: matches = ${escapeHtml(body.matches)}</h4>` +
      `<pre class="out">${escapeHtml(JSON.stringify(body.replayed_result || body, null, 2))}</pre>`;
  } catch (e) {
    output.innerHTML = `<p class="error">Errore durante il replay: ${escapeHtml(e)}</p>`;
  }
}

document.getElementById('match_query').addEventListener('keydown', e => {
  if (e.key === 'Enter') runMatch();
});
document.getElementById('chat_input').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendChat();
});
document.getElementById('history_section').addEventListener('toggle', e => {
  if (e.target.open) loadHistory();
});
loadCalculators();
showWelcome();
</script>
</body>
</html>
"""
