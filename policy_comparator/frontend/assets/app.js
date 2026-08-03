/* Policy Comparator — staff frontend.
 *
 * Deliberately dependency-free: no build step, no bundler, no npm install. The
 * whole tool is `pip install` + two commands, which is what makes it easy to
 * hand to someone who just wants to run it locally.
 *
 * Progress is polled rather than pushed. Provider round trips take tens of
 * seconds, so a two-second poll is entirely adequate and avoids adding a
 * WebSocket lifecycle to an internal tool.
 *
 * Every value that originates from the API is put into the DOM with
 * textContent, never innerHTML. Insurer names, product names and calculation
 * labels all arrive over the wire; treating any of them as markup would make a
 * provider response an injection vector.
 */
(() => {
  'use strict';

  const POLL_MS = 2000;
  const TOKEN_KEY = 'pc.token';
  const REQUEST_KEY = 'pc.request';

  const state = {
    token: localStorage.getItem(TOKEN_KEY),
    email: null,
    requestId: localStorage.getItem(REQUEST_KEY),
    providers: [],
    pollTimer: null,
    view: 'new',
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = (tag, props = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const child of [].concat(children)) {
      if (child === null || child === undefined || child === false) continue;
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    }
    return node;
  };

  // ------------------------------------------------------------ formatting

  const NUM = new Intl.NumberFormat('it-IT', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  const NUM0 = new Intl.NumberFormat('it-IT', { maximumFractionDigits: 0 });

  /** €-suffixed Italian amount, e.g. "1.234,56 €". */
  const money = (value, currency = 'EUR') => {
    if (value === null || value === undefined || value === '') return '—';
    const n = Number(value);
    if (Number.isNaN(n)) return String(value);
    return `${NUM.format(n)} ${currency === 'EUR' ? '€' : currency}`;
  };

  /** Whole-euro amount for large limits, e.g. "6.450.000 €". */
  const limitAmount = (value) => {
    if (value === null || value === undefined || value === '') return '—';
    const n = Number(value);
    if (Number.isNaN(n)) return String(value);
    return `${NUM0.format(n)} €`;
  };

  /** "6.450.000 €" → "6,45 milioni di euro", for the explanatory caption. */
  const millionsCaption = (value) => {
    const n = Number(value);
    if (!n || Number.isNaN(n)) return null;
    if (n >= 1_000_000) {
      const millions = n / 1_000_000;
      const text = new Intl.NumberFormat('it-IT', { maximumFractionDigits: 2 }).format(millions);
      return `${text} milioni di euro`;
    }
    return null;
  };

  const yesNo = (value) => (value === null || value === undefined ? '—' : value ? 'Sì' : 'No');

  const FORMULA_LABELS = {
    free: 'Guida libera',
    expert: 'Guida esperta',
    exclusive: 'Guida esclusiva',
  };

  /** Wording reused wherever a number could be mistaken for a cost. */
  const HELP = {
    premium:
      'Il premio annuo è l’importo che il cliente paga in un anno per la polizza.',
    deductible:
      'La franchigia è la parte di danno che, secondo il contratto, può restare a carico ' +
      'dell’assicurato. Non è un importo pagato alla firma.',
    limitPeople:
      'Il massimale per danni a persone è l’importo massimo che la compagnia può pagare ' +
      'per i danni fisici causati a terzi in un solo sinistro. Non è un importo pagato dal cliente.',
    limitProperty:
      'Il massimale per danni a cose è l’importo massimo che la compagnia può pagare per i ' +
      'danni materiali causati a terzi. Non è un importo pagato dal cliente.',
    instalments:
      'Costo complessivo se la polizza viene pagata a rate: di norma è superiore al premio annuo.',
    optional:
      'Costo delle garanzie accessorie richieste dal cliente e già comprese nel premio annuo.',
    blackBox:
      'La compagnia richiede l’installazione di un dispositivo telematico a bordo del veicolo.',
    repairNetwork:
      'In caso di sinistro la riparazione deve avvenire presso le carrozzerie convenzionate.',
    formula:
      'Indica chi è autorizzato a guidare il veicolo secondo le condizioni di polizza.',
  };

  /** A keyboard-reachable information affordance. */
  const infoIcon = (text) =>
    el('button', {
      class: 'info',
      type: 'button',
      title: text,
      'aria-label': text,
      text: 'i',
      onclick: (event) => {
        event.preventDefault();
        toast(text);
      },
    });

  // ------------------------------------------------------------------ http

  async function api(path, { method = 'GET', body } = {}) {
    const headers = { 'Content-Type': 'application/json' };
    if (state.token) headers.Authorization = `Bearer ${state.token}`;

    const response = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });

    if (response.status === 401) {
      signOut();
      throw new Error('Sessione scaduta. Accedi di nuovo.');
    }

    const text = await response.text();
    const payload = text ? JSON.parse(text) : null;

    if (!response.ok) {
      const detail = payload && payload.detail;
      throw new Error(typeof detail === 'string' ? detail : 'Richiesta non riuscita');
    }
    return payload;
  }

  function toast(message, kind = 'info') {
    const node = $('#toast');
    node.textContent = message;
    node.classList.toggle('is-error', kind === 'error');
    node.classList.toggle('is-success', kind === 'success');
    node.hidden = false;
    clearTimeout(node._timer);
    node._timer = setTimeout(() => {
      node.hidden = true;
    }, 5000);
  }

  // ------------------------------------------------------------------ auth

  function signOut() {
    stopPolling();
    state.token = null;
    state.requestId = null;
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REQUEST_KEY);
    $('#app').hidden = true;
    $('#login').hidden = false;
  }

  async function signIn(email, password) {
    const data = await api('/api/auth/login', {
      method: 'POST',
      body: { email, password },
    });
    state.token = data.access_token;
    state.email = data.email;
    localStorage.setItem(TOKEN_KEY, data.access_token);
    await bootApp();
  }

  // ----------------------------------------------------------------- views

  function show(view) {
    state.view = view;
    for (const section of $$('.view')) {
      section.hidden = section.id !== `view-${view}`;
    }
    const order = ['new', 'missing', 'progress', 'results'];
    const index = order.indexOf(view);
    $$('#stepper li').forEach((li, i) => {
      li.classList.toggle('active', i === index);
      li.classList.toggle('done', i < index);
      if (i === index) li.setAttribute('aria-current', 'step');
      else li.removeAttribute('aria-current');
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // --------------------------------------------------------- new request

  async function loadProviders() {
    const data = await api('/api/providers');
    state.providers = data.providers;

    const picker = $('#provider-picker');
    picker.replaceChildren();

    for (const provider of state.providers) {
      const input = el('input', {
        type: 'checkbox',
        name: 'selected_provider_ids',
        value: provider.provider_id,
        checked: true,
      });
      const meta = [
        provider.provider_type === 'aggregator' ? 'Comparatore' : 'Compagnia',
        provider.live_enabled ? 'collegamento reale' : 'dati dimostrativi',
      ].join(' · ');

      const label = el('label', { class: 'provider-option is-selected' }, [
        input,
        el('span', {}, [
          el('span', { class: 'p-name', text: provider.display_name }),
          el('span', { class: 'p-meta', text: meta }),
        ]),
      ]);
      input.addEventListener('change', () => {
        label.classList.toggle('is-selected', input.checked);
      });
      picker.appendChild(label);
    }
  }

  function clearErrors(form) {
    $$('.field-error', form).forEach((node) => {
      node.textContent = '';
    });
  }

  async function submitNewRequest(event) {
    event.preventDefault();
    const form = event.currentTarget;
    clearErrors(form);

    const selected = $$('input[name="selected_provider_ids"]:checked', form).map((i) => i.value);
    if (!selected.length) {
      $('[data-error="selected_provider_ids"]', form).textContent =
        'Seleziona almeno una compagnia.';
      return;
    }

    const body = {
      vehicle_plate: form.vehicle_plate.value.trim().toUpperCase(),
      owner_date_of_birth: form.owner_date_of_birth.value,
      customer_email: form.customer_email.value.trim(),
      policy_start_date: form.policy_start_date.value,
      privacy_accepted: form.privacy_accepted.checked,
      provider_data_transfer_accepted: form.provider_data_transfer_accepted.checked,
      marketing_accepted: form.marketing_accepted.checked,
      selected_provider_ids: selected,
    };

    if (!body.privacy_accepted || !body.provider_data_transfer_accepted) {
      toast('Entrambi i consensi obbligatori sono necessari per procedere.', 'error');
      return;
    }

    const button = $('button[type="submit"]', form);
    button.disabled = true;
    try {
      const created = await api('/api/quotes', { method: 'POST', body });
      state.requestId = created.request_id;
      localStorage.setItem(REQUEST_KEY, created.request_id);

      // Creating and starting are separate on the server; the staff user's
      // click is what authorizes transmission, so both happen here together.
      await api(`/api/quotes/${created.request_id}/start`, { method: 'POST' });
      toast('Richiesta inviata alle compagnie selezionate.', 'success');
      show('progress');
      startPolling();
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  // ------------------------------------------------------------- progress

  const STATUS_CLASS = {
    quoted: 'badge-ok',
    waiting: 'badge-run',
    running: 'badge-run',
    retrying: 'badge-run',
    missing_information: 'badge-demo',
    manual_action_required: 'badge-demo',
  };

  function renderProgress(data) {
    $('#demo-badge').hidden = !data.demonstration_data;

    const list = $('#progress-list');
    list.replaceChildren();

    for (const provider of data.providers) {
      const badgeClass = STATUS_CLASS[provider.status] || 'badge-err';
      const right = el('span', { class: 'p-right' }, [
        provider.finished ? null : el('span', { class: 'spinner', 'aria-hidden': 'true' }),
        el('span', { class: `badge ${badgeClass}`, text: provider.status_label }),
      ]);

      if (provider.retryable) {
        right.appendChild(
          el('button', {
            class: 'btn btn-ghost btn-sm',
            type: 'button',
            text: 'Riprova',
            onclick: () => retryProvider(provider.provider_id),
          })
        );
      }

      list.appendChild(
        el('div', { class: 'provider-row' }, [
          el('span', { class: 'p-left' }, [
            el('strong', { text: provider.display_name }),
            el('span', {
              class: 'p-msg',
              text:
                provider.error_message ||
                (provider.quotes ? `${provider.quotes} preventivo/i ricevuti` : ' '),
            }),
          ]),
          right,
        ])
      );
    }

    if (data.status === 'awaiting_information') {
      loadMissingFields().then((missing) => {
        if (missing.total_questions > 0 && state.view === 'progress') show('missing');
      });
    }
  }

  async function retryProvider(providerId) {
    try {
      await api(`/api/quotes/${state.requestId}/retry`, {
        method: 'POST',
        body: { provider_id: providerId },
      });
      toast(`Nuovo tentativo per ${providerId}.`);
      startPolling();
    } catch (error) {
      toast(error.message, 'error');
    }
  }

  function startPolling() {
    stopPolling();
    const tick = async () => {
      if (!state.requestId) return;
      try {
        const data = await api(`/api/quotes/${state.requestId}/progress`);
        renderProgress(data);
        if (data.pending === 0) {
          stopPolling();
          if (data.status !== 'awaiting_information') await loadResults();
        }
      } catch (error) {
        stopPolling();
        toast(error.message, 'error');
      }
    };
    tick();
    state.pollTimer = setInterval(tick, POLL_MS);
  }

  function stopPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = null;
  }

  // ------------------------------------------------------- missing fields

  function inputForField(field) {
    const name = field.field_path;
    if (field.input_type === 'choice' && field.choices) {
      return el('select', { name }, [
        el('option', { value: '', text: '—' }),
        ...field.choices.map((c) => el('option', { value: c.value, text: c.label })),
      ]);
    }
    if (field.input_type === 'boolean') {
      return el('select', { name }, [
        el('option', { value: '', text: '—' }),
        el('option', { value: 'true', text: 'Sì' }),
        el('option', { value: 'false', text: 'No' }),
      ]);
    }
    const type =
      field.input_type === 'date' ? 'date' : field.input_type === 'number' ? 'number' : 'text';
    return el('input', { type, name });
  }

  async function loadMissingFields() {
    const data = await api(`/api/quotes/${state.requestId}/missing-fields`);
    const container = $('#missing-groups');
    container.replaceChildren();

    $('#missing-lede').textContent = data.total_questions
      ? `${data.total_questions} informazioni richieste dalle compagnie. ` +
        'Una domanda chiesta da più compagnie compare una volta sola.'
      : 'Nessuna informazione mancante al momento.';

    for (const group of data.groups) {
      const fields = el('div', { class: 'grid grid-2' });
      for (const field of group.fields) {
        fields.appendChild(
          el('label', { class: 'field' }, [
            el('span', { class: 'field-label', text: field.label }),
            inputForField(field),
            field.help_text ? el('span', { class: 'field-hint', text: field.help_text }) : null,
            el('span', {
              class: 'asked-by',
              text: `Richiesto da: ${field.requested_by.join(', ')}`,
            }),
          ])
        );
      }
      const count = group.fields.length;
      container.appendChild(
        el('section', { class: 'missing-group' }, [
          el('h3', { text: group.label }),
          el('p', { class: 'who', text: `${count} camp${count === 1 ? 'o' : 'i'}` }),
          fields,
        ])
      );
    }
    return data;
  }

  async function submitMissingFields(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const updates = {};

    for (const input of $$('input[name], select[name]', form)) {
      const value = input.value.trim();
      if (value !== '') updates[input.name] = value;
    }

    if (!Object.keys(updates).length) {
      toast('Compila almeno un campo.', 'error');
      return;
    }

    const button = $('button[type="submit"]', form);
    button.disabled = true;
    try {
      const data = await api(`/api/quotes/${state.requestId}/missing-fields`, {
        method: 'POST',
        body: { updates },
      });
      toast(
        data.resumed_providers.length
          ? `Ripreso: ${data.resumed_providers.join(', ')}.`
          : 'Dati salvati.',
        'success'
      );
      show('progress');
      startPolling();
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  // -------------------------------------------------------------- results

  function demoBanner() {
    return el('div', { class: 'demo-banner', role: 'note' }, [
      el('span', { class: 'demo-icon', 'aria-hidden': 'true', text: '⚠' }),
      el('div', {}, [
        el('strong', { text: 'Dati dimostrativi — nessuna compagnia è stata contattata.' }),
        el('span', {
          text:
            'I premi mostrati sono calcolati da una formula interna a scopo di ' +
            'dimostrazione e non sono prezzi reali di alcun assicuratore. ' +
            'Non possono essere usati per una quotazione al cliente.',
        }),
      ]),
    ]);
  }

  /**
   * The expandable "how was this calculated" section.
   *
   * A demonstration quote shows the full derivation. A provider-supplied price
   * shows only that the insurer quoted it — inventing a formula for a real
   * price would be fabrication.
   */
  function calculationSection(quote) {
    const isDemo = quote.calculation_source === 'demonstration_formula';
    const breakdown = quote.calculation_breakdown;

    const body = el('div', { class: 'calc-body' });

    if (!isDemo || !breakdown) {
      body.appendChild(
        el('div', { class: 'calc-note provider' }, [
          el('strong', { text: 'Prezzo fornito dalla compagnia. ' }),
          el('span', {
            text:
              'L’importo è stato comunicato direttamente dall’assicuratore. I fattori di ' +
              'tariffazione di dettaglio non sono disponibili, se non inclusi nella risposta ' +
              'del provider.',
          }),
        ])
      );
      return el('details', { class: 'calc' }, [
        el('summary', { text: 'Come è stato calcolato questo importo' }),
        body,
      ]);
    }

    body.appendChild(
      el('div', { class: 'calc-note' }, [
        el('strong', { text: 'Logica dimostrativa. ' }),
        el('span', {
          text:
            'Questi passaggi descrivono una formula interna inventata a scopo di ' +
            'dimostrazione. Non riproducono la tariffa reale di alcuna compagnia.',
        }),
      ])
    );

    const rows = el('tbody');
    for (const step of breakdown.steps || []) {
      const isTotal = step.kind === 'total';
      const label = el('td', {}, [
        el('span', { text: step.label }),
        step.detail ? el('span', { class: 'detail', text: step.detail }) : null,
      ]);
      const factor = el('td', {
        class: 'num',
        text: step.factor ? `× ${step.factor}` : '',
      });
      const value = el('td', {
        class: 'num',
        text:
          step.kind === 'factor' && step.running
            ? money(step.running)
            : step.value !== null && step.value !== undefined
              ? money(step.value)
              : step.running
                ? money(step.running)
                : '',
      });

      rows.appendChild(
        el(
          'tr',
          {
            class: isTotal ? 'is-total' : step.kind === 'rounding' ? 'is-rounding' : null,
          },
          [label, factor, value]
        )
      );
    }

    body.appendChild(
      el('table', { class: 'calc-table' }, [
        el('thead', {}, [
          el('tr', {}, [
            el('th', { text: 'Passaggio' }),
            el('th', { class: 'num', text: 'Fattore' }),
            el('th', { class: 'num', text: 'Valore' }),
          ]),
        ]),
        rows,
      ])
    );

    if (breakdown.rounding) {
      body.appendChild(el('p', { class: 'calc-rounding', text: breakdown.rounding }));
    }
    body.appendChild(
      el('p', {
        class: 'calc-rounding',
        text:
          'I valori intermedi sono mostrati arrotondati a 2 decimali per leggibilità; ' +
          'il calcolo è eseguito in aritmetica decimale esatta.',
      })
    );

    return el('details', { class: 'calc' }, [
      el('summary', { text: 'Come è stato calcolato questo importo' }),
      body,
    ]);
  }

  /**
   * The purchase call to action.
   *
   * A demonstration link points at a non-existent host, so the click is
   * intercepted and explained rather than navigating to a broken page. A real
   * quote opens the provider URL, which the API has already validated as https.
   */
  function purchaseAction(quote) {
    if (quote.purchase_url_is_demonstration) {
      return el('button', {
        class: 'btn btn-primary',
        type: 'button',
        text: 'Prosegui con la compagnia (dimostrazione)',
        onclick: () =>
          toast(
            'Azione dimostrativa: in modalità demo non è possibile acquistare una polizza ' +
              'e nessuna compagnia è stata contattata.',
            'error'
          ),
      });
    }
    if (!quote.purchase_url) return null;
    return el('a', {
      class: 'btn btn-primary',
      href: quote.purchase_url,
      target: '_blank',
      rel: 'noopener noreferrer',
      text: 'Prosegui con la compagnia',
    });
  }

  function kvItem(label, value, help, note) {
    return el('div', {}, [
      el('dt', {}, [el('span', { text: label }), help ? infoIcon(help) : null]),
      el('dd', { text: value }),
      note ? el('span', { class: 'kv-note', text: note }) : null,
    ]);
  }

  function renderRecommendation(data) {
    const host = $('#recommendation');
    host.replaceChildren();

    const best = data.eligible_quotes.find((q) => q.recommended);
    if (!best) {
      host.appendChild(
        el('div', { class: 'card' }, [
          el('h2', { text: 'Nessun preventivo idoneo' }),
          el('p', { class: 'lede', text: data.recommendation_explanation }),
        ])
      );
      return;
    }

    const channel =
      best.source_channel === 'aggregator'
        ? `Tramite ${best.provider_display_name} (comparatore)`
        : `Diretto — ${best.provider_display_name}`;

    const body = el('div', { class: 'reco-body' }, [
      el('div', { class: 'reco-top' }, [
        el('div', {}, [
          el('h2', {
            class: 'reco-title',
            text: `${best.insurer_name}${best.product_name ? ` — ${best.product_name}` : ''}`,
          }),
          el('span', { class: 'reco-sub', text: channel }),
        ]),
        el('div', { class: 'price-block' }, [
          el('div', { class: 'price', text: money(best.annual_total_premium, best.currency) }),
          el('div', { class: 'price-caption', text: 'Premio annuo' }),
        ]),
      ]),
      el('p', { class: 'reco-why', text: data.recommendation_explanation }),
    ]);

    const kv = el('dl', { class: 'kv' }, [
      kvItem('Premio annuo', money(best.annual_total_premium, best.currency), HELP.premium),
      kvItem('Franchigia', money(best.deductible), HELP.deductible),
      kvItem(
        'Massimale danni a persone',
        limitAmount(best.liability_limit_people),
        HELP.limitPeople,
        millionsCaption(best.liability_limit_people)
      ),
      kvItem(
        'Massimale danni a cose',
        limitAmount(best.liability_limit_property),
        HELP.limitProperty,
        millionsCaption(best.liability_limit_property)
      ),
      kvItem('Formula di guida', FORMULA_LABELS[best.driving_formula] || '—', HELP.formula),
      kvItem('Scatola nera obbligatoria', yesNo(best.requires_black_box), HELP.blackBox),
      kvItem(
        'Carrozzerie convenzionate',
        yesNo(best.requires_approved_repair_network),
        HELP.repairNetwork
      ),
    ]);

    if (best.instalments) {
      kv.appendChild(
        kvItem(
          `Pagamento in ${best.instalments.count} rate`,
          `${best.instalments.count} × ${money(best.instalments.amount)}`,
          HELP.instalments,
          `Totale rateale ${money(best.instalments.total)}`
        )
      );
    }

    const optionalIncluded = (best.included_coverages || []).filter(
      (c) => c.code !== 'rc_auto' && c.price && Number(c.price) > 0
    );
    if (optionalIncluded.length) {
      const total = optionalIncluded.reduce((sum, c) => sum + Number(c.price), 0);
      kv.appendChild(
        kvItem(
          'Garanzie accessorie incluse',
          money(total.toFixed(2)),
          HELP.optional,
          optionalIncluded.map((c) => c.label).join(', ')
        )
      );
    }

    body.appendChild(kv);

    if ((best.satisfied_requirements || []).length) {
      body.appendChild(
        el('div', {}, [
          el('h3', { text: 'Requisiti del cliente soddisfatti' }),
          el(
            'ul',
            { class: 'requirement-list ok' },
            best.satisfied_requirements.map((r) => el('li', { text: r }))
          ),
        ])
      );
    } else {
      body.appendChild(
        el('p', {
          class: 'field-hint',
          text:
            'Nessun requisito vincolante impostato: tutti i preventivi ricevuti sono ' +
            'considerati idonei.',
        })
      );
    }

    if ((best.important_exclusions || []).length) {
      body.appendChild(
        el('div', {}, [
          el('h3', { text: 'Esclusioni rilevanti' }),
          el(
            'ul',
            { class: 'requirement-list' },
            best.important_exclusions.map((r) => el('li', { text: r }))
          ),
        ])
      );
    }

    body.appendChild(calculationSection(best));

    const action = purchaseAction(best);
    if (action) body.appendChild(el('div', { class: 'actions' }, [action]));

    host.appendChild(
      el('article', { class: 'reco' }, [
        el('div', {
          class: 'reco-head-bar',
          text: 'Preventivo più economico tra quelli conformi ai requisiti selezionati',
        }),
        body,
      ])
    );
  }

  function renderComparison(quotes) {
    const host = $('#comparison');
    host.replaceChildren();

    if (!quotes.length) {
      host.appendChild(
        el('p', { class: 'empty', text: 'Nessun preventivo idoneo da confrontare.' })
      );
      return;
    }

    const headers = [
      ['Compagnia', null],
      ['Premio annuo', HELP.premium],
      ['Rate', HELP.instalments],
      ['Franchigia', HELP.deductible],
      ['Massimale persone', HELP.limitPeople],
      ['Massimale cose', HELP.limitProperty],
      ['Guida', HELP.formula],
      ['Vincoli', null],
      ['Canale', null],
    ];

    const body = el('tbody');
    for (const quote of quotes) {
      const constraints =
        [
          quote.requires_black_box ? 'Scatola nera' : null,
          quote.requires_approved_repair_network ? 'Carrozzerie convenzionate' : null,
        ]
          .filter(Boolean)
          .join(', ') || 'Nessuno';

      body.appendChild(
        el('tr', { class: quote.recommended ? 'is-reco' : null }, [
          el('td', {}, [
            el('span', { class: 'insurer', text: quote.insurer_name }),
            el('span', { class: 'product', text: quote.product_name || '' }),
            quote.recommended
              ? el('span', { class: 'badge badge-run', text: 'Più economico conforme' })
              : null,
          ]),
          el('td', { class: 'num', text: money(quote.annual_total_premium, quote.currency) }),
          el('td', {
            class: 'num',
            text: quote.instalments
              ? `${quote.instalments.count} × ${money(quote.instalments.amount)}`
              : '—',
          }),
          el('td', { class: 'num', text: money(quote.deductible) }),
          el('td', { class: 'num', text: limitAmount(quote.liability_limit_people) }),
          el('td', { class: 'num', text: limitAmount(quote.liability_limit_property) }),
          el('td', { text: FORMULA_LABELS[quote.driving_formula] || '—' }),
          el('td', { text: constraints }),
          el('td', {
            text:
              quote.source_channel === 'aggregator'
                ? `${quote.provider_display_name} (comparatore)`
                : quote.provider_display_name,
          }),
        ])
      );
    }

    host.appendChild(
      el('table', { class: 'compare' }, [
        el('caption', {
          class: 'visually-hidden',
          text: 'Confronto dei preventivi idonei, ordinati dal più economico.',
        }),
        el('thead', {}, [
          el(
            'tr',
            {},
            headers.map(([label, help]) =>
              el('th', { scope: 'col' }, [el('span', { text: label }), help ? infoIcon(help) : null])
            )
          ),
        ]),
        body,
      ])
    );
  }

  function renderIneligible(quotes) {
    const host = $('#ineligible');
    host.replaceChildren();

    if (!quotes.length) {
      host.appendChild(el('p', { class: 'empty', text: 'Nessun preventivo escluso.' }));
      return;
    }

    for (const quote of quotes) {
      host.appendChild(
        el('div', { class: 'provider-row' }, [
          el('span', { class: 'p-left' }, [
            el('strong', {
              text: `${quote.insurer_name} — ${money(quote.annual_total_premium)}`,
            }),
            el(
              'ul',
              { class: 'reason-list' },
              quote.ineligible_reasons.map((r) =>
                el('li', { text: r.detail ? `${r.message} (${r.detail})` : r.message })
              )
            ),
          ]),
        ])
      );
    }
  }

  function renderUnavailable(providers) {
    const host = $('#unavailable');
    host.replaceChildren();

    if (!providers.length) {
      host.appendChild(el('p', { class: 'empty', text: 'Tutte le compagnie hanno risposto.' }));
      return;
    }

    for (const provider of providers) {
      host.appendChild(
        el('div', { class: 'provider-row' }, [
          el('span', { class: 'p-left' }, [
            el('strong', { text: provider.display_name }),
            el('span', { class: 'p-msg', text: provider.error_message || provider.status_label }),
          ]),
          el('span', { class: 'p-right' }, [
            el('span', { class: 'badge badge-err', text: provider.status_label }),
            el('button', {
              class: 'btn btn-ghost btn-sm',
              type: 'button',
              text: 'Riprova',
              onclick: () => retryProvider(provider.provider_id),
            }),
          ]),
        ])
      );
    }
  }

  /**
   * The requirements editor.
   *
   * The submit button lives *inside* the form element. It previously sat in a
   * sibling container, so clicking it never reached the form's submit handler
   * and the request was silently never sent.
   */
  function renderRequirements(data) {
    const host = $('#requirements-editor');
    const req = data.requirements;
    host.replaceChildren();

    const status = el('p', { class: 'form-status', role: 'status', 'aria-live': 'polite' });

    const fields = el('div', { class: 'grid grid-2' }, [
      el('label', { class: 'field' }, [
        el('span', { class: 'field-label', text: 'Franchigia massima accettata (€)' }),
        el('input', {
          type: 'number',
          name: 'max_acceptable_deductible',
          min: '0',
          step: '50',
          value: req.max_acceptable_deductible || '',
        }),
      ]),
      el('label', { class: 'field' }, [
        el('span', { class: 'field-label', text: 'Formula di guida richiesta' }),
        el('select', { name: 'driving_formula' }, [
          el('option', { value: '', text: 'Indifferente' }),
          ...Object.entries(FORMULA_LABELS).map(([value, label]) =>
            el('option', { value, text: label, selected: req.driving_formula === value })
          ),
        ]),
      ]),
      el('label', { class: 'field' }, [
        el('span', { class: 'field-label', text: 'Accetta la scatola nera' }),
        el('select', { name: 'accepts_black_box' }, [
          el('option', { value: '', text: 'Indifferente' }),
          el('option', { value: 'true', text: 'Sì', selected: req.accepts_black_box === true }),
          el('option', { value: 'false', text: 'No', selected: req.accepts_black_box === false }),
        ]),
      ]),
      el('label', { class: 'field' }, [
        el('span', { class: 'field-label', text: 'Accetta le carrozzerie convenzionate' }),
        el('select', { name: 'accepts_approved_repair_network' }, [
          el('option', { value: '', text: 'Indifferente' }),
          el('option', {
            value: 'true',
            text: 'Sì',
            selected: req.accepts_approved_repair_network === true,
          }),
          el('option', {
            value: 'false',
            text: 'No',
            selected: req.accepts_approved_repair_network === false,
          }),
        ]),
      ]),
    ]);

    const save = el('button', {
      class: 'btn btn-primary',
      type: 'submit',
      id: 'save-requirements',
      text: 'Aggiorna i requisiti',
    });

    const form = el('form', { id: 'form-requirements', novalidate: 'novalidate' }, [
      fields,
      // Inside the form: this is what makes the button actually submit it.
      el('div', { class: 'actions' }, [save]),
      status,
    ]);

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      status.textContent = '';
      status.className = 'form-status';

      const body = {};
      for (const input of $$('input[name], select[name]', form)) {
        const value = input.value.trim();
        if (value === '') {
          body[input.name] = null;
          continue;
        }
        // "false" must survive as the boolean false, not be dropped as falsy:
        // refusing a black box is a real requirement, not an absent one.
        body[input.name] = value === 'true' ? true : value === 'false' ? false : value;
      }

      save.disabled = true;
      const originalLabel = save.textContent;
      save.textContent = 'Salvataggio…';
      try {
        await api(`/api/quotes/${state.requestId}/preferences`, { method: 'PUT', body });
        toast('Requisiti aggiornati.', 'success');
        // Re-render everything the requirements affect: recommendation,
        // eligible quotes, exclusions and the requirements themselves.
        await loadResults();
      } catch (error) {
        status.textContent = `Salvataggio non riuscito: ${error.message}`;
        status.className = 'form-status is-error';
        toast(error.message, 'error');
        save.disabled = false;
        save.textContent = originalLabel;
      }
    });

    host.appendChild(el('h2', { id: 'requirements-heading', text: 'Requisiti del cliente' }));
    host.appendChild(
      el('p', {
        class: 'lede',
        text:
          'Ogni requisito impostato qui è vincolante: i preventivi che non lo rispettano ' +
          'vengono esclusi e non possono essere raccomandati.',
      })
    );
    host.appendChild(form);
  }

  async function loadResults() {
    const data = await api(`/api/quotes/${state.requestId}/results`);
    $('#demo-badge').hidden = !data.demonstration_data;

    const banner = $('#demo-banner-slot');
    banner.replaceChildren();
    if (data.demonstration_data) banner.appendChild(demoBanner());

    $('#results-lede').textContent = data.demonstration_data
      ? 'Confronto basato su dati dimostrativi generati localmente.'
      : 'Preventivi ricevuti dalle compagnie interpellate.';

    renderRecommendation(data);
    renderRequirements(data);
    renderComparison(data.eligible_quotes);
    renderIneligible(data.ineligible_quotes);
    renderUnavailable(data.unavailable_providers);
    show('results');
    return data;
  }

  // --------------------------------------------------------- saved list

  async function loadSaved() {
    const data = await api('/api/quotes');
    const host = $('#saved-list');
    host.replaceChildren();

    if (!data.requests.length) {
      host.appendChild(el('p', { class: 'empty', text: 'Nessuna richiesta salvata.' }));
      return;
    }

    for (const request of data.requests) {
      host.appendChild(
        el('div', { class: 'saved-row' }, [
          el('span', {}, [
            el('strong', { text: new Date(request.created_at).toLocaleString('it-IT') }),
            el('br'),
            el('code', { text: request.request_id.slice(0, 8) }),
            ' · ',
            el('span', { class: 'badge', text: request.status }),
          ]),
          el('button', {
            class: 'btn btn-ghost btn-sm',
            type: 'button',
            text: 'Riprendi',
            onclick: () => {
              state.requestId = request.request_id;
              localStorage.setItem(REQUEST_KEY, request.request_id);
              show('progress');
              startPolling();
            },
          }),
        ])
      );
    }
  }

  // ------------------------------------------------------------- bootstrap

  async function bootApp() {
    $('#login').hidden = true;
    $('#app').hidden = false;

    const me = await api('/api/auth/me');
    state.email = me.email;
    $('#user-email').textContent = me.email || '';

    await loadProviders();

    const start = new Date(Date.now() + 86400000);
    $('#form-new').policy_start_date.value = start.toISOString().slice(0, 10);

    if (state.requestId) {
      show('progress');
      startPolling();
    } else {
      show('new');
    }
  }

  function wire() {
    $('#form-login').addEventListener('submit', async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      $('#login-error').textContent = '';
      try {
        await signIn(form.email.value.trim(), form.password.value);
      } catch (error) {
        $('#login-error').textContent = error.message;
      }
    });

    $('#form-new').addEventListener('submit', submitNewRequest);
    $('#form-missing').addEventListener('submit', submitMissingFields);
    $('#logout').addEventListener('click', signOut);

    $('#btn-open-saved').addEventListener('click', async () => {
      const panel = $('#saved-requests');
      panel.hidden = !panel.hidden;
      if (!panel.hidden) await loadSaved();
    });

    $('#btn-cancel').addEventListener('click', async () => {
      try {
        await api(`/api/quotes/${state.requestId}/cancel`, { method: 'POST' });
        stopPolling();
        toast('Lavori in corso annullati.');
        await api(`/api/quotes/${state.requestId}/progress`).then(renderProgress);
      } catch (error) {
        toast(error.message, 'error');
      }
    });

    for (const button of $$('[data-goto]')) {
      button.addEventListener('click', async () => {
        const target = button.dataset.goto;
        if (target === 'results') await loadResults();
        else show(target);
      });
    }
  }

  wire();

  if (state.token) {
    bootApp().catch(() => signOut());
  } else {
    $('#login').hidden = false;
  }
})();
