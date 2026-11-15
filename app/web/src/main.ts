/**
 * The demo page. It sends the question to the Python server (app/server.py), which runs the
 * harness agent and answers with A2UI messages; the page only feeds those messages to the A2UI
 * MessageProcessor and mounts one <a2ui-surface> per answer.
 */
import {ContextProvider} from '@lit/context';
import {css, html, LitElement, nothing} from 'lit';

import {Context} from '@a2ui/lit/v0_9';
import '@a2ui/lit/v0_9'; // registers <a2ui-surface> and the basic components
import {renderMarkdown} from '@a2ui/markdown-it';
import {type A2uiMessage, type ComponentApi, MessageProcessor, type SurfaceModel} from '@a2ui/web_core/v0_9';

import {A2uiHighchartsElement, appCatalog} from './catalog.js';

interface ModelInfo {
  id: string;
  label: string;
  short: string;
  lora: boolean;
  strict: number;
  available: boolean | null;
}
interface Example {
  question: string;
  hint: string | null;
}
interface Config {
  models: ModelInfo[];
  databases: string[];
  examples: Record<string, Example[]>;
  birdExamples: Record<string, Example[]>;
  cube: boolean;
  ollama: boolean;
}
interface Answer {
  model: ModelInfo;
  state: 'waiting' | 'running' | 'done' | 'failed';
  started?: number;
  surfaceId?: string;
  error?: string;
}
interface Turn {
  id: number;
  question: string;
  hint: string | null;
  db: string;
  answers: Answer[];
}

const ALL = '__all__';

export class CubeDemoApp extends LitElement {
  static properties = {
    config: {state: true},
    loadError: {state: true},
    db: {state: true},
    model: {state: true},
    question: {state: true},
    hint: {state: true},
    showHint: {state: true},
    turns: {state: true},
    busy: {state: true},
    now: {state: true},
  };

  config?: Config;
  loadError = '';
  db = 'formula_1';
  model = 'ollama_chat/qwen14b-cube-lora';
  question = '';
  hint = '';
  showHint = false;
  turns: Turn[] = [];
  busy = false;
  now = Date.now();

  private processor = new MessageProcessor<ComponentApi>([appCatalog]);
  private surfaces = new Map<string, SurfaceModel<ComponentApi>>();
  private nextId = 1;
  private timer?: number;

  constructor() {
    super();
    new ContextProvider(this, {
      context: Context.markdown,
      initialValue: (value: string, options?: unknown) =>
        Promise.resolve(renderMarkdown(value, options as never)),
    });
    this.processor.onSurfaceCreated(s => this.surfaces.set(s.id, s));
  }

  override connectedCallback() {
    super.connectedCallback();
    void this.loadConfig();
    // Charts read colours from CSS variables when drawn, so redraw them on a scheme change.
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () =>
      this.renderRoot.querySelectorAll<A2uiHighchartsElement>('a2ui-highcharts').forEach(c => c.redraw()),
    );
  }

  private async loadConfig() {
    try {
      const r = await fetch('/api/config');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      this.config = await r.json();
    } catch (e) {
      this.loadError = `Cannot reach the app server (${e}). Start it with: python -m app.server`;
    }
  }

  private async ask(question = this.question, hint: string | null = this.hint || null) {
    question = question.trim();
    if (!question || this.busy || !this.config) return;
    const models = this.model === ALL ? this.config.models : this.config.models.filter(m => m.id === this.model);
    const turn: Turn = {
      id: this.nextId++,
      question,
      hint,
      db: this.db,
      answers: models.map(m => ({model: m, state: 'waiting'})),
    };
    this.turns = [turn, ...this.turns];
    this.busy = true;
    this.timer = window.setInterval(() => (this.now = Date.now()), 250);
    // One model at a time: they share the GPU, and the server queues requests anyway.
    for (const a of turn.answers) {
      a.state = 'running';
      a.started = Date.now();
      this.requestUpdate();
      try {
        const r = await fetch('/api/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question, hint, db: turn.db, model: a.model.id}),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        const body = await r.json();
        this.processor.processMessages(body.messages as A2uiMessage[]);
        a.surfaceId = body.surfaceId;
        a.state = 'done';
      } catch (e) {
        a.state = 'failed';
        a.error = String(e);
      }
      this.requestUpdate();
    }
    window.clearInterval(this.timer);
    this.busy = false;
  }

  private pick(ex: Example) {
    this.question = ex.question;
    this.hint = ex.hint ?? '';
    this.showHint = !!ex.hint;
  }

  // ---- rendering ----------------------------------------------------------

  override render() {
    return html`
      <header>
        <div>
          <h1>Ask the BIRD databases</h1>
          <p class="sub">
            Your question goes to an ADK agent that writes a Cube query; the answer comes back as an A2UI
            surface with a Highcharts chart.
          </p>
        </div>
        ${this.renderStatus()}
      </header>
      ${this.loadError ? html`<div class="banner" role="alert">${this.loadError}</div>` : nothing}
      ${this.config ? this.renderForm(this.config) : nothing}
      <section class="feed">${this.turns.map(t => this.renderTurn(t))}</section>
    `;
  }

  private renderStatus() {
    if (!this.config) return nothing;
    const pill = (ok: boolean, label: string) =>
      html`<span class="pill ${ok ? 'ok' : 'bad'}"><span class="dot"></span>${label}</span>`;
    return html`<div class="status">
      ${pill(this.config.cube, this.config.cube ? 'Cube connected' : 'Cube not reachable')}
      ${pill(this.config.ollama, this.config.ollama ? 'Ollama connected' : 'Ollama not reachable')}
    </div>`;
  }

  private renderForm(cfg: Config) {
    const examples = cfg.examples[this.db] ?? [];
    const bird = cfg.birdExamples[this.db] ?? [];
    return html`
      <form
        class="ask"
        @submit=${(e: Event) => {
          e.preventDefault();
          void this.ask();
        }}
      >
        <div class="controls">
          <div class="seg" role="radiogroup" aria-label="Database">
            ${cfg.databases.map(
              d => html`<button
                type="button"
                role="radio"
                aria-checked=${this.db === d}
                class=${this.db === d ? 'on' : ''}
                @click=${() => (this.db = d)}
              >
                ${d}
              </button>`,
            )}
          </div>
          <label class="model">
            <span>Model</span>
            <select @change=${(e: Event) => (this.model = (e.target as HTMLSelectElement).value)}>
              ${cfg.models.map(
                m => html`<option value=${m.id} ?selected=${this.model === m.id}>
                  ${m.label} (${m.strict}% on test)${m.available === false ? ' – not installed' : ''}
                </option>`,
              )}
              <option value=${ALL} ?selected=${this.model === ALL}>Compare all four</option>
            </select>
          </label>
        </div>
        <div class="q">
          <textarea
            rows="2"
            placeholder="e.g. How many races were held in each year?"
            .value=${this.question}
            @input=${(e: Event) => (this.question = (e.target as HTMLTextAreaElement).value)}
            @keydown=${(e: KeyboardEvent) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void this.ask();
              }
            }}
          ></textarea>
          <button class="primary" type="submit" ?disabled=${this.busy || !this.question.trim()}>
            ${this.busy ? 'Working…' : 'Ask'}
          </button>
        </div>
        ${this.showHint
          ? html`<input
              class="hint"
              placeholder="Optional hint, like BIRD's evidence (e.g. full name = forename, surname)"
              .value=${this.hint}
              @input=${(e: Event) => (this.hint = (e.target as HTMLInputElement).value)}
            />`
          : html`<button type="button" class="link" @click=${() => (this.showHint = true)}>+ Add a hint</button>`}
        <div class="examples">
          <span class="label">Try</span>
          ${examples.map(ex => html`<button type="button" class="chip" @click=${() => this.pick(ex)}>${ex.question}</button>`)}
        </div>
        ${bird.length
          ? html`<details class="examples">
              <summary>Questions from the BIRD test set</summary>
              ${bird.map(ex => html`<button type="button" class="chip" @click=${() => this.pick(ex)}>${ex.question}</button>`)}
            </details>`
          : nothing}
      </form>
    `;
  }

  private renderTurn(t: Turn) {
    return html`<article class="turn">
      <div class="turn-head">
        <span class="db">${t.db}</span>
        <h2>${t.question}</h2>
        ${t.hint ? html`<p class="hint-text">Hint: ${t.hint}</p>` : nothing}
      </div>
      <div class="answers ${t.answers.length > 1 ? 'grid' : ''}">${t.answers.map(a => this.renderAnswer(a))}</div>
    </article>`;
  }

  private renderAnswer(a: Answer) {
    const badge = html`<span class="badge ${a.model.lora ? 'lora' : ''}">${a.model.short}</span>`;
    if (a.state === 'done' && a.surfaceId) {
      const s = this.surfaces.get(a.surfaceId);
      return html`<div class="answer">${badge}${s ? html`<a2ui-surface .surface=${s}></a2ui-surface>` : nothing}</div>`;
    }
    if (a.state === 'failed') {
      return html`<div class="answer">${badge}<p class="err">${a.error}</p></div>`;
    }
    const secs = a.started ? ((this.now - a.started) / 1000).toFixed(1) : null;
    return html`<div class="answer pending">
      ${badge}
      <p>${a.model.label}</p>
      <p class="muted">${a.state === 'running' ? html`<span class="spin"></span> Agent working… ${secs} s` : 'Queued'}</p>
    </div>`;
  }

  static override styles = css`
    :host {
      display: block;
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 16px 64px;
      color: var(--text-primary);
      --a2ui-color-primary: var(--accent);
      --a2ui-color-on-primary: #fff;
      --a2ui-color-surface: var(--surface-2);
      --a2ui-color-on-surface: var(--text-primary);
      --a2ui-color-border: var(--grid);
      --a2ui-color-on-background: var(--text-primary);
      --a2ui-text-caption-color: var(--text-secondary);
      --a2ui-border-radius: 10px;
      --a2ui-card-background: var(--surface-1);
      --a2ui-card-border: 1px solid var(--grid);
      --a2ui-card-padding: 14px;
    }
    header {
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }
    h1 {
      font-size: 26px;
      margin: 0 0 6px;
      letter-spacing: -0.01em;
    }
    .sub {
      margin: 0;
      color: var(--text-secondary);
      max-width: 62ch;
      line-height: 1.5;
    }
    .status {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--grid);
      color: var(--text-secondary);
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    .pill.ok .dot {
      background: var(--good);
    }
    .pill.bad .dot {
      background: var(--critical);
    }
    .banner {
      padding: 12px 14px;
      border-radius: 8px;
      border: 1px solid var(--critical);
      margin-bottom: 16px;
    }
    form.ask {
      background: var(--surface-2);
      border: 1px solid var(--grid);
      border-radius: 14px;
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .seg {
      display: inline-flex;
      border: 1px solid var(--grid);
      border-radius: 9px;
      overflow: hidden;
    }
    .seg button {
      border: 0;
      background: transparent;
      color: var(--text-secondary);
      padding: 7px 14px;
      font: inherit;
      font-size: 14px;
      cursor: pointer;
    }
    .seg button.on {
      background: var(--accent);
      color: #fff;
    }
    .model {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      font-size: 14px;
      color: var(--text-secondary);
      min-width: 0;
      max-width: 100%;
    }
    select,
    textarea,
    input.hint {
      font: inherit;
      color: var(--text-primary);
      background: var(--surface-1);
      border: 1px solid var(--grid);
      border-radius: 9px;
      padding: 8px 10px;
      min-width: 0;
      max-width: 100%;
    }
    .q {
      display: flex;
      gap: 10px;
    }
    textarea {
      flex: 1;
      resize: vertical;
      font-size: 15px;
    }
    button.primary {
      font: inherit;
      font-weight: 600;
      border: 0;
      border-radius: 9px;
      padding: 0 22px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
    }
    button.primary:disabled {
      opacity: 0.5;
      cursor: default;
    }
    button.link {
      justify-self: start;
      border: 0;
      background: none;
      color: var(--accent);
      font: inherit;
      font-size: 13px;
      padding: 0;
      cursor: pointer;
    }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      font-size: 13px;
    }
    details.examples {
      display: block;
    }
    details.examples summary {
      cursor: pointer;
      color: var(--text-secondary);
      margin-bottom: 6px;
    }
    details.examples .chip {
      margin: 0 6px 6px 0;
    }
    .label {
      color: var(--text-secondary);
      margin-right: 4px;
    }
    .chip {
      font: inherit;
      font-size: 13px;
      text-align: left;
      border: 1px solid var(--grid);
      background: var(--surface-1);
      color: var(--text-primary);
      border-radius: 999px;
      padding: 4px 10px;
      cursor: pointer;
    }
    .chip:hover {
      border-color: var(--accent);
    }
    .feed {
      display: grid;
      gap: 28px;
      margin-top: 28px;
    }
    .turn-head h2 {
      font-size: 18px;
      margin: 4px 0 10px;
    }
    .db {
      font-size: 12px;
      color: var(--text-secondary);
      font-family: ui-monospace, monospace;
    }
    .hint-text {
      margin: -4px 0 10px;
      font-size: 13px;
      color: var(--text-secondary);
    }
    .answers {
      display: grid;
      gap: 14px;
    }
    .answers.grid {
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr));
    }
    .answer {
      position: relative;
      background: var(--surface-2);
      border: 1px solid var(--grid);
      border-radius: 14px;
      padding: 14px 16px;
      min-width: 0;
    }
    .answer.pending {
      min-height: 120px;
    }
    .badge {
      position: absolute;
      top: 12px;
      right: 14px;
      font-size: 12px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 6px;
      border: 1px solid var(--grid);
      color: var(--text-secondary);
    }
    .badge.lora {
      border-color: var(--accent);
      color: var(--accent);
    }
    .muted {
      color: var(--text-secondary);
      font-size: 14px;
    }
    .err {
      color: var(--critical);
      white-space: pre-wrap;
      font-size: 13px;
    }
    .spin {
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 2px solid var(--grid);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      vertical-align: -1px;
      margin-right: 4px;
    }
    @keyframes spin {
      to {
        transform: rotate(360deg);
      }
    }
    a2ui-surface {
      display: block;
    }
    a2ui-surface pre {
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      background: var(--surface-1);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 10px;
      max-height: 360px;
      overflow: auto;
    }
    @media (max-width: 560px) {
      .q {
        flex-direction: column;
      }
      button.primary {
        padding: 10px;
      }
    }
  `;
}

customElements.define('cube-demo-app', CubeDemoApp);
