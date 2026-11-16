/**
 * The app's A2UI catalog: the basic catalog plus two components the agent's surfaces use.
 *
 *   HighchartsChart  {options: DynamicValue (a Highcharts options object), height?: number}
 *   DataTable        {columns: DynamicValue ([{key, title, numeric}]), rows: DynamicValue ([{...}])}
 *
 * A v0.9 surface uses exactly one catalog, so both are merged under one catalog id, which the
 * server names in createSurface (app/presenter.py CATALOG_ID).
 */
import {css, html, nothing, type PropertyValues} from 'lit';
import Highcharts from 'highcharts';
import 'highcharts/modules/accessibility';
import {z} from 'zod';

import {A2uiLitElement, basicCatalog} from '@a2ui/lit/v0_9';
import {Catalog, DynamicNumberSchema, DynamicValueSchema, type ComponentApi} from '@a2ui/web_core/v0_9';

export const CATALOG_ID = 'https://github.com/shamx9ir/msc-ce893/app/catalog.json';

// ---- HighchartsChart ----------------------------------------------------------------------

export const HighchartsChartApi = {
  name: 'HighchartsChart',
  schema: z
    .object({
      options: DynamicValueSchema.describe('Highcharts options object (JSON, no functions).'),
      height: DynamicNumberSchema.describe('Chart height in pixels. Defaults to 320.').optional(),
      weight: z.number().optional(),
    })
    .describe('Displays a Highcharts chart.')
    .strict(),
} satisfies ComponentApi;

type Opts = Record<string, unknown>;

function isObj(v: unknown): v is Opts {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Deep merge for plain objects; arrays and scalars in `b` replace those in `a`. */
function merge(a: Opts, b: Opts): Opts {
  const out: Opts = {...a};
  for (const [k, v] of Object.entries(b)) {
    out[k] = isObj(v) && isObj(out[k]) ? merge(out[k] as Opts, v) : v;
  }
  return out;
}

/** Theme from the page's CSS variables, so the chart follows light/dark mode. */
function themeFor(el: Element): Opts {
  const s = getComputedStyle(el);
  const v = (name: string, fallback: string) => s.getPropertyValue(name).trim() || fallback;
  const text = v('--text-primary', '#0b0b0b');
  const muted = v('--text-secondary', '#52514e');
  const grid = v('--grid', '#e6e5e0');
  const colors = Array.from({length: 8}, (_, i) => v(`--series-${i + 1}`, '#2a78d6'));
  const axis = {
    labels: {style: {color: muted, fontSize: '12px'}},
    title: {style: {color: muted, fontSize: '12px', fontWeight: '500'}},
    lineColor: grid,
    tickColor: grid,
    gridLineColor: grid,
  };
  return {
    colors,
    chart: {backgroundColor: 'transparent', style: {fontFamily: 'inherit'}, spacing: [8, 8, 8, 8]},
    title: {style: {color: text}},
    xAxis: {...axis, gridLineWidth: 0},
    yAxis: {...axis, gridLineWidth: 1},
    legend: {itemStyle: {color: text, fontWeight: '500'}, itemHoverStyle: {color: text}},
    tooltip: {
      backgroundColor: v('--surface-2', '#ffffff'),
      borderColor: grid,
      style: {color: text},
      valueDecimals: undefined,
    },
    plotOptions: {
      series: {animation: {duration: 300}},
      column: {borderRadius: 4, borderWidth: 0, pointPadding: 0.08, groupPadding: 0.12},
      bar: {borderRadius: 4, borderWidth: 0, pointPadding: 0.08, groupPadding: 0.12},
      line: {lineWidth: 2, marker: {radius: 4, enabledThreshold: 6}},
      scatter: {marker: {radius: 4, symbol: 'circle', lineWidth: 1, lineColor: v('--surface-2', '#ffffff')}},
    },
  };
}

export class A2uiHighchartsElement extends A2uiLitElement<typeof HighchartsChartApi> {
  static readonly tagName = 'a2ui-highcharts';
  private chart?: Highcharts.Chart;
  private chartDiv?: HTMLDivElement;
  private drawnKey?: string;
  private resizeObserver?: ResizeObserver;

  static override styles = css`
    :host {
      display: block;
      width: 100%;
      min-width: 0;
    }
    .hc {
      width: 100%;
    }
  `;

  override render() {
    const props = this.controller.props;
    if (!props) return nothing;
    const h = typeof props.height === 'number' && props.height > 0 ? props.height : 320;
    return html`<div class="hc" style="height:${h}px"></div>`;
  }

  protected override updated(changed: PropertyValues) {
    super.updated(changed);
    this.draw();
  }

  override connectedCallback() {
    super.connectedCallback();
    this.resizeObserver = new ResizeObserver(() => this.chart?.reflow());
    this.resizeObserver.observe(this);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    this.resizeObserver?.disconnect();
    this.chart?.destroy();
    this.chart = undefined;
  }

  /** Redraws with the current theme (called by the app when the colour scheme changes). */
  redraw() {
    this.chart?.destroy();
    this.chart = undefined;
    this.draw();
  }

  private draw() {
    const div = this.renderRoot.querySelector<HTMLDivElement>('.hc');
    const options = this.controller?.props?.options;
    if (!div || !isObj(options)) return;
    // Lit re-renders this element whenever the page updates; only draw when something changed.
    const key = JSON.stringify(options);
    if (this.chart && this.chartDiv === div && this.drawnKey === key) return;
    this.chart?.destroy();
    this.chart = Highcharts.chart(div, merge(themeFor(this), options) as Highcharts.Options);
    this.chartDiv = div;
    this.drawnKey = key;
  }
}

// ---- DataTable ----------------------------------------------------------------------------

export const DataTableApi = {
  name: 'DataTable',
  schema: z
    .object({
      columns: DynamicValueSchema.describe('List of {key, title, numeric}.'),
      rows: DynamicValueSchema.describe('List of row objects keyed by column key.'),
      weight: z.number().optional(),
    })
    .describe('Displays rows as a table.')
    .strict(),
} satisfies ComponentApi;

interface Column {
  key: string;
  title: string;
  numeric?: boolean;
  grouping?: boolean;
  time?: boolean;
}

function fmt(v: unknown, c: Column): string {
  if (v === null || v === undefined || v === '') return '–';
  if (c.numeric) {
    const n = Number(v);
    if (Number.isFinite(n)) return n.toLocaleString(undefined, {maximumFractionDigits: 3, useGrouping: !!c.grouping});
  }
  if (c.time && typeof v === 'string') return v.replace(/T00:00:00(\.000)?$/, '').replace('T', ' ').replace(/\.000$/, '');
  return String(v);
}

export class A2uiDataTableElement extends A2uiLitElement<typeof DataTableApi> {
  static readonly tagName = 'a2ui-data-table';

  static override styles = css`
    :host {
      display: block;
      width: 100%;
      min-width: 0;
    }
    .wrap {
      max-height: 360px;
      overflow: auto;
      border: 1px solid var(--grid, #e6e5e0);
      border-radius: 8px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    th,
    td {
      padding: 6px 10px;
      text-align: left;
      border-bottom: 1px solid var(--grid, #e6e5e0);
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--surface-2, #fff);
      color: var(--text-secondary, #52514e);
      font-weight: 600;
    }
    td.num,
    th.num {
      text-align: right;
    }
    tr:last-child td {
      border-bottom: 0;
    }
  `;

  override render() {
    const props = this.controller.props;
    if (!props) return nothing;
    const cols = (Array.isArray(props.columns) ? props.columns : []) as Column[];
    const rows = (Array.isArray(props.rows) ? props.rows : []) as Opts[];
    if (!rows.length) return html`<p>No rows.</p>`;
    return html`<div class="wrap">
      <table>
        <thead>
          <tr>
            ${cols.map(c => html`<th class=${c.numeric ? 'num' : ''} scope="col">${c.title}</th>`)}
          </tr>
        </thead>
        <tbody>
          ${rows.map(
            r =>
              html`<tr>
                ${cols.map(c => html`<td class=${c.numeric ? 'num' : ''}>${fmt(r[c.key], c)}</td>`)}
              </tr>`,
          )}
        </tbody>
      </table>
    </div>`;
  }
}

// ---- catalog ------------------------------------------------------------------------------

for (const el of [A2uiHighchartsElement, A2uiDataTableElement]) {
  if (!customElements.get(el.tagName)) customElements.define(el.tagName, el);
}

export const appCatalog = new Catalog(
  CATALOG_ID,
  '0.9',
  [
    ...basicCatalog.components.values(),
    {...HighchartsChartApi, tagName: A2uiHighchartsElement.tagName, element: A2uiHighchartsElement},
    {...DataTableApi, tagName: A2uiDataTableElement.tagName, element: A2uiDataTableElement},
  ],
  [...basicCatalog.functions.values()],
  basicCatalog.themeSchema,
);
