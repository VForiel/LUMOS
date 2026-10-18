// Studies (REQ-CTX-10): the dialog starting a sweep or a Monte-Carlo run, and the tab of the bottom panel
// showing its points as a chart (curve of a sweep, histogram of a Monte-Carlo run) and a table.

import { Dices, Download, FlaskConical, Loader2, Plus, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type ContextSummary, type ElementType, type StudyKind, type StudyRequest, type StudyResult, type SweepSpec } from "../api";
import { fieldsOf } from "../lib/schema";
import { formatNumber, parseQuantity } from "../lib/units";
import { useStore } from "../store";
import { Button, Modal } from "./Dialogs";
import { inputClass } from "./Fields";

/** A parameter that can be swept: its path, unit and current value. */
type Candidate = { path: string; unit: string; value: number };
type Row = { path: string; start: string; stop: string; num: number; log: boolean };
/** Defaults of the study dialog stored by a preset in ``ui.study``. */
type StudyDefaults = { kind?: StudyKind; n?: number; outputs?: string[]; metric?: string; parameters?: Record<string, SweepSpec> };

const STATISTICS = ["max", "mean", "min", "sum"] as const;

export function candidatesOf(context: ContextSummary, types: Map<string, ElementType>): Candidate[] {
  const found: Candidate[] = [];
  for (const element of context.elements) {
    const type = types.get(element.type_id);
    if (!type) continue;
    for (const field of fieldsOf(type.schema)) {
      if (field.kind !== "param" || field.input !== "quantity") continue;
      const current = parseQuantity(String(element.params[field.name] ?? ""));
      if (!current) continue;
      found.push({ path: `${element.id}.${field.name}`, unit: current.unit, value: current.value });
    }
  }
  return found;
}

/** Outputs computed by default: those of the context, else the unconnected outputs of its blocks. */
export function defaultOutputs(context: ContextSummary): string[] {
  if (context.outputs.length) return context.outputs;
  const used = new Set(context.connections.map((c) => c.source));
  return context.elements
    .filter((e) => !e.id.includes("."))
    .flatMap((e) => e.outputs.map((port) => `${e.id}.${port}`))
    .filter((port) => !used.has(port));
}

function rowFor(candidate: Candidate | undefined): Row {
  if (!candidate) return { path: "", start: "", stop: "", num: 5, log: false };
  const { value } = candidate;
  const [start, stop] = value === 0 ? [0, 1] : [value * 0.9, value * 1.1];
  return { path: candidate.path, start: formatNumber(start), stop: formatNumber(stop), num: 5, log: false };
}

/** A number typed in the unit of the parameter, or a quantity with its own unit ("190 mm"). */
function withUnit(text: string, unit: string): string {
  const quantity = parseQuantity(text.trim());
  if (quantity && quantity.unit) return text.trim();
  return unit ? `${text.trim()} ${unit}` : text.trim();
}

export function StudyDialog() {
  const { t } = useTranslation();
  const context = useStore((s) => s.context)!;
  const palette = useStore((s) => s.palette);
  const previous = useStore((s) => s.studyRequest);
  const types = useMemo(() => new Map((palette?.types ?? []).map((item) => [item.type_id, item])), [palette]);
  const candidates = useMemo(() => candidatesOf(context, types), [context, types]);
  const defaults = (context.ui.study ?? {}) as StudyDefaults;
  const free = context.elements.flatMap((e) => Object.entries(e.free ?? {}).map(([name, prior]) => ({ path: `${e.id}.${name}`, prior })));
  const noisy = useMemo(() => context.elements.some((e) => types.get(e.type_id)?.schema?.["x-lumos-stochastic"]), [context, types]);

  const [kind, setKind] = useState<StudyKind>(previous?.kind ?? defaults.kind ?? (free.length ? "sample" : "sweep"));
  const [rows, setRows] = useState<Row[]>(() => {
    const listed = previous?.parameters ?? defaults.parameters;
    if (listed && Object.keys(listed).length) {
      return Object.entries(listed).map(([path, spec]) => {
        const candidate = candidates.find((c) => c.path === path);
        if (Array.isArray(spec)) return { ...rowFor(candidate), path };
        const strip = (text: string) => String(parseQuantity(text)?.value ?? text);
        return { path, start: strip(spec.start), stop: strip(spec.stop), num: spec.num, log: Boolean(spec.log) };
      });
    }
    return [rowFor(candidates[0])];
  });
  const [mode, setMode] = useState<"grid" | "zip">(previous?.mode ?? "grid");
  const [n, setN] = useState(previous?.n ?? defaults.n ?? 50);
  const [independent, setIndependent] = useState<boolean | null>(null);
  const available = defaultOutputs(context);
  const [outputs, setOutputs] = useState<string[]>(() => {
    const wanted = previous?.outputs ?? defaults.outputs ?? available;
    return wanted.filter((port) => available.includes(port) || context.outputs.includes(port));
  });
  const noise = independent ?? kind === "sample";
  const close = () => useStore.getState().setDialog(null);

  const count =
    kind === "sample"
      ? n
      : rows.length === 0
        ? 0
        : mode === "zip"
          ? Math.min(...rows.map((r) => r.num))
          : rows.reduce((product, r) => product * Math.max(r.num, 0), 1);
  const unitOf = (path: string) => candidates.find((c) => c.path === path)?.unit ?? "";
  const valid = outputs.length > 0 && count > 0 && (kind === "sample" || rows.every((r) => r.path && r.start !== "" && r.stop !== "" && r.num >= 1));

  const start = async () => {
    const request: StudyRequest = { kind, outputs, noise: noise ? "independent" : "fixed" };
    if (kind === "sample") request.n = n;
    else {
      request.mode = mode;
      request.parameters = Object.fromEntries(
        rows.map((r) => [r.path, { start: withUnit(r.start, unitOf(r.path)), stop: withUnit(r.stop, unitOf(r.path)), num: r.num, log: r.log }]),
      );
    }
    if (await useStore.getState().runStudy(request)) close();
  };

  const update = (index: number, change: Partial<Row>) => setRows(rows.map((row, k) => (k === index ? { ...row, ...change } : row)));

  return (
    <Modal title={t("study.title")} onClose={close} wide testId="study-dialog">
      <div className="space-y-4 p-5 text-sm">
        <div className="flex gap-2" role="group" aria-label={t("study.kind")}>
          {(["sweep", "sample"] as const).map((value) => (
            <button
              key={value}
              type="button"
              data-testid={`study-kind-${value}`}
              aria-pressed={kind === value}
              onClick={() => setKind(value)}
              className={`flex flex-1 items-start gap-2 rounded-xl border p-3 text-left transition-colors ${
                kind === value
                  ? "border-accent-400 bg-accent-50 ring-2 ring-accent-200 dark:border-accent-600 dark:bg-accent-950/40 dark:ring-accent-900"
                  : "border-slate-200 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
              }`}
            >
              {value === "sweep" ? <FlaskConical className="mt-0.5 size-4 shrink-0 text-accent-500" /> : <Dices className="mt-0.5 size-4 shrink-0 text-accent-500" />}
              <span>
                <span className="block font-semibold">{t(`study.${value}`)}</span>
                <span className="block text-xs text-slate-500 dark:text-slate-400">{t(`study.${value}Hint`)}</span>
              </span>
            </button>
          ))}
        </div>

        {kind === "sweep" ? (
          <div className="space-y-2">
            <div className="grid grid-cols-[minmax(10rem,1fr)_5.5rem_5.5rem_4rem_auto_auto] items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <span>{t("study.parameter")}</span>
              <span>{t("study.start")}</span>
              <span>{t("study.stop")}</span>
              <span>{t("study.num")}</span>
              <span>{t("study.log")}</span>
              <span />
            </div>
            {rows.map((row, index) => (
              <div key={index} className="grid grid-cols-[minmax(10rem,1fr)_5.5rem_5.5rem_4rem_auto_auto] items-center gap-2">
                <select
                  data-testid={`study-param-${index}`}
                  className={inputClass}
                  value={row.path}
                  onChange={(event) => {
                    const candidate = candidates.find((c) => c.path === event.target.value);
                    update(index, { ...rowFor(candidate), num: row.num });
                  }}
                >
                  {candidates.map((c) => (
                    <option key={c.path} value={c.path}>
                      {c.path}
                      {c.unit ? ` [${c.unit}]` : ""}
                    </option>
                  ))}
                </select>
                <input data-testid={`study-start-${index}`} className={inputClass} value={row.start} onChange={(e) => update(index, { start: e.target.value })} aria-label={t("study.start")} />
                <input data-testid={`study-stop-${index}`} className={inputClass} value={row.stop} onChange={(e) => update(index, { stop: e.target.value })} aria-label={t("study.stop")} />
                <input
                  data-testid={`study-num-${index}`}
                  className={inputClass}
                  type="number"
                  min={1}
                  value={row.num}
                  onChange={(e) => update(index, { num: Math.max(0, Math.round(Number(e.target.value))) })}
                  aria-label={t("study.num")}
                />
                <input data-testid={`study-log-${index}`} type="checkbox" checked={row.log} onChange={(e) => update(index, { log: e.target.checked })} aria-label={t("study.log")} className="size-4 accent-accent-600" />
                <button type="button" onClick={() => setRows(rows.filter((_, k) => k !== index))} disabled={rows.length === 1} aria-label={t("study.remove")} className="rounded p-1 text-slate-400 hover:text-red-600 disabled:opacity-30">
                  <Trash2 className="size-4" />
                </button>
              </div>
            ))}
            <div className="flex items-center gap-3">
              <button type="button" data-testid="study-add" onClick={() => setRows([...rows, rowFor(candidates.find((c) => !rows.some((r) => r.path === c.path)))])} className="inline-flex items-center gap-1 text-xs font-medium text-accent-600 hover:underline dark:text-accent-300">
                <Plus className="size-3.5" />
                {t("study.add")}
              </button>
              {rows.length > 1 && (
                <label className="ml-auto flex items-center gap-2 text-xs">
                  {t("study.mode")}
                  <select data-testid="study-mode" value={mode} onChange={(e) => setMode(e.target.value as "grid" | "zip")} className={`${inputClass} w-auto`}>
                    <option value="grid">{t("study.grid")}</option>
                    <option value="zip">{t("study.zip")}</option>
                  </select>
                </label>
              )}
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">{t("study.unitsHint")}</p>
          </div>
        ) : (
          <div className="space-y-3">
            <label className="flex items-center gap-3">
              <span className="w-32 text-slate-600 dark:text-slate-300">{t("study.draws")}</span>
              <input data-testid="study-n" type="number" min={1} className={`${inputClass} w-28`} value={n} onChange={(e) => setN(Math.max(0, Math.round(Number(e.target.value))))} />
            </label>
            <div data-testid="study-free" className="rounded-lg border border-slate-200 p-3 dark:border-slate-700">
              <p className="mb-1 text-xs font-semibold text-slate-500 uppercase dark:text-slate-400">{t("study.freeTitle")}</p>
              {free.length ? (
                <ul className="space-y-0.5 font-mono text-xs">
                  {free.map(({ path, prior }) => (
                    <li key={path}>
                      {path} <span className="text-slate-500">{priorText(prior)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">{t(noisy ? "study.noFreeNoise" : "study.noFree")}</p>
              )}
            </div>
          </div>
        )}

        <div className="grid gap-3 sm:grid-cols-2">
          <fieldset className="rounded-lg border border-slate-200 p-3 dark:border-slate-700">
            <legend className="px-1 text-xs font-semibold text-slate-500 uppercase dark:text-slate-400">{t("study.outputs")}</legend>
            {available.length === 0 && <p className="text-xs text-slate-500">{t("study.noOutputs")}</p>}
            {available.map((port) => (
              <label key={port} className="flex items-center gap-2 py-0.5 font-mono text-xs">
                <input
                  type="checkbox"
                  data-testid={`study-output-${port}`}
                  checked={outputs.includes(port)}
                  onChange={(e) => setOutputs(e.target.checked ? [...outputs, port] : outputs.filter((o) => o !== port))}
                  className="size-4 accent-accent-600"
                />
                {port}
              </label>
            ))}
          </fieldset>
          <label className="flex items-start gap-2 rounded-lg border border-slate-200 p-3 text-xs dark:border-slate-700">
            <input type="checkbox" data-testid="study-noise" checked={noise} onChange={(e) => setIndependent(e.target.checked)} className="mt-0.5 size-4 accent-accent-600" />
            <span>
              <span className="block font-semibold text-slate-700 dark:text-slate-200">{t("study.independent")}</span>
              <span className="text-slate-500 dark:text-slate-400">{t("study.independentHint", { seed: context.seed })}</span>
            </span>
          </label>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-slate-100 pt-3 dark:border-slate-800">
          <span data-testid="study-count" className="mr-auto text-xs text-slate-500 dark:text-slate-400">
            {t("study.count", { count })}
          </span>
          <Button onClick={close}>{t("common.cancel")}</Button>
          <Button primary onClick={() => void start()} testId="study-run" disabled={!valid}>
            {t("study.run")}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

export function priorText(prior: Record<string, string> | null | undefined): string {
  if (!prior) return "—";
  if (prior.kind === "normal") return `N(${prior.mean}, ${prior.std})`;
  if (prior.kind === "loguniform") return `logU[${prior.low}, ${prior.high}]`;
  return `U[${prior.low}, ${prior.high}]`;
}

// --- Results -------------------------------------------------------------------------------------------------

/** The value plotted for an output: its single value, or a statistic of its samples. */
function measure(study: StudyResult, index: number, output: string, metric: string): number | null {
  const summary = study.points[index].outputs[output];
  if (!summary) return null;
  if (metric === "value") return summary.values?.length === 1 ? summary.values[0] : null;
  return summary.statistics[metric] ?? null;
}

function scalar(study: StudyResult, output: string): boolean {
  return study.points.every((p) => p.outputs[output]?.values?.length === 1);
}

function ticks(low: number, high: number, count = 5): number[] {
  if (!(high > low)) return [low];
  const step = (high - low) / (count - 1);
  return Array.from({ length: count }, (_, k) => low + k * step);
}

const W = 520;
const H = 200;
const PAD = { left: 64, right: 16, top: 10, bottom: 34 };
const SERIES = ["#7c3aed", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#64748b"];

function Axes({ xs, ys, xLabel, yLabel }: { xs: [number, number]; ys: [number, number]; xLabel: string; yLabel: string }) {
  const sx = (x: number) => PAD.left + ((x - xs[0]) / (xs[1] - xs[0] || 1)) * (W - PAD.left - PAD.right);
  const sy = (y: number) => H - PAD.bottom - ((y - ys[0]) / (ys[1] - ys[0] || 1)) * (H - PAD.top - PAD.bottom);
  return (
    <g className="text-[10px]" fill="currentColor">
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="currentColor" strokeOpacity={0.3} />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="currentColor" strokeOpacity={0.3} />
      {ticks(...xs).map((x) => (
        <text key={`x${x}`} x={sx(x)} y={H - PAD.bottom + 14} textAnchor="middle" fillOpacity={0.7}>
          {formatNumber(Number(x.toPrecision(3)))}
        </text>
      ))}
      {ticks(...ys).map((y) => (
        <g key={`y${y}`}>
          <line x1={PAD.left} y1={sy(y)} x2={W - PAD.right} y2={sy(y)} stroke="currentColor" strokeOpacity={0.08} />
          <text x={PAD.left - 6} y={sy(y) + 3} textAnchor="end" fillOpacity={0.7}>
            {formatNumber(Number(y.toPrecision(3)))}
          </text>
        </g>
      ))}
      <text x={(PAD.left + W - PAD.right) / 2} y={H - 6} textAnchor="middle" fillOpacity={0.85}>
        {xLabel}
      </text>
      <text x={12} y={(PAD.top + H - PAD.bottom) / 2} textAnchor="middle" fillOpacity={0.85} transform={`rotate(-90 12 ${(PAD.top + H - PAD.bottom) / 2})`}>
        {yLabel}
      </text>
    </g>
  );
}

function bounds(values: number[]): [number, number] {
  const low = Math.min(...values);
  const high = Math.max(...values);
  if (high > low) return [low, high];
  const pad = Math.abs(low) * 0.05 || 1;
  return [low - pad, high + pad];
}

function SweepChart({ study, output, metric, x }: { study: StudyResult; output: string; metric: string; x: string }) {
  const unit = study.points[0]?.outputs[output]?.unit ?? "";
  const others = study.parameters.filter((p) => p !== x);
  const series = new Map<string, { x: number; y: number }[]>();
  study.points.forEach((point, index) => {
    const y = measure(study, index, output, metric);
    const value = parseQuantity(point.params[x] ?? "")?.value;
    if (y === null || value === undefined) return;
    const key = others.map((p) => `${p} = ${point.params[p]}`).join(", ");
    series.set(key, [...(series.get(key) ?? []), { x: value, y }]);
  });
  const all = [...series.values()].flat();
  if (!all.length) return null;
  const xs = bounds(all.map((p) => p.x));
  const ys = bounds(all.map((p) => p.y));
  const sx = (v: number) => PAD.left + ((v - xs[0]) / (xs[1] - xs[0])) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - ((v - ys[0]) / (ys[1] - ys[0])) * (H - PAD.top - PAD.bottom);
  const xUnit = parseQuantity(study.points[0].params[x] ?? "")?.unit;
  return (
    <svg data-testid="study-chart" viewBox={`0 0 ${W} ${H}`} className="h-[200px] w-[520px] text-slate-700 dark:text-slate-200" role="img" aria-label={`${output} vs ${x}`}>
      <Axes xs={xs} ys={ys} xLabel={`${x}${xUnit ? ` [${xUnit}]` : ""}`} yLabel={`${output} ${metric === "value" ? "" : metric}${unit ? ` [${unit}]` : ""}`} />
      {[...series.entries()].map(([key, points], k) => {
        const sorted = [...points].sort((a, b) => a.x - b.x);
        const color = SERIES[k % SERIES.length];
        return (
          <g key={key}>
            <path className="curve" d={sorted.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ")} fill="none" stroke={color} strokeWidth={2} />
            {sorted.map((p, i) => (
              <circle key={i} className="point" cx={sx(p.x)} cy={sy(p.y)} r={3.5} fill={color}>
                <title>{`${formatNumber(p.x)} → ${formatNumber(p.y)}${key ? ` (${key})` : ""}`}</title>
              </circle>
            ))}
          </g>
        );
      })}
      {series.size > 1 &&
        [...series.keys()].map((key, k) => (
          <text key={key} x={W - PAD.right} y={PAD.top + 10 + k * 12} textAnchor="end" className="text-[10px]" fill={SERIES[k % SERIES.length]}>
            {key}
          </text>
        ))}
    </svg>
  );
}

function Histogram({ study, output, metric }: { study: StudyResult; output: string; metric: string }) {
  const { t } = useTranslation();
  const values = study.points.map((_, index) => measure(study, index, output, metric)).filter((v): v is number => v !== null);
  if (!values.length) return null;
  const unit = study.points[0]?.outputs[output]?.unit ?? "";
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = values.length > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1)) : 0;
  const xs = bounds(values);
  const bins = Math.min(30, Math.max(5, Math.ceil(Math.sqrt(values.length))));
  const width = (xs[1] - xs[0]) / bins;
  const counts = Array.from({ length: bins }, () => 0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor((v - xs[0]) / width))] += 1;
  const ys: [number, number] = [0, Math.max(...counts)];
  const sx = (v: number) => PAD.left + ((v - xs[0]) / (xs[1] - xs[0])) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - (v / (ys[1] || 1)) * (H - PAD.top - PAD.bottom);
  return (
    <div className="space-y-1">
      <svg data-testid="study-chart" viewBox={`0 0 ${W} ${H}`} className="h-[200px] w-[520px] text-slate-700 dark:text-slate-200" role="img" aria-label={`${output} histogram`}>
        <Axes xs={xs} ys={ys} xLabel={`${output} ${metric === "value" ? "" : metric}${unit ? ` [${unit}]` : ""}`} yLabel={t("study.draws")} />
        {counts.map((count, k) => (
          <rect key={k} className="bar" x={sx(xs[0] + k * width) + 1} y={sy(count)} width={Math.max(sx(xs[0] + width) - sx(xs[0]) - 2, 1)} height={H - PAD.bottom - sy(count)} fill="#7c3aed" fillOpacity={0.75}>
            <title>{`${formatNumber(xs[0] + k * width)} – ${formatNumber(xs[0] + (k + 1) * width)}: ${count}`}</title>
          </rect>
        ))}
        <line x1={sx(mean)} x2={sx(mean)} y1={PAD.top} y2={H - PAD.bottom} stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 3" />
      </svg>
      <p data-testid="study-moments" className="font-mono text-xs text-slate-600 dark:text-slate-300">
        {t("study.moments", { n: values.length, mean: formatNumber(mean), std: formatNumber(std), unit })}
      </p>
    </div>
  );
}

export function StudyPanel() {
  const { t } = useTranslation();
  const studyJob = useStore((s) => s.studyJob);
  const job = useStore((s) => (s.studyJob ? s.jobs[s.studyJob] : undefined));
  const study = useStore((s) => s.study);
  const context = useStore((s) => s.context);
  const defaults = (context?.ui.study ?? {}) as StudyDefaults;
  const [output, setOutput] = useState<string | null>(null);
  const [metric, setMetric] = useState<string | null>(null);
  const [x, setX] = useState<string | null>(null);

  if (!studyJob) {
    return (
      <div data-testid="study" className="flex h-full flex-col items-center justify-center gap-3 p-4 text-sm text-slate-500">
        <p>{t("study.empty")}</p>
        {context && (
          <Button onClick={() => useStore.getState().setDialog({ kind: "study" })} testId="study-open-empty">
            <FlaskConical className="size-4" />
            {t("study.open")}
          </Button>
        )}
      </div>
    );
  }
  if (!study) {
    const failed = job?.status === "failed" || job?.status === "cancelled";
    return (
      <div data-testid="study" className="flex h-full items-center justify-center gap-3 p-4 text-sm text-slate-500">
        {failed ? (
          <p className="text-red-600">{job?.error ?? job?.status}</p>
        ) : (
          <>
            <Loader2 className="size-4 animate-spin text-accent-500" />
            <span>{job?.message || t("run.queued")}</span>
            <span className="font-mono">{Math.round((job?.progress ?? 0) * 100)} %</span>
            <Button onClick={() => void api.cancel(studyJob).then((info) => useStore.getState().onJob(info))}>{t("run.cancel")}</Button>
          </>
        )}
      </div>
    );
  }
  const chosen = output && study.outputs.includes(output) ? output : study.outputs[0];
  const isScalar = scalar(study, chosen);
  const metrics = isScalar ? ["value"] : [...STATISTICS];
  const shownMetric = metric && metrics.includes(metric) ? metric : isScalar ? "value" : defaults.metric && metrics.includes(defaults.metric) ? defaults.metric : "max";
  const axis = x && study.parameters.includes(x) ? x : study.parameters[0];
  const realizations = study.points.some((p) => p.realization);
  const select =
    "max-w-44 rounded-md border border-slate-200 bg-white px-1.5 py-0.5 text-xs outline-none focus:border-accent-400 dark:border-slate-700 dark:bg-slate-800";

  return (
    <div data-testid="study" className="flex h-full gap-4 overflow-auto p-3">
      <div className="flex w-[520px] shrink-0 flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="font-semibold tracking-wider text-accent-700 uppercase dark:text-accent-300">{t(`study.${study.kind}`)}</span>
          <span className="text-slate-500">{t("study.summary", { count: study.points.length, seed: study.seed })}</span>
          <select data-testid="study-output" className={select} value={chosen} onChange={(e) => setOutput(e.target.value)} aria-label={t("study.output")}>
            {study.outputs.map((port) => (
              <option key={port}>{port}</option>
            ))}
          </select>
          <select data-testid="study-metric" className={select} value={shownMetric} onChange={(e) => setMetric(e.target.value)} aria-label={t("study.metric")}>
            {metrics.map((m) => (
              <option key={m} value={m}>
                {t(`study.metrics.${m}`)}
              </option>
            ))}
          </select>
          {study.kind === "sweep" && study.parameters.length > 1 && (
            <select data-testid="study-x" className={select} value={axis} onChange={(e) => setX(e.target.value)} aria-label={t("study.axis")}>
              {study.parameters.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          )}
          <a
            href={api.studyCsvUrl(study.job)}
            download
            data-testid="study-csv"
            className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-1.5 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            <Download className="size-3" />
            CSV
          </a>
        </div>
        {study.kind === "sweep" ? <SweepChart study={study} output={chosen} metric={shownMetric} x={axis} /> : <Histogram study={study} output={chosen} metric={shownMetric} />}
      </div>
      <div className="min-w-0 flex-1 overflow-auto rounded-lg border border-slate-200 dark:border-slate-700">
        <table data-testid="study-table" className="w-full text-left font-mono text-xs whitespace-nowrap">
          <thead className="sticky top-0 bg-slate-50 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
            <tr>
              <th className="px-2 py-1">#</th>
              {realizations && <th className="px-2 py-1">{t("study.realization")}</th>}
              {study.parameters.map((p) => (
                <th key={p} className="px-2 py-1">
                  {p}
                </th>
              ))}
              {study.outputs.map((port) => (
                <th key={port} className="px-2 py-1">
                  {port} {scalar(study, port) ? "" : shownMetric}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {study.points.map((point, index) => (
              <tr key={index} className="border-t border-slate-100 dark:border-slate-800">
                <td className="px-2 py-0.5 text-slate-400">{index + 1}</td>
                {realizations && <td className="px-2 py-0.5 text-slate-400">{point.realization}</td>}
                {study.parameters.map((p) => (
                  <td key={p} className="px-2 py-0.5">
                    {point.params[p]}
                  </td>
                ))}
                {study.outputs.map((port) => {
                  const value = measure(study, index, port, scalar(study, port) ? "value" : shownMetric);
                  const unit = point.outputs[port]?.unit;
                  return (
                    <td key={port} className="px-2 py-0.5">
                      {value === null ? "—" : `${formatNumber(value)}${unit ? ` ${unit}` : ""}`}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
