// Editor of the transfer matrix of a TransferMatrix block (simplified MMI, step P14.19): one cell per output i
// (rows) and input j (columns), holding the modulus and the phase of S_ij; inputs and outputs are added and
// removed; the matrix is said lossless (unitary), lossy (passive) or amplifying (refused by the server).

import { Minus, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { ElementSummary } from "../api";
import { formatNumber } from "../lib/units";
import { parseArray } from "../lib/targets";
import { useStore } from "../store";

/** Same gain tolerance as the server: values typed with 5 or 6 significant digits. */
const TOLERANCE = 1e-4;

type Matrix = number[][];

function matrixOf(text: unknown): Matrix | null {
  const array = parseArray(text);
  if (!array || array.shape.length !== 2) return null;
  const [rows, columns] = array.shape;
  return Array.from({ length: rows }, (_, i) => array.values.slice(i * columns, (i + 1) * columns));
}

/** Largest singular value of S = A e^{iφ} (power iteration on S^H S) and whether S^H S is the identity. */
export function passivity(amplitudes: Matrix, phases: Matrix): { largest: number; unitary: boolean } {
  const rows = amplitudes.length;
  const columns = amplitudes[0]?.length ?? 0;
  const re = amplitudes.map((row, i) => row.map((a, j) => a * Math.cos(phases[i][j])));
  const im = amplitudes.map((row, i) => row.map((a, j) => a * Math.sin(phases[i][j])));
  // G = S^H S, Hermitian of size columns × columns
  const gRe: Matrix = Array.from({ length: columns }, () => Array(columns).fill(0));
  const gIm: Matrix = Array.from({ length: columns }, () => Array(columns).fill(0));
  for (let p = 0; p < columns; p += 1)
    for (let q = 0; q < columns; q += 1)
      for (let i = 0; i < rows; i += 1) {
        gRe[p][q] += re[i][p] * re[i][q] + im[i][p] * im[i][q];
        gIm[p][q] += re[i][p] * im[i][q] - im[i][p] * re[i][q];
      }
  let vRe = Array.from({ length: columns }, (_, k) => 1 + k * 0.1);
  let vIm = Array(columns).fill(0) as number[];
  let value = 0;
  for (let step = 0; step < 200; step += 1) {
    const wRe = gRe.map((row, p) => row.reduce((sum, g, q) => sum + g * vRe[q] - gIm[p][q] * vIm[q], 0));
    const wIm = gRe.map((row, p) => row.reduce((sum, g, q) => sum + g * vIm[q] + gIm[p][q] * vRe[q], 0));
    const norm = Math.sqrt(wRe.reduce((s, x) => s + x * x, 0) + wIm.reduce((s, x) => s + x * x, 0));
    if (!(norm > 0)) break;
    value = norm / Math.sqrt(vRe.reduce((s, x) => s + x * x, 0) + vIm.reduce((s, x) => s + x * x, 0));
    vRe = wRe.map((x) => x / norm);
    vIm = wIm.map((x) => x / norm);
  }
  const unitary =
    rows === columns && gRe.every((row, p) => row.every((g, q) => Math.abs(g - (p === q ? 1 : 0)) < TOLERANCE && Math.abs(gIm[p][q]) < TOLERANCE));
  return { largest: Math.sqrt(value), unitary };
}

function Cell({ value, testId, label, onCommit, disabled }: { value: number; testId: string; label: string; onCommit: (value: number) => void; disabled: boolean }) {
  const [text, setText] = useState(formatNumber(value));
  useEffect(() => {
    setText(formatNumber(value));
  }, [value]);
  const commit = () => {
    const number = Number(text);
    if (Number.isFinite(number) && number !== value) onCommit(number);
    else setText(formatNumber(value));
  };
  return (
    <input
      data-testid={testId}
      aria-label={label}
      value={text}
      disabled={disabled}
      onChange={(event) => setText(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => event.key === "Enter" && commit()}
      className="w-full min-w-0 rounded border border-slate-200 bg-white px-1 py-0.5 text-right font-mono text-[11px] outline-none focus:border-accent-400 dark:border-slate-700 dark:bg-slate-800"
    />
  );
}

export function MatrixEditor({ element, editable }: { element: ElementSummary; editable: boolean }) {
  const { t } = useTranslation();
  const setParam = useStore((s) => s.setParam);
  const amplitudes = matrixOf(element.params.amplitudes);
  const phases = matrixOf(element.params.phases);
  if (!amplitudes || !phases) return null;
  const rows = amplitudes.length;
  const columns = amplitudes[0].length;
  const same = phases.length === rows && phases.every((row) => row.length === columns);
  const status = same ? passivity(amplitudes, phases) : null;
  const write = async (a: Matrix, p: Matrix) => {
    if (JSON.stringify(a) !== JSON.stringify(amplitudes)) await setParam(`${element.id}.amplitudes`, JSON.stringify(a));
    if (JSON.stringify(p) !== JSON.stringify(phases)) await setParam(`${element.id}.phases`, `${JSON.stringify(p)} rad`);
  };
  const change = (which: "a" | "p", i: number, j: number, value: number) => {
    const next = (which === "a" ? amplitudes : phases).map((row) => [...row]);
    next[i][j] = value;
    void (which === "a" ? write(next, phases) : write(amplitudes, next));
  };
  const resize = (dRows: number, dColumns: number) => {
    const shape = (m: Matrix) => {
      let out = m.map((row) => (dColumns > 0 ? [...row, 0] : dColumns < 0 ? row.slice(0, -1) : [...row]));
      if (dRows > 0) out = [...out, Array(out[0].length).fill(0)];
      if (dRows < 0) out = out.slice(0, -1);
      return out;
    };
    void write(shape(amplitudes), shape(phases));
  };
  const button = (testId: string, label: string, icon: typeof Plus, onClick: () => void, disabled = false) => {
    const Icon = icon;
    return (
      <button
        type="button"
        data-testid={testId}
        title={label}
        aria-label={label}
        disabled={!editable || disabled}
        onClick={onClick}
        className="inline-flex items-center gap-0.5 rounded border border-slate-200 px-1 py-0.5 text-[11px] text-slate-600 hover:bg-slate-50 disabled:opacity-40 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
      >
        <Icon className="size-3" />
        {label}
      </button>
    );
  };
  const kind = !status ? "shapeMismatch" : status.largest > 1 + TOLERANCE ? "amplifies" : status.unitary ? "lossless" : "lossy";
  return (
    <section data-testid="matrix-editor" className="space-y-2 border-b border-slate-100 px-4 py-2 text-xs dark:border-slate-800">
      <div className="flex items-center gap-2">
        <span className="font-semibold tracking-wider text-slate-500 uppercase dark:text-slate-400">{t("matrix.title")}</span>
        <span data-testid="matrix-shape" className="ml-auto font-mono text-slate-500">
          {t("matrix.shape", { rows, columns })}
        </span>
      </div>
      <p className="text-slate-500 dark:text-slate-400">{t("matrix.hint")}</p>
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-1">
          <thead>
            <tr>
              <th />
              {Array.from({ length: columns }, (_, j) => (
                <th key={j} className="font-mono text-[10px] font-normal text-slate-400">
                  in {j}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {amplitudes.map((row, i) => (
              <tr key={i}>
                <th className="pr-1 text-right font-mono text-[10px] font-normal whitespace-nowrap text-slate-400">out {i}</th>
                {row.map((a, j) => (
                  <td key={j} className="min-w-[4.5rem] rounded-md bg-slate-50 p-1 align-top dark:bg-slate-800/60">
                    <Cell value={a} testId={`matrix-a-${i}-${j}`} label={t("matrix.amplitude", { i, j })} disabled={!editable} onCommit={(v) => change("a", i, j, v)} />
                    <div className="mt-0.5 flex items-center gap-0.5">
                      <span className="text-[10px] text-slate-400">∠</span>
                      <Cell value={phases[i]?.[j] ?? 0} testId={`matrix-p-${i}-${j}`} label={t("matrix.phase", { i, j })} disabled={!editable || !same} onCommit={(v) => change("p", i, j, v)} />
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-slate-400">{t("matrix.units")}</p>
      <div className="flex flex-wrap gap-1">
        {button("matrix-add-input", t("matrix.addInput"), Plus, () => resize(0, 1), !same)}
        {button("matrix-remove-input", t("matrix.removeInput"), Minus, () => resize(0, -1), !same || columns <= 1)}
        {button("matrix-add-output", t("matrix.addOutput"), Plus, () => resize(1, 0), !same)}
        {button("matrix-remove-output", t("matrix.removeOutput"), Minus, () => resize(-1, 0), !same || rows <= 1)}
      </div>
      <p
        data-testid="matrix-status"
        className={`rounded-md px-2 py-1 ${kind === "lossless" ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300" : kind === "lossy" ? "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300" : "bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300"}`}
      >
        {t(`matrix.${kind}`, { value: status ? formatNumber(Number(status.largest.toPrecision(4))) : "" })}
      </p>
    </section>
  );
}
