// Inputs of the forms, one per kind of field of a schema (REQ-WEB-3): quantities with a unit selector and
// conversion, bounds checked before sending, numbers, booleans, choices, arrays and JSON settings.

import { Info, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { checkBounds, type Field } from "../lib/schema";
import { convert, formatNumber, parseQuantity, type UnitChoices } from "../lib/units";
import { useStore } from "../store";

export const inputClass =
  "w-full rounded-md border border-slate-200 bg-white px-2 py-1 text-sm outline-none focus:border-accent-400 focus:ring-2 focus:ring-accent-200 disabled:bg-slate-50 disabled:text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:focus:ring-accent-900 dark:disabled:bg-slate-900";

export function FieldRow({ field, children, error }: { field: Field; children: ReactNode; error?: ReactNode }) {
  const { t } = useTranslation();
  return (
    <div data-testid={`field-${field.name}`} className="grid grid-cols-[6.5rem_1fr] items-start gap-2 py-1">
      <label className="flex min-w-0 items-center gap-1 pt-1 text-xs text-slate-600 dark:text-slate-300" title={field.doc || field.name}>
        <span className="truncate font-mono">{field.name}</span>
        {field.doc && <Info className="size-3 shrink-0 text-slate-400" />}
        {field.free && (
          <span className="rounded bg-amber-100 px-1 text-[10px] text-amber-700 dark:bg-amber-900/40 dark:text-amber-300" title={t("fields.free")}>
            ∼
          </span>
        )}
      </label>
      <div className="min-w-0">
        {children}
        {error}
      </div>
    </div>
  );
}

function sameValue(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function defaultText(field: Field): string | null {
  const value = field.default as { value?: unknown; unit?: string } | undefined;
  if (value === undefined || value === null) return null;
  if (typeof value === "object" && "value" in value) return `${value.value} ${value.unit ?? ""}`.trim();
  return JSON.stringify(value);
}

function Reset({ field, current, onReset }: { field: Field; current: unknown; onReset: () => void }) {
  const { t } = useTranslation();
  const text = defaultText(field);
  if (text === null) return null;
  const parsedDefault = parseQuantity(text);
  const parsedCurrent = typeof current === "string" ? parseQuantity(current) : null;
  const same =
    parsedDefault && parsedCurrent
      ? parsedDefault.unit === parsedCurrent.unit && Math.abs(parsedDefault.value - parsedCurrent.value) < 1e-12
      : sameValue(field.default, current);
  if (same) return null;
  return (
    <button type="button" onClick={onReset} className="shrink-0 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-700" title={t("fields.reset", { value: text })}>
      <RotateCcw className="size-3.5" />
    </button>
  );
}

/** A quantity: number and unit; the unit selector converts the shown value without changing the quantity. */
export function QuantityInput({
  field,
  value,
  choices,
  disabled,
  onCommit,
}: {
  field: Field;
  value: unknown;
  choices: UnitChoices;
  disabled?: boolean;
  onCommit: (text: string) => Promise<boolean> | void;
}) {
  const { t } = useTranslation();
  const current = typeof value === "string" ? parseQuantity(value) : null;
  const declared = field.unit ?? current?.unit ?? "";
  const units = useMemo(() => {
    const list = choices.length ? choices : ([[declared, 1]] as UnitChoices);
    return current && !list.some(([u]) => u === current.unit) ? [...list, [current.unit, NaN] as [string, number]] : list;
  }, [choices, declared, current]);
  const [unit, setUnit] = useState(current?.unit ?? declared);
  const shown = (quantity: { value: number; unit: string } | null, target: string) => {
    if (!quantity) return typeof value === "string" ? value : "";
    try {
      return formatNumber(convert(quantity.value, quantity.unit, target, units));
    } catch {
      return formatNumber(quantity.value);
    }
  };
  const [text, setText] = useState(shown(current, unit));
  const [problem, setProblem] = useState<string | null>(null);
  const focused = useRef(false);

  useEffect(() => {
    if (!focused.current) setText(shown(current, unit));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, unit]);

  const commit = async () => {
    const parsed = parseQuantity(text);
    if (!parsed) {
      setProblem(t("fields.notANumber"));
      return;
    }
    let chosen = unit;
    if (parsed.unit && parsed.unit !== unit) {
      if (!units.some(([u]) => u === parsed.unit)) {
        setProblem(t("fields.unknownUnit", { unit: parsed.unit }));
        return;
      }
      chosen = parsed.unit;
      setUnit(parsed.unit);
    }
    let inDeclared = parsed.value;
    try {
      inDeclared = convert(parsed.value, chosen, declared, units);
    } catch {
      /* unit without factor: the server checks */
    }
    const outside = checkBounds(field, inDeclared);
    if (outside) {
      const [low, high] = field.bounds ?? [null, null];
      const [lowIn, highIn] = field.inclusive ?? [true, true];
      setProblem(
        outside === "below"
          ? t(lowIn ? "fields.atLeast" : "fields.above", { bound: `${formatNumber(low!)} ${declared}` })
          : t(highIn ? "fields.atMost" : "fields.below", { bound: `${formatNumber(high!)} ${declared}` }),
      );
      return;
    }
    setProblem(null);
    const quantity = chosen ? `${formatNumber(parsed.value)} ${chosen}` : formatNumber(parsed.value);
    if (current && current.unit === chosen && Math.abs(current.value - parsed.value) <= 1e-12 * Math.abs(current.value)) return;
    const ok = await onCommit(quantity);
    if (ok === false) setText(shown(current, unit));
  };

  return (
    <>
      <div className="flex gap-1">
        <input
          data-testid={`value-${field.name}`}
          className={`${inputClass} font-mono ${problem ? "border-red-400 focus:border-red-400 focus:ring-red-200" : ""}`}
          value={text}
          disabled={disabled}
          inputMode="decimal"
          onFocus={() => (focused.current = true)}
          onBlur={() => {
            focused.current = false;
            void commit();
          }}
          onChange={(event) => {
            setText(event.target.value);
            setProblem(null);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") void commit();
            if (event.key === "Escape") {
              setText(shown(current, unit));
              setProblem(null);
              (event.target as HTMLInputElement).blur();
            }
          }}
        />
        {units.length > 1 || declared ? (
          <select
            data-testid={`unit-${field.name}`}
            className={`${inputClass.replace("w-full ", "")} w-[4.75rem] shrink-0 px-1`}
            value={unit}
            disabled={disabled || units.length < 2}
            onChange={(event) => {
              const next = event.target.value;
              const typed = parseQuantity(text);
              if (typed && !typed.unit) {
                try {
                  setText(formatNumber(convert(typed.value, unit, next, units)));
                } catch {
                  /* unit without factor: keep the number */
                }
              }
              setUnit(next);
            }}
            title={t("fields.unit")}
          >
            {units.map(([name]) => (
              <option key={name} value={name}>
                {name ? name.replace(/(^|[^A-Za-z])u(?=(m|W|J|V|A|s|as|rad|Jy|K))/g, "$1µ") : "—"}
              </option>
            ))}
          </select>
        ) : null}
        {!disabled && <Reset field={field} current={value} onReset={() => void onCommit(defaultText(field)!)} />}
      </div>
      {problem && (
        <p data-testid={`bounds-${field.name}`} role="alert" className="mt-0.5 text-xs text-red-600 dark:text-red-400">
          {problem}
        </p>
      )}
    </>
  );
}

function TextInput({
  field,
  value,
  disabled,
  onCommit,
  numeric,
}: {
  field: Field;
  value: unknown;
  disabled?: boolean;
  onCommit: (value: unknown) => Promise<boolean> | void;
  numeric?: "integer" | "number";
}) {
  const { t } = useTranslation();
  const initial = value === null || value === undefined ? "" : String(value);
  const [text, setText] = useState(initial);
  const [problem, setProblem] = useState<string | null>(null);
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setText(initial);
  }, [initial]);
  const commit = () => {
    if (text === initial) return;
    let parsed: unknown = text;
    if (numeric) {
      const number = Number(text);
      if (!Number.isFinite(number) || (numeric === "integer" && !Number.isInteger(number))) {
        setProblem(t(numeric === "integer" ? "fields.notAnInteger" : "fields.notANumber"));
        return;
      }
      parsed = number;
    }
    setProblem(null);
    void onCommit(parsed);
  };
  return (
    <>
      <div className="flex gap-1">
        <input
          data-testid={`value-${field.name}`}
          className={`${inputClass} font-mono`}
          value={text}
          disabled={disabled}
          inputMode={numeric ? "decimal" : undefined}
          onFocus={() => (focused.current = true)}
          onBlur={() => {
            focused.current = false;
            commit();
          }}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && commit()}
        />
        {!disabled && <Reset field={field} current={value} onReset={() => void onCommit(field.default)} />}
      </div>
      {problem && <p className="mt-0.5 text-xs text-red-600 dark:text-red-400">{problem}</p>}
    </>
  );
}

function JsonInput({ field, value, disabled, onCommit }: { field: Field; value: unknown; disabled?: boolean; onCommit: (value: unknown) => void }) {
  const { t } = useTranslation();
  const initial = value === null || value === undefined ? "null" : JSON.stringify(value, null, 1);
  const [text, setText] = useState(initial);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => setText(initial), [initial]);
  return (
    <>
      <textarea
        data-testid={`value-${field.name}`}
        className={`${inputClass} min-h-[2.2rem] font-mono text-xs`}
        rows={Math.min(6, initial.split("\n").length)}
        value={text}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onBlur={() => {
          if (text === initial) return;
          try {
            onCommit(JSON.parse(text));
            setProblem(null);
          } catch {
            setProblem(t("fields.invalidJson"));
          }
        }}
      />
      {problem && <p className="mt-0.5 text-xs text-red-600 dark:text-red-400">{problem}</p>}
    </>
  );
}

/** The input of ``field`` of the element ``element`` (path), editing it through the API. */
export function FieldInput({ element, field, value, disabled }: { element: string; field: Field; value: unknown; disabled?: boolean }) {
  const units = useStore((s) => s.palette?.units);
  const setParam = useStore((s) => s.setParam);
  const path = `${element}.${field.name}`;
  const commit = (next: unknown) => setParam(path, next);
  if (field.kind === "state") {
    return <div className="truncate pt-1 font-mono text-xs text-slate-500 dark:text-slate-400">{String(value ?? "—")}</div>;
  }
  switch (field.input) {
    case "quantity":
      if (typeof value === "string" && !parseQuantity(value)) {
        return <TextInput field={field} value={value} disabled={disabled} onCommit={commit} />;
      }
      return (
        <QuantityInput field={field} value={value} choices={units?.[field.unit ?? ""] ?? []} disabled={disabled} onCommit={commit} />
      );
    case "boolean":
      return (
        <label className="inline-flex cursor-pointer items-center gap-2 pt-1">
          <input
            data-testid={`value-${field.name}`}
            type="checkbox"
            className="peer sr-only"
            checked={Boolean(value)}
            disabled={disabled}
            onChange={(event) => void commit(event.target.checked)}
          />
          <span className="relative h-5 w-9 rounded-full bg-slate-300 transition-colors peer-checked:bg-accent-500 peer-disabled:opacity-50 after:absolute after:top-0.5 after:left-0.5 after:size-4 after:rounded-full after:bg-white after:transition-transform peer-checked:after:translate-x-4 dark:bg-slate-600" />
        </label>
      );
    case "enum":
      return (
        <select data-testid={`value-${field.name}`} className={inputClass} value={String(value)} disabled={disabled} onChange={(event) => void commit(event.target.value)}>
          {(field.choices ?? []).map((choice) => (
            <option key={String(choice)} value={String(choice)}>
              {String(choice)}
            </option>
          ))}
        </select>
      );
    case "integer":
    case "number":
      return <TextInput field={field} value={value} disabled={disabled} onCommit={commit} numeric={field.input} />;
    case "text":
      return <TextInput field={field} value={value} disabled={disabled} onCommit={commit} />;
    default:
      return <JsonInput field={field} value={value} disabled={disabled} onCommit={(next) => void commit(next)} />;
  }
}
