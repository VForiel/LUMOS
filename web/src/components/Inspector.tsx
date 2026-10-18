// Right panel: the selected element (forms generated from its schema), the selection, or the context itself.

import {
  ChevronDown,
  ChevronRight,
  Copy,
  Dices,
  Download,
  FolderOpen,
  FolderTree,
  Layers,
  PackagePlus,
  PanelRightClose,
  PanelRightOpen,
  ScanSearch,
  Trash2,
  Ungroup,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { api, download, type ContextSummary, type ElementSummary, type ElementType } from "../api";
import { category, categoryOf } from "../lib/categories";
import { signalOfPort } from "../lib/ports";
import { fieldsOf, type Field } from "../lib/schema";
import { signalColor } from "../lib/signals";
import { formatNumber, parseQuantity } from "../lib/units";
import { useStore } from "../store";
import { Resizer } from "./Resizer";
import { FieldInput, FieldRow, inputClass, QuantityInput } from "./Fields";
import { plain } from "./Palette";
import { priorText } from "./Study";

function Section({ title, children, defaultOpen = true, extra }: { title: string; children: ReactNode; defaultOpen?: boolean; extra?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-b border-slate-100 px-4 py-2 dark:border-slate-800">
      <div className="flex items-center">
        <button type="button" onClick={() => setOpen(!open)} className="flex flex-1 items-center gap-1 py-1 text-left text-xs font-semibold tracking-wider text-slate-500 uppercase dark:text-slate-400">
          {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
          {title}
        </button>
        {extra}
      </div>
      {open && <div className="pb-1">{children}</div>}
    </section>
  );
}

function ActionButton({ icon, label, onClick, testId, danger, disabled }: { icon: ReactNode; label: string; onClick: () => void; testId?: string; danger?: boolean; disabled?: boolean }) {
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors disabled:opacity-40 ${
        danger
          ? "border-red-200 text-red-600 hover:bg-red-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950"
          : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function NameInput({ path, editable }: { path: string; editable: boolean }) {
  const rename = useStore((s) => s.rename);
  const name = path.split(".").pop()!;
  const [text, setText] = useState(name);
  useEffect(() => {
    setText(name);
  }, [name]);
  const commit = () => {
    if (text && text !== name) void rename(path, text);
    else setText(name);
  };
  return (
    <input
      data-testid="element-name"
      aria-label="name"
      value={text}
      disabled={!editable}
      onChange={(event) => setText(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => event.key === "Enter" && commit()}
      className="w-full rounded-md border border-transparent bg-transparent px-1 text-base font-semibold text-slate-800 outline-none hover:border-slate-200 focus:border-accent-400 dark:text-slate-100 dark:hover:border-slate-700"
    />
  );
}

type PriorKind = "uniform" | "loguniform" | "normal";
const PRIOR_VALUES: Record<PriorKind, [string, string]> = { uniform: ["low", "high"], loguniform: ["low", "high"], normal: ["mean", "std"] };

/** A number typed in the unit of the parameter, or a quantity with its own unit ("2 mm"). */
function inUnit(text: string, unit: string): string {
  const quantity = parseQuantity(text.trim());
  return quantity && quantity.unit ? text.trim() : `${text.trim()} ${unit}`.trim();
}

/** Free a parameter with a prior (Monte-Carlo runs draw it), or fix it again. */
function PriorEditor({ element, field, onClose }: { element: ElementSummary; field: Field; onClose: () => void }) {
  const { t } = useTranslation();
  const setFree = useStore((s) => s.setFree);
  const current = parseQuantity(String(element.params[field.name] ?? "")) ?? { value: 0, unit: field.unit ?? "" };
  const prior = element.free?.[field.name] ?? null;
  const strip = (text: string | undefined) => (text === undefined ? "" : formatNumber(parseQuantity(text)?.value ?? Number(text)));
  const [kind, setKind] = useState<PriorKind>((prior?.kind as PriorKind) ?? "normal");
  const initial = (choice: PriorKind): [string, string] => {
    if (prior && prior.kind === choice) return PRIOR_VALUES[choice].map((key) => strip(prior[key])) as [string, string];
    const v = current.value;
    if (choice === "normal") return [formatNumber(v), formatNumber(Math.abs(v) * 0.01 || 0.1)];
    return v === 0 ? ["0", "1"] : [formatNumber(Math.min(v * 0.9, v * 1.1)), formatNumber(Math.max(v * 0.9, v * 1.1))];
  };
  const [[a, b], setValues] = useState<[string, string]>(initial(kind));
  const [first, second] = PRIOR_VALUES[kind];
  const name = field.name;
  const apply = async () => {
    if (await setFree(`${element.id}.${name}`, true, { kind, [first]: inUnit(a, current.unit), [second]: inUnit(b, current.unit) })) onClose();
  };
  return (
    <div data-testid={`prior-${name}`} className="mt-1 space-y-1.5 rounded-lg border border-amber-200 bg-amber-50/60 p-2 text-xs dark:border-amber-900 dark:bg-amber-950/30">
      <p className="text-slate-600 dark:text-slate-300">{t("prior.hint")}</p>
      <select
        data-testid={`prior-kind-${name}`}
        value={kind}
        onChange={(event) => {
          const next = event.target.value as PriorKind;
          setKind(next);
          setValues(initial(next));
        }}
        className={`${inputClass} py-0.5 text-xs`}
      >
        {(Object.keys(PRIOR_VALUES) as PriorKind[]).map((choice) => (
          <option key={choice} value={choice}>
            {t(`prior.${choice}`)}
          </option>
        ))}
      </select>
      <div className="grid grid-cols-2 gap-1.5">
        {[
          [first, a, (value: string) => setValues([value, b])],
          [second, b, (value: string) => setValues([a, value])],
        ].map(([label, value, change], index) => (
          <label key={label as string} className="block">
            <span className="text-slate-500">
              {t(`prior.${label as string}`)}
              {current.unit ? ` [${current.unit}]` : ""}
            </span>
            <input
              data-testid={`prior-${index ? "b" : "a"}-${name}`}
              value={value as string}
              onChange={(event) => (change as (value: string) => void)(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && void apply()}
              className={`${inputClass} py-0.5 text-xs`}
            />
          </label>
        ))}
      </div>
      <div className="flex justify-end gap-1.5">
        {prior !== null || name in (element.free ?? {}) ? (
          <ActionButton icon={null} label={t("prior.fix")} testId={`prior-remove-${name}`} onClick={() => void setFree(`${element.id}.${name}`, false).then(onClose)} />
        ) : (
          <ActionButton icon={null} label={t("common.cancel")} onClick={onClose} />
        )}
        <ActionButton icon={<Dices className="size-3.5" />} label={t("common.apply")} testId={`prior-apply-${name}`} onClick={() => void apply()} />
      </div>
    </div>
  );
}

/** The row of a parameter: its input, the button freeing it, its prior. */
function ParamRow({ element, field, editable }: { element: ElementSummary; field: Field; editable: boolean }) {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const free = field.name in (element.free ?? {});
  const canFree = field.kind === "param" && field.input === "quantity";
  return (
    <FieldRow field={{ ...field, free }}>
      <div className="flex items-start gap-1">
        <div className="min-w-0 flex-1">
          <FieldInput element={element.id} field={field} value={valueOf(element, field)} disabled={!editable} />
        </div>
        {canFree && editable && (
          <button
            type="button"
            data-testid={`free-${field.name}`}
            aria-pressed={free}
            title={t(free ? "prior.edit" : "prior.free")}
            aria-label={t(free ? "prior.edit" : "prior.free")}
            onClick={() => setEditing(!editing)}
            className={`mt-0.5 grid size-7 shrink-0 place-items-center rounded-md transition-colors ${
              free ? "bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-200" : "text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800"
            }`}
          >
            <Dices className="size-3.5" />
          </button>
        )}
      </div>
      {free && !editing && (
        <p data-testid={`prior-summary-${field.name}`} className="mt-0.5 font-mono text-[11px] text-amber-700 dark:text-amber-300">
          ∼ {priorText(element.free?.[field.name])}
        </p>
      )}
      {editing && <PriorEditor element={element} field={field} onClose={() => setEditing(false)} />}
    </FieldRow>
  );
}

function valueOf(element: ElementSummary, field: Field): unknown {
  const group = field.kind === "param" ? element.params : field.kind === "setting" ? element.settings : element.state;
  return group[field.name];
}

function ElementPanel({ element, type, onInspect }: { element: ElementSummary; type?: ElementType; onInspect: (port: string) => void }) {
  const { t } = useTranslation();
  const context = useStore((s) => s.context)!;
  const role = useStore((s) => s.info?.role);
  const issues = useStore((s) => s.issues).filter((issue) => issue.element === element.id);
  const { ungroup, toggleGroup, saveBlock, copy, removeElements } = useStore.getState();
  const editable = role !== "viewer";
  const isGroup = element.type_id.startsWith("lumos.group") || element.id in (context.groups ?? {});
  const fields = useMemo(() => (type ? fieldsOf(type.schema) : []), [type]);
  const { icon: Icon, color } = category(isGroup ? "group" : categoryOf(element.type_id));
  const groups: [string, Field[]][] = [
    [t("inspector.params"), fields.filter((f) => f.kind === "param")],
    [t("inspector.settings"), fields.filter((f) => f.kind === "setting")],
    [t("inspector.state"), fields.filter((f) => f.kind === "state")],
  ];
  const palette = useStore((s) => s.palette);
  const types = useMemo(() => new Map((palette?.types ?? []).map((item) => [item.type_id, item])), [palette]);
  const signal = (name: string, direction: "inputs" | "outputs") => signalOfPort(context, types, element.id, name, direction);
  const expanded = Boolean(context.ui.expanded?.[element.id]);
  const topLevel = !element.id.includes(".");

  return (
    <>
      <div className="flex items-start gap-3 border-b border-slate-100 px-4 py-3 dark:border-slate-800">
        <span className="mt-1 grid size-9 shrink-0 place-items-center rounded-lg" style={{ background: `${color}22`, color }}>
          {isGroup ? <Layers className="size-5" /> : <Icon className="size-5" />}
        </span>
        <div className="min-w-0 flex-1">
          <NameInput path={element.id} editable={editable} />
          <p className="truncate px-1 text-xs text-slate-500 dark:text-slate-400">{isGroup ? t("inspector.group") : (type?.title ?? element.type_id)}</p>
        </div>
      </div>
      {type?.schema?.description && (
        <p className="border-b border-slate-100 px-4 py-2 text-xs leading-relaxed text-slate-600 dark:border-slate-800 dark:text-slate-400">
          {plain(type.schema.description)}
        </p>
      )}
      {issues.length > 0 && (
        <div className="border-b border-red-100 bg-red-50 px-4 py-2 text-xs text-red-700 dark:border-red-950 dark:bg-red-950/40 dark:text-red-300">
          {issues.map((issue) => (
            <p key={issue.message}>
              <strong>{issue.message}</strong> — {issue.hint}
            </p>
          ))}
        </div>
      )}
      {(isGroup || topLevel) && (
        <div className="flex flex-wrap gap-1.5 border-b border-slate-100 px-4 py-2 dark:border-slate-800">
          {isGroup && topLevel && (
            <ActionButton
              icon={expanded ? <FolderTree className="size-3.5" /> : <FolderOpen className="size-3.5" />}
              label={expanded ? t("inspector.collapse") : t("inspector.expand")}
              onClick={() => void toggleGroup(element.id)}
              testId="toggle-group"
            />
          )}
          {isGroup && topLevel && editable && (
            <ActionButton icon={<Ungroup className="size-3.5" />} label={t("inspector.ungroup")} onClick={() => void ungroup(element.id)} testId="ungroup" />
          )}
          {isGroup && (
            <ActionButton icon={<PackagePlus className="size-3.5" />} label={t("inspector.saveBlock")} onClick={() => void saveBlock(element.id)} testId="save-block" />
          )}
          {isGroup && (
            <ActionButton
              icon={<Download className="size-3.5" />}
              label={t("inspector.downloadBlock")}
              onClick={() => void api.exportElement(context.id, element.id).then((text) => download(`${element.id.split(".").pop()}.lumos`, text))}
            />
          )}
          {topLevel && <ActionButton icon={<Copy className="size-3.5" />} label={t("inspector.copy")} onClick={() => void copy([element.id])} />}
          {topLevel && editable && (
            <ActionButton icon={<Trash2 className="size-3.5" />} label={t("inspector.delete")} onClick={() => void removeElements([element.id])} danger testId="delete" />
          )}
        </div>
      )}
      {groups.map(([title, list]) =>
        list.length ? (
          <Section key={title} title={title} defaultOpen={title !== t("inspector.state")}>
            {list.map((field) =>
              field.kind === "param" ? (
                <ParamRow key={field.name} element={element} field={field} editable={editable} />
              ) : (
                <FieldRow key={field.name} field={field}>
                  <FieldInput element={element.id} field={field} value={valueOf(element, field)} disabled={!editable} />
                </FieldRow>
              ),
            )}
          </Section>
        ) : null,
      )}
      {(element.inputs.length > 0 || element.outputs.length > 0) && (
        <Section title={t("inspector.ports")}>
          <ul className="space-y-1 text-xs">
            {element.inputs.map((port) => (
              <li key={`in-${port}`} className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
                <span className="size-2.5 rounded-full" style={{ background: signalColor(signal(port, "inputs")) }} />
                <span className="font-mono">{port}</span>
                <span className="text-slate-400">← {signal(port, "inputs")}</span>
              </li>
            ))}
            {element.outputs.map((port) => (
              <li key={`out-${port}`} className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
                <span className="size-2.5 rounded-full" style={{ background: signalColor(signal(port, "outputs")) }} />
                <span className="font-mono">{port}</span>
                <span className="flex-1 text-slate-400">→ {signal(port, "outputs")}</span>
                <button
                  type="button"
                  data-testid={`inspect-${port}`}
                  onClick={() => onInspect(`${element.id}.${port}`)}
                  className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-accent-600 hover:bg-accent-50 dark:text-accent-300 dark:hover:bg-accent-900/30"
                >
                  <ScanSearch className="size-3.5" />
                  {t("inspector.inspect")}
                </button>
              </li>
            ))}
          </ul>
        </Section>
      )}
    </>
  );
}

const LENGTH: Field = { name: "band", kind: "param", input: "quantity", doc: "", unit: "um", bounds: [0, null], inclusive: [false, true] };

function ContextPanel({ context }: { context: ContextSummary }) {
  const { t } = useTranslation();
  const role = useStore((s) => s.info?.role);
  const units = useStore((s) => s.palette?.units);
  const { setContextInfo, setObservation } = useStore.getState();
  const editable = role !== "viewer";
  const [name, setName] = useState(context.name);
  const [description, setDescription] = useState(context.description);
  useEffect(() => {
    setName(context.name);
  }, [context.name]);
  useEffect(() => {
    setDescription(context.description);
  }, [context.description]);
  const band = context.band?.map((value) => parseQuantity(value)) ?? null;
  const toMicrons = (q: { value: number; unit: string } | null | undefined) => {
    if (!q) return null;
    const factor = new Map(units?.um ?? [["um", 1]]).get(q.unit);
    return factor ? `${formatNumber(q.value * factor)} um` : `${q.value} ${q.unit}`;
  };
  const [minimum, setMinimum] = useState(toMicrons(band?.[0]) ?? "1 um");
  const [maximum, setMaximum] = useState(toMicrons(band?.[1]) ?? "2 um");
  const [n, setN] = useState(context.n_wavelengths);
  useEffect(() => {
    if (band) {
      setMinimum(toMicrons(band[0]) ?? "1 um");
      setMaximum(toMicrons(band[1]) ?? "2 um");
    }
    setN(context.n_wavelengths);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [context.band?.join(), context.n_wavelengths]);
  const counts = context.elements.filter((e) => !e.id.includes(".")).length;

  return (
    <>
      <div className="border-b border-slate-100 px-4 py-3 dark:border-slate-800">
        <input
          data-testid="context-title"
          value={name}
          disabled={!editable}
          onChange={(event) => setName(event.target.value)}
          onBlur={() => name !== context.name && void setContextInfo({ name })}
          onKeyDown={(event) => event.key === "Enter" && (event.target as HTMLInputElement).blur()}
          className="w-full rounded-md border border-transparent bg-transparent px-1 text-base font-semibold outline-none hover:border-slate-200 focus:border-accent-400 dark:hover:border-slate-700"
        />
        <p className="px-1 text-xs text-slate-500 dark:text-slate-400">
          {t("inspector.summary", { count: counts, mode: t(`mode.${context.mode}`, { defaultValue: context.mode }) })}
        </p>
      </div>
      <Section title={t("inspector.description")}>
        <textarea
          data-testid="context-description"
          value={description}
          disabled={!editable}
          placeholder={t("inspector.describe")}
          rows={3}
          onChange={(event) => setDescription(event.target.value)}
          onBlur={() => description !== context.description && void setContextInfo({ description })}
          className={`${inputClass} text-xs`}
        />
      </Section>
      <Section title={t("inspector.observation")}>
        <p className="mb-1 text-xs text-slate-500 dark:text-slate-400">{t("inspector.observationHint")}</p>
        <FieldRow field={{ ...LENGTH, name: "minimum" }}>
          <QuantityInput field={{ ...LENGTH, name: "minimum" }} value={minimum} choices={units?.um ?? []} disabled={!editable} onCommit={(text) => setMinimum(text)} />
        </FieldRow>
        <FieldRow field={{ ...LENGTH, name: "maximum" }}>
          <QuantityInput field={{ ...LENGTH, name: "maximum" }} value={maximum} choices={units?.um ?? []} disabled={!editable} onCommit={(text) => setMaximum(text)} />
        </FieldRow>
        <FieldRow field={{ name: "wavelengths", kind: "param", input: "integer", doc: "" }}>
          <input data-testid="value-wavelengths" type="number" min={1} max={512} className={inputClass} value={n} disabled={!editable} onChange={(event) => setN(Number(event.target.value))} />
        </FieldRow>
        {editable && (
          <div className="mt-1 flex justify-end">
            <ActionButton icon={null} label={t("common.apply")} testId="apply-band" onClick={() => void setObservation(minimum, maximum, n)} />
          </div>
        )}
      </Section>
      <Section title={t("inspector.randomness")}>
        <SeedInput context={context} editable={editable} />
      </Section>
      {context.outputs.length > 0 && (
        <Section title={t("inspector.outputs")}>
          <ul className="space-y-0.5 font-mono text-xs text-slate-600 dark:text-slate-300">
            {context.outputs.map((output) => (
              <li key={output}>{output}</li>
            ))}
          </ul>
        </Section>
      )}
    </>
  );
}

/** The seed of every random process of the context: type it, or draw a new one. */
function SeedInput({ context, editable }: { context: ContextSummary; editable: boolean }) {
  const { t } = useTranslation();
  const setContextInfo = useStore((s) => s.setContextInfo);
  const [text, setText] = useState(String(context.seed));
  useEffect(() => {
    setText(String(context.seed));
  }, [context.seed]);
  const commit = (value: string) => {
    const seed = Number(value);
    if (Number.isInteger(seed) && seed >= 0 && seed !== context.seed) void setContextInfo({ seed });
    else setText(String(context.seed));
  };
  return (
    <>
      <p className="mb-1 text-xs text-slate-500 dark:text-slate-400">{t("inspector.randomnessHint")}</p>
      <div className="grid grid-cols-[6.5rem_1fr] items-center gap-2 py-1">
        <label htmlFor="seed" className="font-mono text-xs text-slate-600 dark:text-slate-300">
          seed
        </label>
        <div className="flex gap-1">
          <input
            id="seed"
            data-testid="seed"
            type="number"
            min={0}
            step={1}
            value={text}
            disabled={!editable}
            onChange={(event) => setText(event.target.value)}
            onBlur={() => commit(text)}
            onKeyDown={(event) => event.key === "Enter" && commit(text)}
            className={inputClass}
          />
          {editable && (
            <button
              type="button"
              data-testid="new-seed"
              title={t("inspector.newSeed")}
              aria-label={t("inspector.newSeed")}
              onClick={() => void setContextInfo({ seed: Math.floor(Math.random() * 2 ** 31) })}
              className="grid size-8 shrink-0 place-items-center rounded-md border border-slate-200 text-slate-500 hover:bg-slate-50 hover:text-accent-600 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              <Dices className="size-4" />
            </button>
          )}
        </div>
      </div>
    </>
  );
}

function SelectionPanel({ ids }: { ids: string[] }) {
  const { t } = useTranslation();
  const role = useStore((s) => s.info?.role);
  const { group, copy, removeElements } = useStore.getState();
  const top = ids.filter((id) => !id.includes("."));
  return (
    <div className="space-y-3 px-4 py-4">
      <p className="font-semibold">{t("inspector.selected", { count: ids.length })}</p>
      <ul className="flex flex-wrap gap-1">
        {ids.map((id) => (
          <li key={id} className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs dark:bg-slate-800">
            {id}
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap gap-1.5">
        {role !== "viewer" && (
          <ActionButton icon={<Layers className="size-3.5" />} label={t("inspector.groupSelection")} onClick={() => group(top)} testId="group" disabled={!top.length} />
        )}
        <ActionButton icon={<Copy className="size-3.5" />} label={t("inspector.copy")} onClick={() => void copy(top)} />
        {role !== "viewer" && (
          <ActionButton icon={<Trash2 className="size-3.5" />} label={t("inspector.delete")} onClick={() => void removeElements(top)} danger />
        )}
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400">{t("inspector.selectionHint")}</p>
    </div>
  );
}

export function Inspector({ onInspect }: { onInspect: (port: string) => void }) {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  const selection = useStore((s) => s.selection);
  const palette = useStore((s) => s.palette);
  const layout = useStore((s) => s.layout);
  const setLayout = useStore((s) => s.setLayout);
  const open = layout.inspectorOpen;
  const setOpen = (inspectorOpen: boolean) => setLayout({ inspectorOpen });
  const types = useMemo(() => new Map((palette?.types ?? []).map((item) => [item.type_id, item])), [palette]);
  const element = selection.length === 1 ? context?.elements.find((e) => e.id === selection[0]) : undefined;

  if (!open) {
    return (
      <aside className="flex w-10 shrink-0 flex-col items-center border-l border-slate-200 bg-white pt-2 dark:border-slate-800 dark:bg-slate-900">
        <button type="button" data-testid="show-inspector" onClick={() => setOpen(true)} title={t("inspector.show")} aria-label={t("inspector.show")} className="rounded p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
          <PanelRightOpen className="size-4" />
        </button>
      </aside>
    );
  }
  return (
    <aside data-testid="inspector" data-tour="inspector" className="relative flex shrink-0 flex-col border-l border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900" style={{ width: layout.inspector }}>
      <Resizer side="left" size={layout.inspector} onResize={(inspector) => setLayout({ inspector })} testId="resize-inspector" />
      <div className="flex h-9 items-center justify-between border-b border-slate-100 px-3 dark:border-slate-800">
        <h2 className="text-xs font-semibold tracking-wider text-slate-500 uppercase dark:text-slate-400">{t("inspector.title")}</h2>
        <button type="button" data-testid="hide-inspector" onClick={() => setOpen(false)} title={t("inspector.hide")} aria-label={t("inspector.hide")} className="rounded p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800">
          <PanelRightClose className="size-4" />
        </button>
      </div>
      <div className="scrollbar-thin flex-1 overflow-y-auto">
        {!context ? (
          <p className="p-4 text-sm text-slate-500">{t("common.loading")}</p>
        ) : element ? (
          <ElementPanel key={element.id} element={element} type={types.get(element.type_id)} onInspect={onInspect} />
        ) : selection.length > 1 ? (
          <SelectionPanel ids={selection} />
        ) : (
          <ContextPanel context={context} />
        )}
      </div>
    </aside>
  );
}
