// Editor of declarative components (P15.15, ADR-0025): a form filling a template, its problems listed while it
// is typed (the checks are run by the server), the document it makes, and on local instances a code view.

import { Code2, FileJson, ListChecks, Plus, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, download, type ComponentDocument } from "../api";
import { useStore } from "../store";
import { Button, Modal } from "./Dialogs";

const TEMPLATES = ["gain_phase", "spectral_transmission", "transfer_matrix", "composite"] as const;
type Template = (typeof TEMPLATES)[number];

type ParamRow = { name: string; unit: string; default: string; min: string; max: string; doc: string };
type CheckRow = { wavelength: string; power: string; expect: string; rtol: string };
type Form = {
  author: string;
  name: string;
  version: number;
  title: string;
  description: string;
  license: string;
  template: Template;
  params: ParamRow[];
  gain: string;
  phase: string;
  amplifier: boolean;
  wavelengths: string;
  transmission: string;
  /** Body of the templates without dedicated fields (transfer_matrix, composite), as JSON. */
  body: string;
  checks: CheckRow[];
};

const EXAMPLE_BODIES: Record<string, string> = {
  transfer_matrix: JSON.stringify({ amplitudes: [[0.7071, 0.7071], [0.7071, 0.7071]], phases: [[0, 0], [0, 3.1416]], wavelength: "1.55 um" }, null, 2),
  composite: JSON.stringify({ group: { type: "lumos.group", name: "group", children: [], connections: [], exposed_ports: {} } }, null, 2),
};

const EMPTY: Form = {
  author: "",
  name: "",
  version: 1,
  title: "",
  description: "",
  license: "",
  template: "gain_phase",
  params: [],
  gain: "1",
  phase: "0 rad",
  amplifier: false,
  wavelengths: "[1.0, 1.5] um",
  transmission: "[1, 1]",
  body: "",
  checks: [],
};

const CODE_EXAMPLE = `from lumos.core.element import Element
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal


@register("plugins.me.Halver")
class Halver(Element):
    """Halve the power."""

    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values, inputs):
        return {"out": inputs["in"] * 0.5**0.5}
`;

/** A number when the text is one, else the text (a quantity such as "3 dB" or a "$parameter"). */
function value(text: string): number | string {
  const trimmed = text.trim();
  return trimmed !== "" && !Number.isNaN(Number(trimmed)) ? Number(trimmed) : trimmed;
}

function bound(text: string): number | null {
  return text.trim() === "" || Number.isNaN(Number(text)) ? null : Number(text);
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** The ``.lumos`` document of kind ``component`` described by the form. */
export function toDocument(form: Form): ComponentDocument {
  const params: Record<string, unknown> = {};
  for (const row of form.params) {
    if (!row.name.trim()) continue;
    const declared: Record<string, unknown> = { unit: row.unit.trim(), default: value(row.default), doc: row.doc };
    if (row.min.trim() || row.max.trim()) declared.bounds = [bound(row.min), bound(row.max)];
    params[row.name.trim()] = declared;
  }
  let body: unknown;
  if (form.template === "gain_phase") {
    body = { gain: value(form.gain), phase: value(form.phase), ...(form.amplifier ? { amplifier: true } : {}) };
  } else if (form.template === "spectral_transmission") {
    body = { wavelengths: value(form.wavelengths), transmission: value(form.transmission) };
  } else {
    body = parseJson(form.body);
  }
  const checks = form.checks.map((row) => ({
    kind: "power",
    wavelength: row.wavelength.trim(),
    power: row.power.trim(),
    expect: row.expect.trim(),
    rtol: Number(row.rtol) || 1e-6,
  }));
  return {
    lumos_format: "1.3",
    kind: "component",
    component: {
      id: `${form.author.trim()}.${form.name.trim()}`,
      version: form.version,
      title: form.title,
      description: form.description,
      license: form.license,
      authors: form.author.trim() ? [form.author.trim()] : [],
      template: form.template,
      params,
      body,
      checks,
    },
  };
}

const text = (v: unknown) => (v === undefined || v === null ? "" : typeof v === "string" ? v : JSON.stringify(v));

/** The form of a component document (to edit a saved component). */
export function fromDocument(document: ComponentDocument): Form {
  const c = document.component;
  const [author, ...rest] = String(c.id ?? "").split(".");
  const template = (TEMPLATES as readonly string[]).includes(c.template) ? (c.template as Template) : "gain_phase";
  const body = c.body ?? {};
  return {
    ...EMPTY,
    author,
    name: rest.join("."),
    version: Number(c.version) || 1,
    title: c.title ?? "",
    description: c.description ?? "",
    license: c.license ?? "",
    template,
    params: Object.entries(c.params ?? {}).map(([name, declared]) => {
      const d = declared as Record<string, unknown>;
      const [min, max] = (d.bounds as (number | null)[] | undefined) ?? [null, null];
      return { name, unit: text(d.unit), default: text(d.default), min: text(min), max: text(max), doc: text(d.doc) };
    }),
    gain: text(body.gain ?? EMPTY.gain),
    phase: text(body.phase ?? EMPTY.phase),
    amplifier: Boolean(body.amplifier),
    wavelengths: text(body.wavelengths ?? EMPTY.wavelengths),
    transmission: text(body.transmission ?? EMPTY.transmission),
    body: template === "transfer_matrix" || template === "composite" ? JSON.stringify(body, null, 2) : "",
    checks: (c.checks ?? []).map((check: Record<string, unknown>) => ({
      wavelength: text(check.wavelength),
      power: text(check.power),
      expect: text(check.expect),
      rtol: text(check.rtol ?? "1e-6"),
    })),
  };
}

const INPUT =
  "w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-sm outline-none focus:border-accent-400 focus:ring-2 focus:ring-accent-200 dark:border-slate-700 dark:bg-slate-800 dark:focus:ring-accent-900";

function Field({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-300">{label}</span>
      {children}
      {help && <span className="mt-0.5 block text-[11px] text-slate-400">{help}</span>}
    </label>
  );
}

export function ComponentEditor({ id }: { id?: string }) {
  const { t } = useTranslation();
  const scripts = useStore((s) => s.info?.scripts ?? false);
  const [form, setForm] = useState<Form>(EMPTY);
  const [view, setView] = useState<"form" | "document" | "code">("form");
  const [problems, setProblems] = useState<string[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [code, setCode] = useState(CODE_EXAMPLE);
  const close = () => useStore.getState().setDialog(null);
  const change = (patch: Partial<Form>) => setForm((current) => ({ ...current, ...patch }));
  const document = useMemo(() => toDocument(form), [form]);

  useEffect(() => {
    if (id) void api.component(id).then((found) => setForm(fromDocument(found)), (error) => useStore.getState().fail(error));
  }, [id]);

  // the problems of the component, checked by the server while it is typed
  useEffect(() => {
    let current = true;
    const timer = window.setTimeout(() => {
      api.checkComponent(document).then(
        (found) => current && setProblems(found),
        (error: Error) => current && setProblems([String(error.message ?? error)]),
      );
    }, 300);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [document]);

  const save = async () => {
    setSaving(true);
    try {
      const saved = await api.saveComponent(document);
      await useStore.getState().refreshPalette();
      useStore.getState().toast({ kind: "success", title: "componentSaved", values: { name: saved.title || saved.id } });
      close();
    } catch (error) {
      useStore.getState().fail(error);
    } finally {
      setSaving(false);
    }
  };

  const runCode = async () => {
    try {
      const types = await api.codeComponent(code);
      await useStore.getState().refreshPalette();
      useStore.getState().toast({ kind: "success", title: "componentDefined", values: { types: types.join(", ") || "—" } });
    } catch (error) {
      useStore.getState().fail(error);
    }
  };

  const setParam = (index: number, patch: Partial<ParamRow>) =>
    change({ params: form.params.map((row, k) => (k === index ? { ...row, ...patch } : row)) });
  const setCheck = (index: number, patch: Partial<CheckRow>) =>
    change({ checks: form.checks.map((row, k) => (k === index ? { ...row, ...patch } : row)) });

  const tab = (name: typeof view, label: string, Icon: typeof Code2) => (
    <button
      type="button"
      data-testid={`component-view-${name}`}
      onClick={() => setView(name)}
      className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium ${
        view === name ? "bg-accent-50 text-accent-700 dark:bg-accent-900/40 dark:text-accent-300" : "text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
      }`}
    >
      <Icon className="size-3.5" /> {label}
    </button>
  );

  return (
    <Modal title={t(id ? "component.editTitle" : "component.title")} onClose={close} wide testId="component-editor">
      <div className="space-y-4 p-5">
        <div className="flex flex-wrap items-center gap-1">
          {tab("form", t("component.form"), ListChecks)}
          {tab("document", t("component.document"), FileJson)}
          {scripts && tab("code", t("component.code"), Code2)}
        </div>

        {view === "form" && (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label={t("component.author")} help={t("component.authorHelp")}>
                <input data-testid="component-author" className={INPUT} value={form.author} onChange={(e) => change({ author: e.target.value })} />
              </Field>
              <Field label={t("component.name")} help={t("component.nameHelp")}>
                <input data-testid="component-name" className={INPUT} value={form.name} onChange={(e) => change({ name: e.target.value })} />
              </Field>
              <Field label={t("component.version")}>
                <input
                  data-testid="component-version"
                  type="number"
                  min={1}
                  className={INPUT}
                  value={form.version}
                  onChange={(e) => change({ version: Math.max(1, Math.round(Number(e.target.value) || 1)) })}
                />
              </Field>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label={t("component.titleField")}>
                <input data-testid="component-title" className={INPUT} value={form.title} onChange={(e) => change({ title: e.target.value })} />
              </Field>
              <Field label={t("component.license")} help={t("component.licenseHelp")}>
                <input data-testid="component-license" className={INPUT} value={form.license} onChange={(e) => change({ license: e.target.value })} />
              </Field>
            </div>
            <Field label={t("component.description")}>
              <textarea
                data-testid="component-description"
                rows={2}
                className={INPUT}
                value={form.description}
                onChange={(e) => change({ description: e.target.value })}
              />
            </Field>

            <section className="space-y-2">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{t("component.params")}</h3>
                <button
                  type="button"
                  data-testid="component-add-param"
                  onClick={() => change({ params: [...form.params, { name: "", unit: "", default: "", min: "", max: "", doc: "" }] })}
                  className="ml-auto inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-accent-600 hover:bg-accent-50 dark:hover:bg-accent-900/30"
                >
                  <Plus className="size-3.5" /> {t("component.addParam")}
                </button>
              </div>
              {form.params.length === 0 && <p className="text-xs text-slate-400">{t("component.paramsHint")}</p>}
              {form.params.map((row, index) => (
                <div key={index} className="grid grid-cols-2 gap-2 rounded-lg border border-slate-200 p-2 sm:grid-cols-[1fr_5rem_5rem_4rem_4rem_2fr_auto] dark:border-slate-700">
                  <input data-testid={`param-name-${index}`} placeholder={t("component.paramName")} className={INPUT} value={row.name} onChange={(e) => setParam(index, { name: e.target.value })} />
                  <input data-testid={`param-unit-${index}`} placeholder={t("component.unit")} className={INPUT} value={row.unit} onChange={(e) => setParam(index, { unit: e.target.value })} />
                  <input data-testid={`param-default-${index}`} placeholder={t("component.default")} className={INPUT} value={row.default} onChange={(e) => setParam(index, { default: e.target.value })} />
                  <input placeholder="min" className={INPUT} value={row.min} onChange={(e) => setParam(index, { min: e.target.value })} />
                  <input placeholder="max" className={INPUT} value={row.max} onChange={(e) => setParam(index, { max: e.target.value })} />
                  <input data-testid={`param-doc-${index}`} placeholder={t("component.paramDoc")} className={INPUT} value={row.doc} onChange={(e) => setParam(index, { doc: e.target.value })} />
                  <button
                    type="button"
                    onClick={() => change({ params: form.params.filter((_, k) => k !== index) })}
                    title={t("component.remove")}
                    className="rounded p-1 text-slate-400 hover:text-red-500"
                  >
                    <Trash2 className="size-4" />
                  </button>
                </div>
              ))}
            </section>

            <section className="space-y-2">
              <h3 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{t("component.behaviour")}</h3>
              <Field label={t("component.template")} help={t(`component.templates.${form.template}`)}>
                <select
                  data-testid="component-template"
                  className={INPUT}
                  value={form.template}
                  onChange={(e) => {
                    const template = e.target.value as Template;
                    change({ template, body: form.body || EXAMPLE_BODIES[template] || "" });
                  }}
                >
                  {TEMPLATES.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </Field>
              {form.template === "gain_phase" && (
                <div className="grid gap-3 sm:grid-cols-3">
                  <Field label={t("component.gain")} help={t("component.valueHelp")}>
                    <input data-testid="component-gain" className={INPUT} value={form.gain} onChange={(e) => change({ gain: e.target.value })} />
                  </Field>
                  <Field label={t("component.phase")}>
                    <input data-testid="component-phase" className={INPUT} value={form.phase} onChange={(e) => change({ phase: e.target.value })} />
                  </Field>
                  <label className="flex items-center gap-2 pt-5 text-sm">
                    <input type="checkbox" data-testid="component-amplifier" checked={form.amplifier} onChange={(e) => change({ amplifier: e.target.checked })} />
                    {t("component.amplifier")}
                  </label>
                </div>
              )}
              {form.template === "spectral_transmission" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <Field label={t("component.wavelengths")} help='[1.0, 1.5] um'>
                    <input data-testid="component-wavelengths" className={INPUT} value={form.wavelengths} onChange={(e) => change({ wavelengths: e.target.value })} />
                  </Field>
                  <Field label={t("component.transmission")} help="[0.9, 0.8]">
                    <input data-testid="component-transmission" className={INPUT} value={form.transmission} onChange={(e) => change({ transmission: e.target.value })} />
                  </Field>
                </div>
              )}
              {(form.template === "transfer_matrix" || form.template === "composite") && (
                <Field label={t("component.body")} help={t("component.bodyHelp")}>
                  <textarea
                    data-testid="component-body"
                    rows={8}
                    spellCheck={false}
                    className={`${INPUT} font-mono text-xs`}
                    value={form.body}
                    onChange={(e) => change({ body: e.target.value })}
                  />
                </Field>
              )}
            </section>

            <section className="space-y-2">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{t("component.checks")}</h3>
                <button
                  type="button"
                  data-testid="component-add-check"
                  onClick={() => change({ checks: [...form.checks, { wavelength: "", power: "", expect: "", rtol: "1e-6" }] })}
                  className="ml-auto inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-accent-600 hover:bg-accent-50 dark:hover:bg-accent-900/30"
                >
                  <Plus className="size-3.5" /> {t("component.addCheck")}
                </button>
              </div>
              <p className="text-xs text-slate-400">{t("component.checksHint")}</p>
              {form.checks.map((row, index) => (
                <div key={index} className="grid grid-cols-2 gap-2 rounded-lg border border-slate-200 p-2 sm:grid-cols-[1fr_1fr_1fr_5rem_auto] dark:border-slate-700">
                  <input data-testid={`check-wavelength-${index}`} placeholder={t("component.checkWavelength")} className={INPUT} value={row.wavelength} onChange={(e) => setCheck(index, { wavelength: e.target.value })} />
                  <input data-testid={`check-power-${index}`} placeholder={t("component.checkPower")} className={INPUT} value={row.power} onChange={(e) => setCheck(index, { power: e.target.value })} />
                  <input data-testid={`check-expect-${index}`} placeholder={t("component.checkExpect")} className={INPUT} value={row.expect} onChange={(e) => setCheck(index, { expect: e.target.value })} />
                  <input placeholder="rtol" className={INPUT} value={row.rtol} onChange={(e) => setCheck(index, { rtol: e.target.value })} />
                  <button
                    type="button"
                    onClick={() => change({ checks: form.checks.filter((_, k) => k !== index) })}
                    title={t("component.remove")}
                    className="rounded p-1 text-slate-400 hover:text-red-500"
                  >
                    <Trash2 className="size-4" />
                  </button>
                </div>
              ))}
            </section>
          </>
        )}

        {view === "document" && (
          <pre data-testid="component-json" className="scrollbar-thin max-h-96 overflow-auto rounded-lg bg-slate-50 p-3 font-mono text-xs dark:bg-slate-800">
            {JSON.stringify(document, null, 2)}
          </pre>
        )}

        {view === "code" && scripts && (
          <div className="space-y-2">
            <p className="text-xs text-slate-500">{t("component.codeHelp")}</p>
            <textarea
              data-testid="component-source"
              rows={16}
              spellCheck={false}
              className={`${INPUT} font-mono text-xs`}
              value={code}
              onChange={(e) => setCode(e.target.value)}
            />
            <div className="flex justify-end">
              <Button primary testId="component-run" onClick={() => void runCode()}>
                {t("component.run")}
              </Button>
            </div>
          </div>
        )}

        {view !== "code" && (
          <>
            {problems && problems.length > 0 && (
              <ul data-testid="component-problems" className="space-y-0.5 rounded-lg border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
                {problems.map((problem) => (
                  <li key={problem}>• {problem}</li>
                ))}
              </ul>
            )}
            <div className="flex flex-wrap justify-end gap-2">
              <Button
                testId="component-download"
                onClick={() => download(`${form.name.trim() || "component"}.lumos`, JSON.stringify(document, null, 2))}
              >
                {t("component.download")}
              </Button>
              <Button primary testId="component-save" disabled={saving || problems === null || problems.length > 0} onClick={() => void save()}>
                {t("component.save")}
              </Button>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
}
