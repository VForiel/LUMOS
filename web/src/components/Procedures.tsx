// The Procedures tab (step P15.12, ADR-0024): the procedures of the context edited with blocks (Blockly).
// The blocks only show the JSON of a procedure (lib/procedures): every change is converted back and saved in
// the context through the API, where the engine checks it; runs are jobs whose outputs are the named results.

import { AlertTriangle, Check, Code2, FileCode2, Play, Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type PortSummary, type ProcedureInfo } from "../api";
import {
  defaultStep,
  FUNCTIONS,
  OPERATORS,
  procedureToWorkspace,
  STEP_KINDS,
  workspaceToProcedure,
  type ProcedureJson,
  type StepKind,
  type WorkspaceJson,
} from "../lib/procedures";
import { useStore } from "../store";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type BlocklyModule = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Workspace = any;

/** Colours of the blocks, by family of steps. */
const COLOURS: Record<StepKind | "procedure", string> = {
  procedure: "#475569",
  action: "#f59e0b",
  measure: "#3b82f6",
  repeat: "#16a34a",
  for_each: "#16a34a",
  process: "#8b5cf6",
  set: "#0d9488",
  use: "#0d9488",
  wait: "#d97706",
  if: "#d97706",
  save: "#64748b",
  show: "#64748b",
};

const TOOL =
  "inline-flex items-center gap-1 rounded-lg px-2 py-1 text-sm text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800";

const text = (name: string, value = "") => ({ type: "field_input", name, text: value });

/** Blockly definitions of the steps (fields named as in lib/procedures). */
function definitions(t: (key: string) => string) {
  const step = (kind: StepKind, message: string, args: object[], extra: object = {}) => ({
    type: `lumos_${kind}`,
    message0: message,
    args0: args,
    previousStatement: null,
    nextStatement: null,
    colour: COLOURS[kind],
    tooltip: t(`procedures.help.${kind}`),
    ...extra,
  });
  return [
    {
      type: "lumos_procedure",
      message0: "%1 %2 %3",
      args0: [{ type: "field_label_serializable", name: "NAME", class: "lumos-procedure-name" }, text("DESCRIPTION"), { type: "input_dummy" }],
      message1: "%1",
      args1: [{ type: "input_statement", name: "STEPS" }],
      colour: COLOURS.procedure,
      tooltip: t("procedures.help.procedure"),
    },
    step("action", `${t("procedures.block.action")} %1 . %2 ( %3 %4 )`, [text("ELEMENT"), text("ACTION"), text("ARGS"), text("KWARGS")]),
    step("measure", `${t("procedures.block.measure")} %1 → %2 ${t("procedures.block.collect")} %3`, [
      text("OUTPUTS"),
      text("AS"),
      { type: "field_checkbox", name: "COLLECT", checked: false },
    ]),
    {
      ...step("repeat", `${t("procedures.block.repeat")} %1 ×`, [{ type: "field_number", name: "TIMES", value: 2, min: 0, precision: 1 }]),
      message1: "%1",
      args1: [{ type: "input_statement", name: "STEPS" }],
    },
    {
      ...step("for_each", `${t("procedures.block.for_each")} %1 ${t("procedures.block.in")} %2 %3 → %4 … %5 (%6, log %7)`, [
        text("AS", "value"),
        {
          type: "field_dropdown",
          name: "MODE",
          options: [
            [t("procedures.block.list"), "list"],
            [t("procedures.block.range"), "range"],
          ],
        },
        text("VALUES"),
        text("START"),
        text("STOP"),
        { type: "field_number", name: "NUM", value: 2, min: 1, precision: 1 },
        { type: "field_checkbox", name: "LOG", checked: false },
      ]),
      message1: "%1",
      args1: [{ type: "input_statement", name: "STEPS" }],
    },
    step("process", `%1 = %2 ( %3 )`, [
      text("AS"),
      { type: "field_dropdown", name: "FUNCTION", options: FUNCTIONS.map((name) => [name, name]) },
      text("ARGS"),
    ]),
    step("set", `${t("procedures.block.set")} %1 = %2`, [text("TARGET"), text("VALUE")]),
    step("use", `${t("procedures.block.use")} %1 → %2`, [text("RESULT"), text("PORT")]),
    step("wait", `${t("procedures.block.wait")} %1`, [text("DURATION", "1 s")]),
    {
      ...step("if", `${t("procedures.block.if")} %1 %2 %3`, [
        text("LEFT"),
        { type: "field_dropdown", name: "OP", options: OPERATORS.map((op) => [op, op]) },
        text("RIGHT"),
      ]),
      message1: "%1",
      args1: [{ type: "input_statement", name: "THEN" }],
      message2: `${t("procedures.block.else")} %1`,
      args2: [{ type: "input_statement", name: "ELSE" }],
    },
    step("save", `${t("procedures.block.save")} %1 → %2`, [text("RESULTS"), text("ARCHIVE")]),
    step("show", `${t("procedures.block.show")} %1`, [text("RESULTS")]),
  ];
}

let loaded: Promise<BlocklyModule> | null = null;
/** Load Blockly once (a large library, only needed in this tab) and define the blocks of the steps. */
function loadBlockly(t: (key: string) => string): Promise<BlocklyModule> {
  loaded ??= import("blockly").then((Blockly) => {
    Blockly.common.defineBlocksWithJsonArray(definitions(t));
    return Blockly;
  });
  return loaded;
}

function toolbox(t: (key: string) => string) {
  return {
    kind: "flyoutToolbox",
    contents: STEP_KINDS.map((kind) => ({ kind: "block", type: `lumos_${kind}`, tooltip: t(`procedures.kind.${kind}`) })),
  };
}

type Run = { job: string; status: string; message: string; results: PortSummary[] };

export function ProceduresView() {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  const info = useStore((s) => s.info);
  const role = info?.role;
  const editable = role !== "viewer";
  const canRun = role === "simulator" || role === "operator";
  const [list, setList] = useState<ProcedureInfo[]>([]);
  const [current, setCurrent] = useState("");
  const [procedure, setProcedure] = useState<ProcedureJson | null>(null);
  const [saved, setSaved] = useState(false);
  const [run, setRun] = useState<Run | null>(null);
  const host = useRef<HTMLDivElement>(null);
  const workspace = useRef<Workspace>(null);
  const timer = useRef<number>(0);
  const contextId = context?.id;
  const selected = useMemo(() => list.find((p) => p.name === current), [list, current]);

  const refresh = useCallback(async () => {
    if (!contextId) return [];
    const found = await api.procedures(contextId);
    setList(found);
    return found;
  }, [contextId]);

  // the procedures of the context; the first one is opened
  useEffect(() => {
    void refresh().then((found) => setCurrent((name) => (found.some((p) => p.name === name) ? name : (found[0]?.name ?? ""))));
  }, [refresh]);

  useEffect(() => {
    setRun(null);
    if (!contextId || !current) {
      setProcedure(null);
      return;
    }
    void api.procedure(contextId, current).then((json) => setProcedure({ ...json, steps: json.steps ?? [] }));
  }, [contextId, current]);

  const save = useCallback(
    async (json: ProcedureJson) => {
      if (!contextId || !editable) return;
      const body = (json.script !== undefined
        ? { name: json.name, description: json.description, script: json.script }
        : { name: json.name, description: json.description, steps: json.steps }) as ProcedureJson;
      const updated = await api.putProcedure(contextId, body);
      setList((items) => items.map((item) => (item.name === updated.name ? updated : item)));
      setSaved(true);
    },
    [contextId, editable],
  );

  // the workspace: created once, filled with the procedure opened
  useEffect(() => {
    if (!host.current || !procedure || procedure.script !== undefined) return;
    let cancelled = false;
    void loadBlockly(t).then((Blockly) => {
      if (cancelled || !host.current) return;
      if (!workspace.current) {
        workspace.current = Blockly.inject(host.current, {
          toolbox: toolbox(t),
          readOnly: !editable,
          trashcan: true,
          zoom: { controls: true, wheel: true, startScale: 0.9 },
          move: { scrollbars: true, drag: true, wheel: false },
          renderer: "thrasos",
        });
        workspace.current.addChangeListener((event: { isUiEvent: boolean }) => {
          if (event.isUiEvent) return;
          setSaved(false);
          window.clearTimeout(timer.current);
          timer.current = window.setTimeout(() => {
            const json = workspaceToProcedure(Blockly.serialization.workspaces.save(workspace.current) as WorkspaceJson);
            if (json.name) void save(json);
          }, 600);
        });
      }
      // no change events while the procedure is drawn: they would save it again and mark the context changed
      Blockly.Events.disable();
      try {
        Blockly.serialization.workspaces.load(procedureToWorkspace(procedure), workspace.current);
      } finally {
        Blockly.Events.enable();
      }
    });
    return () => {
      cancelled = true;
    };
  }, [procedure, editable, save, t]);

  // the workspace follows the size of its box (the results panel opens beside it)
  useEffect(() => {
    if (!host.current) return;
    const observer = new ResizeObserver(() => {
      if (workspace.current) void loadBlockly(t).then((Blockly) => Blockly.svgResize(workspace.current));
    });
    observer.observe(host.current);
    return () => observer.disconnect();
  }, [procedure, t]);

  useEffect(
    () => () => {
      workspace.current?.dispose();
      workspace.current = null;
    },
    [],
  );

  const create = (script: boolean) =>
    useStore.getState().setDialog({
      kind: "prompt",
      title: script ? "procedures.newScript" : "procedures.new",
      label: "procedures.name",
      value: "",
      confirm: "procedures.create",
      onConfirm: (name) => {
        if (!contextId || !name.trim()) return;
        const json: ProcedureJson = script ? { name, description: "", steps: [], script: "results['answer'] = 42\n" } : { name, description: "", steps: [] };
        const body = script ? { name, description: "", script: json.script } : json;
        void api
          .putProcedure(contextId, body as ProcedureJson)
          .then(() => refresh())
          .then(() => setCurrent(name))
          .catch((error) => useStore.getState().fail(error));
      },
    });

  const remove = async () => {
    if (!contextId || !current) return;
    await api.removeProcedure(contextId, current);
    const found = await refresh();
    setCurrent(found[0]?.name ?? "");
  };

  const add = (kind: StepKind) => {
    if (!procedure) return;
    const json = { ...procedure, steps: [...procedure.steps, defaultStep(kind)] };
    setProcedure(json);
    void save(json);
  };

  const python = async () => {
    if (!contextId || !current) return;
    const code = await api.exportProcedure(contextId, current);
    useStore.getState().setDialog({ kind: "python", code, testId: "procedure-code", fileName: `${current}.py` });
  };

  const start = async (confirmed = false) => {
    if (!contextId || !current) return;
    if (selected?.devices.length && !confirmed) {
      useStore.getState().setDialog({
        kind: "elicit",
        message: t("procedures.confirm", { devices: selected.devices.join(", ") }),
        answer: (accepted) => {
          if (accepted) void start(true);
        },
      });
      return;
    }
    try {
      const job = await api.runProcedure(contextId, current, confirmed);
      setRun({ job: job.id, status: job.status, message: "", results: [] });
      let state = job;
      while (state.status === "queued" || state.status === "running") {
        await new Promise((resolve) => window.setTimeout(resolve, 400));
        state = await api.job(job.id);
        setRun({ job: job.id, status: state.status, message: state.message, results: [] });
      }
      const results = state.status === "done" ? await Promise.all(state.outputs.map((name) => api.result(job.id, name))) : [];
      setRun({ job: job.id, status: state.status, message: state.error ?? state.message, results });
      void useStore.getState().refresh(); // the procedure moved the clock and changed states
    } catch (error) {
      useStore.getState().fail(error);
    }
  };

  if (!context) return null;
  const problems = selected?.problems ?? [];
  return (
    <div data-testid="procedures" className="flex min-h-0 flex-1 flex-col bg-white dark:bg-slate-950">
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 px-3 py-2 dark:border-slate-800">
        <select
          data-testid="procedure-select"
          className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm dark:border-slate-700 dark:bg-slate-900"
          value={current}
          onChange={(event) => setCurrent(event.target.value)}
        >
          {list.length === 0 && <option value="">{t("procedures.none")}</option>}
          {list.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        {editable && (
          <button type="button" data-testid="procedure-new" className={TOOL} onClick={() => create(false)} title={t("procedures.new")}>
            <Plus className="size-4" /> {t("procedures.new")}
          </button>
        )}
        {editable && info?.scripts && (
          <button type="button" data-testid="procedure-new-script" className={TOOL} onClick={() => create(true)} title={t("procedures.newScriptHelp")}>
            <FileCode2 className="size-4" /> {t("procedures.newScript")}
          </button>
        )}
        {editable && current && (
          <button type="button" data-testid="procedure-delete" className={TOOL} onClick={() => void remove()} title={t("procedures.delete")}>
            <Trash2 className="size-4" />
          </button>
        )}
        <div className="flex-1" />
        {saved && (
          <span data-testid="procedure-saved" className="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400">
            <Check className="size-3.5" /> {t("procedures.saved")}
          </span>
        )}
        {current && procedure?.script === undefined && (
          <button type="button" data-testid="procedure-python" className={TOOL} onClick={() => void python()}>
            <Code2 className="size-4" /> {t("procedures.python")}
          </button>
        )}
        {current && canRun && (
          <button
            type="button"
            data-testid="procedure-run"
            disabled={problems.length > 0 || run?.status === "running" || run?.status === "queued"}
            className="inline-flex items-center gap-1.5 rounded-lg bg-accent-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-accent-700 disabled:opacity-50"
            onClick={() => void start()}
          >
            <Play className="size-4" /> {t("procedures.run")}
          </button>
        )}
      </div>
      {problems.length > 0 && (
        <ul data-testid="procedure-problems" className="space-y-0.5 border-b border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300">
          {problems.map((problem) => (
            <li key={problem} className="flex gap-1.5">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" /> {problem}
            </li>
          ))}
        </ul>
      )}
      {editable && current && procedure?.script === undefined && (
        <div className="flex flex-wrap gap-1 border-b border-slate-100 px-3 py-1.5 dark:border-slate-800" aria-label={t("procedures.add")}>
          <span className="mr-1 self-center text-xs text-slate-500">{t("procedures.add")}</span>
          {STEP_KINDS.map((kind) => (
            <button
              key={kind}
              type="button"
              data-testid={`procedure-add-${kind}`}
              onClick={() => add(kind)}
              className="rounded-md border px-2 py-0.5 text-xs hover:bg-slate-50 dark:hover:bg-slate-800"
              style={{ borderColor: COLOURS[kind], color: COLOURS[kind] }}
              title={t(`procedures.help.${kind}`)}
            >
              {t(`procedures.kind.${kind}`)}
            </button>
          ))}
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        {procedure?.script !== undefined ? (
          <ScriptEditor procedure={procedure} editable={editable && Boolean(info?.scripts)} onSave={(json) => void save(json)} />
        ) : (
          <div ref={host} data-testid="procedure-blocks" className="relative min-h-0 flex-1" />
        )}
        {!current && <p className="m-auto max-w-md p-6 text-center text-sm text-slate-500">{t("procedures.empty")}</p>}
        {run && (
          <aside data-testid="procedure-results" className="w-72 shrink-0 overflow-auto border-l border-slate-200 p-3 text-sm dark:border-slate-800">
            <h3 className="mb-2 font-semibold">
              {t("procedures.results")} <span className="text-xs font-normal text-slate-500">({t(`job.${run.status}`, run.status)})</span>
            </h3>
            {run.message && <p className="mb-2 text-xs text-slate-500">{run.message}</p>}
            <ul className="space-y-1.5">
              {run.results.map((result) => (
                <li key={result.port} className="rounded-lg border border-slate-200 px-2 py-1 dark:border-slate-800">
                  <div className="font-mono text-xs font-semibold">{result.port}</div>
                  <div className="text-xs text-slate-500">
                    {result.kind}
                    {result.shape?.length ? ` [${result.shape.join(" × ")}]` : ""}
                    {result.values?.length === 1 ? ` = ${result.values[0].toPrecision(5)} ${result.unit ?? ""}` : ""}
                    {result.statistics?.mean !== undefined ? ` · mean ${Number(result.statistics.mean).toPrecision(4)} ${result.unit ?? ""}` : ""}
                  </div>
                </li>
              ))}
            </ul>
          </aside>
        )}
      </div>
    </div>
  );
}

function ScriptEditor({ procedure, editable, onSave }: { procedure: ProcedureJson; editable: boolean; onSave: (json: ProcedureJson) => void }) {
  const { t } = useTranslation();
  const [code, setCode] = useState(procedure.script ?? "");
  useEffect(() => setCode(procedure.script ?? ""), [procedure]);
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
      <p className="text-xs text-slate-500">{t("procedures.scriptHelp")}</p>
      <textarea
        data-testid="procedure-script"
        className="min-h-0 flex-1 rounded-lg border border-slate-200 bg-slate-950 p-3 font-mono text-xs text-slate-100 dark:border-slate-700"
        value={code}
        readOnly={!editable}
        spellCheck={false}
        onChange={(event) => setCode(event.target.value)}
        onBlur={() => editable && code !== procedure.script && onSave({ name: procedure.name, description: procedure.description, steps: [], script: code })}
      />
    </div>
  );
}
