// Procedures (ADR-0024) as Blockly workspaces and back (step P15.12). The engine reads the procedure JSON;
// the editor only shows it as blocks: one block per step, nested steps in statement inputs. These pure
// functions work on the JSON serialisation of Blockly, so they are tested without a browser.

export type Value = string | number | boolean | null;
export type Step = { do: string; [key: string]: unknown };
export type ProcedureJson = { name: string; description: string; steps: Step[]; script?: string };

/** A block of the Blockly JSON serialisation (the subset used here). */
export type BlockJson = {
  type: string;
  id?: string;
  x?: number;
  y?: number;
  fields: Record<string, string | number | boolean>;
  inputs?: Record<string, { block: BlockJson }>;
  next?: { block: BlockJson };
};
export type WorkspaceJson = { blocks: { languageVersion: number; blocks: BlockJson[] } };

/** Kinds of steps, in the order of the toolbox. */
export const STEP_KINDS = ["action", "measure", "repeat", "for_each", "process", "set", "use", "wait", "if", "save", "show"] as const;
export type StepKind = (typeof STEP_KINDS)[number];

/** Processing functions of the engine (lumos.core.procedure.FUNCTIONS). */
export const FUNCTIONS = [
  "mean",
  "median",
  "stack",
  "subtract",
  "divide",
  "scale",
  "subtract_dark",
  "flat_field",
  "crop",
  "rebin",
  "total",
  "maximum",
  "minimum",
  "average",
  "element",
  "photometry",
] as const;

export const OPERATORS = ["<", "<=", ">", ">=", "==", "!="] as const;

// --- Values written in text fields ----------------------------------------------------------------------

/** Text of a value: strings as they are ("2 mm", "$dark"), other values in JSON. */
export function valueText(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value ?? null);
}

/** Value of a text field: JSON numbers, booleans and null, else the text itself. */
export function textValue(text: string): Value {
  const trimmed = text.trim();
  if (/^(-?\d+(\.\d+)?([eE][+-]?\d+)?|true|false|null)$/.test(trimmed)) return JSON.parse(trimmed) as Value;
  return text;
}

/** Split ``a, [1, 2] nm, b`` at the commas outside brackets and quotes. */
export function splitTop(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quote = "";
  let current = "";
  for (const char of text) {
    if (quote) {
      if (char === quote) quote = "";
    } else if (char === '"' || char === "'") quote = char;
    else if (char === "[" || char === "{" || char === "(") depth += 1;
    else if (char === "]" || char === "}" || char === ")") depth -= 1;
    else if (char === "," && depth === 0) {
      parts.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

const listText = (values: unknown[]) => values.map(valueText).join(", ");
const textList = (text: string) => splitTop(text).map(textValue);
const namesText = (names: unknown) => (Array.isArray(names) ? names.join(", ") : "");
const textNames = (text: string) => splitTop(text);
const pairsText = (pairs: Record<string, unknown>) =>
  Object.entries(pairs)
    .map(([key, value]) => `${key}=${valueText(value)}`)
    .join(", ");
function textPairs(text: string): Record<string, Value> {
  const pairs: Record<string, Value> = {};
  for (const part of splitTop(text)) {
    const at = part.indexOf("=");
    if (at > 0) pairs[part.slice(0, at).trim()] = textValue(part.slice(at + 1).trim());
  }
  return pairs;
}

// --- Procedure -> workspace -----------------------------------------------------------------------------

function chain(steps: Step[]): { block: BlockJson } | undefined {
  let next: { block: BlockJson } | undefined;
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const block = stepBlock(steps[index]);
    if (next) block.next = next;
    next = { block };
  }
  return next;
}

function withSteps(block: BlockJson, name: string, steps: unknown): BlockJson {
  const first = chain(Array.isArray(steps) ? (steps as Step[]) : []);
  if (first) block.inputs = { ...(block.inputs ?? {}), [name]: first };
  return block;
}

export function stepBlock(step: Step): BlockJson {
  const type = `lumos_${step.do}`;
  const s = step as Record<string, unknown>;
  switch (step.do) {
    case "action":
      return {
        type,
        fields: {
          ELEMENT: String(s.element ?? ""),
          ACTION: String(s.action ?? ""),
          ARGS: listText((s.args as unknown[]) ?? []),
          KWARGS: pairsText((s.kwargs as Record<string, unknown>) ?? {}),
        },
      };
    case "measure":
      return { type, fields: { OUTPUTS: namesText(s.outputs), AS: String(s.as ?? ""), COLLECT: Boolean(s.collect) } };
    case "repeat":
      return withSteps({ type, fields: { TIMES: Number(s.times ?? 1) } }, "STEPS", s.steps);
    case "for_each": {
      const values = s.values;
      const range = values && !Array.isArray(values) ? (values as Record<string, unknown>) : null;
      const fields = range
        ? {
            MODE: "range",
            VALUES: "",
            START: valueText(range.start),
            STOP: valueText(range.stop),
            NUM: Number(range.num ?? 2),
            LOG: Boolean(range.log),
            AS: String(s.as ?? ""),
          }
        : { MODE: "list", VALUES: listText((values as unknown[]) ?? []), START: "", STOP: "", NUM: 2, LOG: false, AS: String(s.as ?? "") };
      return withSteps({ type, fields: { ...fields, PARALLEL: Boolean(s.parallel) } }, "STEPS", s.steps);
    }
    case "process":
      return {
        type,
        fields: { FUNCTION: String(s.function ?? "mean"), ARGS: pairsText((s.args as Record<string, unknown>) ?? {}), AS: String(s.as ?? "") },
      };
    case "set":
      return { type, fields: { TARGET: String(s.target ?? ""), VALUE: valueText(s.value) } };
    case "use":
      return { type, fields: { RESULT: s.result == null ? "" : String(s.result), PORT: String(s.port ?? "") } };
    case "wait":
      return { type, fields: { DURATION: String(s.duration ?? "1 s") } };
    case "if": {
      const block: BlockJson = { type, fields: { LEFT: valueText(s.left), OP: String(s.op ?? ">"), RIGHT: valueText(s.right) } };
      withSteps(block, "THEN", s.then);
      return withSteps(block, "ELSE", s.else);
    }
    case "save":
      return { type, fields: { RESULTS: namesText(s.results), ARCHIVE: String(s.archive ?? "") } };
    case "show":
      return { type, fields: { RESULTS: namesText(s.results) } };
    default:
      return { type: "lumos_unknown", fields: { JSON: JSON.stringify(step) } };
  }
}

export function procedureToWorkspace(procedure: ProcedureJson): WorkspaceJson {
  const top = withSteps(
    { type: "lumos_procedure", x: 20, y: 20, fields: { NAME: procedure.name, DESCRIPTION: procedure.description ?? "" } },
    "STEPS",
    procedure.steps,
  );
  return { blocks: { languageVersion: 0, blocks: [top] } };
}

// --- Workspace -> procedure -----------------------------------------------------------------------------

function chainSteps(first: { block: BlockJson } | undefined): Step[] {
  const steps: Step[] = [];
  for (let link = first; link; link = link.block.next) {
    const step = blockStep(link.block);
    if (step) steps.push(step);
  }
  return steps;
}

const field = (block: BlockJson, name: string) => String(block.fields?.[name] ?? "");

export function blockStep(block: BlockJson): Step | null {
  const inner = (name: string) => chainSteps(block.inputs?.[name]);
  switch (block.type) {
    case "lumos_action":
      return {
        do: "action",
        element: field(block, "ELEMENT"),
        action: field(block, "ACTION"),
        args: textList(field(block, "ARGS")),
        kwargs: textPairs(field(block, "KWARGS")),
      };
    case "lumos_measure":
      return { do: "measure", outputs: textNames(field(block, "OUTPUTS")), as: field(block, "AS"), collect: block.fields.COLLECT === true || block.fields.COLLECT === "TRUE" };
    case "lumos_repeat":
      return { do: "repeat", times: Number(block.fields.TIMES ?? 1), steps: inner("STEPS") };
    case "lumos_for_each": {
      const values =
        field(block, "MODE") === "range"
          ? {
              start: textValue(field(block, "START")),
              stop: textValue(field(block, "STOP")),
              num: Number(block.fields.NUM ?? 2),
              log: block.fields.LOG === true || block.fields.LOG === "TRUE",
            }
          : textList(field(block, "VALUES"));
      const parallel = block.fields.PARALLEL === true || block.fields.PARALLEL === "TRUE";
      return { do: "for_each", values, as: field(block, "AS"), steps: inner("STEPS"), ...(parallel ? { parallel: true } : {}) };
    }
    case "lumos_process":
      return { do: "process", function: field(block, "FUNCTION"), args: textPairs(field(block, "ARGS")), as: field(block, "AS") };
    case "lumos_set":
      return { do: "set", target: field(block, "TARGET"), value: textValue(field(block, "VALUE")) };
    case "lumos_use":
      return { do: "use", result: field(block, "RESULT") || null, port: field(block, "PORT") };
    case "lumos_wait":
      return { do: "wait", duration: field(block, "DURATION") };
    case "lumos_if":
      return { do: "if", left: textValue(field(block, "LEFT")), op: field(block, "OP"), right: textValue(field(block, "RIGHT")), then: inner("THEN"), else: inner("ELSE") };
    case "lumos_save":
      return { do: "save", results: textNames(field(block, "RESULTS")), archive: field(block, "ARCHIVE") };
    case "lumos_show":
      return { do: "show", results: textNames(field(block, "RESULTS")) };
    case "lumos_unknown":
      return JSON.parse(field(block, "JSON")) as Step;
    default:
      return null;
  }
}

/** The procedure of a workspace: its procedure block (blocks left aside are not part of it). */
export function workspaceToProcedure(workspace: WorkspaceJson): ProcedureJson {
  const top = workspace.blocks?.blocks?.find((block) => block.type === "lumos_procedure");
  if (!top) return { name: "", description: "", steps: [] };
  return { name: field(top, "NAME"), description: field(top, "DESCRIPTION"), steps: chainSteps(top.inputs?.STEPS) };
}

/** A new step of a kind, as added from the quick buttons of the tab. */
export function defaultStep(kind: StepKind): Step {
  const defaults: Record<StepKind, Step> = {
    action: { do: "action", element: "", action: "", args: [], kwargs: {} },
    measure: { do: "measure", outputs: [], as: "result", collect: false },
    repeat: { do: "repeat", times: 2, steps: [] },
    for_each: { do: "for_each", values: [], as: "value", steps: [] },
    process: { do: "process", function: "mean", args: {}, as: "processed" },
    set: { do: "set", target: "", value: "" },
    use: { do: "use", result: null, port: "" },
    wait: { do: "wait", duration: "1 s" },
    if: { do: "if", left: "", op: ">", right: "", then: [], else: [] },
    save: { do: "save", results: [], archive: "" },
    show: { do: "show", results: [] },
  };
  return structuredClone(defaults[kind]);
}
