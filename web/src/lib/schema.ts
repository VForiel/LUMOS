// Forms and ports generated from the JSON Schema of an element type (REQ-WEB-2): no code per element.

export type FieldKind = "param" | "setting" | "state";
export type FieldInput = "quantity" | "number" | "integer" | "boolean" | "enum" | "text" | "json";

export type Field = {
  name: string;
  kind: FieldKind;
  input: FieldInput;
  doc: string;
  unit?: string;
  dimension?: string;
  bounds?: [number | null, number | null];
  inclusive?: [boolean, boolean];
  choices?: unknown[];
  default?: unknown;
  free?: boolean;
};

/** A port of an element type; ``count`` names the setting repeating it (``name`` is then a pattern "in{i}"). */
export type Port = {
  name: string;
  signal: string;
  multiple: boolean;
  doc: string;
  /** Whether the port carries a bus of lanes (drawn as pins grouped by a brace). */
  bus?: boolean;
  /** Whether an input may stay unconnected (an empty lane for a Mux). */
  optional?: boolean;
  count?: string;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

function boundValue(bound: Json): number | null {
  if (bound === null || bound === undefined) return null;
  return typeof bound === "object" ? Number(bound.value) : Number(bound);
}

function fieldOf(name: string, kind: FieldKind, schema: Json): Field {
  const doc = String(schema.description ?? "");
  if (schema["x-lumos-unit"] !== undefined) {
    const shape = schema["x-lumos-shape"];
    const scalar = Array.isArray(shape) && shape.length === 0;
    const bounds = schema["x-lumos-bounds"] ?? [null, null];
    return {
      name,
      kind,
      doc,
      input: scalar ? "quantity" : "text",
      unit: String(schema["x-lumos-unit"]),
      dimension: schema["x-lumos-dimension"],
      bounds: [boundValue(bounds[0]), boundValue(bounds[1])],
      inclusive: schema["x-lumos-inclusive"] ?? [true, true],
      default: schema.default,
      free: Boolean(schema["x-lumos-free"]),
    };
  }
  const field: Field = { name, kind, doc, input: "json", default: schema.default };
  if (Array.isArray(schema.enum)) return { ...field, input: "enum", choices: schema.enum };
  switch (schema.type) {
    case "integer":
      return { ...field, input: "integer" };
    case "number":
      return { ...field, input: "number" };
    case "boolean":
      return { ...field, input: "boolean" };
    case "string":
      return { ...field, input: "text" };
  }
  const value = schema.default;
  if (value && typeof value === "object" && "value" in value && typeof value.unit === "string") {
    return { ...field, input: "quantity", unit: value.unit };
  }
  return field;
}

/** Fields of an element type: parameters, then settings, then state (read-only). */
export function fieldsOf(schema: Json): Field[] {
  const groups: [string, FieldKind][] = [
    ["params", "param"],
    ["settings", "setting"],
    ["state", "state"],
  ];
  const fields: Field[] = [];
  for (const [group, kind] of groups) {
    const properties = schema?.properties?.[group]?.properties ?? {};
    for (const [name, field] of Object.entries(properties)) fields.push(fieldOf(name, kind, field));
  }
  return fields;
}

/** Inputs and outputs of an element type with their signal classes. */
export function portsOf(schema: Json): { inputs: Port[]; outputs: Port[] } {
  const ports = schema?.["x-lumos-ports"] ?? {};
  return { inputs: ports.inputs ?? [], outputs: ports.outputs ?? [] };
}

/** The declaration of the port ``name`` in ``ports``: by name, or a numbered port of a pattern "in{i}". */
export function portNamed(ports: Port[], name: string): Port | undefined {
  const exact = ports.find((port) => port.name === name);
  if (exact) return exact;
  for (const port of ports) {
    const at = port.name.indexOf("{i}");
    if (at < 0) continue;
    const prefix = port.name.slice(0, at);
    const suffix = port.name.slice(at + 3);
    const middle = name.slice(prefix.length, name.length - suffix.length);
    if (name.startsWith(prefix) && name.endsWith(suffix) && /^\d+$/.test(middle)) return { ...port, name };
  }
  return undefined;
}

/** "below" or "above" when ``value`` (in the unit of the field) leaves its bounds, else null. */
export function checkBounds(field: Field, value: number): "below" | "above" | null {
  if (!field.bounds) return null;
  const [low, high] = field.bounds;
  const [lowIn, highIn] = field.inclusive ?? [true, true];
  if (low !== null && (lowIn ? value < low : value <= low)) return "below";
  if (high !== null && (highIn ? value > high : value >= high)) return "above";
  return null;
}
