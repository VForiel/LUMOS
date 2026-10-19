// Free targets (ADR-0022): what a sweep or a Monte-Carlo run may vary. A physical parameter, a numeric
// setting, or a component "field[i]" / "field[i, j]" of an array parameter, read from the values the server
// writes ("[0.25, 0.5] nm", "[[1, 0], [0, 1]]").

import type { Field } from "./schema";

export type ArrayValue = { values: number[]; shape: number[]; unit: string };
export type Component = { name: string; value: number; unit: string };

/** A full array written by the server, flattened in C order; null for scalars and truncated arrays. */
export function parseArray(text: unknown): ArrayValue | null {
  if (typeof text !== "string") return null;
  const match = /^\s*(\[.*\])\s*([^\]]*)$/s.exec(text);
  if (!match || match[2].includes("...") || match[2].includes("(")) return null;
  let data: unknown;
  try {
    data = JSON.parse(match[1]);
  } catch {
    return null;
  }
  const shape: number[] = [];
  for (let level: unknown = data; Array.isArray(level); level = level[0]) shape.push(level.length);
  const values: number[] = [];
  const walk = (node: unknown, depth: number): boolean => {
    if (depth === shape.length) {
      if (typeof node !== "number") return false;
      values.push(node);
      return true;
    }
    return Array.isArray(node) && node.length === shape[depth] && node.every((child) => walk(child, depth + 1));
  };
  if (!shape.length || shape.includes(0) || !walk(data, 0)) return null;
  return { values, shape, unit: match[2].trim() };
}

/** The components of an array value, named "field[i]" or "field[i, j]" (C order). */
export function componentsOf(field: string, text: unknown): Component[] {
  const array = parseArray(text);
  if (!array) return [];
  return array.values.map((value, flat) => {
    const index: number[] = [];
    let rest = flat;
    for (let k = array.shape.length - 1; k >= 0; k -= 1) {
      index.unshift(rest % array.shape[k]);
      rest = Math.floor(rest / array.shape[k]);
    }
    return { name: `${field}[${index.join(", ")}]`, value, unit: array.unit };
  });
}

/** A setting a sweep or a Monte-Carlo run may vary: an integer or a real number, without a list of choices. */
export function isNumericSetting(field: Field): boolean {
  return field.kind === "setting" && (field.input === "integer" || field.input === "number") && !field.choices?.length;
}

/** An array parameter, whose components may be free. */
export function isArrayParam(field: Field): boolean {
  return field.kind === "param" && field.input === "text" && field.unit !== undefined;
}
