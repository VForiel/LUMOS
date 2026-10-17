// Signal carried by a port of an element of a context; for a group, the signal of the inner port it exposes.

import type { ContextSummary, ElementType } from "../api";
import { portsOf } from "./schema";

export function splitPort(reference: string): [string, string] {
  const index = reference.lastIndexOf(".");
  return [reference.slice(0, index), reference.slice(index + 1)];
}

export function signalOfPort(
  context: ContextSummary | null,
  types: Map<string, ElementType>,
  path: string,
  port: string,
  direction: "inputs" | "outputs",
): string {
  if (!context) return "Signal";
  const group = context.groups?.[path];
  if (group) {
    const inner = group.exposed[port];
    if (!inner) return "Signal";
    const [child, childPort] = splitPort(inner);
    return signalOfPort(context, types, `${path}.${child}`, childPort, direction);
  }
  const element = context.elements.find((e) => e.id === path);
  const type = element && types.get(element.type_id);
  return (type && portsOf(type.schema)[direction].find((p) => p.name === port)?.signal) ?? "Signal";
}
