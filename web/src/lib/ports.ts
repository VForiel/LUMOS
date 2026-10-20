// Signal carried by a port of an element of a context; for a group, the signal of the inner port it exposes.

import type { ContextSummary, ElementType } from "../api";
import { portNamed, portsOf, type Port } from "./schema";
import { ANY_OPTICAL } from "./signals";

export function splitPort(reference: string): [string, string] {
  const index = reference.lastIndexOf(".");
  return [reference.slice(0, index), reference.slice(index + 1)];
}

/** Declaration of a port of an element of a context; for a group, that of the inner port it exposes. */
export function portOf(
  context: ContextSummary | null,
  types: Map<string, ElementType>,
  path: string,
  port: string,
  direction: "inputs" | "outputs",
): Port | undefined {
  if (!context) return undefined;
  const group = context.groups?.[path];
  if (group) {
    const inner = group.exposed[port];
    if (!inner) return undefined;
    const [child, childPort] = splitPort(inner);
    return portOf(context, types, `${path}.${child}`, childPort, direction);
  }
  const element = context.elements.find((e) => e.id === path);
  const type = element && types.get(element.type_id);
  return type ? portNamed(portsOf(type.schema)[direction], port) : undefined;
}

export function signalOfPort(
  context: ContextSummary | null,
  types: Map<string, ElementType>,
  path: string,
  port: string,
  direction: "inputs" | "outputs",
): string {
  const declared = portOf(context, types, path, port, direction);
  if (declared?.generic) return ANY_OPTICAL;
  return declared?.signal ?? "Signal";
}
