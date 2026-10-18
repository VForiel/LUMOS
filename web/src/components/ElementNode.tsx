// A block of the graph: an element with its ports (colour = signal), a few values and its problems.

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { AlertTriangle, Dices, Layers } from "lucide-react";
import { memo } from "react";
import { useTranslation } from "react-i18next";
import type { ElementSummary, ElementType, Issue } from "../api";
import { category, categoryOf } from "../lib/categories";
import { portsOf, type Port } from "../lib/schema";
import { signalColor } from "../lib/signals";

export type ElementData = {
  element: ElementSummary;
  type?: ElementType;
  issues: Issue[];
  running: boolean;
  onInspect?: (port: string) => void;
  /** Name shown instead of the path (elements inside an expanded group). */
  label?: string;
  group?: boolean;
};

export type ElementFlowNode = Node<ElementData, "element">;

const PORT_GAP = 26;
const HEADER = 54;

function ports(element: ElementSummary, type?: ElementType): { inputs: Port[]; outputs: Port[] } {
  const declared = type ? portsOf(type.schema) : { inputs: [], outputs: [] };
  const byName = (list: Port[], names: string[]) =>
    names.map((name) => list.find((p) => p.name === name) ?? { name, signal: "Signal", multiple: false, doc: "" });
  return { inputs: byName(declared.inputs, element.inputs), outputs: byName(declared.outputs, element.outputs) };
}

function preview(element: ElementSummary): [string, string][] {
  return Object.entries(element.params)
    .filter(([, value]) => typeof value === "string" && value.length < 22)
    .slice(0, 3) as [string, string][];
}

export const ElementNode = memo(function ElementNode({ data, selected }: NodeProps<ElementFlowNode>) {
  const { t } = useTranslation();
  const { element, type, issues } = data;
  const group = Boolean(data.group) || element.type_id.startsWith("lumos.group");
  const name = group ? "group" : categoryOf(element.type_id);
  const { icon: Icon, color } = category(name);
  const { inputs, outputs } = ports(element, type);
  const rows = Math.max(inputs.length, outputs.length, 1);
  const values = preview(element);
  const errors = issues.filter((issue) => issue.severity === "error").length;
  const title = type?.title ?? element.type_id.split(".").pop();

  return (
    <div
      className={`w-[210px] overflow-hidden rounded-xl border bg-white text-left shadow-sm transition-shadow dark:bg-slate-900 ${
        selected ? "shadow-lg" : "hover:shadow-md"
      } ${errors ? "border-red-300 dark:border-red-800" : "border-slate-200 dark:border-slate-700"} ${
        data.running ? "animate-pulse" : ""
      }`}
      title={type?.summary}
    >
      <div className="h-1" style={{ background: color }} />
      <div className="flex items-center gap-2 px-3 pt-2 pb-1.5" style={{ minHeight: HEADER - 4 }}>
        <span className="grid size-7 shrink-0 place-items-center rounded-lg" style={{ background: `${color}22`, color }}>
          {group ? <Layers className="size-4" /> : <Icon className="size-4" />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm leading-tight font-semibold text-slate-800 dark:text-slate-100">
            {data.label ?? element.id}
          </span>
          <span className="block truncate text-[11px] leading-tight text-slate-500 dark:text-slate-400">{title}</span>
        </span>
        {Object.keys(element.free ?? {}).length > 0 && (
          <span data-testid="free-badge" title={t("node.free", { names: Object.keys(element.free ?? {}).join(", ") })} className="text-amber-500">
            <Dices className="size-3.5" />
          </span>
        )}
        {errors > 0 && (
          <span title={issues.map((issue) => issue.message).join("\n")} className="text-red-500">
            <AlertTriangle className="size-4" aria-label={t("node.problems", { count: errors })} />
          </span>
        )}
      </div>
      <div className="relative" style={{ height: rows * PORT_GAP + 6 }}>
        {inputs.map((port, index) => (
          <div
            key={`in-${port.name}`}
            className="absolute left-0 flex items-center pl-3 text-[11px] text-slate-500 dark:text-slate-400"
            style={{ top: index * PORT_GAP + 4, height: PORT_GAP - 6 }}
            title={`${port.name}: ${port.signal}${port.doc ? ` — ${port.doc}` : ""}`}
          >
            <Handle
              id={port.name}
              type="target"
              position={Position.Left}
              style={{ background: signalColor(port.signal), top: "50%" }}
            />
            {port.name}
          </div>
        ))}
        {outputs.map((port, index) => (
          <div
            key={`out-${port.name}`}
            className="absolute right-0 flex items-center pr-3 text-[11px] text-slate-500 dark:text-slate-400"
            style={{ top: index * PORT_GAP + 4, height: PORT_GAP - 6 }}
            title={`${port.name}: ${port.signal}${port.doc ? ` — ${port.doc}` : ""}`}
          >
            <button
              type="button"
              className="nodrag rounded px-1 hover:bg-accent-100 hover:text-accent-700 dark:hover:bg-accent-900/40 dark:hover:text-accent-300"
              onClick={(event) => {
                event.stopPropagation();
                data.onInspect?.(`${element.id}.${port.name}`);
              }}
              title={t("node.inspect", { port: `${element.id}.${port.name}` })}
            >
              {port.name}
            </button>
            <Handle
              id={port.name}
              type="source"
              position={Position.Right}
              style={{ background: signalColor(port.signal), top: "50%" }}
            />
          </div>
        ))}
      </div>
      {values.length > 0 && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-2 border-t border-slate-100 px-3 py-1.5 text-[11px] dark:border-slate-800">
          {values.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="truncate text-slate-400">{key}</dt>
              <dd className="truncate text-right font-mono text-slate-600 dark:text-slate-300">{value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
});
