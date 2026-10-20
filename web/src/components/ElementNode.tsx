// A block of the graph: an element with its ports (colour = signal), a few values and its problems.

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { AlertTriangle, Dices, Layers, ScanSearch } from "lucide-react";
import { memo, type CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import type { ElementSummary, ElementType, Issue } from "../api";
import { category, categoryOf } from "../lib/categories";
import { portNamed, portsOf, type Port } from "../lib/schema";
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
/** Spacing of the pins of a bus port, and the most pins drawn (a wider bus shows its count). */
const PIN_GAP = 7;
const MAX_PINS = 8;

function ports(element: ElementSummary, type?: ElementType): { inputs: Port[]; outputs: Port[] } {
  const declared = type ? portsOf(type.schema) : { inputs: [], outputs: [] };
  const byName = (list: Port[], names: string[]) =>
    names.map((name) => portNamed(list, name) ?? { name, signal: "Signal", multiple: false, doc: "" });
  return { inputs: byName(declared.inputs, element.inputs), outputs: byName(declared.outputs, element.outputs) };
}

/** Height of the row of a port: a bus of known width needs room for its pins. */
function rowHeight(port: Port, lanes?: number): number {
  if (!port.bus || port.count || !lanes || lanes < 2) return PORT_GAP; // a repeated port is one lane
  return Math.max(PORT_GAP, Math.min(lanes, MAX_PINS) * PIN_GAP + 12);
}

function offsets(list: Port[], lanes: Record<string, number>): number[] {
  const tops: number[] = [];
  let top = 4;
  for (const port of list) {
    tops.push(top);
    top += rowHeight(port, lanes[port.name]);
  }
  return tops;
}

/** Pins of a bus port on the edge of the block, gathered by a brace into the handle (one link carries them all). */
function BusPins({ side, lanes, height, color }: { side: "left" | "right"; lanes?: number; height: number; color: string }) {
  const n = lanes && lanes > 1 ? Math.min(lanes, MAX_PINS) : 0;
  const middle = height / 2;
  const span = (n - 1) * PIN_GAP;
  const top = middle - span / 2;
  const mirror = side === "right" ? "scale(-1 1) translate(-18 0)" : undefined;
  return (
    <svg
      data-testid="bus-pins"
      data-lanes={lanes ?? ""}
      aria-hidden
      className="pointer-events-none absolute overflow-visible"
      style={{ [side]: -18, top: 0, width: 18, height }}
    >
      <g transform={mirror} stroke={color} fill="none" strokeWidth={1.5} strokeLinecap="round">
        {n > 0 ? (
          <>
            {Array.from({ length: n }, (_, k) => (
              <line key={k} x1={0} x2={6} y1={top + k * PIN_GAP} y2={top + k * PIN_GAP} />
            ))}
            {/* brace from the first to the last pin, its tip on the handle */}
            <path
              d={`M6 ${top} Q10 ${top} 10 ${top + 4} L10 ${middle - 3} Q10 ${middle} 14 ${middle} Q10 ${middle} 10 ${middle + 3} L10 ${top + span - 4} Q10 ${top + span} 6 ${top + span}`}
            />
          </>
        ) : (
          // a bus of any number of lanes: a double stroke
          <>
            <line x1={2} x2={14} y1={middle - 2} y2={middle - 2} />
            <line x1={2} x2={14} y1={middle + 2} y2={middle + 2} />
          </>
        )}
      </g>
      {lanes && lanes > MAX_PINS && (
        <text x={side === "left" ? 0 : 18} y={top + span + 11} fontSize={8} fill={color} textAnchor={side === "left" ? "start" : "end"}>
          {lanes}
        </text>
      )}
    </svg>
  );
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
  const lanes = element.lanes ?? {};
  const inputTops = offsets(inputs, lanes);
  const outputTops = offsets(outputs, lanes);
  const bottom = (list: Port[], tops: number[]) =>
    list.length ? tops[list.length - 1] + rowHeight(list[list.length - 1], lanes[list[list.length - 1].name]) : PORT_GAP + 4;
  const height = Math.max(bottom(inputs, inputTops), bottom(outputs, outputTops)) + 2;
  const values = preview(element);
  const errors = issues.filter((issue) => issue.severity === "error").length;
  const title = type?.title ?? element.type_id.split(".").pop();

  return (
    <div
      className={`w-[210px] rounded-xl border bg-white text-left shadow-sm transition-shadow dark:bg-slate-900 ${
        selected ? "shadow-lg" : "hover:shadow-md"
      } ${errors ? "border-red-300 dark:border-red-800" : "border-slate-200 dark:border-slate-700"} ${
        data.running ? "animate-pulse" : ""
      }`}
      title={type?.summary}
    >
      <div className="h-1 rounded-t-xl" style={{ background: color }} />
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
        {outputs.length > 0 && data.onInspect && (
          <button
            type="button"
            data-testid="node-inspect"
            className="nodrag -mr-1 grid size-6 shrink-0 place-items-center rounded-md text-slate-400 hover:bg-accent-100 hover:text-accent-700 dark:hover:bg-accent-900/40 dark:hover:text-accent-300"
            onClick={(event) => {
              event.stopPropagation();
              data.onInspect?.(`${element.id}.${outputs[0].name}`);
            }}
            title={t("node.inspect", { port: `${element.id}.${outputs[0].name}` })}
            aria-label={t("node.inspect", { port: `${element.id}.${outputs[0].name}` })}
          >
            <ScanSearch className="size-3.5" />
          </button>
        )}
      </div>
      <div className="relative" style={{ height }}>
        {/* the whole label of a port is its handle: drag it, or click it then another port, to connect */}
        {inputs.map((port, index) => (
          <div
            key={`target-${port.name}`}
            className="pointer-events-none absolute inset-x-0"
            style={{ top: inputTops[index], height: rowHeight(port, lanes[port.name]) - 6 }}
          >
            {port.bus && !port.count && (
              <BusPins side="left" lanes={lanes[port.name]} height={rowHeight(port, lanes[port.name]) - 6} color={signalColor(port.signal)} />
            )}
            <Handle
              id={port.name}
              type="target"
              position={Position.Left}
              className="lumos-port"
              style={{ "--port-color": signalColor(port.signal) } as CSSProperties}
              title={`${port.name}: ${port.bus ? `bus of ${lanes[port.name] ?? "N"} × ` : ""}${port.signal}${port.doc ? ` — ${port.doc}` : ""}`}
            >
              {port.name}
            </Handle>
          </div>
        ))}
        {outputs.map((port, index) => (
          <div
            key={`source-${port.name}`}
            className="pointer-events-none absolute inset-x-0"
            style={{ top: outputTops[index], height: rowHeight(port, lanes[port.name]) - 6 }}
          >
            {port.bus && !port.count && (
              <BusPins side="right" lanes={lanes[port.name]} height={rowHeight(port, lanes[port.name]) - 6} color={signalColor(port.signal)} />
            )}
            <Handle
              id={port.name}
              type="source"
              position={Position.Right}
              className="lumos-port"
              style={{ "--port-color": signalColor(port.signal) } as CSSProperties}
              title={`${port.name}: ${port.bus ? `bus of ${lanes[port.name] ?? "N"} × ` : ""}${port.signal}${port.doc ? ` — ${port.doc}` : ""}`}
            >
              {port.name}
            </Handle>
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
