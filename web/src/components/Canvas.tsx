// The graph editor (REQ-WEB-1): blocks, typed connections coloured by signal, drag and drop from the palette.

import {
  applyNodeChanges,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
  type Connection,
  type Edge,
  type IsValidConnection,
  type NodeChange,
} from "@xyflow/react";
import { MousePointerClick } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import { category, categoryOf } from "../lib/categories";
import { portsOf } from "../lib/schema";
import { compatible, signalColor } from "../lib/signals";
import { useStore } from "../store";
import { ElementNode, type ElementFlowNode } from "./ElementNode";
import { DRAG_TYPE } from "./Palette";

const nodeTypes = { element: ElementNode };

/** "laser.out" → ["laser", "out"] (element paths may contain dots inside groups). */
export function splitPort(reference: string): [string, string] {
  const index = reference.lastIndexOf(".");
  return [reference.slice(0, index), reference.slice(index + 1)];
}

export function Canvas({ onInspect }: { onInspect: (port: string) => void }) {
  const { t } = useTranslation();
  const flow = useReactFlow();
  const context = useStore((s) => s.context);
  const palette = useStore((s) => s.palette);
  const positions = useStore((s) => s.positions);
  const selection = useStore((s) => s.selection);
  const issues = useStore((s) => s.issues);
  const jobs = useStore((s) => s.jobs);
  const role = useStore((s) => s.info?.role);
  const { addElement, connect, disconnect, removeElements, move, select } = useStore.getState();
  const [nodes, setNodes] = useState<ElementFlowNode[]>([]);
  const fitted = useRef<string | null>(null);

  const types = useMemo(() => new Map((palette?.types ?? []).map((item) => [item.type_id, item])), [palette]);
  const running = Object.values(jobs).some(
    (job) => job.context === context?.id && (job.status === "running" || job.status === "queued"),
  );
  const editable = role !== "viewer";

  useEffect(() => {
    const elements = (context?.elements ?? []).filter((element) => !element.id.includes("."));
    setNodes((previous) => {
      const known = new Map(previous.map((node) => [node.id, node]));
      return elements.map((element) => {
        const old = known.get(element.id);
        const [x, y] = positions[element.id] ?? [0, 0];
        return {
          ...(old ?? {}),
          id: element.id,
          type: "element" as const,
          position: { x, y },
          selected: selection.includes(element.id),
          data: {
            element,
            type: types.get(element.type_id),
            issues: issues.filter((issue) => issue.element === element.id),
            running,
            onInspect,
          },
        };
      });
    });
  }, [context, positions, selection, issues, types, running, onInspect]);

  useEffect(() => {
    if (context && nodes.length && fitted.current !== context.id) {
      fitted.current = context.id;
      window.requestAnimationFrame(() => void flow.fitView({ padding: 0.15, duration: 300, maxZoom: 1 }));
    }
  }, [context, nodes.length, flow]);

  const signalOf = useCallback(
    (node: string, handle: string, direction: "inputs" | "outputs") => {
      const element = context?.elements.find((e) => e.id === node);
      const type = element && types.get(element.type_id);
      const port = type ? portsOf(type.schema)[direction].find((p) => p.name === handle) : undefined;
      return port?.signal ?? "Signal";
    },
    [context, types],
  );

  const edges: Edge[] = useMemo(
    () =>
      (context?.connections ?? []).map(({ source, target }) => {
        const [sourceNode, sourceHandle] = splitPort(source);
        const [targetNode, targetHandle] = splitPort(target);
        const color = signalColor(signalOf(sourceNode, sourceHandle, "outputs"));
        return {
          id: `${source}->${target}`,
          source: sourceNode,
          sourceHandle,
          target: targetNode,
          targetHandle,
          className: running ? "running" : undefined,
          style: { stroke: color },
          data: { source, target },
        };
      }),
    [context, signalOf, running],
  );

  const isValidConnection: IsValidConnection = useCallback(
    (connection) => {
      if (!palette || connection.source === connection.target) return false;
      const from = signalOf(connection.source, connection.sourceHandle ?? "", "outputs");
      const to = signalOf(connection.target, connection.targetHandle ?? "", "inputs");
      return compatible(from, to, palette.signals);
    },
    [palette, signalOf],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<ElementFlowNode>[]) => {
      setNodes((current) => applyNodeChanges(changes, current));
      for (const change of changes) {
        if (change.type === "position" && change.position && !change.dragging) {
          move(change.id, [Math.round(change.position.x), Math.round(change.position.y)]);
        }
      }
    },
    [move],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      void connect(`${connection.source}.${connection.sourceHandle}`, `${connection.target}.${connection.targetHandle}`);
    },
    [connect],
  );

  const onDrop = useCallback(
    (event: DragEvent) => {
      const typeId = event.dataTransfer.getData(DRAG_TYPE);
      if (!typeId) return;
      event.preventDefault();
      const point = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      void addElement(typeId, [Math.round(point.x - 115), Math.round(point.y - 30)]);
    },
    [flow, addElement],
  );

  const empty = (context?.elements.length ?? 0) === 0;

  return (
    <div data-testid="canvas" data-tour="canvas" className="relative min-h-0 flex-1">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        onSelectionChange={({ nodes: chosen }) => select(chosen.map((node) => node.id))}
        onNodesDelete={(deleted) => void removeElements(deleted.map((node) => node.id))}
        onEdgesDelete={(deleted) => {
          for (const edge of deleted) {
            const data = edge.data as { source: string; target: string };
            void disconnect(data.source, data.target);
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDrop={onDrop}
        deleteKeyCode={editable ? ["Delete", "Backspace"] : null}
        nodesConnectable={editable}
        multiSelectionKeyCode={["Shift", "Meta", "Control"]}
        selectionKeyCode="Shift"
        snapToGrid
        snapGrid={[10, 10]}
        minZoom={0.15}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{ interactionWidth: 18 }}
        connectionRadius={28}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} color="var(--canvas-dots)" />
        <Controls showInteractive={false} position="bottom-left" />
        <MiniMap
          pannable
          zoomable
          position="bottom-right"
          className="!rounded-lg !border !border-slate-200 dark:!border-slate-700"
          nodeColor={(node) => category(categoryOf((node as ElementFlowNode).data.element.type_id)).color}
          nodeStrokeWidth={0}
        />
      </ReactFlow>
      {empty && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <div className="max-w-sm rounded-2xl border border-dashed border-slate-300 bg-white/80 p-6 text-center backdrop-blur dark:border-slate-700 dark:bg-slate-900/80">
            <MousePointerClick className="mx-auto mb-2 size-8 text-accent-500" />
            <p className="font-semibold text-slate-700 dark:text-slate-200">{t("canvas.empty")}</p>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{t("canvas.hint")}</p>
          </div>
        </div>
      )}
    </div>
  );
}
