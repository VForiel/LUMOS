// The graph editor (REQ-WEB-1, REQ-WEB-4): blocks, typed connections coloured by signal, drag and drop from
// the palette, groups shown collapsed (one block) or expanded (a frame holding their elements).

import {
  applyNodeChanges,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  SelectionMode,
  useReactFlow,
  useUpdateNodeInternals,
  type Connection,
  type Edge,
  type EdgeChange,
  type FinalConnectionState,
  type IsValidConnection,
  type NodeChange,
} from "@xyflow/react";
import { BoxSelect, Hand, MousePointerClick, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import type { ContextSummary } from "../api";
import { category, categoryOf } from "../lib/categories";
import { autoLayout } from "../lib/layout";
import { portOf, signalOfPort, splitPort } from "../lib/ports";
import { compatible, signalColor } from "../lib/signals";
import { useStore } from "../store";
import { attachedCentre, ElementNode, portCentre, type ElementFlowNode } from "./ElementNode";
import { FRAME_HEADER, FRAME_PAD, GroupFrame, type FrameFlowNode } from "./GroupFrame";
import { FeedbackEdge, FeedbackMarker } from "./FeedbackEdge";
import { LinkEdge } from "./LinkEdge";
import { BLOCK_TYPE, DRAG_TYPE } from "./Palette";
import { PresetGallery } from "./Presets";

const nodeTypes = { element: ElementNode, frame: GroupFrame };
const edgeTypes = { link: LinkEdge, feedback: FeedbackEdge };
const PAN_BUTTONS = [1, 2]; // in selection mode, the middle and right buttons still move the view
const NODE_WIDTH = 210;
/** Width of a Mux or Demux drawn against its block, and the gap between them (P15.10). */
const ATTACHED_WIDTH = 78;
const ATTACHED_GAP = 14;
const NODE_HEIGHT = 170;
const NONE: Record<string, boolean> = {};
const DELETE_KEYS = ["Delete", "Backspace"];
const MULTI_KEYS = ["Shift", "Meta", "Control"];
const SNAP: [number, number] = [10, 10];
const EDGE_OPTIONS = { interactionWidth: 18 };

type FlowNode = ElementFlowNode | FrameFlowNode;

export { splitPort };

/** Positions of the elements of an expanded group, relative to its frame. */
function insideLayout(context: ContextSummary, group: string): Record<string, [number, number]> {
  const children = context.elements.filter((e) => e.id.startsWith(`${group}.`) && !e.id.slice(group.length + 1).includes("."));
  const saved = context.ui.positions ?? {};
  const names = children.map((c) => c.id.slice(group.length + 1));
  let layout: Record<string, [number, number]>;
  if (names.every((name) => saved[`${group}.${name}`])) {
    layout = Object.fromEntries(names.map((name) => [name, saved[`${group}.${name}`]]));
  } else {
    const edges = (context.groups[group]?.connections ?? []).map((c) => [splitPort(c.source)[0], splitPort(c.target)[0]] as [string, string]);
    layout = autoLayout(names, edges);
  }
  const xs = Object.values(layout).map(([x]) => x);
  const ys = Object.values(layout).map(([, y]) => y);
  const [x0, y0] = [Math.min(...xs), Math.min(...ys)];
  return Object.fromEntries(Object.entries(layout).map(([name, [x, y]]) => [name, [x - x0 + FRAME_PAD + 70, y - y0 + FRAME_HEADER]]));
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
  const { addElement, connect, disconnect, removeElements, moved, select, setQuickAdd, insertBlock } = useStore.getState();
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const fitted = useRef<string | null>(null);
  const welcome = useStore((s) => s.welcome);
  const pointer = useStore((s) => s.pointer);
  const [selectedEdges, setSelectedEdges] = useState<ReadonlySet<string>>(new Set());

  const types = useMemo(() => new Map((palette?.types ?? []).map((item) => [item.type_id, item])), [palette]);
  const running = Object.values(jobs).some((job) => job.context === context?.id && (job.status === "running" || job.status === "queued"));
  const editable = role !== "viewer";
  const expanded = context?.ui.expanded ?? NONE;

  const signalOf = useCallback(
    (path: string, port: string, direction: "inputs" | "outputs") => signalOfPort(context, types, path, port, direction),
    [context, types],
  );

  useEffect(() => {
    if (!context) {
      setNodes([]);
      return;
    }
    const built: FlowNode[] = [];
    const later: FlowNode[] = [];
    const nodeIssues = (id: string) => issues.filter((issue) => issue.element === id);
    const attached = (context.ui.attached ?? {}) as Record<string, { to: string; port: string; side: "in" | "out" }>;
    const attachedTo = (id: string) => {
      const owner = attached[id]?.to;
      return Boolean(owner && context.elements.some((e) => e.id === owner) && !(context.groups?.[owner] && expanded[owner]));
    };
    for (const element of context.elements.filter((e) => !e.id.includes("."))) {
      const [x, y] = positions[element.id] ?? [0, 0];
      const isGroup = Boolean(context.groups?.[element.id]);
      if (isGroup && expanded[element.id]) {
        const inside = insideLayout(context, element.id);
        const xs = Object.values(inside).map(([cx]) => cx);
        const ys = Object.values(inside).map(([, cy]) => cy);
        const signals = Object.fromEntries([
          ...element.inputs.map((p) => [p, signalOf(element.id, p, "inputs")]),
          ...element.outputs.map((p) => [p, signalOf(element.id, p, "outputs")]),
        ]);
        built.push({
          id: element.id,
          type: "frame",
          position: { x, y },
          selected: selection.includes(element.id),
          zIndex: -1,
          data: {
            element,
            signals,
            width: Math.max(...xs, 0) + NODE_WIDTH + FRAME_PAD + 70,
            height: Math.max(...ys, 0) + NODE_HEIGHT + FRAME_PAD,
          },
        });
        for (const child of context.elements.filter((e) => e.id.startsWith(`${element.id}.`) && !e.id.slice(element.id.length + 1).includes("."))) {
          const name = child.id.slice(element.id.length + 1);
          const [cx, cy] = inside[name] ?? [FRAME_PAD, FRAME_HEADER];
          built.push({
            id: child.id,
            type: "element",
            parentId: element.id,
            extent: "parent",
            position: { x: cx, y: cy },
            connectable: false,
            deletable: false,
            selected: selection.includes(child.id),
            data: { element: child, type: types.get(child.type_id), issues: nodeIssues(child.id), running, onInspect, label: name },
          });
        }
      } else if (attachedTo(element.id)) {
        // a split of a bus port (P15.10): drawn against its block, moving and deleted with it
        const { to, port, side } = attached[element.id];
        const owner = context.elements.find((e) => e.id === to);
        const index = Math.max(0, (side === "in" ? owner?.inputs : owner?.outputs)?.indexOf(port) ?? 0);
        const count = Math.max(element.inputs.length, element.outputs.length);
        later.push({
          id: element.id,
          type: "element",
          parentId: to,
          position: {
            x: side === "in" ? -ATTACHED_WIDTH - ATTACHED_GAP : NODE_WIDTH + ATTACHED_GAP,
            y: portCentre(index) - attachedCentre(count),
          },
          draggable: false,
          selected: selection.includes(element.id),
          data: { element, type: types.get(element.type_id), issues: nodeIssues(element.id), running, onInspect, attached: side },
        });
      } else {
        built.push({
          id: element.id,
          type: "element",
          position: { x, y },
          selected: selection.includes(element.id),
          data: {
            element,
            type: types.get(element.type_id),
            issues: nodeIssues(element.id),
            running,
            onInspect,
            group: isGroup,
            split: Object.values(attached)
              .filter((entry) => entry.to === element.id)
              .map((entry) => entry.port),
          },
        });
      }
    }
    built.push(...later); // React Flow wants a parent before its children
    setNodes((previous) => {
      const known = new Map(previous.map((node) => [node.id, node]));
      return built.map((node) => {
        const old = known.get(node.id);
        return old && old.type === node.type ? ({ ...old, ...node, measured: old.measured } as FlowNode) : node;
      });
    });
  }, [context, positions, selection, issues, types, running, onInspect, expanded, signalOf]);

  // ports may move when a context changes (values shown, problems, groups): measure the handles again
  const updateNodeInternals = useUpdateNodeInternals();
  const nodeIds = nodes.map((node) => node.id).join(",");
  useEffect(() => {
    if (!nodeIds) return;
    const frame = window.requestAnimationFrame(() => updateNodeInternals(nodeIds.split(",")));
    return () => window.cancelAnimationFrame(frame);
  }, [context, nodeIds, updateNodeInternals]);

  useEffect(() => {
    // fit the view to a context opened with elements; a context built from empty keeps the view of the user
    if (!context || fitted.current === context.id) return;
    if (context.elements.length === 0) {
      fitted.current = context.id;
      return;
    }
    if (nodes.length) {
      fitted.current = context.id;
      window.requestAnimationFrame(() => void flow.fitView({ padding: 0.15, duration: 300, maxZoom: 1 }));
    }
  }, [context, nodes.length, flow]);

  const edges: Edge[] = useMemo(() => {
    if (!context) return [];
    const list: Edge[] = context.connections.map(({ source, target }) => {
      const [sourceNode, sourceHandle] = splitPort(source);
      const [targetNode, targetHandle] = splitPort(target);
      return {
        id: `${source}->${target}`,
        type: "link",
        selected: selectedEdges.has(`${source}->${target}`),
        source: sourceNode,
        sourceHandle,
        target: targetNode,
        targetHandle,
        className: running ? "running" : undefined,
        style: {
          stroke: signalColor(signalOf(sourceNode, sourceHandle, "outputs")),
          // a bus (several lanes on one link) is drawn thicker
          ...(portOf(context, types, sourceNode, sourceHandle, "outputs")?.bus ? { strokeWidth: 3.5 } : {}),
        },
        // the button removing a link shows on a link selected alone, not on those of a box selection
        data: { source, target, removable: selectedEdges.size === 1 && selection.length === 0 },
      };
    });
    const shown = new Set(context.elements.filter((e) => !e.id.includes(".")).map((e) => e.id));
    for (const { source, target, mode } of context.feedbacks ?? []) {
      // the block driven: the element of the field, or the collapsed group holding it
      const owner = target.slice(0, target.lastIndexOf("."));
      const targetNode = shown.has(owner) || expanded[owner.split(".")[0]] ? owner : owner.split(".")[0];
      const [sourceNode, sourceHandle] = splitPort(source);
      const element = context.elements.find((e) => e.id === targetNode);
      list.push({
        id: `feedback:${source}->${target}`,
        type: "feedback",
        className: "feedback",
        source: sourceNode,
        sourceHandle,
        target: targetNode,
        targetHandle: element?.inputs[0] ?? null,
        selected: selectedEdges.has(`feedback:${source}->${target}`),
        zIndex: 2,
        data: { source, target, mode, field: target.slice(target.lastIndexOf(".") + 1) },
      });
    }
    for (const [group, info] of Object.entries(context.groups ?? {})) {
      if (!expanded[group]) continue;
      for (const { source, target } of info.connections) {
        const [sourceNode, sourceHandle] = splitPort(source);
        const [targetNode, targetHandle] = splitPort(target);
        list.push({
          id: `${group}:${source}->${target}`,
          source: `${group}.${sourceNode}`,
          sourceHandle,
          target: `${group}.${targetNode}`,
          targetHandle,
          deletable: false,
          selectable: false,
          zIndex: 1,
          style: { stroke: signalColor(signalOf(`${group}.${sourceNode}`, sourceHandle, "outputs")), strokeDasharray: "4 3" },
        });
      }
    }
    return list;
  }, [context, signalOf, running, expanded, selectedEdges, selection]);

  useEffect(() => {
    // forget the selected links that were removed
    const known = new Set([
      ...(context?.connections.map(({ source, target }) => `${source}->${target}`) ?? []),
      ...(context?.feedbacks?.map(({ source, target }) => `feedback:${source}->${target}`) ?? []),
    ]);
    setSelectedEdges((current) => ([...current].every((id) => known.has(id)) ? current : new Set([...current].filter((id) => known.has(id)))));
  }, [context]);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    const chosen = changes.filter((change) => change.type === "select");
    if (!chosen.length) return;
    setSelectedEdges((current) => {
      const next = new Set(current);
      for (const change of chosen) {
        if (change.selected) next.add(change.id);
        else next.delete(change.id);
      }
      return next;
    });
  }, []);

  const isValidConnection: IsValidConnection = useCallback(
    (connection) => {
      if (!palette || connection.source === connection.target) return false;
      const from = signalOf(connection.source, connection.sourceHandle ?? "", "outputs");
      const to = signalOf(connection.target, connection.targetHandle ?? "", "inputs");
      return compatible(from, to, palette.signals);
    },
    [palette, signalOf],
  );

  const onNodesChange = useCallback((changes: NodeChange<FlowNode>[]) => {
    setNodes((current) => applyNodeChanges(changes, current));
  }, []);

  const onNodeDragStop = useCallback(
    (_: unknown, __: FlowNode, dragged: FlowNode[]) => {
      if (!context) return;
      const updates: Record<string, [number, number]> = {};
      for (const node of dragged) {
        const position: [number, number] = [Math.round(node.position.x), Math.round(node.position.y)];
        if (node.parentId) {
          const group = node.parentId;
          const inside = insideLayout(context, group);
          const saved = context.ui.positions ?? {};
          const names = Object.keys(inside);
          if (names.some((name) => !saved[`${group}.${name}`])) {
            // first move inside this group: save the whole arrangement in frame coordinates
            for (const [name, xy] of Object.entries(inside)) updates[`${group}.${name}`] = xy;
            updates[node.id] = position;
          } else {
            const name = node.id.slice(group.length + 1);
            const [ix, iy] = inside[name];
            const [sx, sy] = saved[node.id];
            updates[node.id] = [sx + position[0] - ix, sy + position[1] - iy];
          }
        } else {
          updates[node.id] = position;
        }
      }
      void moved(updates);
    },
    [context, moved],
  );

  const onSelectionChange = useCallback(
    ({ nodes: chosen }: { nodes: FlowNode[] }) => select(chosen.map((node) => node.id).sort()),
    [select],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      void connect(`${connection.source}.${connection.sourceHandle}`, `${connection.target}.${connection.targetHandle}`);
    },
    [connect],
  );

  const onConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      if (state.isValid || !state.fromNode || !state.fromHandle || !editable || !context || !palette) return;
      const point = "changedTouches" in event ? event.changedTouches[0] : event;
      const fromId = state.fromNode.id;
      const fromPort = state.fromHandle.id ?? "";
      const outgoing = state.fromHandle.type === "source";
      // dropped on a block: connect to its first compatible port that is free
      const below = document.elementFromPoint(point.clientX, point.clientY)?.closest(".react-flow__node") as HTMLElement | null;
      const onto = below?.dataset.id;
      if (onto && onto !== fromId && !onto.includes(".")) {
        const element = context.elements.find((e) => e.id === onto);
        const fromSignal = signalOf(fromId, fromPort, outgoing ? "outputs" : "inputs");
        const busy = new Set(context.connections.map((c) => (outgoing ? c.target : c.source)));
        const candidates = outgoing ? (element?.inputs ?? []) : (element?.outputs ?? []);
        const port = candidates.find((name) => {
          const signal = signalOf(onto, name, outgoing ? "inputs" : "outputs");
          const ok = outgoing ? compatible(fromSignal, signal, palette.signals) : compatible(signal, fromSignal, palette.signals);
          return ok && (!outgoing || !busy.has(`${onto}.${name}`));
        });
        if (port) {
          if (outgoing) void connect(`${fromId}.${fromPort}`, `${onto}.${port}`);
          else void connect(`${onto}.${port}`, `${fromId}.${fromPort}`);
        }
        return;
      }
      if (state.toNode) return;
      const position = flow.screenToFlowPosition({ x: point.clientX, y: point.clientY });
      const direction = outgoing ? "out" : "in";
      const port = `${fromId}.${fromPort}`;
      const signal = signalOf(fromId, fromPort, outgoing ? "outputs" : "inputs");
      setQuickAdd({
        position: [position.x - (direction === "out" ? 0 : NODE_WIDTH), position.y - 30],
        from: { port, direction, signal },
      });
    },
    [flow, signalOf, setQuickAdd, editable, context, palette, connect],
  );

  const onDrop = useCallback(
    (event: DragEvent) => {
      const point = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const position: [number, number] = [Math.round(point.x - NODE_WIDTH / 2), Math.round(point.y - 30)];
      const typeId = event.dataTransfer.getData(DRAG_TYPE);
      const block = event.dataTransfer.getData(BLOCK_TYPE);
      if (!typeId && !block) return;
      event.preventDefault();
      if (typeId) void addElement(typeId, position);
      else {
        const found = useStore.getState().blocks.find((b) => b.name === block);
        if (found) void insertBlock(found.document, position);
      }
    },
    [flow, addElement, insertBlock],
  );

  const empty = (context?.elements.length ?? 0) === 0;

  return (
    <div
      data-testid="canvas"
      data-tour="canvas"
      className="relative min-h-0 flex-1"
      onDoubleClick={(event) => {
        if (!editable || !(event.target as HTMLElement).classList.contains("react-flow__pane")) return;
        const point = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
        setQuickAdd({ position: [point.x - NODE_WIDTH / 2, point.y - 30] });
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStop={onNodeDragStop}
        onNodeClick={() => {
          // a block clicked shows the inspector when it was hidden
          if (!useStore.getState().layout.inspectorOpen) useStore.getState().setLayout({ inspectorOpen: true });
        }}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        isValidConnection={isValidConnection}
        onSelectionChange={onSelectionChange}
        onNodesDelete={(deleted) =>
          void removeElements(
            deleted
              // the inside of a group goes with it; a split attached to a deleted block goes with the block
              .filter((node) => !node.parentId || ("attached" in node.data && node.data.attached && !deleted.some((d) => d.id === node.parentId)))
              .map((node) => node.id),
          )
        }
        onEdgesDelete={(deleted) => {
          for (const edge of deleted) {
            const data = edge.data as { source: string; target: string; mode?: string } | undefined;
            if (!data) continue;
            if (edge.type === "feedback") void useStore.getState().removeFeedback(data.source, data.target);
            else void disconnect(data.source, data.target);
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDrop={onDrop}
        deleteKeyCode={editable ? DELETE_KEYS : null}
        nodesConnectable={editable}
        nodesDraggable={editable}
        multiSelectionKeyCode={MULTI_KEYS}
        selectionKeyCode="Shift"
        panOnDrag={pointer === "pan" ? true : PAN_BUTTONS}
        selectionOnDrag={pointer === "select"}
        selectionMode={pointer === "select" ? SelectionMode.Partial : SelectionMode.Full}
        zoomOnDoubleClick={false}
        snapToGrid
        snapGrid={SNAP}
        minZoom={0.15}
        maxZoom={2.5}
        attributionPosition="top-right"
        defaultEdgeOptions={EDGE_OPTIONS}
        connectionRadius={28}
      >
        <FeedbackMarker />
        <Panel position="top-left" className="!m-2">
          <div role="group" aria-label={t("canvas.pointer")} className="flex rounded-lg border border-slate-200 bg-white/90 p-0.5 shadow-sm backdrop-blur dark:border-slate-700 dark:bg-slate-900/90">
            {(
              [
                ["pan", Hand, "canvas.pointerPan", "H"],
                ["select", BoxSelect, "canvas.pointerSelect", "V"],
              ] as const
            ).map(([mode, Icon, label, key]) => (
              <button
                key={mode}
                type="button"
                data-testid={`pointer-${mode}`}
                aria-pressed={pointer === mode}
                title={`${t(label)} (${key})`}
                aria-label={t(label)}
                onClick={() => useStore.getState().setPointer(mode)}
                className={`grid size-7 place-items-center rounded-md transition-colors ${
                  pointer === mode
                    ? "bg-accent-100 text-accent-700 dark:bg-accent-900/60 dark:text-accent-200"
                    : "text-slate-500 hover:bg-slate-100 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
                }`}
              >
                <Icon className="size-4" />
              </button>
            ))}
          </div>
        </Panel>
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} color="var(--canvas-dots)" />
        <Controls showInteractive={false} position="bottom-left" />
        <MiniMap
          pannable
          zoomable
          position="bottom-right"
          style={{ width: 150, height: 96 }}
          className="!rounded-lg !border !border-slate-200 dark:!border-slate-700"
          nodeColor={(node) => {
            const data = node.data as { element?: { type_id: string } };
            return node.type === "frame" ? "#c7d2fe" : category(categoryOf(data.element?.type_id ?? "")).color;
          }}
          nodeStrokeWidth={0}
        />
      </ReactFlow>
      {(welcome && (empty || !context)) && (
        <div className="pointer-events-none absolute inset-0 overflow-y-auto p-6">
          <div className="pointer-events-auto relative mx-auto max-w-3xl rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-xl backdrop-blur dark:border-slate-700 dark:bg-slate-900/90">
            <button
              type="button"
              onClick={() => useStore.setState({ welcome: false })}
              className="absolute top-2 right-2 rounded p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
              aria-label={t("common.close")}
              data-testid="hide-welcome"
            >
              <X className="size-4" />
            </button>
            <PresetGallery welcome />
          </div>
        </div>
      )}
      {empty && !welcome && (
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
