// An expanded group: a frame holding its elements, with the ports it exposes on its border.

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { FolderTree, Layers } from "lucide-react";
import { memo } from "react";
import { useTranslation } from "react-i18next";
import type { ElementSummary } from "../api";
import { signalColor } from "../lib/signals";
import { useStore } from "../store";

export type FrameData = {
  element: ElementSummary;
  width: number;
  height: number;
  signals: Record<string, string>;
};

export type FrameFlowNode = Node<FrameData, "frame">;

export const FRAME_HEADER = 44;
export const FRAME_PAD = 24;

export const GroupFrame = memo(function GroupFrame({ data, selected }: NodeProps<FrameFlowNode>) {
  const { t } = useTranslation();
  const { element, width, height, signals } = data;
  const port = (name: string, index: number, side: "in" | "out") => (
    <div
      key={`${side}-${name}`}
      className={`absolute flex items-center text-[11px] text-slate-500 dark:text-slate-400 ${side === "in" ? "left-0 pl-3" : "right-0 pr-3"}`}
      style={{ top: FRAME_HEADER + index * 24, height: 20 }}
    >
      <Handle
        id={name}
        type={side === "in" ? "target" : "source"}
        position={side === "in" ? Position.Left : Position.Right}
        style={{ background: signalColor(signals[name] ?? "Signal"), top: "50%" }}
      />
      {name}
    </div>
  );
  return (
    <div
      className={`relative rounded-2xl border-2 border-dashed bg-indigo-50/40 dark:bg-indigo-950/20 ${selected ? "border-accent-500" : "border-indigo-200 dark:border-indigo-900"}`}
      style={{ width, height }}
    >
      <div className="flex h-10 items-center gap-2 px-3">
        <Layers className="size-4 text-indigo-500" />
        <span className="font-semibold text-slate-700 dark:text-slate-200">{element.id}</span>
        <span className="text-xs text-slate-400">{t("inspector.group")}</span>
        <button
          type="button"
          className="nodrag ml-auto rounded p-1 text-slate-400 hover:bg-white hover:text-slate-600 dark:hover:bg-slate-800"
          onClick={(event) => {
            event.stopPropagation();
            void useStore.getState().toggleGroup(element.id);
          }}
          title={t("inspector.collapse")}
        >
          <FolderTree className="size-4" />
        </button>
      </div>
      {element.inputs.map((name, index) => port(name, index, "in"))}
      {element.outputs.map((name, index) => port(name, index, "out"))}
    </div>
  );
});
