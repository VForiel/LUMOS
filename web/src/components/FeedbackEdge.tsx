// A feedback of a loop (ADR-0020): a dashed amber curve from an output port back over the top of the block
// whose field it drives, labelled with the field and the mode ("+=" adds, "=" sets).

import { BaseEdge, EdgeLabelRenderer, useInternalNode, type Edge, type EdgeProps } from "@xyflow/react";
import { Repeat, Unlink } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useStore } from "../store";

export type FeedbackData = { source: string; target: string; mode: string; field: string };
export type FeedbackFlowEdge = Edge<FeedbackData, "feedback">;

export function FeedbackEdge({ sourceX, sourceY, target, selected, data, interactionWidth }: EdgeProps<FeedbackFlowEdge>) {
  const { t } = useTranslation();
  const node = useInternalNode(target);
  const editable = useStore((s) => s.info?.role !== "viewer");
  if (!node || !data) return null;
  const width = node.measured.width ?? 210;
  const tx = node.internals.positionAbsolute.x + width / 2;
  const ty = node.internals.positionAbsolute.y;
  const top = Math.min(sourceY, ty) - 70;
  const path = `M ${sourceX},${sourceY} C ${sourceX + 70},${sourceY} ${sourceX + 70},${top} ${(sourceX + tx) / 2},${top} S ${tx},${top} ${tx},${ty - 2}`;
  // the label sits on the top of the curve, nearer the driven block (away from the blocks in between)
  const labelX = tx + (sourceX - tx) * 0.25;
  const labelY = top;
  return (
    <>
      <BaseEdge
        path={path}
        interactionWidth={interactionWidth}
        style={{ stroke: "#f59e0b", strokeWidth: selected ? 3 : 2, strokeDasharray: "6 4" }}
        markerEnd="url(#lumos-feedback-arrow)"
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan pointer-events-auto absolute flex items-center gap-1 rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-800 shadow-sm dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200"
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          title={t("feedback.title", { source: data.source, target: data.target, mode: t(`feedback.modes.${data.mode}`) })}
        >
          <Repeat className="size-3" />
          <span className="font-mono">
            {data.field} {data.mode === "add" ? "+=" : "="}
          </span>
          {selected && editable && (
            <button
              type="button"
              data-testid="remove-feedback-edge"
              aria-label={t("feedback.remove")}
              title={t("feedback.remove")}
              onClick={(event) => {
                event.stopPropagation();
                void useStore.getState().removeFeedback(data.source, data.target);
              }}
              className="ml-0.5 rounded-full p-0.5 hover:bg-amber-200 dark:hover:bg-amber-800"
            >
              <Unlink className="size-3" />
            </button>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

/** The arrow head of feedback edges, rendered once in the canvas. */
export function FeedbackMarker() {
  return (
    <svg style={{ position: "absolute", width: 0, height: 0 }} aria-hidden="true">
      <defs>
        <marker id="lumos-feedback-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b" />
        </marker>
      </defs>
    </svg>
  );
}
