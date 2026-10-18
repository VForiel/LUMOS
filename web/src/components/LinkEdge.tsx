// A connection of the canvas: a Bézier curve coloured by its signal; once selected, a button at its middle
// removes the link alone (its blocks stay).

import { BaseEdge, EdgeLabelRenderer, getBezierPath, type Edge, type EdgeProps } from "@xyflow/react";
import { Unlink } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useStore } from "../store";

export type LinkData = { source: string; target: string; removable?: boolean };
export type LinkFlowEdge = Edge<LinkData, "link">;

export function LinkEdge({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, markerEnd, selected, data, interactionWidth }: EdgeProps<LinkFlowEdge>) {
  const { t } = useTranslation();
  const editable = useStore((s) => s.info?.role !== "viewer");
  const [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  return (
    <>
      <BaseEdge path={path} style={selected ? { ...style, strokeWidth: 3 } : style} markerEnd={markerEnd} interactionWidth={interactionWidth} />
      {selected && editable && data?.removable && (
        <EdgeLabelRenderer>
          <button
            type="button"
            data-testid="remove-link"
            title={t("canvas.removeLink")}
            aria-label={t("canvas.removeLink")}
            onClick={(event) => {
              event.stopPropagation();
              void useStore.getState().disconnect(data.source, data.target);
            }}
            className="nodrag nopan pointer-events-auto absolute grid size-7 place-items-center rounded-full border border-slate-200 bg-white text-slate-500 shadow-md transition hover:border-red-300 hover:bg-red-50 hover:text-red-600 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-red-950"
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            <Unlink className="size-3.5" />
          </button>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
