// The edge of a panel, dragged to resize it (sizes are kept by the store, in the browser).

import { useRef } from "react";
import { useTranslation } from "react-i18next";

/**
 * A grip along an edge of a panel.
 *
 * ``side`` is the edge it sits on; ``onResize`` gets the size the panel would have, computed from the size
 * at the start of the drag and the movement of the pointer (towards the inside of the panel shrinks it).
 */
export function Resizer({ side, size, onResize, testId }: { side: "left" | "right" | "top"; size: number; onResize: (size: number) => void; testId: string }) {
  const { t } = useTranslation();
  const start = useRef<{ pointer: number; size: number } | null>(null);
  const vertical = side !== "top";
  const position = side === "left" ? "left-0 -translate-x-1/2" : side === "right" ? "right-0 translate-x-1/2" : "top-0 -translate-y-1/2";
  return (
    <div
      role="separator"
      aria-orientation={vertical ? "vertical" : "horizontal"}
      aria-label={t("panels.resize")}
      aria-valuenow={size}
      data-testid={testId}
      tabIndex={0}
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        start.current = { pointer: vertical ? event.clientX : event.clientY, size };
      }}
      onPointerMove={(event) => {
        if (!start.current) return;
        const moved = (vertical ? event.clientX : event.clientY) - start.current.pointer;
        // the panel grows when its edge moves away from it
        onResize(start.current.size + (side === "right" ? moved : -moved));
      }}
      onPointerUp={(event) => {
        start.current = null;
        event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onKeyDown={(event) => {
        const step = event.shiftKey ? 40 : 10;
        const grow = vertical ? (side === "right" ? "ArrowRight" : "ArrowLeft") : "ArrowUp";
        const shrink = vertical ? (side === "right" ? "ArrowLeft" : "ArrowRight") : "ArrowDown";
        if (event.key === grow) onResize(size + step);
        else if (event.key === shrink) onResize(size - step);
        else return;
        event.preventDefault();
      }}
      className={`group absolute z-20 ${position} ${vertical ? "inset-y-0 w-2 cursor-col-resize" : "inset-x-0 h-2 cursor-row-resize"} outline-none`}
    >
      <span
        className={`absolute bg-accent-400 opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 group-active:opacity-100 ${
          vertical ? "inset-y-0 left-1/2 w-0.5 -translate-x-1/2" : "inset-x-0 top-1/2 h-0.5 -translate-y-1/2"
        }`}
      />
    </div>
  );
}
