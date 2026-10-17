// Keyboard shortcuts of the editor (listed in the "?" dialog).

import { useReactFlow } from "@xyflow/react";
import { useEffect } from "react";
import { freeSpot } from "./components/Palette";
import { useStore } from "./store";

function typing(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  if (!element) return false;
  return element.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
}

export function canvasCentre(flow: ReturnType<typeof useReactFlow>): [number, number] {
  const canvas = document.querySelector('[data-testid="canvas"]')?.getBoundingClientRect();
  if (!canvas) return [0, 0];
  const point = flow.screenToFlowPosition({ x: canvas.left + canvas.width / 2, y: canvas.top + canvas.height / 2 });
  const taken = Object.entries(useStore.getState().positions).filter(([id]) => !id.includes(".")).map(([, xy]) => xy);
  return freeSpot([point.x - 115, point.y - 60], taken);
}

export function useShortcuts(actions: { run?: () => void; open: () => void }): void {
  const flow = useReactFlow();
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      const store = useStore.getState();
      if (store.dialog || store.quickAdd || typing(event.target)) return;
      const key = event.key.toLowerCase();
      const mod = event.ctrlKey || event.metaKey;
      const editable = store.info?.role !== "viewer";
      const top = store.selection.filter((id) => !id.includes("."));
      let done = true;
      if (mod && key === "z" && !event.shiftKey) void store.undo();
      else if (mod && (key === "y" || (key === "z" && event.shiftKey))) void store.redo();
      else if (mod && key === "s") void store.save();
      else if (mod && key === "o") actions.open();
      else if (mod && key === "k" && editable) store.setQuickAdd({ position: canvasCentre(flow) });
      else if (mod && key === "g" && editable && top.length) store.group(top);
      else if (mod && key === "c" && top.length) void store.copy(top);
      else if (mod && key === "v" && editable) void store.paste();
      else if (mod && key === "d" && editable && top.length) void store.copy(top).then(() => store.paste());
      else if (mod && key === "a") store.select(store.context?.elements.filter((e) => !e.id.includes(".")).map((e) => e.id) ?? []);
      else if (mod && key === "enter") actions.run?.();
      else if (!mod && key === "f") void flow.fitView({ padding: 0.15, duration: 300, maxZoom: 1 });
      else if (!mod && event.key === "?") store.setDialog({ kind: "shortcuts" });
      else if (key === "escape") store.select([]);
      else done = false;
      if (done) event.preventDefault();
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [flow, actions]);
}
