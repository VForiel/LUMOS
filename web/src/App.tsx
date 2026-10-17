import { ReactFlowProvider } from "@xyflow/react";
import { useCallback, useEffect } from "react";
import { Canvas } from "./components/Canvas";
import { Palette } from "./components/Palette";
import { Toasts } from "./components/Toasts";
import { TopBar } from "./components/TopBar";
import { connectEvents, disconnectEvents } from "./events";
import { useStore } from "./store";

export function App() {
  useEffect(() => {
    let cancelled = false;
    void useStore
      .getState()
      .start()
      .then(() => {
        if (!cancelled) connectEvents();
      });
    return () => {
      cancelled = true;
      disconnectEvents();
    };
  }, []);

  const inspect = useCallback((port: string) => {
    useStore.getState().toast({ kind: "info", title: port });
  }, []);

  return (
    <ReactFlowProvider>
      <div className="flex h-full flex-col">
        <TopBar />
        <div className="flex min-h-0 flex-1">
          <Palette />
          <main className="relative flex min-w-0 flex-1 flex-col">
            <Canvas onInspect={inspect} />
          </main>
        </div>
        <Toasts />
      </div>
    </ReactFlowProvider>
  );
}
