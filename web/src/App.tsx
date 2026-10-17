import { ReactFlowProvider } from "@xyflow/react";
import { Bot, Code2, Cpu, FilePlus2, FolderOpen, GraduationCap, Keyboard, LayoutGrid, Redo2, Save, Undo2 } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AccountMenu, LoginDialog } from "./components/Account";
import { BottomPanel } from "./components/BottomPanel";
import { Canvas } from "./components/Canvas";
import { Dialogs } from "./components/Dialogs";
import { Inspector } from "./components/Inspector";
import { ConfirmAction, ElicitDialog, EmergencyStop, OperationPanel } from "./components/Operation";
import { Palette } from "./components/Palette";
import { PresetsDialog } from "./components/Presets";
import { QuickAdd } from "./components/QuickAdd";
import { RunButton } from "./components/RunButton";
import { Toasts } from "./components/Toasts";
import { ToolButton, TopBar } from "./components/TopBar";
import { connectEvents, disconnectEvents } from "./events";
import { useShortcuts } from "./shortcuts";
import { startTour } from "./tour";
import { useStore } from "./store";

const AssistantPanel = lazy(() => import("./components/Assistant").then((module) => ({ default: module.AssistantPanel })));

function Separator() {
  return <div className="mx-1 h-5 w-px bg-slate-200 dark:bg-slate-700" />;
}

function Editor() {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  const role = useStore((s) => s.info?.role);
  const store = useStore.getState();
  const fileInput = useRef<HTMLInputElement>(null);
  const ready = useStore((s) => s.ready);
  const info = useStore((s) => s.info);
  const showAssistant = Boolean(info?.assistant && (info.role === "simulator" || info.role === "operator"));
  const showOperation = Boolean(info?.operation || context?.mode === "operation" || context?.elements.some((e) => e.actions.length));
  const editable = role !== "viewer";

  const inspect = useCallback((port: string) => void useStore.getState().inspect(port), []);
  const actions = useMemo(() => ({ open: () => fileInput.current?.click(), run: () => void useStore.getState().run() }), []);
  useShortcuts(actions);

  return (
    <div className="flex h-full flex-col" data-ready={ready}>
      {!ready && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-slate-50/70 backdrop-blur-sm dark:bg-slate-950/70" data-testid="loading">
          <div className="flex items-center gap-3 rounded-xl bg-white px-4 py-3 shadow-lg dark:bg-slate-900">
            <span className="size-4 animate-spin rounded-full border-2 border-accent-500 border-t-transparent" />
            <span className="text-sm">{t("common.loading")}</span>
          </div>
        </div>
      )}
      <TopBar
        right={
          <>
            {(info?.operation || context?.mode === "operation") && <EmergencyStop />}
            <RunButton />
            <Separator />
            <div className="flex items-center" data-tour="help">
              <ToolButton icon={<GraduationCap className="size-4" />} label={t("toolbar.tour")} onClick={() => void startTour()} testId="tour" />
              <ToolButton icon={<Keyboard className="size-4" />} label={t("toolbar.shortcuts")} onClick={() => store.setDialog({ kind: "shortcuts" })} shortcut="?" />
            </div>
            <AccountMenu />
          </>
        }
      >
        <Separator />
        <ToolButton icon={<LayoutGrid className="size-4" />} label={t("toolbar.presets")} onClick={() => store.setDialog({ kind: "presets" })} testId="open-presets">
          <span className="hidden xl:inline">{t("toolbar.presetsShort")}</span>
        </ToolButton>
        <div className="flex items-center" data-tour="files">
        <ToolButton icon={<FilePlus2 className="size-4" />} label={t("toolbar.new")} onClick={() => void store.newContext()} testId="new" disabled={!editable} />
        <ToolButton icon={<FolderOpen className="size-4" />} label={t("toolbar.open")} onClick={() => fileInput.current?.click()} testId="open" shortcut="Ctrl+O" />
        <input
          ref={fileInput}
          type="file"
          accept=".lumos,application/json"
          className="hidden"
          data-testid="open-file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void store.openFile(file);
            event.target.value = "";
          }}
        />
        <ToolButton icon={<Save className="size-4" />} label={t("toolbar.save")} onClick={() => void store.save()} testId="save" shortcut="Ctrl+S" />
        <ToolButton icon={<Code2 className="size-4" />} label={t("toolbar.python")} onClick={() => void store.python()} testId="python" />
        </div>
        <Separator />
        <ToolButton icon={<Undo2 className="size-4" />} label={t("toolbar.undo")} onClick={() => void store.undo()} testId="undo" disabled={!context?.can_undo || !editable} shortcut="Ctrl+Z" />
        <ToolButton icon={<Redo2 className="size-4" />} label={t("toolbar.redo")} onClick={() => void store.redo()} testId="redo" disabled={!context?.can_redo || !editable} shortcut="Ctrl+Shift+Z" />
      </TopBar>
      <div className="flex min-h-0 flex-1">
        <Palette />
        <main className="relative flex min-w-0 flex-1 flex-col">
          <Canvas onInspect={inspect} />
          <BottomPanel
            extraTabs={[
              ...(showOperation ? [{ tab: "operation" as const, icon: Cpu, label: "tabs.operation" }] : []),
              ...(showAssistant ? [{ tab: "assistant" as const, icon: Bot, label: "tabs.assistant" }] : []),
            ]}
            renderExtra={(tab) => (tab === "operation" ? <OperationPanel /> : tab === "assistant" ? (
                <Suspense fallback={null}>
                  <AssistantPanel />
                </Suspense>
              ) : null)}
          />
        </main>
        <Inspector onInspect={inspect} />
      </div>
      <QuickAdd />
      <Dialogs
        extra={(dialog) =>
          dialog.kind === "presets" ? (
            <PresetsDialog />
          ) : dialog.kind === "login" ? (
            <LoginDialog />
          ) : dialog.kind === "elicit" ? (
            <ElicitDialog message={dialog.message} answer={dialog.answer} />
          ) : dialog.kind === "confirm-action" ? (
            <ConfirmAction element={dialog.element} action={dialog.action} device={dialog.device} onConfirm={dialog.onConfirm} />
          ) : null
        }
      />
      <Toasts />
    </div>
  );
}

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

  return (
    <ReactFlowProvider>
      <Editor />
    </ReactFlowProvider>
  );
}
