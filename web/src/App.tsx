import { ResultsBrowser } from "./components/Browser";
import { ReactFlowProvider } from "@xyflow/react";
import { Bot, Code2, Cpu, FilePlus2, FlaskConical, FolderOpen, GraduationCap, Keyboard, LayoutGrid, Redo2, Save, Undo2 } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { AccountMenu, LoginDialog } from "./components/Account";
import { ArchivesDialog } from "./components/Archives";
import { ComponentEditor } from "./components/ComponentEditor";
import { BottomPanel } from "./components/BottomPanel";
import { Canvas } from "./components/Canvas";
import { Dialogs } from "./components/Dialogs";
import { Inspector } from "./components/Inspector";
import { ConfirmAction, ElicitDialog, EmergencyStop, OperationPanel } from "./components/Operation";
import { Palette } from "./components/Palette";
import { PresetsDialog } from "./components/Presets";
import { QuickAdd } from "./components/QuickAdd";
import { RunButton } from "./components/RunButton";
import { StudyDialog } from "./components/Study";
import { Toasts } from "./components/Toasts";
import { ToolButton, TopBar } from "./components/TopBar";
import { connectEvents, disconnectEvents } from "./events";
import { useShortcuts } from "./shortcuts";
import { startTour } from "./tour";
import { isNarrow, useStore } from "./store";

const ProceduresView = lazy(() => import("./components/Procedures").then((module) => ({ default: module.ProceduresView })));

/** Switch between the graph of the context and its procedures. */
function ViewSwitch() {
  const { t } = useTranslation();
  const view = useStore((s) => s.view);
  const context = useStore((s) => s.context);
  const count = context?.procedures?.length ?? 0;
  const tab = (name: "graph" | "procedures", label: string) => (
    <button
      type="button"
      role="tab"
      aria-selected={view === name}
      data-testid={`view-${name}`}
      onClick={() => useStore.getState().setView(name)}
      className={`rounded-md px-3 py-1 text-xs font-medium ${
        view === name ? "bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white" : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
      }`}
    >
      {label}
    </button>
  );
  return (
    <div role="tablist" className="flex items-center gap-0.5 self-start rounded-lg bg-slate-100 p-0.5 m-2 mb-0 dark:bg-slate-800">
      {tab("graph", t("view.graph"))}
      {tab("procedures", count ? `${t("view.procedures")} (${count})` : t("view.procedures"))}
    </div>
  );
}

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
  const view = useStore((s) => s.view);

  // a window becoming narrow keeps one side column only
  useEffect(() => {
    const fit = () => {
      const { layout, setLayout } = useStore.getState();
      if (isNarrow() && layout.paletteOpen && layout.inspectorOpen) setLayout({ inspectorOpen: false });
    };
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

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
            <ToolButton
              icon={<FlaskConical className="size-4" />}
              label={t("toolbar.study")}
              onClick={() => store.setDialog({ kind: "study" })}
              testId="open-study"
              disabled={!context || !(role === "simulator" || role === "operator")}
              tour="study"
            >
              <span className="hidden xl:inline">{t("toolbar.studyShort")}</span>
            </ToolButton>
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
          <ViewSwitch />
          <div className={view === "graph" ? "flex min-h-0 flex-1 flex-col" : "hidden"}>
            <Canvas onInspect={inspect} />
          </div>
          {view === "procedures" && (
            <Suspense fallback={<div className="flex-1" />}>
              <ProceduresView />
            </Suspense>
          )}
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
          ) : dialog.kind === "study" ? (
            <StudyDialog />
          ) : dialog.kind === "browser" ? (
            <ResultsBrowser />
          ) : dialog.kind === "archives" ? (
            <ArchivesDialog />
          ) : dialog.kind === "component" ? (
            <ComponentEditor id={dialog.id} />
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

/** Closing or reloading the page with unsaved changes asks first (the browser shows its own dialog). */
function useLeaveGuard() {
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => {
      if (!useStore.getState().dirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, []);
}

/** The title of the tab shows the context and whether it has unsaved changes. */
function useTitle() {
  const name = useStore((s) => s.context?.name);
  const dirty = useStore((s) => s.dirty);
  useEffect(() => {
    document.title = `${dirty ? "• " : ""}${name ? `${name} — ` : ""}LUMOS`;
  }, [name, dirty]);
}

export function App() {
  useLeaveGuard();
  useTitle();
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
