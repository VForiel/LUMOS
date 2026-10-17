// State of the interface: the instance, the palette, the open context and what the user does with it.
// The server is the source of truth: every edit goes through the API, then the context is described again.

import { create } from "zustand";
import {
  api,
  ApiError,
  download,
  fileName,
  type ContextSummary,
  type Fragment,
  type Info,
  type Issue,
  type JobInfo,
  type JournalEntry,
  type Palette,
  type PortSummary,
} from "./api";
import { autoLayout, COLUMN } from "./lib/layout";

export type Toast = { id: number; kind: "error" | "info" | "success"; title: string; hint?: string; values?: Record<string, unknown> };
export type Inspected = { context: string; port: string; summary: PortSummary | null; figure: string | null; loading: boolean; error?: string; job?: string };
export type BottomTab = "results" | "journal" | "problems" | "operation" | "assistant";
export type Dialog =
  | { kind: "prompt"; title: string; label: string; value: string; confirm: string; onConfirm: (value: string) => void }
  | { kind: "python"; code: string }
  | { kind: "shortcuts" }
  | { kind: "presets" }
  | { kind: "login" }
  | { kind: "confirm-action"; element: string; action: string; device: string; onConfirm: () => void };
export type QuickAdd = { position: [number, number]; from?: { port: string; direction: "out" | "in"; signal: string } };
export type Block = { name: string; document: string; saved: string };

type State = {
  info: Info | null;
  palette: Palette | null;
  context: ContextSummary | null;
  positions: Record<string, [number, number]>;
  selection: string[];
  issues: Issue[];
  jobs: Record<string, JobInfo>;
  journal: JournalEntry[];
  inspected: Inspected | null;
  toasts: Toast[];
  bottom: BottomTab | null;
  dialog: Dialog | null;
  quickAdd: QuickAdd | null;
  blocks: Block[];
};

type Actions = {
  start: () => Promise<void>;
  show: (summary: ContextSummary) => Promise<void>;
  refresh: () => Promise<void>;
  open: (summary: ContextSummary) => Promise<void>;
  newContext: () => Promise<void>;
  openFile: (file: File) => Promise<void>;
  save: () => Promise<void>;
  python: () => Promise<void>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  addElement: (typeId: string, position?: [number, number]) => Promise<string | null>;
  removeElements: (ids: string[]) => Promise<void>;
  connect: (source: string, target: string) => Promise<boolean>;
  disconnect: (source: string, target: string) => Promise<void>;
  setParam: (path: string, value: unknown) => Promise<boolean>;
  setContextInfo: (change: { name?: string; description?: string }) => Promise<void>;
  setObservation: (minimum: string, maximum: string, n: number) => Promise<boolean>;
  rename: (path: string, name: string) => Promise<void>;
  group: (ids: string[]) => void;
  ungroup: (path: string) => Promise<void>;
  toggleGroup: (path: string) => Promise<void>;
  copy: (ids: string[]) => Promise<void>;
  paste: () => Promise<void>;
  saveBlock: (path: string) => Promise<void>;
  insertBlock: (document: string, position?: [number, number]) => Promise<void>;
  forgetBlock: (name: string) => void;
  moved: (positions: Record<string, [number, number]>) => Promise<void>;
  select: (ids: string[]) => void;
  toast: (toast: Omit<Toast, "id">) => void;
  dismiss: (id: number) => void;
  fail: (error: unknown) => void;
  setBottom: (tab: BottomTab | null) => void;
  setDialog: (dialog: Dialog | null) => void;
  setQuickAdd: (quickAdd: QuickAdd | null) => void;
  onJob: (job: JobInfo) => void;
  onJournal: (context: string, entry: JournalEntry) => void;
};

const CONTEXT_KEY = "lumos.context";
const DOCUMENT_KEY = "lumos.document";
const CLIPBOARD_KEY = "lumos.clipboard";
const BLOCKS_KEY = "lumos.blocks";

let toastId = 0;
let copying: Promise<void> = Promise.resolve();

export function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function readBlocks(): Block[] {
  try {
    return JSON.parse(storage()?.getItem(BLOCKS_KEY) ?? "[]");
  } catch {
    return [];
  }
}

/** Keep a copy of the open context in the browser, to restore it after a restart of the server. */
async function remember(contextId: string): Promise<void> {
  try {
    const text = await api.exportLumos(contextId);
    storage()?.setItem(DOCUMENT_KEY, text);
    storage()?.setItem(CONTEXT_KEY, contextId);
  } catch {
    /* best effort */
  }
}

/** Positions of the blocks: those saved in the context, an automatic layout for the others. */
export function placed(summary: ContextSummary): Record<string, [number, number]> {
  const saved = summary.ui?.positions ?? {};
  const ids = summary.elements.filter((e) => !e.id.includes(".")).map((e) => e.id);
  const edges = summary.connections.map((c) => [c.source.split(".")[0], c.target.split(".")[0]] as [string, string]);
  const layout = autoLayout(ids, edges);
  const anyKnown = ids.some((id) => saved[id]);
  const right = anyKnown ? Math.max(...ids.filter((id) => saved[id]).map((id) => saved[id][0])) + COLUMN : 0;
  const positions: Record<string, [number, number]> = { ...saved };
  for (const id of ids) {
    if (saved[id]) continue;
    positions[id] = anyKnown ? [right, layout[id][1]] : layout[id];
  }
  return positions;
}

const isEditable = (info: Info | null) => info?.role !== "viewer" && info?.role !== null;

export const useStore = create<State & Actions>((set, get) => {
  /** Run an edit on the open context, then show the context as the server describes it. */
  async function edit<T>(change: (id: string) => Promise<T>): Promise<T | undefined> {
    const context = get().context;
    if (!context) return undefined;
    try {
      const result = await change(context.id);
      await get().refresh();
      return result;
    } catch (error) {
      get().fail(error);
      await get().refresh();
      return undefined;
    }
  }

  async function copyNow(ids: string[]): Promise<void> {
    const context = get().context;
    if (!context || !ids.length) return;
    try {
      const fragment = await api.copy(context.id, ids.filter((id) => !id.includes(".")));
      storage()?.setItem(CLIPBOARD_KEY, JSON.stringify(fragment));
      get().toast({ kind: "info", title: "copied", values: { count: fragment.elements.length } });
    } catch (error) {
      get().fail(error);
    }
  }

  return {
    info: null,
    palette: null,
    context: null,
    positions: {},
    selection: [],
    issues: [],
    jobs: {},
    journal: [],
    inspected: null,
    toasts: [],
    bottom: null,
    dialog: null,
    quickAdd: null,
    blocks: readBlocks(),

    async start() {
      try {
        const [info, palette] = await Promise.all([api.info(), api.palette()]);
        set({ info, palette });
      } catch (error) {
        get().fail(error);
        return;
      }
      const store = storage();
      const id = store?.getItem(CONTEXT_KEY);
      if (id) {
        try {
          await get().open(await api.describe(id));
          return;
        } catch {
          /* the server forgot it (restart, expired session): restore the copy */
        }
      }
      const document = store?.getItem(DOCUMENT_KEY);
      if (document) {
        try {
          await get().open(await api.importLumos(document));
          get().toast({ kind: "info", title: "restored" });
          return;
        } catch {
          store?.removeItem(DOCUMENT_KEY);
        }
      }
      if (isEditable(get().info)) await get().newContext();
      else set({ dialog: { kind: "presets" } });
    },

    async show(summary) {
      const issues = await api.validate(summary.id).catch(() => [] as Issue[]);
      const known = new Set(summary.elements.map((e) => e.id));
      set({ context: summary, issues, positions: placed(summary), selection: get().selection.filter((id) => known.has(id)) });
      void remember(summary.id);
    },

    async refresh() {
      const context = get().context;
      if (!context) return;
      try {
        await get().show(await api.describe(context.id));
      } catch (error) {
        get().fail(error);
      }
    },

    async open(summary) {
      const previous = get().context;
      set({ selection: [], inspected: null, journal: [] });
      await get().show(summary);
      if (previous && previous.id !== summary.id) void api.close(previous.id).catch(() => undefined);
    },

    async newContext() {
      try {
        await get().open(await api.createContext("untitled"));
      } catch (error) {
        get().fail(error);
      }
    },

    async openFile(file) {
      try {
        const text = await file.text();
        await get().open(await api.importLumos(text));
        get().toast({ kind: "success", title: "opened", values: { name: file.name } });
      } catch (error) {
        get().fail(error);
      }
    },

    async save() {
      const context = get().context;
      if (!context) return;
      try {
        download(fileName(context.name, ".lumos"), await api.exportLumos(context.id));
      } catch (error) {
        get().fail(error);
      }
    },

    async python() {
      const context = get().context;
      if (!context) return;
      try {
        set({ dialog: { kind: "python", code: await api.exportPython(context.id) } });
      } catch (error) {
        get().fail(error);
      }
    },

    async undo() {
      const context = get().context;
      if (!context?.can_undo) return;
      try {
        await get().show(await api.undo(context.id));
      } catch (error) {
        get().fail(error);
      }
    },

    async redo() {
      const context = get().context;
      if (!context?.can_redo) return;
      try {
        await get().show(await api.redo(context.id));
      } catch (error) {
        get().fail(error);
      }
    },

    async addElement(typeId, position) {
      const element = await edit(async (id) => {
        const added = await api.addElement(id, typeId);
        if (position) await api.updateUi(id, { positions: { [added.id]: position.map(Math.round) } }, false);
        return added;
      });
      if (!element) return null;
      set({ selection: [element.id] });
      return element.id;
    },

    async removeElements(ids) {
      await edit(async (id) => {
        for (const element of ids) await api.removeElement(id, element);
      });
      set({ selection: [] });
    },

    async connect(source, target) {
      return (await edit((id) => api.connect(id, source, target).then(() => true))) ?? false;
    },

    async disconnect(source, target) {
      await edit((id) => api.disconnect(id, source, target));
    },

    async setParam(path, value) {
      return (await edit((id) => api.setParam(id, path, value).then(() => true))) ?? false;
    },

    async setContextInfo(change) {
      await edit((id) => api.setContext(id, change));
    },

    async setObservation(minimum, maximum, n) {
      return (await edit((id) => api.setObservation(id, minimum, maximum, n).then(() => true))) ?? false;
    },

    async rename(path, name) {
      const done = await edit((id) => api.rename(id, path, name).then(() => true));
      if (done) set({ selection: [path.includes(".") ? `${path.slice(0, path.lastIndexOf("."))}.${name}` : name] });
    },

    group(ids) {
      const taken = new Set(get().context?.elements.map((e) => e.id));
      let name = "group";
      for (let n = 1; taken.has(name); n += 1) name = `group${n}`;
      set({
        dialog: {
          kind: "prompt",
          title: "dialog.group.title",
          label: "dialog.group.label",
          value: name,
          confirm: "dialog.group.confirm",
          onConfirm: (value) => {
            void edit((id) => api.group(id, ids, value)).then((summary) => {
              if (summary) set({ selection: [value] });
            });
          },
        },
      });
    },

    async ungroup(path) {
      await edit((id) => api.ungroup(id, path));
      set({ selection: [] });
    },

    async toggleGroup(path) {
      const expanded = { ...(get().context?.ui.expanded ?? {}) };
      expanded[path] = !expanded[path];
      await edit((id) => api.updateUi(id, { expanded }, false));
    },

    copy(ids) {
      copying = copying.then(() => copyNow(ids));
      return copying;
    },

    async paste() {
      await copying;
      const text = storage()?.getItem(CLIPBOARD_KEY);
      if (!text) return;
      const fragment = JSON.parse(text) as Fragment;
      const pasted = await edit((id) => api.paste(id, fragment));
      if (pasted) set({ selection: pasted.ids });
    },

    async saveBlock(path) {
      const context = get().context;
      if (!context) return;
      try {
        const document = await api.exportElement(context.id, path);
        const name = path.split(".").pop()!;
        const blocks = [...get().blocks.filter((b) => b.name !== name), { name, document, saved: new Date().toISOString() }];
        storage()?.setItem(BLOCKS_KEY, JSON.stringify(blocks));
        set({ blocks });
        get().toast({ kind: "success", title: "blockSaved", values: { name } });
      } catch (error) {
        get().fail(error);
      }
    },

    async insertBlock(document, position) {
      const element = await edit(async (id) => {
        const added = await api.insertBlock(id, document);
        if (position) await api.updateUi(id, { positions: { [added.id]: position.map(Math.round) } }, false);
        return added;
      });
      if (element) set({ selection: [element.id] });
    },

    forgetBlock(name) {
      const blocks = get().blocks.filter((b) => b.name !== name);
      storage()?.setItem(BLOCKS_KEY, JSON.stringify(blocks));
      set({ blocks });
    },

    async moved(positions) {
      set({ positions: { ...get().positions, ...positions } });
      const context = get().context;
      if (!context || !isEditable(get().info)) return;
      try {
        const summary = await api.updateUi(context.id, { positions: { ...(context.ui.positions ?? {}), ...positions } });
        set({ context: summary });
      } catch (error) {
        get().fail(error);
      }
    },

    select(ids) {
      ids = [...ids].sort();
      const current = [...get().selection].sort();
      if (current.length === ids.length && current.every((id, k) => id === ids[k])) return;
      set({ selection: ids });
    },

    toast(toast) {
      const id = ++toastId;
      set({ toasts: [...get().toasts.slice(-4), { ...toast, id }] });
      window.setTimeout(() => get().dismiss(id), toast.kind === "error" ? 9000 : 3500);
    },

    dismiss(id) {
      set({ toasts: get().toasts.filter((t) => t.id !== id) });
    },

    fail(error) {
      if (error instanceof ApiError) get().toast({ kind: "error", title: error.message, hint: error.hint });
      else get().toast({ kind: "error", title: String(error) });
    },

    setBottom(tab) {
      set({ bottom: tab });
    },

    setDialog(dialog) {
      set({ dialog });
    },

    setQuickAdd(quickAdd) {
      set({ quickAdd });
    },

    onJob(job) {
      set({ jobs: { ...get().jobs, [job.id]: job } });
    },

    onJournal(context, entry) {
      if (get().context?.id !== context) return;
      set({ journal: [...get().journal.slice(-499), entry] });
    },
  };
});
