// State of the interface: the instance, the palette, the open context and what the user does with it.

import { create } from "zustand";
import { api, ApiError, type ContextSummary, type Info, type Issue, type JobInfo, type JournalEntry, type Palette, type PortSummary } from "./api";
import { autoLayout, COLUMN } from "./lib/layout";

export type Toast = { id: number; kind: "error" | "info" | "success"; title: string; hint?: string };
export type Inspected = { port: string; summary: PortSummary | null; figure: string | null; loading: boolean; error?: string };
export type BottomTab = "results" | "journal" | "problems" | "operation" | "assistant";

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
  busy: boolean;
  bottom: BottomTab | null;
  dirty: boolean;
};

type Actions = {
  start: () => Promise<void>;
  refresh: () => Promise<void>;
  open: (summary: ContextSummary) => Promise<void>;
  newContext: () => Promise<void>;
  addElement: (typeId: string, position?: [number, number]) => Promise<string | null>;
  removeElements: (ids: string[]) => Promise<void>;
  connect: (source: string, target: string) => Promise<boolean>;
  disconnect: (source: string, target: string) => Promise<void>;
  move: (id: string, position: [number, number]) => void;
  select: (ids: string[]) => void;
  toast: (toast: Omit<Toast, "id">) => void;
  dismiss: (id: number) => void;
  fail: (error: unknown) => void;
  setBottom: (tab: BottomTab | null) => void;
  onJob: (job: JobInfo) => void;
  onJournal: (context: string, entry: JournalEntry) => void;
};

const CONTEXT_KEY = "lumos.context";
const DOCUMENT_KEY = "lumos.document";

let toastId = 0;

function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** Keep a copy of the open context in the browser, to restore it after a restart of the server. */
export async function remember(contextId: string): Promise<void> {
  try {
    const text = await api.exportLumos(contextId);
    storage()?.setItem(DOCUMENT_KEY, text);
    storage()?.setItem(CONTEXT_KEY, contextId);
  } catch {
    /* best effort */
  }
}

function placed(summary: ContextSummary, known: Record<string, [number, number]>): Record<string, [number, number]> {
  const saved = { ...(summary.ui?.positions ?? {}), ...known };
  const ids = summary.elements.filter((e) => !e.id.includes(".")).map((e) => e.id);
  const missing = ids.filter((id) => !saved[id]);
  if (!missing.length) return Object.fromEntries(ids.map((id) => [id, saved[id]]));
  const edges = summary.connections.map((c) => [c.source.split(".")[0], c.target.split(".")[0]] as [string, string]);
  const layout = autoLayout(ids, edges);
  const anyKnown = ids.some((id) => saved[id]);
  const positions: Record<string, [number, number]> = {};
  for (const id of ids) {
    if (saved[id]) positions[id] = saved[id];
    else if (!anyKnown) positions[id] = layout[id];
    else {
      const right = Math.max(...Object.values(saved).map(([x]) => x)) + COLUMN;
      positions[id] = [right, layout[id][1]];
    }
  }
  return positions;
}

export const useStore = create<State & Actions>((set, get) => ({
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
  busy: false,
  bottom: null,
  dirty: false,

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
    await get().newContext();
  },

  async refresh() {
    const context = get().context;
    if (!context) return;
    try {
      const [summary, issues] = await Promise.all([api.describe(context.id), api.validate(context.id)]);
      set({ context: summary, issues, positions: placed(summary, get().positions) });
      void remember(summary.id);
    } catch (error) {
      get().fail(error);
    }
  },

  async open(summary) {
    const issues = await api.validate(summary.id).catch(() => []);
    set({ context: summary, issues, positions: placed(summary, {}), selection: [], inspected: null, journal: [], dirty: false });
    void remember(summary.id);
  },

  async newContext() {
    try {
      await get().open(await api.createContext("untitled"));
    } catch (error) {
      get().fail(error);
    }
  },

  async addElement(typeId, position) {
    const context = get().context;
    if (!context) return null;
    try {
      const element = await api.addElement(context.id, typeId);
      const positions = { ...get().positions };
      positions[element.id] = position ?? [0, 0];
      set({ positions, selection: [element.id], dirty: true });
      await get().refresh();
      return element.id;
    } catch (error) {
      get().fail(error);
      return null;
    }
  },

  async removeElements(ids) {
    const context = get().context;
    if (!context || !ids.length) return;
    try {
      for (const id of ids) await api.removeElement(context.id, id);
      const positions = { ...get().positions };
      for (const id of ids) delete positions[id];
      set({ positions, selection: [], dirty: true });
    } catch (error) {
      get().fail(error);
    }
    await get().refresh();
  },

  async connect(source, target) {
    const context = get().context;
    if (!context) return false;
    try {
      await api.connect(context.id, source, target);
      set({ dirty: true });
      await get().refresh();
      return true;
    } catch (error) {
      get().fail(error);
      return false;
    }
  },

  async disconnect(source, target) {
    const context = get().context;
    if (!context) return;
    try {
      await api.disconnect(context.id, source, target);
      set({ dirty: true });
    } catch (error) {
      get().fail(error);
    }
    await get().refresh();
  },

  move(id, position) {
    set({ positions: { ...get().positions, [id]: position } });
  },

  select(ids) {
    const current = get().selection;
    if (current.length === ids.length && current.every((id, k) => id === ids[k])) return;
    set({ selection: ids });
  },

  toast(toast) {
    const id = ++toastId;
    set({ toasts: [...get().toasts, { ...toast, id }] });
    window.setTimeout(() => get().dismiss(id), toast.kind === "error" ? 9000 : 4000);
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

  onJob(job) {
    set({ jobs: { ...get().jobs, [job.id]: job } });
  },

  onJournal(context, entry) {
    if (get().context?.id !== context) return;
    set({ journal: [...get().journal.slice(-499), entry] });
  },
}));
