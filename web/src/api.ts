// Typed client of the HTTP API of LUMOS (lumos.web, ADR-0018).

import type { Signals } from "./lib/signals";
import type { UnitChoices } from "./lib/units";

export type Info = {
  name: string;
  version: string;
  profile: "local" | "public";
  operation: boolean;
  languages: string[];
  user: string | null;
  role: Role | null;
  auth: { enabled: boolean; required: boolean; methods: string[] };
  assistant: boolean;
};

export type Role = "viewer" | "simulator" | "operator";

export type ElementType = {
  type_id: string;
  title: string;
  summary: string;
  category: string;
  inputs: string[];
  outputs: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  schema: any;
};

export type Palette = { types: ElementType[]; signals: Signals; units: Record<string, UnitChoices> };

export type ElementSummary = {
  id: string;
  type_id: string;
  params: Record<string, unknown>;
  settings: Record<string, unknown>;
  state: Record<string, unknown>;
  inputs: string[];
  outputs: string[];
  actions: string[];
  /** Free parameters and their priors (values in the unit of the parameter), e.g. {kind: "normal", mean: "200 mm", std: "2 mm"}. */
  free?: Record<string, Prior | null>;
};
export type Prior = { kind: "uniform" | "loguniform" | "normal"; [value: string]: string };
export type StudyKind = "sweep" | "sample" | "loop";
export type Feedback = { source: string; target: string; mode: "set" | "add" };
export type LoopRequest = { kind: "loop"; steps: number; period: string; outputs: string[] };
/** A sweep range or list of values, by path "element.field". */
export type SweepSpec = { start: string; stop: string; num: number; log?: boolean } | string[];
export type StudyRequest = {
  kind: "sweep" | "sample";
  outputs: string[];
  parameters?: Record<string, SweepSpec>;
  mode?: "grid" | "zip";
  n?: number;
  noise?: "fixed" | "independent";
};
export type StudyPoint = { params: Record<string, string>; realization: number; outputs: Record<string, PortSummary> };
export type StudyResult = {
  job: string;
  context: string;
  kind: StudyKind;
  mode: string | null;
  noise: string;
  seed: number;
  parameters: string[];
  outputs: string[];
  points: StudyPoint[];
};

export type Connection = { source: string; target: string };

export type GroupInfo = { connections: Connection[]; exposed: Record<string, string> };

/** The context in the foreground of the session, who focused it, and a count of the changes. */
export type Focus = { context: string | null; origin: string | null; count: number };

/** Disk used by the data of the session (archives, recordings), its limit and warnings (P14.21). */
export type StorageInfo = { used_bytes: number; limit_bytes: number | null; disk_free_bytes: number; disk_total_bytes: number; warnings: string[] };

export type ContextSummary = {
  id: string;
  name: string;
  description: string;
  mode: string;
  twin: boolean;
  seed: number;
  band: string[] | null;
  n_wavelengths: number;
  elements: ElementSummary[];
  connections: Connection[];
  feedbacks?: Feedback[];
  outputs: string[];
  groups: Record<string, GroupInfo>;
  ui: { positions?: Record<string, [number, number]>; expanded?: Record<string, boolean>; [key: string]: unknown };
  can_undo: boolean;
  can_redo: boolean;
  /** Grows at every change of the context, whoever made it. */
  revision?: number;
};

export type Issue = { severity: string; element: string | null; message: string; hint: string };

export type JobInfo = {
  id: string;
  kind?: "run" | StudyKind;
  context: string;
  outputs: string[];
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  message: string;
  error: string | null;
  hint: string | null;
  seconds: number | null;
  archive: string | null;
  warnings?: string[];
};

export type PortSummary = {
  port: string;
  kind: string;
  unit: string | null;
  shape: number[];
  values: number[] | null;
  statistics: Record<string, number>;
  details: Record<string, string>;
};

export type Preset = { name: string; title: string; description: string; outputs: string[]; elements: string[] };

export type JournalEntry = {
  kind: string;
  elapsed: number;
  date: string;
  origin: string;
  element?: string | null;
  action?: string | null;
  args?: string;
  mode?: string | null;
  measured?: number | null;
  predicted?: number | null;
  logger?: string | null;
  level?: string | null;
  message?: string;
  wall_time?: string;
};

export type Fragment = {
  kind: "fragment";
  elements: unknown[];
  connections: { from: string; to: string }[];
  positions: Record<string, [number, number]>;
};

export type Device = { vendor: string; model: string; serial: string; firmware: string; url: string };
export type DeviceStatus = { element: string; driver?: string; connected?: boolean; state?: string; [key: string]: unknown };
export type ActionResult = { element: string; action: string; mode: string; predicted_s: number; measured_s: number | null; value: string | null };

export class ApiError extends Error {
  constructor(
    message: string,
    public hint: string,
    public code: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, raw = false): Promise<T> {
  const init: RequestInit = { method, credentials: "same-origin", headers: {} };
  if (body !== undefined) {
    init.body = typeof body === "string" ? body : JSON.stringify(body);
    (init.headers as Record<string, string>)["content-type"] = "application/json";
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let error = { error: response.statusText, hint: "", code: "failed" };
    try {
      error = await response.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(error.error, error.hint, error.code, response.status);
  }
  if (response.status === 204) return undefined as T;
  if (raw) return (await response.text()) as T;
  return (await response.json()) as T;
}

const enc = encodeURIComponent;
const ctx = (id: string) => `/api/contexts/${enc(id)}`;

export const api = {
  info: () => request<Info>("GET", "/api/info"),
  palette: () => request<Palette>("GET", "/api/palette"),
  presets: () => request<Preset[]>("GET", "/api/presets"),
  loadPreset: (name: string) => request<ContextSummary>("POST", `/api/presets/${enc(name)}/load`),
  contexts: () => request<ContextSummary[]>("GET", "/api/contexts"),
  createContext: (name = "", mode = "simulation") => request<ContextSummary>("POST", "/api/contexts", { name, mode }),
  describe: (id: string) => request<ContextSummary>("GET", ctx(id)),
  close: (id: string) => request<void>("DELETE", ctx(id)),
  /** Bring a context to the foreground of the session: the assistant works on it by default. */
  focus: (id: string | null) => request<Focus>("PUT", "/api/focus", { context: id }),
  storage: () => request<StorageInfo>("GET", "/api/storage"),
  setContext: (id: string, change: { name?: string; description?: string; seed?: number }) =>
    request<ContextSummary>("PATCH", ctx(id), change),
  setFree: (id: string, path: string, free: boolean, prior?: Record<string, unknown> | null) =>
    request<ContextSummary>("PUT", `${ctx(id)}/free`, { path, free, prior: prior ?? null }),
  study: (id: string, body: StudyRequest) => request<JobInfo>("POST", `${ctx(id)}/studies`, body),
  loop: (id: string, body: LoopRequest) =>
    request<JobInfo>("POST", `${ctx(id)}/loops`, { steps: body.steps, period: body.period, outputs: body.outputs }),
  assistantLink: () => request<{ url: string | null }>("GET", "/api/assistant/link"),
  createAssistantLink: () => request<{ url: string | null }>("POST", "/api/assistant/link"),
  revokeAssistantLink: () => request<void>("DELETE", "/api/assistant/link"),
  addFeedback: (id: string, source: string, target: string, mode: string) =>
    request<ContextSummary>("POST", `${ctx(id)}/feedbacks`, { source, target, mode }),
  removeFeedback: (id: string, source: string, target: string) =>
    request<ContextSummary>("DELETE", `${ctx(id)}/feedbacks?source=${enc(source)}&target=${enc(target)}`),
  studyResult: (job: string) => request<StudyResult>("GET", `/api/jobs/${enc(job)}/study`),
  studyCsvUrl: (job: string) => `/api/jobs/${enc(job)}/study.csv`,
  importLumos: (text: string) => request<ContextSummary>("POST", "/api/contexts/import", text),
  exportLumos: (id: string) => request<string>("GET", `${ctx(id)}/export.lumos`, undefined, true),
  exportPython: (id: string) => request<string>("GET", `${ctx(id)}/export.py`, undefined, true),
  addElement: (id: string, type_id: string, name?: string, params?: Record<string, unknown>) =>
    request<ElementSummary>("POST", `${ctx(id)}/elements`, { type_id, name, params: params ?? {} }),
  removeElement: (id: string, path: string) => request<void>("DELETE", `${ctx(id)}/elements/${path}`),
  connect: (id: string, source: string, target: string) =>
    request<void>("POST", `${ctx(id)}/connections`, { source, target }),
  disconnect: (id: string, source: string, target: string) =>
    request<void>("DELETE", `${ctx(id)}/connections?source=${enc(source)}&target=${enc(target)}`),
  setParam: (id: string, path: string, value: unknown) =>
    request<ElementSummary>("PATCH", `${ctx(id)}/params`, { path, value }),
  setObservation: (id: string, minimum: string, maximum: string, n_wavelengths: number) =>
    request<ContextSummary>("PUT", `${ctx(id)}/observation`, { minimum, maximum, n_wavelengths }),
  updateUi: (id: string, ui: Record<string, unknown>, record = true) =>
    request<ContextSummary>("PUT", `${ctx(id)}/ui${record ? "" : "?record=false"}`, ui),
  undo: (id: string) => request<ContextSummary>("POST", `${ctx(id)}/undo`),
  redo: (id: string) => request<ContextSummary>("POST", `${ctx(id)}/redo`),
  rename: (id: string, path: string, name: string) =>
    request<ContextSummary>("POST", `${ctx(id)}/elements/${path}/rename`, { name }),
  group: (id: string, elements: string[], name: string) =>
    request<ContextSummary>("POST", `${ctx(id)}/groups`, { elements, name }),
  ungroup: (id: string, path: string) => request<ContextSummary>("POST", `${ctx(id)}/groups/${path}/ungroup`),
  copy: (id: string, elements: string[]) => request<Fragment>("POST", `${ctx(id)}/copy`, { elements }),
  paste: (id: string, fragment: Fragment) => request<{ ids: string[] }>("POST", `${ctx(id)}/paste`, { fragment }),
  exportElement: (id: string, path: string) =>
    request<string>("GET", `${ctx(id)}/elements/${path}/export.lumos`, undefined, true),
  insertBlock: (id: string, text: string) => request<ElementSummary>("POST", `${ctx(id)}/blocks`, text),
  validate: (id: string) => request<Issue[]>("GET", `${ctx(id)}/validate`),
  run: (id: string, outputs: string[] = [], archive = false) =>
    request<JobInfo>("POST", `${ctx(id)}/runs`, { outputs, archive }),
  job: (job: string) => request<JobInfo>("GET", `/api/jobs/${enc(job)}`),
  result: (job: string, port: string) => request<PortSummary>("GET", `/api/jobs/${enc(job)}/results/${port}`),
  inspect: (id: string, port: string) => request<PortSummary>("GET", `${ctx(id)}/ports/${port}`),
  figureUrl: (id: string, port: string) => `${ctx(id)}/ports/${port}/figure.png`,
  resultFigureUrl: (job: string, port: string) => `/api/jobs/${enc(job)}/results/${port}/figure.png`,
  portDownloadUrl: (id: string, port: string, format: string) => `${ctx(id)}/ports/${port}/download?format=${format}`,
  resultDownloadUrl: (job: string, port: string, format: string) =>
    `/api/jobs/${enc(job)}/results/${port}/download?format=${format}`,
  archiveUrl: (path: string) => `/api/archives/download?path=${enc(path)}`,
  cancel: (job: string) => request<JobInfo>("POST", `/api/jobs/${enc(job)}/cancel`),
  journal: (id: string, limit = 200) => request<JournalEntry[]>("GET", `${ctx(id)}/journal?limit=${limit}`),
  discover: (tcp: string[], serial = false, visa = false) =>
    request<Device[]>("POST", "/api/devices/discover", { tcp, serial, visa }),
  associate: (id: string) => request<Record<string, Device>>("POST", `${ctx(id)}/devices/associate`, {}),
  devices: (id: string) => request<DeviceStatus[]>("GET", `${ctx(id)}/devices`),
  confirmation: (id: string, element: string) =>
    request<{ device: string | null }>("GET", `${ctx(id)}/elements/${element}/confirmation`),
  action: (id: string, element: string, action: string, args: unknown[], confirmed: boolean) =>
    request<ActionResult>("POST", `${ctx(id)}/actions`, { element, action, args, kwargs: {}, confirmed }),
  stop: (id: string) => request<void>("POST", `${ctx(id)}/stop`),
  me: () => request<{ user: string | null; role: Role | null }>("GET", "/api/auth/me"),
  login: (username: string, password: string) =>
    request<{ user: string; role: Role }>("POST", "/api/auth/login", { username, password }),
  challenge: (username: string) =>
    request<{ challenge: string; namespace: string; expires_in: number }>("POST", "/api/auth/challenge", {
      username,
    }),
  sshLogin: (username: string, challenge: string, signature: string) =>
    request<{ user: string; role: Role }>("POST", "/api/auth/ssh", { username, challenge, signature }),
  logout: () => request<void>("POST", "/api/auth/logout"),
};

/** Save ``content`` as a file in the downloads of the browser. */
export function download(filename: string, content: string | Blob, type = "application/json"): void {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A file name from a context name: "Vortex coronagraph" → "Vortex_coronagraph". */
export function fileName(name: string, suffix: string): string {
  const stem = name.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[_.]+|[_.]+$/g, "") || "context";
  return `${stem}${suffix}`;
}
