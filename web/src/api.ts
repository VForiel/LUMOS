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
};

export type ContextSummary = {
  id: string;
  name: string;
  description: string;
  mode: string;
  twin: boolean;
  band: string[] | null;
  n_wavelengths: number;
  elements: ElementSummary[];
  connections: { source: string; target: string }[];
  outputs: string[];
  ui?: { positions?: Record<string, [number, number]>; [key: string]: unknown };
  can_undo?: boolean;
  can_redo?: boolean;
};

export type Issue = { severity: string; element: string | null; message: string; hint: string };

export type JobInfo = {
  id: string;
  context: string;
  outputs: string[];
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  message: string;
  error: string | null;
  hint: string | null;
  seconds: number | null;
  archive: string | null;
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

export type Preset = { name: string; description: string; outputs: string[] };

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
    if (typeof body === "string") {
      init.body = body;
      (init.headers as Record<string, string>)["content-type"] = "application/json";
    } else {
      init.body = JSON.stringify(body);
      (init.headers as Record<string, string>)["content-type"] = "application/json";
    }
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
  validate: (id: string) => request<Issue[]>("GET", `${ctx(id)}/validate`),
  run: (id: string, outputs: string[] = [], archive = false) =>
    request<JobInfo>("POST", `${ctx(id)}/runs`, { outputs, archive }),
  job: (job: string) => request<JobInfo>("GET", `/api/jobs/${enc(job)}`),
  result: (job: string, port: string) => request<PortSummary>("GET", `/api/jobs/${enc(job)}/results/${port}`),
  inspect: (id: string, port: string) => request<PortSummary>("GET", `${ctx(id)}/ports/${port}`),
  figureUrl: (id: string, port: string) => `${ctx(id)}/ports/${port}/figure.png`,
  journal: (id: string, limit = 200) => request<JournalEntry[]>("GET", `${ctx(id)}/journal?limit=${limit}`),
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
