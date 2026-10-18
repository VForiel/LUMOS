// The assistant of the interface (REQ-WEB-12): a language model called from the browser with the key of the
// user, using the tools of the MCP server of this instance (/mcp, same session as the interface). Keys stay in
// the browser tab and only go to their provider; LUMOS never sees them.
//
// Providers: Claude (Anthropic Messages API) and every provider of the OpenAI chat completions API with tools:
// OpenAI, Google Gemini (OpenAI-compatible endpoint), OpenRouter (key, or login with OAuth PKCE) and any
// compatible endpoint (Mistral, Groq, a local Ollama or vLLM...).

import Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { ElicitRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import i18n from "./i18n";
import { OPENROUTER_CALLBACK } from "./oauth";
import { storage, useStore } from "./store";

export const MODEL = "claude-opus-5-5";

export type ProviderId = "anthropic" | "openai" | "gemini" | "openrouter" | "custom";
export type Provider = {
  id: ProviderId;
  label: string;
  protocol: "anthropic" | "openai";
  baseUrl: string;
  model: string;
  keyRequired: boolean;
  keyPlaceholder: string;
  keysUrl: string;
};

export const PROVIDERS: Provider[] = [
  { id: "anthropic", label: "Claude (Anthropic)", protocol: "anthropic", baseUrl: "https://api.anthropic.com", model: MODEL, keyRequired: true, keyPlaceholder: "sk-ant-…", keysUrl: "https://console.anthropic.com/settings/keys" },
  { id: "openai", label: "OpenAI", protocol: "openai", baseUrl: "https://api.openai.com/v1", model: "gpt-5", keyRequired: true, keyPlaceholder: "sk-…", keysUrl: "https://platform.openai.com/api-keys" },
  { id: "gemini", label: "Google Gemini", protocol: "openai", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", model: "gemini-2.5-pro", keyRequired: true, keyPlaceholder: "AIza…", keysUrl: "https://aistudio.google.com/apikey" },
  { id: "openrouter", label: "OpenRouter", protocol: "openai", baseUrl: "https://openrouter.ai/api/v1", model: "openrouter/auto", keyRequired: true, keyPlaceholder: "sk-or-…", keysUrl: "https://openrouter.ai/settings/keys" },
  { id: "custom", label: "OpenAI-compatible", protocol: "openai", baseUrl: "", model: "", keyRequired: false, keyPlaceholder: "", keysUrl: "" },
];

export function provider(id: ProviderId): Provider {
  return PROVIDERS.find((p) => p.id === id) ?? PROVIDERS[0];
}

/** What the user chose: provider, model and endpoint (kept by the browser), and the key (kept by the tab). */
export type Settings = { provider: ProviderId; model: string; baseUrl: string; key: string | null };

const PROVIDER_KEY = "lumos.assistant.provider";
const keyName = (id: ProviderId) => (id === "anthropic" ? "lumos.anthropicKey" : `lumos.assistant.${id}.key`);

function session(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function savedSettings(id?: ProviderId): Settings {
  const chosen = id ?? ((storage()?.getItem(PROVIDER_KEY) as ProviderId | null) ?? "anthropic");
  const known = provider(chosen);
  return {
    provider: known.id,
    model: storage()?.getItem(`lumos.assistant.${known.id}.model`) || known.model,
    baseUrl: storage()?.getItem(`lumos.assistant.${known.id}.baseUrl`) || known.baseUrl,
    key: session()?.getItem(keyName(known.id)) ?? null,
  };
}

/** Keep the settings: the key in the tab only (session storage), the rest in the browser. */
export function saveSettings(settings: Settings): void {
  try {
    storage()?.setItem(PROVIDER_KEY, settings.provider);
    storage()?.setItem(`lumos.assistant.${settings.provider}.model`, settings.model);
    if (settings.provider === "custom") storage()?.setItem("lumos.assistant.custom.baseUrl", settings.baseUrl);
    if (settings.key) session()?.setItem(keyName(settings.provider), settings.key);
    else session()?.removeItem(keyName(settings.provider));
  } catch {
    /* storage disabled: the settings live in memory only */
  }
}

/** Whether the assistant can be used with these settings. */
export function ready(settings: Settings): boolean {
  const known = provider(settings.provider);
  return Boolean(settings.model && settings.baseUrl && (settings.key || !known.keyRequired));
}

// --- OpenRouter login (OAuth PKCE, RFC 7636) ----------------------------------------------------------------

function base64url(bytes: ArrayBuffer | Uint8Array): string {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let text = "";
  for (const byte of data) text += String.fromCharCode(byte);
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/**
 * Log in to OpenRouter in a popup: the user authorises LUMOS on openrouter.ai, which creates a key on their
 * account (with their own limits and credits); the code returned to the callback is exchanged for that key.
 */
export async function openRouterLogin(): Promise<string> {
  const verifier = base64url(crypto.getRandomValues(new Uint8Array(48)));
  const challenge = base64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
  const state = base64url(crypto.getRandomValues(new Uint8Array(16)));
  const callback = `${window.location.origin}${OPENROUTER_CALLBACK}`;
  const url = `https://openrouter.ai/auth?callback_url=${encodeURIComponent(callback)}&code_challenge=${challenge}&code_challenge_method=S256&state=${state}`;
  const popup = window.open(url, "lumos-openrouter", "popup,width=520,height=720");
  if (!popup) throw new Error(i18n.t("assistant.popupBlocked"));
  const code = await new Promise<string>((resolve, reject) => {
    const listen = (event: MessageEvent) => {
      if (event.origin !== window.location.origin || event.data?.type !== "lumos-openrouter") return;
      window.removeEventListener("message", listen);
      window.clearInterval(watch);
      if (event.data.state !== state) reject(new Error(i18n.t("assistant.loginFailed")));
      else if (!event.data.code) reject(new Error(i18n.t("assistant.loginCancelled")));
      else resolve(String(event.data.code));
    };
    const watch = window.setInterval(() => {
      if (popup.closed) {
        window.clearInterval(watch);
        window.setTimeout(() => {
          window.removeEventListener("message", listen);
          reject(new Error(i18n.t("assistant.loginCancelled")));
        }, 500);
      }
    }, 400);
    window.addEventListener("message", listen);
  });
  const response = await fetch("https://openrouter.ai/api/v1/auth/keys", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ code, code_verifier: verifier, code_challenge_method: "S256" }),
  });
  if (!response.ok) throw new Error(`${i18n.t("assistant.loginFailed")} (${response.status})`);
  const { key } = (await response.json()) as { key?: string };
  if (!key) throw new Error(i18n.t("assistant.loginFailed"));
  return key;
}

// --- Conversation ---------------------------------------------------------------------------------------------

type Message = Anthropic.Beta.BetaMessageParam;
type ContentBlock = Anthropic.Beta.BetaContentBlock;
type ToolResultContent = Anthropic.Beta.BetaToolResultBlockParam["content"];
type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
};
type McpTool = { name: string; description: string; schema: Record<string, unknown> };
type ToolOutcome = { text: string; image?: { data: string; mimeType: string }; failed: boolean };

export type Turn =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string }
  | { kind: "tool"; name: string; status: "running" | "done" | "error"; summary?: string; image?: string }
  | { kind: "notice"; text: string };

function systemPrompt(): string {
  const context = useStore.getState().context;
  const open = context
    ? `The user has the context "${context.id}" (${context.name || "untitled"}) open in the interface: pass context="${context.id}" to the tools unless the user names another context. Its elements: ${context.elements.map((e) => `${e.id} (${e.type_id})`).join(", ") || "none yet"}.`
    : "No context is open.";
  return [
    "You are the assistant of the LUMOS web interface. LUMOS simulates light from a source or a scene to detected data through optical elements connected in a graph (a context), and can drive optical benches.",
    "The interface shows the context the user is editing; what you change through the tools appears there at once.",
    open,
    "Write every physical quantity with its unit (e.g. \"0.5 mm\", \"1.6 um\"). Check a context with validate_context before running it.",
    `Answer in the language of the user (the interface is in ${i18n.language === "fr" ? "French" : "English"}), briefly: the interface already shows figures and values.`,
  ].join("\n");
}

const MAX_STEPS = 25;

/** A conversation with the assistant: history (append-only), MCP connection, tool loop. */
export class Conversation {
  private history: Message[] = [];
  private chat: ChatMessage[] = [];
  private mcp: Client | null = null;
  private tools: McpTool[] = [];
  private abort: AbortController | null = null;

  constructor(private onTurn: (turn: Turn, replace?: boolean) => void) {}

  private async connect(): Promise<void> {
    if (this.mcp) return;
    const client = new Client({ name: "lumos-web", version: "1" }, { capabilities: { elicitation: {} } });
    client.setRequestHandler(ElicitRequestSchema, async (request) => {
      const accepted = await useStore.getState().ask(String(request.params.message ?? ""));
      return accepted ? { action: "accept", content: { confirm: true } } : { action: "decline" };
    });
    await client.connect(new StreamableHTTPClientTransport(new URL("/mcp", window.location.origin)));
    const { tools } = await client.listTools();
    this.tools = tools.map((tool) => ({ name: tool.name, description: tool.description ?? "", schema: tool.inputSchema as Record<string, unknown> }));
    this.mcp = client;
  }

  stop(): void {
    this.abort?.abort();
  }

  /** Send a message of the user and run the tool loop until the model answers. */
  async send(text: string, settings: Settings): Promise<void> {
    this.onTurn({ kind: "user", text });
    this.abort = new AbortController();
    try {
      await this.connect();
      if (provider(settings.provider).protocol === "anthropic") await this.claude(text, settings);
      else await this.completions(text, settings);
    } catch (error) {
      if (this.abort?.signal.aborted) this.onTurn({ kind: "notice", text: i18n.t("assistant.stopped") });
      else {
        const message =
          error instanceof Anthropic.AuthenticationError || (error instanceof HttpError && (error.status === 401 || error.status === 403))
            ? i18n.t("assistant.badKey")
            : error instanceof Error
              ? error.message
              : String(error);
        this.onTurn({ kind: "notice", text: message });
      }
      this.repair();
    } finally {
      this.abort = null;
    }
  }

  private async claude(text: string, settings: Settings): Promise<void> {
    const claude = new Anthropic({ apiKey: settings.key ?? "", dangerouslyAllowBrowser: true });
    const tools = this.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.schema as Anthropic.Beta.BetaTool["input_schema"] }));
    this.history.push({ role: "user", content: text });
    for (let step = 0; step < MAX_STEPS; step += 1) {
      const response = await claude.beta.messages.create(
        {
          model: settings.model || MODEL,
          max_tokens: 16000,
          thinking: { type: "adaptive" },
          output_config: { effort: "medium" },
          system: systemPrompt(),
          tools,
          messages: this.history,
          betas: ["server-side-fallback-2026-07-01"],
          fallbacks: "default",
        },
        { signal: this.abort!.signal },
      );
      this.history.push({ role: "assistant", content: response.content as Anthropic.Beta.BetaContentBlockParam[] });
      const said = response.content.filter((b: ContentBlock) => b.type === "text").map((b) => (b as Anthropic.Beta.BetaTextBlock).text).join("\n").trim();
      if (said) this.onTurn({ kind: "assistant", text: said });
      if (response.stop_reason === "refusal") return this.onTurn({ kind: "notice", text: i18n.t("assistant.refused") });
      if (response.stop_reason === "max_tokens") return this.onTurn({ kind: "notice", text: i18n.t("assistant.truncated") });
      if (response.stop_reason === "pause_turn") continue;
      const calls = response.content.filter((b: ContentBlock): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use");
      if (response.stop_reason !== "tool_use" || calls.length === 0) return;
      const results: Anthropic.Beta.BetaToolResultBlockParam[] = [];
      for (const call of calls) {
        const outcome = await this.call(call.name, call.input as Record<string, unknown>);
        const content: ToolResultContent = [{ type: "text", text: outcome.text }];
        if (outcome.image) {
          const media = outcome.image.mimeType as "image/png" | "image/jpeg" | "image/gif" | "image/webp";
          content.push({ type: "image", source: { type: "base64", media_type: media, data: outcome.image.data } });
        }
        results.push({ type: "tool_result", tool_use_id: call.id, content, is_error: outcome.failed });
      }
      this.history.push({ role: "user", content: results });
      await useStore.getState().refresh();
    }
    this.onTurn({ kind: "notice", text: i18n.t("assistant.tooLong") });
  }

  /** The tool loop of the OpenAI chat completions API (OpenAI, Gemini, OpenRouter, compatible endpoints). */
  private async completions(text: string, settings: Settings): Promise<void> {
    const tools = this.tools.map((tool) => ({ type: "function", function: { name: tool.name, description: tool.description, parameters: tool.schema } }));
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (settings.key) headers.authorization = `Bearer ${settings.key}`;
    if (settings.provider === "openrouter") {
      headers["HTTP-Referer"] = window.location.origin;
      headers["X-Title"] = "LUMOS";
    }
    const url = `${settings.baseUrl.replace(/\/+$/, "")}/chat/completions`;
    this.chat.push({ role: "user", content: text });
    for (let step = 0; step < MAX_STEPS; step += 1) {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify({ model: settings.model, messages: [{ role: "system", content: systemPrompt() }, ...this.chat], tools }),
        signal: this.abort!.signal,
      });
      if (!response.ok) {
        let detail = response.statusText;
        try {
          const body = await response.json();
          detail = body?.error?.message ?? body?.message ?? JSON.stringify(body);
        } catch {
          /* not JSON */
        }
        throw new HttpError(response.status, `${response.status}: ${detail}`);
      }
      const body = await response.json();
      const choice = body?.choices?.[0];
      const message = (choice?.message ?? { role: "assistant", content: "" }) as ChatMessage;
      this.chat.push({ role: "assistant", content: message.content ?? null, ...(message.tool_calls?.length ? { tool_calls: message.tool_calls } : {}) });
      const said = (message.content ?? "").trim();
      if (said) this.onTurn({ kind: "assistant", text: said });
      if (choice?.finish_reason === "length") return this.onTurn({ kind: "notice", text: i18n.t("assistant.truncated") });
      if (!message.tool_calls?.length) return;
      for (const call of message.tool_calls) {
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse(call.function.arguments || "{}");
        } catch {
          /* the model wrote invalid JSON: the tool reports the missing arguments */
        }
        const outcome = await this.call(call.function.name, args);
        const note = outcome.image ? `\n[${i18n.t("assistant.figureShown")}]` : "";
        this.chat.push({ role: "tool", tool_call_id: call.id, content: `${outcome.failed ? "Error: " : ""}${outcome.text}${note}` });
      }
      await useStore.getState().refresh();
    }
    this.onTurn({ kind: "notice", text: i18n.t("assistant.tooLong") });
  }

  /** After an interruption, drop an unanswered tool call so that the history stays valid. */
  private repair(): void {
    const last = this.history[this.history.length - 1];
    if (last?.role === "assistant" && Array.isArray(last.content) && last.content.some((b) => b.type === "tool_use")) this.history.pop();
    if (this.history[this.history.length - 1]?.role === "user") this.history.pop();
    while (this.chat.length && (this.chat[this.chat.length - 1].role !== "assistant" || this.chat[this.chat.length - 1].tool_calls)) this.chat.pop();
  }

  private async call(name: string, input: Record<string, unknown>): Promise<ToolOutcome> {
    this.onTurn({ kind: "tool", name, status: "running" });
    try {
      const result = await this.mcp!.callTool({ name, arguments: input });
      const texts: string[] = [];
      let image: ToolOutcome["image"];
      for (const block of (result.content ?? []) as { type: string; text?: string; data?: string; mimeType?: string }[]) {
        if (block.type === "text" && block.text) texts.push(block.text);
        else if (block.type === "image" && block.data && block.mimeType) image = { data: block.data, mimeType: block.mimeType };
      }
      const text = texts.join("\n") || JSON.stringify(result.structuredContent ?? {});
      const failed = Boolean(result.isError);
      this.onTurn({ kind: "tool", name, status: failed ? "error" : "done", summary: (texts[0] ?? "").slice(0, 300), image: image && `data:${image.mimeType};base64,${image.data}` }, true);
      return { text, image, failed };
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      this.onTurn({ kind: "tool", name, status: "error", summary: text }, true);
      return { text, failed: true };
    }
  }
}

class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
