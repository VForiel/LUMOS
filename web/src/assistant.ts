// The assistant of the interface (REQ-WEB-12): Claude, called from the browser with the API key of the user,
// using the tools of the MCP server of this instance (/mcp, same session as the interface). The key stays in
// the browser tab and only goes to the Claude API; LUMOS never sees it.

import Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { ElicitRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import i18n from "./i18n";
import { useStore } from "./store";

export const MODEL = "claude-opus-5-5";
const KEY = "lumos.anthropicKey";

type Message = Anthropic.Beta.BetaMessageParam;
type ContentBlock = Anthropic.Beta.BetaContentBlock;
type ToolResultContent = Anthropic.Beta.BetaToolResultBlockParam["content"];

export type Turn =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string }
  | { kind: "tool"; name: string; status: "running" | "done" | "error"; summary?: string; image?: string }
  | { kind: "notice"; text: string };

export function savedKey(): string | null {
  try {
    return window.sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function saveKey(key: string | null): void {
  try {
    if (key) window.sessionStorage.setItem(KEY, key);
    else window.sessionStorage.removeItem(KEY);
  } catch {
    /* storage disabled: the key lives in memory only */
  }
}

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

/** A conversation with the assistant: history (append-only, thinking blocks kept), MCP connection, loop. */
export class Conversation {
  private history: Message[] = [];
  private mcp: Client | null = null;
  private tools: Anthropic.Beta.BetaTool[] = [];
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
    this.tools = tools.map((tool) => ({
      name: tool.name,
      description: tool.description ?? "",
      input_schema: tool.inputSchema as Anthropic.Beta.BetaTool["input_schema"],
    }));
    this.mcp = client;
  }

  stop(): void {
    this.abort?.abort();
  }

  /** Send a message of the user and run the tool loop until Claude answers. */
  async send(text: string, apiKey: string): Promise<void> {
    this.onTurn({ kind: "user", text });
    this.abort = new AbortController();
    try {
      await this.connect();
      const claude = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
      this.history.push({ role: "user", content: text });
      for (let step = 0; step < 25; step += 1) {
        const response = await claude.beta.messages.create(
          {
            model: MODEL,
            max_tokens: 16000,
            thinking: { type: "adaptive" },
            output_config: { effort: "medium" },
            system: systemPrompt(),
            tools: this.tools,
            messages: this.history,
            betas: ["server-side-fallback-2026-07-01"],
            fallbacks: "default",
          },
          { signal: this.abort.signal },
        );
        this.history.push({ role: "assistant", content: response.content as Anthropic.Beta.BetaContentBlockParam[] });
        const said = response.content.filter((b: ContentBlock) => b.type === "text").map((b) => (b as Anthropic.Beta.BetaTextBlock).text).join("\n").trim();
        if (said) this.onTurn({ kind: "assistant", text: said });
        if (response.stop_reason === "refusal") {
          this.onTurn({ kind: "notice", text: i18n.t("assistant.refused") });
          return;
        }
        if (response.stop_reason === "max_tokens") {
          this.onTurn({ kind: "notice", text: i18n.t("assistant.truncated") });
          return;
        }
        if (response.stop_reason === "pause_turn") continue;
        const calls = response.content.filter((b: ContentBlock): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use");
        if (response.stop_reason !== "tool_use" || calls.length === 0) return;
        const results: Anthropic.Beta.BetaToolResultBlockParam[] = [];
        for (const call of calls) results.push(await this.call(call));
        this.history.push({ role: "user", content: results });
        await useStore.getState().refresh();
      }
      this.onTurn({ kind: "notice", text: i18n.t("assistant.tooLong") });
    } catch (error) {
      if (this.abort?.signal.aborted) {
        this.onTurn({ kind: "notice", text: i18n.t("assistant.stopped") });
        this.repair();
        return;
      }
      const message = error instanceof Anthropic.AuthenticationError ? i18n.t("assistant.badKey") : error instanceof Error ? error.message : String(error);
      this.onTurn({ kind: "notice", text: message });
      this.repair();
    } finally {
      this.abort = null;
    }
  }

  /** After an interruption, drop an unanswered tool call so that the history stays valid. */
  private repair(): void {
    const last = this.history[this.history.length - 1];
    if (last?.role === "assistant" && Array.isArray(last.content) && last.content.some((b) => b.type === "tool_use")) {
      this.history.pop();
    }
    if (this.history[this.history.length - 1]?.role === "user") this.history.pop();
  }

  private async call(call: Anthropic.Beta.BetaToolUseBlock): Promise<Anthropic.Beta.BetaToolResultBlockParam> {
    this.onTurn({ kind: "tool", name: call.name, status: "running" });
    try {
      const result = await this.mcp!.callTool({ name: call.name, arguments: call.input as Record<string, unknown> });
      const content: ToolResultContent = [];
      let summary = "";
      let image: string | undefined;
      for (const block of (result.content ?? []) as { type: string; text?: string; data?: string; mimeType?: string }[]) {
        if (block.type === "text" && block.text) {
          content.push({ type: "text", text: block.text });
          summary ||= block.text;
        } else if (block.type === "image" && block.data && block.mimeType) {
          const media = block.mimeType as "image/png" | "image/jpeg" | "image/gif" | "image/webp";
          content.push({ type: "image", source: { type: "base64", media_type: media, data: block.data } });
          image = `data:${block.mimeType};base64,${block.data}`;
        }
      }
      if (!content.length) content.push({ type: "text", text: JSON.stringify(result.structuredContent ?? {}) });
      const failed = Boolean(result.isError);
      this.onTurn({ kind: "tool", name: call.name, status: failed ? "error" : "done", summary: summary.slice(0, 300), image }, true);
      return { type: "tool_result", tool_use_id: call.id, content, is_error: failed };
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      this.onTurn({ kind: "tool", name: call.name, status: "error", summary: text }, true);
      return { type: "tool_result", tool_use_id: call.id, content: text, is_error: true };
    }
  }
}
