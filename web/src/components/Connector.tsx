// Connector link (P14.15): the MCP server of this session at a secret URL, to add LUMOS as a connector of
// Claude or ChatGPT and use their subscription instead of an API key. What the assistant there does appears
// in this interface.

import { Check, Copy, Link2, Link2Off, Loader2, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../api";
import { useStore } from "../store";
import { inputClass } from "./Fields";

export function ConnectorLink() {
  const { t } = useTranslation();
  const [url, setUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    void api
      .assistantLink()
      .then((answer) => setUrl(answer.url))
      .catch(() => undefined);
  }, []);
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
  const run = async (action: () => Promise<{ url: string | null } | void>) => {
    setBusy(true);
    try {
      const answer = await action();
      setUrl(answer ? answer.url : null);
    } catch (error) {
      useStore.getState().fail(error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section data-testid="assistant-connector" className="space-y-2 rounded-xl border border-slate-200 p-3 dark:border-slate-700">
      <p className="flex items-center gap-2 font-semibold">
        <Link2 className="size-4 text-accent-500" />
        {t("connector.title")}
      </p>
      <p className="text-xs text-slate-600 dark:text-slate-400">{t("connector.hint")}</p>
      {url ? (
        <>
          <div className="flex gap-1.5">
            <input data-testid="assistant-connector-url" readOnly value={url} className={`${inputClass} font-mono text-xs`} onFocus={(e) => e.target.select()} />
            <button
              type="button"
              title={t("connector.copy")}
              aria-label={t("connector.copy")}
              onClick={() => {
                void navigator.clipboard?.writeText(url).then(() => setCopied(true));
                window.setTimeout(() => setCopied(false), 1500);
              }}
              className="grid size-8 shrink-0 place-items-center rounded-md border border-slate-200 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              {copied ? <Check className="size-4 text-emerald-500" /> : <Copy className="size-4" />}
            </button>
            <button
              type="button"
              data-testid="assistant-connector-revoke"
              title={t("connector.revoke")}
              aria-label={t("connector.revoke")}
              disabled={busy}
              onClick={() => void run(() => api.revokeAssistantLink())}
              className="grid size-8 shrink-0 place-items-center rounded-md border border-slate-200 text-red-600 hover:bg-red-50 dark:border-slate-700 dark:hover:bg-red-950"
            >
              <Link2Off className="size-4" />
            </button>
          </div>
          <p className="text-[11px] text-amber-700 dark:text-amber-300">{t("connector.secret")}</p>
          {local && (
            <p className="flex items-start gap-1.5 text-[11px] text-amber-700 dark:text-amber-300">
              <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
              {t("connector.local")}
            </p>
          )}
          <ul className="list-disc space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-300">
            <li>{t("connector.claude")}</li>
            <li>{t("connector.chatgpt")}</li>
            <li>
              {t("connector.code")} <code className="rounded bg-slate-100 px-1 font-mono text-[11px] dark:bg-slate-800">claude mcp add --transport http lumos {url}</code>
            </li>
          </ul>
        </>
      ) : (
        <button
          type="button"
          data-testid="assistant-connector-create"
          disabled={busy}
          onClick={() => void run(() => api.createAssistantLink())}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Link2 className="size-4" />}
          {t("connector.create")}
        </button>
      )}
    </section>
  );
}
