// Assistant panel (REQ-WEB-12): a conversation with Claude using the tools of LUMOS on the open context.

import { Bot, KeyRound, Loader2, Send, Square, Wrench } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Conversation, MODEL, saveKey, savedKey, type Turn } from "../assistant";
import { inputClass } from "./Fields";

export function AssistantPanel() {
  const { t } = useTranslation();
  const [key, setKey] = useState<string | null>(savedKey());
  const [draftKey, setDraftKey] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const conversation = useRef<Conversation | null>(null);
  const end = useRef<HTMLDivElement>(null);

  if (!conversation.current) {
    conversation.current = new Conversation((turn, replace) =>
      setTurns((previous) => (replace && previous.length ? [...previous.slice(0, -1), turn] : [...previous, turn])),
    );
  }
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [turns.length, busy]);

  const send = async () => {
    const message = text.trim();
    if (!message || !key || busy) return;
    setText("");
    setBusy(true);
    try {
      await conversation.current!.send(message, key);
    } finally {
      setBusy(false);
    }
  };

  if (!key) {
    return (
      <div data-testid="assistant" className="mx-auto flex h-full max-w-xl flex-col justify-center gap-3 p-4 text-sm">
        <p className="flex items-center gap-2 font-semibold">
          <KeyRound className="size-4 text-accent-500" />
          {t("assistant.keyTitle")}
        </p>
        <p className="text-slate-600 dark:text-slate-400">{t("assistant.keyHint")}</p>
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (!draftKey.trim()) return;
            saveKey(draftKey.trim());
            setKey(draftKey.trim());
            setDraftKey("");
          }}
        >
          <input data-testid="assistant-key" type="password" autoComplete="off" placeholder="sk-ant-…" className={inputClass} value={draftKey} onChange={(e) => setDraftKey(e.target.value)} />
          <button type="submit" data-testid="assistant-key-save" className="rounded-lg bg-accent-600 px-3 text-white hover:bg-accent-700">
            {t("common.apply")}
          </button>
        </form>
      </div>
    );
  }

  return (
    <div data-testid="assistant" className="flex h-full flex-col text-sm">
      <div className="flex-1 space-y-2 overflow-y-auto p-3">
        {turns.length === 0 && (
          <p className="flex items-center gap-2 text-slate-500">
            <Bot className="size-4" />
            {t("assistant.empty")}
          </p>
        )}
        {turns.map((turn, index) => {
          switch (turn.kind) {
            case "user":
              return (
                <p key={index} className="ml-auto max-w-[75%] rounded-xl bg-accent-600 px-3 py-1.5 whitespace-pre-wrap text-white">
                  {turn.text}
                </p>
              );
            case "assistant":
              return (
                <p key={index} className="max-w-[85%] rounded-xl bg-slate-100 px-3 py-1.5 whitespace-pre-wrap dark:bg-slate-800">
                  {turn.text}
                </p>
              );
            case "tool":
              return (
                <div key={index} data-testid="tool-call" className="flex max-w-[85%] items-start gap-2 text-xs text-slate-500">
                  {turn.status === "running" ? <Loader2 className="mt-0.5 size-3.5 animate-spin" /> : <Wrench className={`mt-0.5 size-3.5 ${turn.status === "error" ? "text-red-500" : "text-emerald-500"}`} />}
                  <span className="min-w-0">
                    <span className="font-mono">{turn.name}</span>
                    {turn.summary && <span className="line-clamp-2 text-slate-400">{turn.summary}</span>}
                    {turn.image && <img src={turn.image} alt={turn.name} className="mt-1 h-28 rounded border border-slate-200 dark:border-slate-700" />}
                  </span>
                </div>
              );
            default:
              return (
                <p key={index} className="text-xs text-amber-700 dark:text-amber-300">
                  {turn.text}
                </p>
              );
          }
        })}
        {busy && <Loader2 className="size-4 animate-spin text-accent-500" />}
        <div ref={end} />
      </div>
      <div className="flex items-end gap-2 border-t border-slate-100 p-2 dark:border-slate-800">
        <textarea
          data-testid="assistant-input"
          rows={2}
          value={text}
          placeholder={t("assistant.placeholder")}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          className={`${inputClass} resize-none`}
        />
        {busy ? (
          <button type="button" onClick={() => conversation.current?.stop()} className="grid size-9 place-items-center rounded-lg border border-slate-200 dark:border-slate-700" aria-label={t("run.cancel")}>
            <Square className="size-4" />
          </button>
        ) : (
          <button type="button" data-testid="assistant-send" onClick={() => void send()} className="grid size-9 place-items-center rounded-lg bg-accent-600 text-white hover:bg-accent-700" aria-label={t("assistant.send")}>
            <Send className="size-4" />
          </button>
        )}
      </div>
      <p className="flex justify-between px-3 pb-1 text-[10px] text-slate-400">
        <span>{t("assistant.footer", { model: MODEL })}</span>
        <button
          type="button"
          data-testid="assistant-key-forget"
          onClick={() => {
            saveKey(null);
            setKey(null);
          }}
          className="underline hover:text-slate-600"
        >
          {t("assistant.forget")}
        </button>
      </p>
    </div>
  );
}
