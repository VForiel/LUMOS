// Assistant panel (REQ-WEB-12): a conversation with a language model using the tools of LUMOS on the open
// context. The user chooses the provider (Claude, OpenAI, Gemini, OpenRouter or any compatible endpoint) and
// gives a key, or logs in to OpenRouter.

import { Bot, ExternalLink, KeyRound, Loader2, LogIn, Send, Settings2, Square, Wrench } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Conversation, openRouterLogin, provider, PROVIDERS, ready, saveSettings, savedSettings, type ProviderId, type Settings, type Turn } from "../assistant";
import { inputClass } from "./Fields";

function SettingsForm({ initial, onDone }: { initial: Settings; onDone: (settings: Settings) => void }) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<Settings>(initial);
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const known = provider(draft.provider);
  const choose = (id: ProviderId) => {
    setDraft(savedSettings(id));
    setKey("");
    setProblem(null);
  };
  const apply = (settings: Settings) => {
    if (!ready(settings)) {
      setProblem(t(settings.baseUrl ? (settings.model ? "assistant.keyMissing" : "assistant.modelMissing") : "assistant.urlMissing"));
      return;
    }
    saveSettings(settings);
    onDone(settings);
  };
  const login = async () => {
    setBusy(true);
    setProblem(null);
    try {
      apply({ ...draft, key: await openRouterLogin() });
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div data-testid="assistant" className="h-full overflow-y-auto">
      <form
        className="mx-auto flex max-w-xl flex-col gap-3 p-4 text-sm"
        onSubmit={(event) => {
          event.preventDefault();
          apply({ ...draft, key: key.trim() || draft.key });
        }}
      >
        <p className="flex items-center gap-2 font-semibold">
          <KeyRound className="size-4 text-accent-500" />
          {t("assistant.settingsTitle")}
        </p>
        <p className="text-slate-600 dark:text-slate-400">{t("assistant.settingsHint")}</p>
        <div className="grid grid-cols-[7rem_1fr] items-center gap-2">
          <label htmlFor="assistant-provider" className="text-slate-600 dark:text-slate-300">
            {t("assistant.provider")}
          </label>
          <select id="assistant-provider" data-testid="assistant-provider" className={inputClass} value={draft.provider} onChange={(e) => choose(e.target.value as ProviderId)}>
            {PROVIDERS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id === "custom" ? t("assistant.custom") : p.label}
              </option>
            ))}
          </select>
          {draft.provider === "custom" && (
            <>
              <label htmlFor="assistant-base-url" className="text-slate-600 dark:text-slate-300">
                {t("assistant.baseUrl")}
              </label>
              <input
                id="assistant-base-url"
                data-testid="assistant-base-url"
                className={inputClass}
                placeholder="http://localhost:11434/v1"
                value={draft.baseUrl}
                onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value.trim() })}
              />
            </>
          )}
          <label htmlFor="assistant-model" className="text-slate-600 dark:text-slate-300">
            {t("assistant.model")}
          </label>
          <input id="assistant-model" data-testid="assistant-model" className={inputClass} value={draft.model} onChange={(e) => setDraft({ ...draft, model: e.target.value.trim() })} />
          <label htmlFor="assistant-key" className="text-slate-600 dark:text-slate-300">
            {t(known.keyRequired ? "assistant.key" : "assistant.keyOptional")}
          </label>
          <input
            id="assistant-key"
            data-testid="assistant-key"
            type="password"
            autoComplete="off"
            placeholder={draft.key ? t("assistant.keyKept") : known.keyPlaceholder}
            className={inputClass}
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
        </div>
        {known.keysUrl && (
          <a href={known.keysUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 self-start text-xs text-accent-600 hover:underline dark:text-accent-300">
            <ExternalLink className="size-3" />
            {t("assistant.getKey", { provider: known.label })}
          </a>
        )}
        <p className="text-xs text-slate-500 dark:text-slate-400">{t(`assistant.about.${draft.provider}`)}</p>
        {problem && <p className="text-xs text-red-600 dark:text-red-400">{problem}</p>}
        <div className="flex flex-wrap justify-end gap-2">
          {draft.provider === "openrouter" && (
            <button
              type="button"
              data-testid="assistant-openrouter-login"
              disabled={busy}
              onClick={() => void login()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 font-medium hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              {busy ? <Loader2 className="size-4 animate-spin" /> : <LogIn className="size-4" />}
              {t("assistant.openrouterLogin")}
            </button>
          )}
          <button type="submit" data-testid="assistant-key-save" className="rounded-lg bg-accent-600 px-3 py-1.5 font-medium text-white hover:bg-accent-700">
            {t("common.apply")}
          </button>
        </div>
      </form>
    </div>
  );
}

export function AssistantPanel() {
  const { t } = useTranslation();
  const [settings, setSettings] = useState<Settings>(savedSettings());
  const [editing, setEditing] = useState(!ready(settings));
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const conversation = useRef<Conversation | null>(null);
  const end = useRef<HTMLDivElement>(null);

  const restart = () => {
    conversation.current = new Conversation((turn, replace) =>
      setTurns((previous) => (replace && previous.length ? [...previous.slice(0, -1), turn] : [...previous, turn])),
    );
  };
  if (!conversation.current) restart();
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [turns.length, busy]);

  const send = async () => {
    const message = text.trim();
    if (!message || busy) return;
    setText("");
    setBusy(true);
    try {
      await conversation.current!.send(message, settings);
    } finally {
      setBusy(false);
    }
  };

  if (editing) {
    return (
      <SettingsForm
        initial={settings}
        onDone={(chosen) => {
          if (chosen.provider !== settings.provider || chosen.model !== settings.model || chosen.baseUrl !== settings.baseUrl) {
            restart(); // another model: a new conversation
            setTurns([]);
          }
          setSettings(chosen);
          setEditing(false);
        }}
      />
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
      <p className="flex items-center justify-between gap-2 px-3 pb-1 text-[10px] text-slate-400">
        <span className="truncate">{t("assistant.footer", { model: `${provider(settings.provider).id === "custom" ? settings.baseUrl : provider(settings.provider).label} · ${settings.model}` })}</span>
        <span className="flex shrink-0 gap-3">
          <button type="button" data-testid="assistant-settings" onClick={() => setEditing(true)} className="inline-flex items-center gap-1 underline hover:text-slate-600">
            <Settings2 className="size-3" />
            {t("assistant.change")}
          </button>
          <button
            type="button"
            data-testid="assistant-key-forget"
            onClick={() => {
              const forgotten = { ...settings, key: null };
              saveSettings(forgotten);
              setSettings(forgotten);
              setEditing(true);
            }}
            className="underline hover:text-slate-600"
          >
            {t("assistant.forget")}
          </button>
        </span>
      </p>
    </div>
  );
}
