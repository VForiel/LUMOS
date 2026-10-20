// Modal dialogs: names, Python script, keyboard shortcuts.

import { AlertTriangle, Check, Copy, Download, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { download, fileName } from "../api";
import { useStore, type Dialog } from "../store";
import { inputClass } from "./Fields";

export function Modal({ title, children, onClose, wide, testId }: { title: string; children: ReactNode; onClose: () => void; wide?: boolean; testId?: string }) {
  const { t } = useTranslation();
  useEffect(() => {
    const close = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-40 grid place-items-center bg-slate-900/40 p-4 backdrop-blur-sm" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        data-testid={testId}
        className={`flex max-h-[85vh] w-full flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900 ${wide ? "max-w-4xl" : "max-w-md"}`}
      >
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3 dark:border-slate-800">
          <h2 className="font-semibold">{title}</h2>
          <button type="button" onClick={onClose} aria-label={t("common.close")} className="rounded p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="size-4" />
          </button>
        </div>
        <div className="scrollbar-thin overflow-y-auto">{children}</div>
      </div>
    </div>
  );
}

export function Button({ children, onClick, primary, testId, disabled, type = "button" }: { children: ReactNode; onClick?: () => void; primary?: boolean; testId?: string; disabled?: boolean; type?: "button" | "submit" }) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      data-testid={testId}
      className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-50 ${
        primary
          ? "bg-accent-600 text-white shadow-sm hover:bg-accent-700"
          : "border border-slate-200 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
      }`}
    >
      {children}
    </button>
  );
}

function PromptDialog({ dialog }: { dialog: Extract<Dialog, { kind: "prompt" }> }) {
  const { t } = useTranslation();
  const close = () => useStore.getState().setDialog(null);
  const [value, setValue] = useState(dialog.value);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    input.current?.select();
  }, []);
  const confirm = () => {
    if (!value.trim()) return;
    close();
    dialog.onConfirm(value.trim());
  };
  return (
    <Modal title={t(dialog.title)} onClose={close}>
      <form
        className="space-y-4 p-5"
        onSubmit={(event) => {
          event.preventDefault();
          confirm();
        }}
      >
        <label className="block text-sm">
          <span className="mb-1 block text-slate-600 dark:text-slate-300">{t(dialog.label)}</span>
          <input ref={input} data-testid="dialog-input" className={inputClass} value={value} onChange={(event) => setValue(event.target.value)} />
        </label>
        <div className="flex justify-end gap-2">
          <Button onClick={close}>{t("common.cancel")}</Button>
          <Button primary type="submit" testId="dialog-confirm">
            {t(dialog.confirm)}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

function PythonDialog({ code, testId, fileName: name }: { code: string; testId?: string; fileName?: string }) {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  const [copied, setCopied] = useState(false);
  const close = () => useStore.getState().setDialog(null);
  return (
    <Modal title={t("dialog.python.title")} onClose={close} wide>
      <div className="space-y-3 p-5">
        <p className="text-sm text-slate-600 dark:text-slate-400">{t("dialog.python.hint")}</p>
        <pre data-testid={testId ?? "python-code"} className="max-h-[55vh] overflow-auto rounded-xl bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-100">
          {code}
        </pre>
        <div className="flex justify-end gap-2">
          <Button
            onClick={() => {
              void navigator.clipboard?.writeText(code).then(() => setCopied(true));
              window.setTimeout(() => setCopied(false), 1500);
            }}
          >
            {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
            {copied ? t("dialog.python.copied") : t("dialog.python.copy")}
          </Button>
          <Button primary onClick={() => download(name ?? fileName(context?.name ?? "context", ".py"), code, "text/x-python")}>
            <Download className="size-4" />
            {t("dialog.python.download")}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

export const SHORTCUTS: [string, string][] = [
  ["Ctrl+K", "shortcuts.add"],
  ["Ctrl+Z", "shortcuts.undo"],
  ["Ctrl+Shift+Z / Ctrl+Y", "shortcuts.redo"],
  ["Ctrl+C / Ctrl+V", "shortcuts.copy"],
  ["Ctrl+D", "shortcuts.duplicate"],
  ["Ctrl+G", "shortcuts.group"],
  ["Delete", "shortcuts.delete"],
  ["Ctrl+A", "shortcuts.selectAll"],
  ["Ctrl+S", "shortcuts.save"],
  ["Ctrl+O", "shortcuts.open"],
  ["Ctrl+Enter", "shortcuts.run"],
  ["F", "shortcuts.fit"],
  ["Shift+drag", "shortcuts.box"],
  ["H / V", "shortcuts.pointer"],
  ["Space+drag", "shortcuts.pan"],
  ["?", "shortcuts.help"],
];

function DiscardDialog({ answer }: { answer: (choice: "save" | "discard" | "cancel") => void }) {
  const { t } = useTranslation();
  const name = useStore((s) => s.context?.name) || t("context.untitled");
  return (
    <Modal title={t("dialog.discard.title")} onClose={() => answer("cancel")} testId="discard-dialog">
      <div className="space-y-4 p-5 text-sm">
        <p className="flex items-start gap-2 text-slate-600 dark:text-slate-300">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-500" />
          <span>{t("dialog.discard.text", { name })}</span>
        </p>
        <div className="flex flex-wrap justify-end gap-2">
          <Button onClick={() => answer("cancel")} testId="discard-cancel">
            {t("common.cancel")}
          </Button>
          <Button onClick={() => answer("discard")} testId="discard-confirm">
            {t("dialog.discard.discard")}
          </Button>
          <Button primary onClick={() => answer("save")} testId="discard-save">
            <Download className="size-4" />
            {t("dialog.discard.save")}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

function ShortcutsDialog() {
  const { t } = useTranslation();
  return (
    <Modal title={t("shortcuts.title")} onClose={() => useStore.getState().setDialog(null)}>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 p-5 text-sm">
        {SHORTCUTS.map(([keys, label]) => (
          <div key={keys} className="contents">
            <dt>
              <kbd className="rounded border border-slate-300 bg-slate-50 px-1.5 py-0.5 font-mono text-xs dark:border-slate-600 dark:bg-slate-800">{keys}</kbd>
            </dt>
            <dd className="text-slate-600 dark:text-slate-300">{t(label)}</dd>
          </div>
        ))}
      </dl>
    </Modal>
  );
}

export function Dialogs({ extra }: { extra?: (dialog: Dialog) => ReactNode }) {
  const dialog = useStore((s) => s.dialog);
  if (!dialog) return null;
  switch (dialog.kind) {
    case "prompt":
      return <PromptDialog dialog={dialog} />;
    case "python":
      return <PythonDialog code={dialog.code} testId={dialog.testId} fileName={dialog.fileName} />;
    case "shortcuts":
      return <ShortcutsDialog />;
    case "discard":
      return <DiscardDialog answer={dialog.answer} />;
    default:
      return <>{extra?.(dialog)}</>;
  }
}
