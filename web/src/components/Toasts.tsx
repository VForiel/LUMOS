// Notifications: errors keep the hint of the server, so the user knows how to fix the request.

import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useStore } from "../store";

const ICONS = { error: AlertCircle, info: Info, success: CheckCircle2 };
const STYLES = {
  error: "border-red-200 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-100",
  info: "border-sky-200 bg-sky-50 text-sky-900 dark:border-sky-900 dark:bg-sky-950 dark:text-sky-100",
  success: "border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-100",
};

export function Toasts() {
  const { t } = useTranslation();
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismiss);
  return (
    <div className="pointer-events-none fixed top-14 right-4 z-50 flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2" aria-live="polite">
      {toasts.map((toast) => {
        const Icon = ICONS[toast.kind];
        return (
          <div
            key={toast.id}
            role={toast.kind === "error" ? "alert" : "status"}
            data-testid={`toast-${toast.kind}`}
            className={`pointer-events-auto flex gap-3 rounded-xl border p-3 text-sm shadow-lg ${STYLES[toast.kind]}`}
          >
            <Icon className="mt-0.5 size-4 shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="font-medium break-words">{t(`toast.${toast.title}`, { defaultValue: toast.title, ...toast.values })}</p>
              {toast.hint && <p className="mt-0.5 break-words opacity-80">{toast.hint}</p>}
            </div>
            <button type="button" onClick={() => dismiss(toast.id)} aria-label={t("common.close")} className="opacity-60 hover:opacity-100">
              <X className="size-4" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
