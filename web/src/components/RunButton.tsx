// Run the context (REQ-WEB-5): progress while it runs, cancellation, archive option.

import { Archive, Loader2, Play, Square } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useStore } from "../store";

export function RunButton() {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  const job = useStore((s) => (s.lastJob ? s.jobs[s.lastJob] : undefined));
  const archive = useStore((s) => s.archive);
  const role = useStore((s) => s.info?.role);
  const errors = useStore((s) => s.issues.filter((i) => i.severity === "error").length);
  const { run, cancel } = useStore.getState();
  const running = job && (job.status === "queued" || job.status === "running") && job.context === context?.id;
  const allowed = role === "simulator" || role === "operator";

  return (
    <div className="flex shrink-0 items-center gap-1.5" data-tour="run">
      <button
        type="button"
        onClick={() => useStore.getState().setDialog({ kind: "archives" })}
        title={t("run.archive")}
        aria-pressed={archive}
        data-testid="archive-toggle"
        className={`hidden size-8 place-items-center rounded-lg md:grid ${archive ? "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300" : "text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"}`}
      >
        <Archive className="size-4" />
      </button>
      {running ? (
        <div className="flex items-center gap-2">
          <div className="w-36" title={job.message}>
            <div className="mb-0.5 flex justify-between text-[10px] text-slate-500">
              <span className="truncate">{job.message || t("run.queued")}</span>
              <span>{Math.round(job.progress * 100)} %</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
              <div className="h-full rounded-full bg-accent-500 transition-[width] duration-300" style={{ width: `${Math.max(job.progress * 100, 4)}%` }} />
            </div>
          </div>
          <button
            type="button"
            data-testid="cancel-run"
            onClick={() => void cancel()}
            className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 text-sm font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            <Square className="size-3.5" />
            {t("run.cancel")}
          </button>
        </div>
      ) : (
        <button
          type="button"
          data-testid="run"
          disabled={!context || !allowed}
          onClick={() => void run()}
          title={errors ? t("run.problems", { count: errors }) : `${t("run.run")} (Ctrl+Enter)`}
          className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent-600 px-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-accent-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {job?.status === "queued" ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4 fill-current" />}
          {t("run.run")}
        </button>
      )}
    </div>
  );
}
