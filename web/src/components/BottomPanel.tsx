// Bottom panel (REQ-WEB-5): results of the last run and of inspected ports, live journal, problems.

import { AlertTriangle, Archive, CheckCircle2, ChevronDown, ChevronUp, Download, FileJson, Image, ListTree, Loader2, ScrollText, Table2, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { api, type JournalEntry, type PortSummary } from "../api";
import { formatNumber } from "../lib/units";
import { FIGURE_KINDS, useStore, type BottomTab } from "../store";

function DownloadLink({ href, label, testId, icon }: { href: string; label: string; testId?: string; icon?: ReactNode }) {
  return (
    <a
      href={href}
      download
      data-testid={testId}
      className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-1.5 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
    >
      {icon ?? <Download className="size-3" />}
      {label}
    </a>
  );
}

function Statistics({ summary }: { summary: PortSummary }) {
  const { t } = useTranslation();
  const unit = summary.unit ? ` ${summary.unit}` : "";
  return (
    <div data-testid="statistics" className="space-y-1 text-xs">
      <p className="text-slate-500 dark:text-slate-400">
        {summary.kind}
        {summary.shape.length > 0 && ` · ${summary.shape.join(" × ")}`}
        {summary.unit && ` · ${summary.unit}`}
      </p>
      {summary.values && summary.values.length <= 8 && (
        <p className="font-mono text-sm font-semibold text-slate-800 dark:text-slate-100">
          {summary.values.map((v) => formatNumber(v)).join(", ")}
          {unit}
        </p>
      )}
      {Object.keys(summary.statistics).length > 0 && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 font-mono">
          {Object.entries(summary.statistics).map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-slate-400">{t(`results.${key}`, { defaultValue: key })}</dt>
              <dd className="text-slate-700 dark:text-slate-200">
                {formatNumber(value)}
                {unit}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {Object.keys(summary.details).length > 0 && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3">
          {Object.entries(summary.details)
            .filter(([, value]) => value)
            .map(([key, value]) => (
              <div key={key} className="contents">
                <dt className="text-slate-400">{key}</dt>
                <dd className="font-mono break-words text-slate-700 dark:text-slate-200">{value}</dd>
              </div>
            ))}
        </dl>
      )}
    </div>
  );
}

function Downloads({ make, figure }: { make: (format: string) => string; figure: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap gap-1">
      {figure && <DownloadLink href={make("png")} label="PNG" testId="download-png" icon={<Image className="size-3" />} />}
      {figure && <DownloadLink href={make("plt")} label="PltEdit" testId="download-plt" />}
      <DownloadLink href={make("npz")} label={t("results.data")} testId="download-npz" icon={<Table2 className="size-3" />} />
      <DownloadLink href={make("json")} label="JSON" testId="download-json" icon={<FileJson className="size-3" />} />
    </div>
  );
}

function Figure({ src, alt }: { src: string | null; alt: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setFailed(false);
  }, [src]);
  if (!src || failed) return null;
  return (
    <a href={src} target="_blank" rel="noreferrer" className="block shrink-0">
      <img src={src} alt={alt} onError={() => setFailed(true)} className="h-44 rounded-lg border border-slate-200 bg-white object-contain dark:border-slate-700" />
    </a>
  );
}

function Results() {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  const inspected = useStore((s) => s.inspected);
  const lastJob = useStore((s) => s.lastJob);
  const job = useStore((s) => (s.lastJob ? s.jobs[s.lastJob] : undefined));
  const results = useStore((s) => s.results);
  const close = () => useStore.setState({ inspected: null });

  return (
    <div data-testid="results" className="flex h-full gap-4 overflow-x-auto p-3">
      {inspected && inspected.context === context?.id && (
        <section data-testid="inspected" className="flex min-w-[26rem] shrink-0 gap-3 rounded-xl border border-accent-200 bg-accent-50/50 p-3 dark:border-accent-900 dark:bg-accent-950/20">
          <div className="flex min-w-48 flex-1 flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold tracking-wider text-accent-700 uppercase dark:text-accent-300">{t("results.inspected")}</span>
              <span className="font-mono text-sm font-semibold">{inspected.port}</span>
              <button type="button" onClick={close} className="ml-auto rounded p-0.5 text-slate-400 hover:text-slate-600" aria-label={t("common.close")}>
                <X className="size-3.5" />
              </button>
            </div>
            {inspected.loading && (
              <p className="flex items-center gap-2 text-xs text-slate-500">
                <Loader2 className="size-4 animate-spin" />
                {t("results.computing")}
              </p>
            )}
            {inspected.error && <p className="text-xs text-red-600 dark:text-red-400">{inspected.error}</p>}
            {inspected.summary && <Statistics summary={inspected.summary} />}
            {inspected.summary && (
              <Downloads make={(format) => api.portDownloadUrl(inspected.context, inspected.port, format)} figure={Boolean(inspected.figure)} />
            )}
          </div>
          <Figure src={inspected.figure} alt={inspected.port} />
        </section>
      )}
      {job ? (
        <section className="flex shrink-0 gap-3">
          <div className="flex w-44 shrink-0 flex-col gap-1 text-xs">
            <span className="font-semibold tracking-wider text-slate-500 uppercase">{t("results.lastRun")}</span>
            <span className="flex items-center gap-1.5">
              {job.status === "done" ? (
                <CheckCircle2 className="size-4 text-emerald-500" />
              ) : job.status === "failed" || job.status === "cancelled" ? (
                <AlertTriangle className="size-4 text-red-500" />
              ) : (
                <Loader2 className="size-4 animate-spin text-accent-500" />
              )}
              {t(`results.status.${job.status}`)}
              {job.seconds !== null && job.status !== "running" && ` · ${job.seconds.toFixed(1)} s`}
            </span>
            {(job.status === "running" || job.status === "queued") && (
              <span className="text-slate-500">{job.message}</span>
            )}
            {job.error && <span className="text-red-600 dark:text-red-400">{job.error}</span>}
            {job.hint && job.status === "failed" && <span className="text-slate-500">{job.hint}</span>}
            {job.archive && <DownloadLink href={api.archiveUrl(job.archive)} label={t("results.archive")} icon={<Archive className="size-3" />} testId="download-archive" />}
          </div>
          {job.status === "done" &&
            job.outputs.map((port) => {
              const summary = results[port];
              const figure = summary && FIGURE_KINDS.includes(summary.kind) && summary.shape.length <= 4;
              return (
                <article key={port} data-testid="output-card" className="flex w-[25rem] shrink-0 gap-3 rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
                  <div className="flex min-w-0 flex-1 flex-col gap-2">
                    <span className="font-mono text-sm font-semibold">{port}</span>
                    {summary ? <Statistics summary={summary} /> : <Loader2 className="size-4 animate-spin text-slate-400" />}
                    {summary && <Downloads make={(format) => api.resultDownloadUrl(job.id, port, format)} figure={Boolean(figure)} />}
                  </div>
                  {figure && <Figure src={api.resultFigureUrl(job.id, port)} alt={port} />}
                </article>
              );
            })}
        </section>
      ) : (
        !inspected && <p className="m-auto max-w-md text-center text-sm text-slate-500">{lastJob ? t("common.loading") : t("results.empty")}</p>
      )}
    </div>
  );
}

function Journal() {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  const live = useStore((s) => s.journal);
  const [history, setHistory] = useState<JournalEntry[]>([]);
  const end = useRef<HTMLDivElement>(null);
  // read again when a run ends: its entries may precede the live connection
  const finished = useStore((s) => Object.values(s.jobs).filter((j) => j.context === s.context?.id && j.status !== "queued" && j.status !== "running").length);
  useEffect(() => {
    if (!context) return;
    const read = () => void api.journal(context.id).then(setHistory).catch(() => undefined);
    read();
    const timer = window.setInterval(read, 2000); // also when the live connection is down
    return () => window.clearInterval(timer);
  }, [context?.id, finished]); // eslint-disable-line react-hooks/exhaustive-deps
  const entries = [...history.filter((h) => !live.some((l) => l.wall_time === h.wall_time && l.kind === h.kind)), ...live];
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [entries.length]);
  return (
    <div data-testid="journal" className="h-full overflow-y-auto p-2 font-mono text-xs">
      {entries.length === 0 && <p className="p-2 font-sans text-sm text-slate-500">{t("journal.empty")}</p>}
      {entries.map((entry, index) => (
        <div key={`${entry.wall_time}-${index}`} className="flex gap-2 border-b border-slate-100 py-0.5 dark:border-slate-800">
          <span className="text-slate-400">{entry.wall_time?.slice(11, 19)}</span>
          <span className={`w-14 shrink-0 ${entry.kind === "run" ? "text-emerald-600" : entry.kind === "action" ? "text-accent-600" : entry.level === "WARNING" || entry.level === "ERROR" ? "text-amber-600" : "text-slate-500"}`}>
            {entry.kind}
          </span>
          <span className="w-10 shrink-0 text-slate-400">{entry.origin}</span>
          <span className="min-w-0 flex-1 break-words text-slate-700 dark:text-slate-300">
            {entry.kind === "run" && `${entry.args} (${formatNumber(entry.measured ?? 0)} s)`}
            {entry.kind === "action" && `${entry.element}.${entry.action}(${entry.args}) · ${entry.mode}`}
            {entry.kind === "log" && `${entry.logger}: ${entry.message}`}
            {!["run", "action", "log"].includes(entry.kind) && (entry.message || entry.args)}
          </span>
        </div>
      ))}
      <div ref={end} />
    </div>
  );
}

function Problems() {
  const { t } = useTranslation();
  const issues = useStore((s) => s.issues);
  const select = useStore((s) => s.select);
  if (!issues.length) {
    return (
      <div data-testid="problems" className="flex h-full items-center justify-center gap-2 text-sm text-emerald-600">
        <CheckCircle2 className="size-4" />
        {t("problems.none")}
      </div>
    );
  }
  return (
    <ul data-testid="problems" className="h-full space-y-1 overflow-y-auto p-2 text-sm">
      {issues.map((issue, index) => (
        <li key={index} className="flex items-start gap-2 rounded-md px-2 py-1 hover:bg-slate-50 dark:hover:bg-slate-800">
          <AlertTriangle className={`mt-0.5 size-4 shrink-0 ${issue.severity === "error" ? "text-red-500" : "text-amber-500"}`} />
          <div className="min-w-0 flex-1">
            <p>{issue.message}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">{issue.hint}</p>
          </div>
          {issue.element && (
            <button type="button" onClick={() => select([issue.element!])} className="shrink-0 rounded border border-slate-200 px-1.5 py-0.5 font-mono text-xs hover:bg-white dark:border-slate-700 dark:hover:bg-slate-900">
              {issue.element}
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}

const TABS: { tab: BottomTab; icon: typeof ListTree; label: string }[] = [
  { tab: "results", icon: ListTree, label: "tabs.results" },
  { tab: "journal", icon: ScrollText, label: "tabs.journal" },
  { tab: "problems", icon: AlertTriangle, label: "tabs.problems" },
];

export function BottomPanel({ extraTabs, renderExtra }: { extraTabs?: typeof TABS; renderExtra?: (tab: BottomTab) => ReactNode }) {
  const { t } = useTranslation();
  const bottom = useStore((s) => s.bottom);
  const setBottom = useStore((s) => s.setBottom);
  const issues = useStore((s) => s.issues);
  const errors = issues.filter((i) => i.severity === "error").length;
  const tabs = [...TABS, ...(extraTabs ?? [])];
  return (
    <div data-tour="results" className="flex shrink-0 flex-col border-t border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900" style={{ height: bottom ? 280 : 36 }}>
      <div className="flex h-9 shrink-0 items-center gap-0.5 border-b border-slate-100 px-2 dark:border-slate-800">
        {tabs.map(({ tab, icon: Icon, label }) => (
          <button
            key={tab}
            type="button"
            data-testid={`tab-${tab}`}
            onClick={() => setBottom(bottom === tab ? null : tab)}
            className={`inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-xs font-medium ${bottom === tab ? "bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-white" : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"}`}
          >
            <Icon className="size-3.5" />
            {t(label)}
            {tab === "problems" && errors > 0 && <span className="rounded-full bg-red-500 px-1.5 text-[10px] text-white">{errors}</span>}
          </button>
        ))}
        <button type="button" onClick={() => setBottom(bottom ? null : "results")} className="ml-auto rounded p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800" aria-label={t(bottom ? "tabs.hide" : "tabs.show")}>
          {bottom ? <ChevronDown className="size-4" /> : <ChevronUp className="size-4" />}
        </button>
      </div>
      {bottom && (
        <div className="min-h-0 flex-1">
          {bottom === "results" && <Results />}
          {bottom === "journal" && <Journal />}
          {bottom === "problems" && <Problems />}
          {renderExtra?.(bottom)}
        </div>
      )}
    </div>
  );
}
