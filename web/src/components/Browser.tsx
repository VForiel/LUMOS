// Results browser (step P14.20): the runs and studies of the session, newest first. A run shows the figures
// of its outputs; a study is browsed point by point (steps of a loop, points of a sweep or draws of a
// Monte-Carlo run) with a slider and a play button, for the ports recorded (study recorded, Recorder blocks).

import { FlaskConical, Pause, Play, Repeat, Dices, Rocket, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type JobInfo, type PortSummary, type StudyResult } from "../api";
import { formatNumber } from "../lib/units";
import { FIGURE_KINDS, useStore } from "../store";
import { Modal } from "./Dialogs";

const ICONS = { run: Rocket, sweep: FlaskConical, sample: Dices, loop: Repeat } as const;
const STATISTIC = ["max", "mean", "min", "sum"] as const;

function Values({ summary }: { summary: PortSummary | undefined }) {
  if (!summary) return null;
  const unit = summary.unit ? ` ${summary.unit}` : "";
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 font-mono text-xs">
      {summary.values && summary.values.length <= 8 && (
        <div className="contents">
          <dt className="text-slate-400">value</dt>
          <dd>
            {summary.values.map((v) => formatNumber(v)).join(", ")}
            {unit}
          </dd>
        </div>
      )}
      {Object.entries(summary.statistics).map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="text-slate-400">{key}</dt>
          <dd>
            {formatNumber(value)}
            {unit}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** The value of a port at each point, to draw its curve under the slider. */
function series(study: StudyResult, port: string): number[] {
  return study.points.map((point) => {
    const summary = point.outputs[port];
    if (!summary) return NaN;
    if (summary.values?.length === 1) return summary.values[0];
    for (const key of STATISTIC) if (key in summary.statistics) return summary.statistics[key];
    return NaN;
  });
}

function Sparkline({ values, index, onPick }: { values: number[]; index: number; onPick: (k: number) => void }) {
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return null;
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  const W = 560;
  const H = 56;
  const x = (k: number) => (values.length === 1 ? W / 2 : (k / (values.length - 1)) * (W - 8) + 4);
  const y = (v: number) => H - 6 - ((v - low) / (high - low || 1)) * (H - 12);
  const path = values.map((v, k) => (Number.isFinite(v) ? `${k && Number.isFinite(values[k - 1]) ? "L" : "M"}${x(k)},${y(v)}` : "")).join(" ");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-14 w-full cursor-pointer text-accent-600 dark:text-accent-300"
      onClick={(event) => {
        const box = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
        onPick(Math.round(((event.clientX - box.left) / box.width) * (values.length - 1)));
      }}
    >
      <path d={path} fill="none" stroke="currentColor" strokeWidth={1.5} />
      {Number.isFinite(values[index]) && <circle cx={x(index)} cy={y(values[index])} r={4} fill="#f59e0b" />}
    </svg>
  );
}

function StudyView({ job }: { job: JobInfo }) {
  const { t } = useTranslation();
  const [study, setStudy] = useState<StudyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [port, setPort] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    setStudy(null);
    setIndex(0);
    setPlaying(false);
    api
      .studyResult(job.id)
      .then(setStudy)
      .catch((reason) => setError(String(reason?.message ?? reason)));
  }, [job.id]);
  const recorded = study?.recorded ?? {};
  const ports = study ? [...Object.keys(recorded), ...study.outputs.filter((p) => !(p in recorded))] : [];
  const chosen = port && ports.includes(port) ? port : ports[0];
  const count = study?.points.length ?? 0;
  useEffect(() => {
    if (!playing || !count) return;
    const timer = window.setInterval(() => {
      setIndex((k) => {
        if (k >= count - 1) {
          setPlaying(false);
          return k;
        }
        return k + 1;
      });
    }, 450);
    return () => window.clearInterval(timer);
  }, [playing, count]);
  const values = useMemo(() => (study && chosen ? series(study, chosen) : []), [study, chosen]);
  if (error) return <p className="p-4 text-sm text-red-600">{error}</p>;
  if (!study || !chosen) return <p className="p-4 text-sm text-slate-500">{t("common.loading")}</p>;
  const point = study.points[index];
  // the nearest recorded point at or before the slider (a Recorder may keep one point out of several)
  const frames = recorded[chosen] ?? [];
  const frame = [...frames].reverse().find((k) => k <= index) ?? frames[0];
  const where = Object.entries(point.params)
    .map(([name, value]) => `${name} = ${value}`)
    .join(" · ");
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <select
          data-testid="browser-port"
          value={chosen}
          onChange={(event) => setPort(event.target.value)}
          className="rounded-md border border-slate-200 bg-white px-1.5 py-1 font-mono dark:border-slate-700 dark:bg-slate-800"
        >
          {ports.map((p) => (
            <option key={p} value={p}>
              {p}
              {p in recorded ? " ●" : ""}
            </option>
          ))}
        </select>
        <span className="text-slate-500">{chosen in recorded ? t("browser.recorded", { count: frames.length }) : t("browser.notRecorded")}</span>
      </div>
      <div className="flex items-center gap-2">
        <button type="button" onClick={() => setIndex(0)} aria-label={t("browser.first")} className="rounded p-1 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
          <SkipBack className="size-4" />
        </button>
        <button
          type="button"
          data-testid="browser-play"
          onClick={() => {
            if (index >= count - 1) setIndex(0);
            setPlaying(!playing);
          }}
          aria-label={t(playing ? "browser.pause" : "browser.play")}
          className="grid size-8 place-items-center rounded-full bg-accent-600 text-white hover:bg-accent-700"
        >
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        </button>
        <button type="button" onClick={() => setIndex(count - 1)} aria-label={t("browser.last")} className="rounded p-1 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
          <SkipForward className="size-4" />
        </button>
        <input
          data-testid="browser-slider"
          type="range"
          min={0}
          max={Math.max(count - 1, 0)}
          step={1}
          value={index}
          onChange={(event) => {
            setPlaying(false);
            setIndex(Number(event.target.value));
          }}
          className="min-w-0 flex-1 accent-accent-600"
          aria-label={t("browser.slider")}
        />
      </div>
      <p data-testid="browser-step" className="font-mono text-xs text-slate-600 dark:text-slate-300">
        {t(study.kind === "loop" ? "browser.step" : "browser.point", { index: index + 1, count })}
        {where && ` · ${where}`}
      </p>
      <Sparkline values={values} index={index} onPick={(k) => setIndex(Math.min(Math.max(k, 0), count - 1))} />
      <div className="flex min-h-0 flex-1 gap-4">
        {frame !== undefined ? (
          <img
            data-testid="browser-figure"
            src={api.pointFigureUrl(job.id, frame, chosen)}
            alt={`${chosen} #${frame + 1}`}
            className="max-h-[45vh] min-h-0 rounded-lg border border-slate-200 bg-white object-contain dark:border-slate-700"
          />
        ) : (
          <p className="max-w-xs text-xs text-slate-500">{t("browser.howToRecord")}</p>
        )}
        <div className="min-w-0 flex-1">
          <Values summary={point.outputs[chosen]} />
        </div>
      </div>
    </div>
  );
}

function RunView({ job }: { job: JobInfo }) {
  const { t } = useTranslation();
  const [summaries, setSummaries] = useState<Record<string, PortSummary>>({});
  useEffect(() => {
    setSummaries({});
    if (job.status !== "done") return;
    void Promise.all(job.outputs.map((port) => api.result(job.id, port).then((s) => [port, s] as const)))
      .then((pairs) => setSummaries(Object.fromEntries(pairs)))
      .catch(() => undefined);
  }, [job.id, job.status, job.outputs]);
  if (job.status !== "done") return <p className="p-4 text-sm text-slate-500">{job.error ?? t(`results.status.${job.status}`)}</p>;
  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto p-4 md:grid-cols-2">
      {job.outputs.map((port) => {
        const summary = summaries[port];
        const figure = summary && FIGURE_KINDS.includes(summary.kind) && summary.shape.length <= 4;
        return (
          <article key={port} data-testid="browser-output" className="flex flex-col gap-2 rounded-xl border border-slate-200 p-3 dark:border-slate-700">
            <span className="font-mono text-sm font-semibold">{port}</span>
            {figure && <img src={api.resultFigureUrl(job.id, port)} alt={port} className="max-h-64 rounded-lg border border-slate-200 bg-white object-contain dark:border-slate-700" />}
            <Values summary={summary} />
          </article>
        );
      })}
    </div>
  );
}

export function ResultsBrowser() {
  const { t } = useTranslation();
  const live = useStore((s) => s.jobs);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  useEffect(() => {
    void api
      .jobs()
      .then((list) => setJobs([...list].reverse()))
      .catch(() => setJobs(Object.values(live).reverse()));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const shown = jobs.map((job) => live[job.id] ?? job);
  const job = shown.find((j) => j.id === chosen) ?? shown[0];
  const close = () => useStore.getState().setDialog(null);
  return (
    <Modal title={t("browser.title")} onClose={close} wide testId="results-browser">
      <div className="flex h-[70vh] min-h-0">
        <ul className="scrollbar-thin w-52 shrink-0 space-y-1 overflow-y-auto border-r border-slate-100 p-2 dark:border-slate-800">
          {shown.length === 0 && <li className="p-2 text-xs text-slate-500">{t("browser.empty")}</li>}
          {shown.map((item) => {
            const Icon = ICONS[(item.kind ?? "run") as keyof typeof ICONS] ?? Rocket;
            const active = item.id === job?.id;
            return (
              <li key={item.id}>
                <button
                  type="button"
                  data-testid="browser-job"
                  aria-pressed={active}
                  onClick={() => setChosen(item.id)}
                  className={`flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left text-xs ${active ? "bg-accent-50 text-accent-800 dark:bg-accent-950/50 dark:text-accent-200" : "hover:bg-slate-50 dark:hover:bg-slate-800"}`}
                >
                  <Icon className="mt-0.5 size-3.5 shrink-0" />
                  <span className="min-w-0 flex-1">
                    <span className="block font-semibold">{t(`browser.kinds.${item.kind ?? "run"}`)}</span>
                    <span className="block truncate font-mono text-[10px] text-slate-500">{item.outputs.join(", ")}</span>
                    <span className="block text-[10px] text-slate-400">
                      {t(`results.status.${item.status}`)}
                      {item.seconds !== null && ` · ${item.seconds.toFixed(1)} s`}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {job && (job.kind === "run" || !job.kind ? <RunView job={job} /> : job.status === "done" ? <StudyView key={job.id} job={job} /> : <p className="p-4 text-sm text-slate-500">{job.error ?? t(`results.status.${job.status}`)}</p>)}
        </div>
      </div>
    </Modal>
  );
}
