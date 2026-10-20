// Archives of the session (P15.17, review of 20/10/2026): the switch "archive each run" and a browser of the
// archives written so far: download, files, a view of the data (summary and figure) and deletion.

import { Archive, Download, FileText, Image as ImageIcon, Table2, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type ArchiveFile, type ArchiveInfo, type PortSummary } from "../api";
import { formatNumber } from "../lib/units";
import { useStore } from "../store";
import { Modal } from "./Dialogs";

const q = (params: Record<string, string>) => new URLSearchParams(params).toString();

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

const ICONS = { data: Table2, figure: ImageIcon } as Record<string, typeof FileText>;

export function ArchivesDialog() {
  const { t } = useTranslation();
  const archive = useStore((s) => s.archive);
  const role = useStore((s) => s.info?.role);
  const [archives, setArchives] = useState<ArchiveInfo[] | null>(null);
  const [selected, setSelected] = useState<ArchiveInfo | null>(null);
  const [files, setFiles] = useState<ArchiveFile[]>([]);
  const [preview, setPreview] = useState<{ file: ArchiveFile; summary?: PortSummary; error?: string } | null>(null);
  const close = () => useStore.getState().setDialog(null);

  const load = useCallback(async () => {
    try {
      setArchives(await api.archives());
    } catch (error) {
      useStore.getState().fail(error);
      setArchives([]);
    }
  }, []);
  useEffect(() => void load(), [load]);

  useEffect(() => {
    setPreview(null);
    setFiles([]);
    if (selected) void api.archiveFiles(selected.path).then(setFiles);
  }, [selected]);

  const open = async (file: ArchiveFile) => {
    if (!selected) return;
    if (file.kind !== "data") {
      setPreview({ file });
      return;
    }
    setPreview({ file });
    try {
      setPreview({ file, summary: await api.archiveData(selected.path, file.name) });
    } catch (error) {
      setPreview({ file, error: String((error as Error).message ?? error) });
    }
  };

  const remove = async () => {
    if (!selected) return;
    await api.deleteArchive(selected.path);
    setSelected(null);
    await load();
  };

  const fileUrl = (file: ArchiveFile) => (selected ? `/api/archives/file?${q({ path: selected.path, name: file.name })}` : "#");

  return (
    <Modal title={t("archives.title")} onClose={close} wide testId="archives">
      <div className="space-y-4 p-5">
        <label className="flex items-start gap-3 rounded-xl border border-slate-200 p-3 dark:border-slate-700">
          <input
            type="checkbox"
            data-testid="archive-switch"
            className="mt-1"
            checked={archive}
            disabled={role === "viewer"}
            onChange={(event) => useStore.getState().setArchive(event.target.checked)}
          />
          <span>
            <span className="block text-sm font-medium">{t("archives.switch")}</span>
            <span className="block text-xs text-slate-500">{t("archives.switchHelp")}</span>
          </span>
        </label>
        <div className="grid min-h-[16rem] gap-4 md:grid-cols-[minmax(0,15rem)_1fr]">
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{t("archives.list")}</h3>
            {archives && archives.length === 0 && <p className="text-sm text-slate-500">{t("archives.none")}</p>}
            <ul className="space-y-1">
              {archives?.map((item) => (
                <li key={item.path}>
                  <button
                    type="button"
                    data-testid="archive-item"
                    onClick={() => setSelected(item)}
                    className={`w-full rounded-lg border px-2.5 py-1.5 text-left text-sm ${
                      selected?.path === item.path
                        ? "border-accent-400 bg-accent-50 dark:border-accent-700 dark:bg-accent-900/30"
                        : "border-slate-200 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
                    }`}
                  >
                    <span className="flex items-center gap-1.5 font-medium">
                      <Archive className="size-3.5 shrink-0 text-amber-600" />
                      <span className="truncate">{item.name}</span>
                    </span>
                    <span className="block text-xs text-slate-500">
                      {item.created_at.slice(0, 19).replace("T", " ")} · {size(item.size_bytes)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <div className="min-w-0">
            {!selected && archives && archives.length > 0 && <p className="text-sm text-slate-500">{t("archives.pick")}</p>}
            {selected && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="mr-auto truncate font-semibold">{selected.name}</h3>
                  <a
                    data-testid="archive-download"
                    href={`/api/archives/download?${q({ path: selected.path })}`}
                    download
                    className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2 py-1 text-xs hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
                  >
                    <Download className="size-3.5" /> {t("archives.download")}
                  </a>
                  {role !== "viewer" && (
                    <button
                      type="button"
                      data-testid="archive-delete"
                      onClick={() => void remove()}
                      className="inline-flex items-center gap-1 rounded-lg border border-red-200 px-2 py-1 text-xs text-red-600 hover:bg-red-50 dark:border-red-900 dark:hover:bg-red-950/40"
                    >
                      <Trash2 className="size-3.5" /> {t("archives.delete")}
                    </button>
                  )}
                </div>
                <ul className="grid gap-1 sm:grid-cols-2">
                  {files.map((file) => {
                    const Icon = ICONS[file.kind] ?? FileText;
                    return (
                      <li key={file.name} className="flex items-center gap-1">
                        <button
                          type="button"
                          data-testid="archive-file"
                          onClick={() => void open(file)}
                          className={`flex min-w-0 flex-1 items-center gap-1.5 rounded-md px-2 py-1 text-left text-xs ${
                            preview?.file.name === file.name ? "bg-slate-100 dark:bg-slate-800" : "hover:bg-slate-50 dark:hover:bg-slate-800"
                          }`}
                        >
                          <Icon className="size-3.5 shrink-0 text-slate-400" />
                          <span className="truncate font-mono">{file.name}</span>
                          <span className="ml-auto shrink-0 text-slate-400">{size(file.size_bytes)}</span>
                        </button>
                        <a href={fileUrl(file)} download className="rounded p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800" title={t("archives.downloadFile")}>
                          <Download className="size-3.5" />
                        </a>
                      </li>
                    );
                  })}
                </ul>
                {preview && (
                  <div data-testid="archive-preview" className="rounded-xl border border-slate-200 p-3 text-sm dark:border-slate-700">
                    <p className="mb-2 font-mono text-xs font-semibold">{preview.file.name}</p>
                    {preview.error && <p className="text-xs text-red-600">{preview.error}</p>}
                    {preview.file.kind === "figure" && /\.(png|jpg|svg)$/i.test(preview.file.name) && (
                      <img src={fileUrl(preview.file)} alt={preview.file.name} className="max-h-80 rounded" />
                    )}
                    {preview.summary && (
                      <div className="grid gap-3 md:grid-cols-[1fr_auto]">
                        <dl className="grid grid-cols-[auto_1fr] content-start gap-x-3 text-xs">
                          <dt className="text-slate-400">{t("archives.kind")}</dt>
                          <dd>{preview.summary.kind}</dd>
                          {preview.summary.shape.length > 0 && (
                            <>
                              <dt className="text-slate-400">{t("archives.shape")}</dt>
                              <dd className="font-mono">{preview.summary.shape.join(" × ")}</dd>
                            </>
                          )}
                          {Object.entries(preview.summary.statistics).map(([key, value]) => (
                            <div key={key} className="contents">
                              <dt className="text-slate-400">{t(`results.${key}`, { defaultValue: key })}</dt>
                              <dd className="font-mono">
                                {formatNumber(value)} {preview.summary?.unit ?? ""}
                              </dd>
                            </div>
                          ))}
                        </dl>
                        <img
                          src={`/api/archives/figure.png?${q({ path: selected.path, name: preview.file.name })}`}
                          alt={preview.file.name}
                          className="max-h-72 rounded"
                          onError={(event) => ((event.target as HTMLImageElement).style.display = "none")}
                        />
                      </div>
                    )}
                    {preview.file.kind !== "data" && preview.file.kind !== "figure" && (
                      <a href={fileUrl(preview.file)} target="_blank" rel="noreferrer" className="text-xs text-accent-600 underline">
                        {t("archives.openFile")}
                      </a>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
}
