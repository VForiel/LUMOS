// The Catalogue tab of the palette (P15.16, ADR-0026): contributions of the community, shipped with LUMOS. A
// declarative component is imported in one click (then it is an element of the palette); a preset opens.

import { BookOpen, Download, FolderOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type CatalogueEntry } from "../api";
import { useStore } from "../store";

export function CatalogueList({ query }: { query: string }) {
  const { t } = useTranslation();
  const role = useStore((s) => s.info?.role);
  const [entries, setEntries] = useState<CatalogueEntry[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    void api.catalogue().then(setEntries, (error) => {
      useStore.getState().fail(error);
      setEntries([]);
    });
  }, []);

  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = (entries ?? []).filter((entry) => {
    const text = `${entry.id} ${entry.title} ${entry.description} ${entry.authors.join(" ")}`.toLowerCase();
    return words.every((word) => text.includes(word));
  });

  const use = async (entry: CatalogueEntry) => {
    const store = useStore.getState();
    setBusy(entry.id);
    try {
      if (entry.kind === "component") {
        const saved = await api.useCatalogueComponent(entry.id);
        await store.refreshPalette();
        store.toast({ kind: "success", title: "componentSaved", values: { name: saved.title || saved.id } });
      } else {
        await store.replace(() => api.openCataloguePreset(entry.id));
      }
    } catch (error) {
      store.fail(error);
    } finally {
      setBusy(null);
    }
  };

  if (entries === null) return <p className="px-2 py-1 text-xs text-slate-400">{t("common.loading")}</p>;
  if (shown.length === 0) return <p className="px-2 py-1 text-xs text-slate-400">{t("catalogue.empty")}</p>;
  return (
    <ul className="space-y-1.5" data-testid="catalogue">
      {shown.map((entry) => (
        <li
          key={`${entry.kind}:${entry.id}`}
          data-testid="catalogue-entry"
          className="rounded-lg border border-slate-200 p-2 text-sm dark:border-slate-700"
        >
          <div className="flex items-start gap-2">
            {entry.kind === "component" ? (
              <BookOpen className="mt-0.5 size-4 shrink-0 text-indigo-400" />
            ) : (
              <FolderOpen className="mt-0.5 size-4 shrink-0 text-amber-500" />
            )}
            <div className="min-w-0 flex-1">
              <p className="font-medium leading-tight">{entry.title || entry.id}</p>
              <p className="truncate font-mono text-[10px] text-slate-400">{entry.id}</p>
            </div>
            <button
              type="button"
              data-testid="catalogue-use"
              disabled={busy !== null || (entry.kind === "component" && role === "viewer")}
              onClick={() => void use(entry)}
              title={t(entry.kind === "component" ? "catalogue.import" : "catalogue.open")}
              className="inline-flex shrink-0 items-center gap-1 rounded-md border border-slate-200 px-1.5 py-0.5 text-xs text-accent-700 hover:bg-accent-50 disabled:opacity-40 dark:border-slate-700 dark:text-accent-300 dark:hover:bg-accent-900/30"
            >
              <Download className="size-3.5" />
              {t(entry.kind === "component" ? "catalogue.importShort" : "catalogue.openShort")}
            </button>
          </div>
          {entry.description && <p className="mt-1 line-clamp-3 text-xs text-slate-500 dark:text-slate-400">{entry.description}</p>}
          <p className="mt-1 text-[10px] text-slate-400">
            {[entry.authors.join(", "), entry.version ? `v${entry.version}` : "", entry.license, t(`catalogue.kinds.${entry.kind}`)].filter(Boolean).join(" · ")}
          </p>
        </li>
      ))}
    </ul>
  );
}
