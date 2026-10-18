// Gallery of presets (REQ-WEB-7): the same .lumos files as the notebooks, with what they contain.

import { ArrowRight, GraduationCap } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type Preset } from "../api";
import { category, categoryOf } from "../lib/categories";
import { useStore } from "../store";
import { startTour } from "../tour";
import { Modal } from "./Dialogs";

function PresetCard({ preset, onOpen }: { preset: Preset; onOpen: () => void }) {
  const categories = [...new Set(preset.elements.map(categoryOf))];
  return (
    <button
      type="button"
      data-testid={`preset-${preset.name}`}
      onClick={onOpen}
      className="group flex flex-col gap-2 rounded-xl border border-slate-200 bg-white p-3 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-accent-300 hover:shadow-md dark:border-slate-700 dark:bg-slate-900 dark:hover:border-accent-700"
    >
      <div className="flex items-center gap-1">
        {categories.slice(0, 6).map((name) => {
          const { icon: Icon, color } = category(name);
          return (
            <span key={name} className="grid size-6 place-items-center rounded-md" style={{ background: `${color}1f`, color }}>
              <Icon className="size-3.5" />
            </span>
          );
        })}
        <ArrowRight className="ml-auto size-4 text-slate-300 transition group-hover:translate-x-0.5 group-hover:text-accent-500" />
      </div>
      <span className="font-semibold text-slate-800 dark:text-slate-100">{preset.title}</span>
      <span className="line-clamp-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">{preset.description}</span>
    </button>
  );
}

export function PresetGallery({ welcome }: { welcome?: boolean }) {
  const { t } = useTranslation();
  const [presets, setPresets] = useState<Preset[]>([]);
  const role = useStore((s) => s.info?.role);
  useEffect(() => {
    void api.presets().then(setPresets).catch((error) => useStore.getState().fail(error));
  }, []);
  const open = async (name: string) => {
    const store = useStore.getState();
    store.setDialog(null);
    await store.replace(() => api.loadPreset(name));
  };
  return (
    <div data-testid="presets" className="space-y-4">
      {welcome && (
        <div className="flex flex-wrap items-center gap-3 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 p-4 text-white shadow-lg">
          <GraduationCap className="size-8 shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="font-semibold">{t("presets.welcome")}</p>
            <p className="text-sm text-indigo-100">{t("presets.welcomeHint")}</p>
          </div>
          <button
            type="button"
            data-testid="start-tour"
            onClick={() => void startTour()}
            className="rounded-lg bg-white px-3 py-1.5 text-sm font-semibold text-indigo-700 shadow hover:bg-indigo-50"
          >
            {t("presets.tour")}
          </button>
        </div>
      )}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] gap-3">
        {presets.map((preset) => (
          <PresetCard key={preset.name} preset={preset} onOpen={() => void open(preset.name)} />
        ))}
      </div>
      {welcome && role !== "viewer" && <p className="text-center text-sm text-slate-500 dark:text-slate-400">{t("presets.orBuild")}</p>}
    </div>
  );
}

export function PresetsDialog() {
  const { t } = useTranslation();
  return (
    <Modal title={t("presets.title")} onClose={() => useStore.getState().setDialog(null)} wide>
      <div className="p-5">
        <PresetGallery />
      </div>
    </Modal>
  );
}
