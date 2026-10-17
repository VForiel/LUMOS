// Palette of element types, generated from the registry (REQ-WEB-2): search, categories, drag or click to add.

import { useReactFlow } from "@xyflow/react";
import { ChevronDown, ChevronRight, FileUp, Package, Search, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { ElementType } from "../api";
import { category, CATEGORY_ORDER } from "../lib/categories";
import { useStore } from "../store";

export const DRAG_TYPE = "application/x-lumos-type";
export const BLOCK_TYPE = "application/x-lumos-block";

/** Docstring text without reStructuredText markup (``code``, :math:`x`). */
export function plain(text: string): string {
  return text.replace(/:\w+:`([^`]*)`/g, "$1").replace(/``([^`]*)``/g, "$1").replace(/`([^`]*)`/g, "$1");
}

function matches(item: ElementType, words: string[]): boolean {
  const text = `${item.type_id} ${item.title} ${item.summary}`.toLowerCase();
  return words.every((word) => text.includes(word));
}

/** A free spot near the centre of the visible canvas, in flow coordinates (new blocks never overlap). */
export function freeSpot(centre: [number, number], taken: [number, number][]): [number, number] {
  const [cx, cy] = centre;
  const free = ([x, y]: [number, number]) => taken.every(([tx, ty]) => Math.abs(tx - x) > 250 || Math.abs(ty - y) > 180);
  for (let ring = 0; ring < 12; ring += 1) {
    for (let dx = -ring; dx <= ring; dx += 1) {
      for (let dy = -ring; dy <= ring; dy += 1) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== ring) continue;
        const spot: [number, number] = [Math.round(cx + dx * 270), Math.round(cy + dy * 200)];
        if (free(spot)) return spot;
      }
    }
  }
  return [Math.round(cx), Math.round(cy)];
}

export function useCanvasCentre(): () => [number, number] {
  const flow = useReactFlow();
  return () => {
    const canvas = document.querySelector('[data-testid="canvas"]')?.getBoundingClientRect();
    if (!canvas) return [0, 0];
    const point = flow.screenToFlowPosition({ x: canvas.left + canvas.width / 2, y: canvas.top + canvas.height / 2 });
    const positions = useStore.getState().positions;
    const taken = Object.entries(positions).filter(([id]) => !id.includes(".")).map(([, xy]) => xy);
    return freeSpot([point.x - 115, point.y - 60], taken);
  };
}

export function Palette() {
  const { t } = useTranslation();
  const palette = useStore((s) => s.palette);
  const addElement = useStore((s) => s.addElement);
  const role = useStore((s) => s.info?.role);
  const centre = useCanvasCentre();
  const [query, setQuery] = useState("");
  const [closed, setClosed] = useState<Record<string, boolean>>({});

  const groups = useMemo(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    const found = new Map<string, ElementType[]>();
    for (const item of palette?.types ?? []) {
      if (words.length && !matches(item, words)) continue;
      found.set(item.category, [...(found.get(item.category) ?? []), item]);
    }
    const order = (name: string) => {
      const index = CATEGORY_ORDER.indexOf(name);
      return index < 0 ? CATEGORY_ORDER.length : index;
    };
    return [...found.entries()].sort(([a], [b]) => order(a) - order(b) || a.localeCompare(b));
  }, [palette, query]);

  const editable = role !== "viewer";
  const blocks = useStore((s) => s.blocks);
  const { insertBlock, forgetBlock } = useStore.getState();
  const shownBlocks = blocks.filter((b) => !query || b.name.toLowerCase().includes(query.toLowerCase()));

  return (
    <aside
      data-testid="palette"
      className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"
      data-tour="palette"
    >
      <div className="border-b border-slate-200 p-3 dark:border-slate-800">
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="text-xs font-semibold tracking-wider text-slate-500 uppercase dark:text-slate-400">
            {t("palette.title")}
          </h2>
          <span className="text-xs text-slate-400">{palette?.types.length ?? 0}</span>
        </div>
        <label className="relative block">
          <Search className="pointer-events-none absolute top-2 left-2.5 size-4 text-slate-400" />
          <input
            data-testid="palette-search"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("palette.search")}
            className="w-full rounded-lg border border-slate-200 bg-slate-50 py-1.5 pr-7 pl-8 text-sm outline-none placeholder:text-slate-400 focus:border-accent-400 focus:ring-2 focus:ring-accent-200 dark:border-slate-700 dark:bg-slate-800 dark:focus:ring-accent-900"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="absolute top-1.5 right-1.5 rounded p-0.5 text-slate-400 hover:text-slate-600"
              aria-label={t("common.clear")}
            >
              <X className="size-4" />
            </button>
          )}
        </label>
      </div>
      <div className="scrollbar-thin flex-1 overflow-y-auto p-2">
        {editable && (
          <section className="mb-2 rounded-lg border border-dashed border-slate-200 p-1 dark:border-slate-700" data-testid="blocks">
            <div className="flex items-center gap-2 px-2 py-1 text-xs font-semibold text-slate-600 dark:text-slate-300">
              <Package className="size-4 text-indigo-500" />
              <span className="flex-1">{t("palette.blocks")}</span>
              <label className="cursor-pointer rounded p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800" title={t("palette.importBlock")}>
                <FileUp className="size-3.5" />
                <input
                  type="file"
                  accept=".lumos,application/json"
                  className="hidden"
                  data-testid="import-block"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void file.text().then((text) => insertBlock(text, centre()));
                    event.target.value = "";
                  }}
                />
              </label>
            </div>
            {shownBlocks.length === 0 ? (
              <p className="px-2 pb-1 text-[11px] text-slate-400">{t("palette.blocksHint")}</p>
            ) : (
              <ul className="space-y-0.5">
                {shownBlocks.map((block) => (
                  <li key={block.name} className="group flex items-center">
                    <button
                      type="button"
                      data-testid="block-item"
                      draggable
                      onDragStart={(event) => {
                        event.dataTransfer.setData(BLOCK_TYPE, block.name);
                        event.dataTransfer.effectAllowed = "copy";
                      }}
                      onClick={() => void insertBlock(block.document, centre())}
                      className="flex-1 truncate rounded-md px-2 py-1 pl-7 text-left text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800"
                    >
                      {block.name}
                    </button>
                    <button
                      type="button"
                      onClick={() => forgetBlock(block.name)}
                      title={t("palette.forgetBlock")}
                      className="invisible rounded p-1 text-slate-400 group-hover:visible hover:text-red-500"
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}
        {groups.length === 0 && <p className="p-3 text-sm text-slate-500">{t("palette.empty")}</p>}
        {groups.map(([name, items]) => {
          const { icon: Icon, color } = category(name);
          const open = !closed[name] || query !== "";
          return (
            <section key={name} data-testid="palette-category" className="mb-1">
              <button
                type="button"
                onClick={() => setClosed({ ...closed, [name]: open })}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs font-semibold text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                <Icon className="size-4" style={{ color }} />
                <span className="flex-1">{t(`categories.${name}`, { defaultValue: name })}</span>
                <span className="font-normal text-slate-400">{items.length}</span>
              </button>
              {open && (
                <ul className="mt-0.5 mb-2 space-y-0.5">
                  {items.map((item) => (
                    <li key={item.type_id}>
                      <button
                        type="button"
                        data-testid="palette-item"
                        data-type={item.type_id}
                        draggable={editable}
                        disabled={!editable}
                        onDragStart={(event) => {
                          event.dataTransfer.setData(DRAG_TYPE, item.type_id);
                          event.dataTransfer.effectAllowed = "copy";
                        }}
                        onClick={() => void addElement(item.type_id, centre())}
                        title={`${plain(item.summary)}\n\n${t("palette.hint")}`}
                        className="group flex w-full cursor-grab items-start gap-2 rounded-md border border-transparent px-2 py-1.5 pl-7 text-left text-sm hover:border-slate-200 hover:bg-slate-50 active:cursor-grabbing disabled:cursor-not-allowed disabled:opacity-60 dark:hover:border-slate-700 dark:hover:bg-slate-800"
                      >
                        <span className="mt-1.5 size-1.5 shrink-0 rounded-full" style={{ background: color }} />
                        <span className="min-w-0">
                          <span className="block truncate font-medium text-slate-700 dark:text-slate-200">
                            {item.title}
                          </span>
                          <span className="line-clamp-1 text-xs text-slate-500 dark:text-slate-400">
                            {plain(item.summary)}
                          </span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          );
        })}
      </div>
    </aside>
  );
}
