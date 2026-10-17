// Add an element by name (Ctrl+K, double click on the canvas, or a connection dropped on empty space, which
// offers the compatible elements and connects the new one).

import { CornerDownLeft, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { ElementType } from "../api";
import { category } from "../lib/categories";
import { portsOf } from "../lib/schema";
import { compatible } from "../lib/signals";
import { useStore } from "../store";
import { plain } from "./Palette";

function score(item: ElementType, words: string[]): number {
  const title = item.title.toLowerCase();
  const text = `${item.type_id} ${item.summary}`.toLowerCase();
  let total = 0;
  for (const word of words) {
    if (title.startsWith(word)) total += 4;
    else if (title.includes(word)) total += 3;
    else if (text.includes(word)) total += 1;
    else return -1;
  }
  return total;
}

export function QuickAdd() {
  const { t } = useTranslation();
  const quick = useStore((s) => s.quickAdd);
  const palette = useStore((s) => s.palette);
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (quick) {
      setQuery("");
      setIndex(0);
      window.setTimeout(() => input.current?.focus(), 0);
    }
  }, [quick]);

  const matches = useMemo(() => {
    if (!palette || !quick) return [];
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    const from = quick.from;
    return palette.types
      .filter((item) => {
        if (!from) return true;
        const ports = portsOf(item.schema);
        return from.direction === "out"
          ? ports.inputs.some((p) => compatible(from.signal, p.signal, palette.signals))
          : ports.outputs.some((p) => compatible(p.signal, from.signal, palette.signals));
      })
      .map((item) => [item, words.length ? score(item, words) : 0] as const)
      .filter(([, value]) => value >= 0)
      .sort((a, b) => b[1] - a[1] || a[0].title.localeCompare(b[0].title))
      .slice(0, 12)
      .map(([item]) => item);
  }, [palette, quick, query]);

  if (!quick) return null;
  const close = () => useStore.getState().setQuickAdd(null);

  const choose = async (item: ElementType) => {
    close();
    const store = useStore.getState();
    const id = await store.addElement(item.type_id, quick.position);
    const from = quick.from;
    if (!id || !from || !palette) return;
    const ports = portsOf(item.schema);
    if (from.direction === "out") {
      const port = ports.inputs.find((p) => compatible(from.signal, p.signal, palette.signals));
      if (port) await store.connect(from.port, `${id}.${port.name}`);
    } else {
      const port = ports.outputs.find((p) => compatible(p.signal, from.signal, palette.signals));
      if (port) await store.connect(`${id}.${port.name}`, from.port);
    }
  };

  return (
    <div className="fixed inset-0 z-40 bg-slate-900/20 backdrop-blur-[1px]" onMouseDown={(event) => event.target === event.currentTarget && close()}>
      <div className="mx-auto mt-[12vh] w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900">
        <label className="flex items-center gap-2 border-b border-slate-100 px-4 dark:border-slate-800">
          <Search className="size-4 text-slate-400" />
          <input
            ref={input}
            data-testid="quick-add"
            value={query}
            placeholder={quick.from ? t("quickAdd.compatible", { port: quick.from.port }) : t("quickAdd.placeholder")}
            onChange={(event) => {
              setQuery(event.target.value);
              setIndex(0);
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") setIndex((i) => Math.min(i + 1, matches.length - 1));
              else if (event.key === "ArrowUp") setIndex((i) => Math.max(i - 1, 0));
              else if (event.key === "Enter" && matches[index]) void choose(matches[index]);
              else if (event.key === "Escape") close();
              else return;
              event.preventDefault();
            }}
            className="h-12 flex-1 bg-transparent text-base outline-none placeholder:text-slate-400"
          />
        </label>
        <ul className="max-h-80 overflow-y-auto p-1.5">
          {matches.length === 0 && <li className="px-3 py-4 text-sm text-slate-500">{t("palette.empty")}</li>}
          {matches.map((item, k) => {
            const { icon: Icon, color } = category(item.category);
            return (
              <li key={item.type_id}>
                <button
                  type="button"
                  onMouseEnter={() => setIndex(k)}
                  onClick={() => void choose(item)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left ${k === index ? "bg-accent-50 dark:bg-accent-900/30" : ""}`}
                >
                  <Icon className="size-4 shrink-0" style={{ color }} />
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-medium">{item.title}</span>
                    <span className="block truncate text-xs text-slate-500 dark:text-slate-400">{plain(item.summary)}</span>
                  </span>
                  {k === index && <CornerDownLeft className="size-4 text-slate-400" />}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
