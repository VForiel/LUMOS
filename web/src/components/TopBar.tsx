// Top bar: brand, open context, language and theme. Controls without room in a narrow window go into the
// "More" menu (P15.19): each is shown in the bar from a breakpoint, and in the menu below it.

import { Heart, Languages, Monitor, Moon, MoreHorizontal, Sun } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { LANGUAGES, setLanguage, type Language } from "../i18n";
import { useStore } from "../store";
import { savedTheme, setTheme, type Theme } from "../theme";

export function Logo() {
  return (
    <svg viewBox="0 0 64 64" className="size-7" aria-hidden="true">
      <defs>
        <radialGradient id="lumos-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0" stopColor="#fef3c7" />
          <stop offset="0.45" stopColor="#a78bfa" />
          <stop offset="1" stopColor="#4f46e5" />
        </radialGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="#0f172a" />
      <circle cx="32" cy="32" r="13" fill="url(#lumos-glow)" />
      <path
        d="M32 6v10M32 48v10M6 32h10M48 32h10M13.6 13.6l7 7M43.4 43.4l7 7M13.6 50.4l7-7M43.4 20.6l7-7"
        stroke="#c4b5fd"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function ToolButton({
  icon,
  label,
  onClick,
  disabled,
  testId,
  active,
  shortcut,
  children,
  tour,
}: {
  icon: ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  testId?: string;
  active?: boolean;
  shortcut?: string;
  children?: ReactNode;
  tour?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      data-testid={testId}
      data-tour={tour}
      title={shortcut ? `${label} (${shortcut})` : label}
      aria-label={label}
      className={`inline-flex h-8 items-center gap-1.5 rounded-lg px-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        active
          ? "bg-accent-100 text-accent-700 dark:bg-accent-900/50 dark:text-accent-200"
          : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}

/** Breakpoint from which a control is in the bar (below it, in the More menu). */
export type Below = "sm" | "md" | "lg" | "xl";
// literal class names, seen by Tailwind
const IN_BAR: Record<Below, string> = { sm: "hidden sm:flex", md: "hidden md:flex", lg: "hidden lg:flex", xl: "hidden xl:flex" };
const IN_MENU: Record<Below, string> = { sm: "sm:hidden", md: "md:hidden", lg: "lg:hidden", xl: "xl:hidden" };

/** Controls shown in the bar from the breakpoint ``below`` (and in the More menu under it). */
export function Collapse({ below, children }: { below: Below; children: ReactNode }) {
  return <div className={`${IN_BAR[below]} shrink-0 items-center gap-0.5`}>{children}</div>;
}

export type MoreItem = { id: string; icon: ReactNode; label: string; onClick: () => void; disabled?: boolean; below: Below };

function MoreMenu({ items }: { items: MoreItem[] }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", away);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("mousedown", away);
      window.removeEventListener("keydown", escape);
    };
  }, [open]);
  return (
    <div ref={box} className="relative shrink-0 xl:hidden">
      <ToolButton icon={<MoreHorizontal className="size-4" />} label={t("toolbar.more")} testId="more" active={open} onClick={() => setOpen(!open)} />
      {open && (
        <div
          role="menu"
          data-testid="more-menu"
          className="absolute right-0 z-40 mt-1 max-h-[80vh] w-60 overflow-y-auto rounded-xl border border-slate-200 bg-white p-1 shadow-lg dark:border-slate-700 dark:bg-slate-900"
        >
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              role="menuitem"
              data-testid={`more-${item.id}`}
              disabled={item.disabled}
              onClick={() => {
                setOpen(false);
                item.onClick();
              }}
              className={`${IN_MENU[item.below]} flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-sm text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 dark:text-slate-200 dark:hover:bg-slate-800`}
            >
              <span className="text-slate-500">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Sponsor page of the project (GitHub Sponsors). */
export const SPONSOR_URL = "https://github.com/sponsors/VForiel";

const THEMES: { theme: Theme; icon: ReactNode }[] = [
  { theme: "light", icon: <Sun className="size-4" /> },
  { theme: "dark", icon: <Moon className="size-4" /> },
  { theme: "system", icon: <Monitor className="size-4" /> },
];

export function TopBar({ children, right, more = [] }: { children?: ReactNode; right?: ReactNode; more?: MoreItem[] }) {
  const { t, i18n } = useTranslation();
  const context = useStore((s) => s.context);
  const info = useStore((s) => s.info);
  const dirty = useStore((s) => s.dirty);
  const [theme, chooseTheme] = useState<Theme>(savedTheme());
  const next = THEMES[(THEMES.findIndex((item) => item.theme === theme) + 1) % THEMES.length];
  const switchTheme = () => {
    chooseTheme(next.theme);
    setTheme(next.theme);
  };
  const other = LANGUAGES.find((language) => language !== i18n.language) ?? LANGUAGES[0];
  const items: MoreItem[] = [
    ...more,
    { id: "theme", icon: next.icon, label: t("theme.switch", { theme: t(`theme.${next.theme}`) }), onClick: switchTheme, below: "xl" },
    { id: "language", icon: <Languages className="size-4" />, label: other === "en" ? "English" : "Français", onClick: () => setLanguage(other as Language), below: "xl" },
    { id: "support", icon: <Heart className="size-4" />, label: t("toolbar.support"), onClick: () => window.open(SPONSOR_URL, "_blank", "noopener"), below: "xl" },
  ];

  return (
    <header className="flex h-12 shrink-0 items-center gap-1 border-b border-slate-200 bg-white px-2 sm:gap-2 sm:px-3 dark:border-slate-800 dark:bg-slate-900">
      <div className="flex shrink-0 items-center gap-2 sm:pr-2">
        <Logo />
        <div className="hidden leading-none sm:block">
          <span className="block text-sm font-bold tracking-wide text-slate-900 dark:text-white">LUMOS</span>
          <span className="block text-[10px] text-slate-400">{info ? `v${info.version.split("+")[0]}` : ""}</span>
        </div>
      </div>
      <div className="hidden h-6 w-px bg-slate-200 md:block dark:bg-slate-700" />
      <div className="hidden min-w-0 items-center gap-1.5 px-1 md:flex">
        <span className="truncate text-sm font-medium text-slate-700 dark:text-slate-200" data-testid="context-name">
          {context?.name || t("context.untitled")}
        </span>
        {dirty && (
          <span
            data-testid="unsaved"
            title={t("context.unsaved")}
            aria-label={t("context.unsaved")}
            className="inline-flex shrink-0 items-center gap-1 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800 dark:bg-amber-900/50 dark:text-amber-200"
          >
            <span className="size-1.5 rounded-full bg-amber-500" />
            <span className="hidden lg:inline">{t("context.unsavedShort")}</span>
          </span>
        )}
      </div>
      <div className="flex min-w-0 flex-1 items-center gap-0.5">{children}</div>
      {right}
      <Collapse below="xl">
      <a
        href={SPONSOR_URL}
        target="_blank"
        rel="noopener noreferrer"
        data-testid="support"
        title={t("toolbar.support")}
        aria-label={t("toolbar.support")}
        className="inline-flex h-8 items-center rounded-lg px-2 text-slate-600 transition-colors hover:bg-pink-50 hover:text-pink-600 dark:text-slate-300 dark:hover:bg-pink-950/40 dark:hover:text-pink-400"
      >
        <Heart className="size-4" />
      </a>
      <ToolButton
        icon={THEMES.find((item) => item.theme === theme)!.icon}
        label={t("theme.switch", { theme: t(`theme.${next.theme}`) })}
        testId="theme"
        onClick={switchTheme}
      />
      <select
        data-testid="language"
        aria-label={t("toolbar.language")}
        value={i18n.language}
        onChange={(event) => setLanguage(event.target.value as Language)}
        className="h-8 rounded-lg border border-slate-200 bg-white px-2 text-sm dark:border-slate-700 dark:bg-slate-800"
      >
        {LANGUAGES.map((language) => (
          <option key={language} value={language}>
            {language === "en" ? "English" : "Français"}
          </option>
        ))}
      </select>
      </Collapse>
      <MoreMenu items={items} />
    </header>
  );
}
