// Top bar: brand, open context, language and theme.

import { Monitor, Moon, Sun } from "lucide-react";
import { useState, type ReactNode } from "react";
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

const THEMES: { theme: Theme; icon: ReactNode }[] = [
  { theme: "light", icon: <Sun className="size-4" /> },
  { theme: "dark", icon: <Moon className="size-4" /> },
  { theme: "system", icon: <Monitor className="size-4" /> },
];

export function TopBar({ children, right }: { children?: ReactNode; right?: ReactNode }) {
  const { t, i18n } = useTranslation();
  const context = useStore((s) => s.context);
  const info = useStore((s) => s.info);
  const dirty = useStore((s) => s.dirty);
  const [theme, chooseTheme] = useState<Theme>(savedTheme());
  const next = THEMES[(THEMES.findIndex((item) => item.theme === theme) + 1) % THEMES.length];

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b border-slate-200 bg-white px-3 dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center gap-2 pr-2">
        <Logo />
        <div className="leading-none">
          <span className="block text-sm font-bold tracking-wide text-slate-900 dark:text-white">LUMOS</span>
          <span className="block text-[10px] text-slate-400">{info ? `v${info.version.split("+")[0]}` : ""}</span>
        </div>
      </div>
      <div className="h-6 w-px bg-slate-200 dark:bg-slate-700" />
      <div className="flex min-w-0 items-center gap-1.5 px-1">
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
      <div className="flex flex-1 items-center gap-0.5">{children}</div>
      {right}
      <ToolButton
        icon={THEMES.find((item) => item.theme === theme)!.icon}
        label={t("theme.switch", { theme: t(`theme.${next.theme}`) })}
        testId="theme"
        onClick={() => {
          chooseTheme(next.theme);
          setTheme(next.theme);
        }}
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
    </header>
  );
}
