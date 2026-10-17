// Light and dark themes: follow the system unless the user chose one.

export type Theme = "light" | "dark" | "system";

const KEY = "lumos.theme";

export function savedTheme(): Theme {
  try {
    const value = window.localStorage.getItem(KEY);
    if (value === "light" || value === "dark") return value;
  } catch {
    /* storage disabled */
  }
  return "system";
}

export function applyTheme(theme: Theme = savedTheme()): void {
  const dark = theme === "dark" || (theme === "system" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", Boolean(dark));
}

export function setTheme(theme: Theme): void {
  try {
    if (theme === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, theme);
  } catch {
    /* storage disabled */
  }
  applyTheme(theme);
}

window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => applyTheme());
