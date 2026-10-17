// English and French texts of the interface (REQ-WEB-8); the choice is remembered in the browser.

import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en.json";
import fr from "./locales/fr.json";

export const LANGUAGES = ["en", "fr"] as const;
export type Language = (typeof LANGUAGES)[number];

const KEY = "lumos.language";

function initial(): Language {
  try {
    const saved = window.localStorage.getItem(KEY);
    if (saved === "en" || saved === "fr") return saved;
  } catch {
    /* storage disabled */
  }
  const preferred = navigator.languages?.length ? navigator.languages : [navigator.language];
  for (const language of preferred) {
    const code = language.toLowerCase().slice(0, 2);
    if (code === "fr" || code === "en") return code;
  }
  return "en";
}

export function setLanguage(language: Language): void {
  void i18n.changeLanguage(language);
  try {
    window.localStorage.setItem(KEY, language);
  } catch {
    /* storage disabled */
  }
}

i18n.on("languageChanged", (language) => {
  document.documentElement.lang = language;
});

void i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, fr: { translation: fr } },
  lng: initial(),
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});
document.documentElement.lang = i18n.language;

export default i18n;
