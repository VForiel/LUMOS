import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import "./i18n";
import { App } from "./App";
import { applyTheme } from "./theme";
import { finishOpenRouterLogin } from "./oauth";

applyTheme();

// the popup of the OpenRouter login only hands its code back to the interface, then closes
if (!finishOpenRouterLogin()) {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
