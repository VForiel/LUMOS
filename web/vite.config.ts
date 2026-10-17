/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = process.env.LUMOS_BACKEND ?? "http://127.0.0.1:8000";

// The interface is built into the Python package, which serves it at "/" (lumos web).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // absolute: the tests run from the root of the repository (test.root)
  cacheDir: fileURLToPath(new URL("./node_modules/.vite", import.meta.url)),
  build: {
    outDir: fileURLToPath(new URL("../src/lumos/web/static", import.meta.url)),
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    // `npm run dev` with `lumos web` running: the API and its WebSocket are proxied to the backend
    proxy: { "/api": { target: backend, ws: true } },
  },
  test: {
    root: fileURLToPath(new URL("..", import.meta.url)),
    include: ["tests/frontend/**/*.test.ts"],
  },
});
