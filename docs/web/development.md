# Developing the interface

The interface lives in `web/` (Vite, React, TypeScript, `@xyflow/react`, Tailwind CSS, i18next, driver.js).
It is built into `src/lumos/web/static/`, which `lumos web` serves at `/`.

## Build once

```console
$ cd web
$ npm ci
$ npm run build
$ lumos web --open
```

## Work on it

Run the backend and the Vite development server side by side; Vite reloads the page as you edit and forwards
`/api` (and its WebSocket) to the backend:

```console
$ lumos web                 # terminal 1: http://localhost:8000
$ cd web && npm run dev     # terminal 2: http://localhost:5173
```

Set `LUMOS_BACKEND=http://127.0.0.1:9000` before `npm run dev` if the backend listens elsewhere.

## Principles

- **No code per element.** The palette, the blocks, their ports, the forms and the units come from
  `/api/palette` (JSON Schemas of the registry, signal classes, units with their conversion factors). A new
  element type appears in the interface as soon as it is registered.
- **The backend decides.** Connections are checked in the browser for comfort (colours, refusal while
  dragging), and again by the service, whose message and hint are shown when it refuses.
- **Every text is translated.** Texts are in `web/src/locales/en.json` and `fr.json`, with the same keys
  (checked by the tests). Element names and their documentation come from the library, in English.

## Tests

| Command | What |
|---------|------|
| `npm test` (in `web/`) | unit tests of the pure logic (`tests/frontend`, Vitest): units, forms from schemas, typed ports, layout, translations |
| `npm run typecheck` | TypeScript |
| `pytest tests/unit/web` | HTTP API, authentication, configuration |
| `pytest -m e2e tests/e2e` | the interface in Chromium (Playwright), in English and in French |

The browser tests need the built interface and a browser: `pip install -e ".[e2e]"` then
`playwright install chromium`.
