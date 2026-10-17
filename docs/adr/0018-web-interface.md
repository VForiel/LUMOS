# ADR-0018: Web interface

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P14.1, P14.4 (REQ-WEB-1 to REQ-WEB-13, REQ-SVC-1)

## Context

The web interface lets people build a context with blocks, run it and inspect its outputs without writing
Python (UC5: a student opens the public instance, loads a preset and follows a tutorial). It must give the same
results as the Python API and the AI assistant (objective O4), serve a single user on their own machine as well
as anonymous visitors on a public server, and never let a web page drive real equipment without authentication.

## Decision

- **Backend** `lumos.web`: a FastAPI application built by `create_app(WebConfig)`, served by `lumos web`
  (uvicorn). Each route translates one HTTP request into one call of `lumos.service.Service` (ADR-0015) and
  nothing else; a `ServiceError` keeps its message, hint and code and gets an HTTP status (`invalid` 400,
  `forbidden` 403, `not_found` 404, `failed` 422, `quota` 429). The routes are under `/api` (OpenAPI page at
  `/api/docs`).
- **Sessions**: each browser gets an opaque random token in an `HttpOnly`, `SameSite=Strict` cookie (`Secure`
  over HTTPS), bound to its own session of the service (origin `web`). The token is not the identifier of the
  service session. Sessions expire after an idle time and close their contexts.
- **Live updates**: the WebSocket `/api/ws` pushes the changes of the runs of the session (status, progress) and
  the new entries of the journals of its contexts (actions, runs, captured logs). It polls the service every
  0.2 s: runs last seconds to minutes, and polling keeps the service free of callbacks into an event loop.
- **Local safety**: `lumos web` listens on `127.0.0.1` by default. In the `local` profile the backend refuses
  HTTP requests addressed to a host name other than `localhost`, `127.0.0.1` or `::1` (a page of another site
  rebound to `127.0.0.1` by DNS cannot use it) and WebSockets opened from a non-local origin (cookies are not
  enough to protect a WebSocket).
- **Palette**: `/api/palette` gives every element type with its JSON Schema, its category (the second part of
  its identifier, `lumos.<category>.<Name>`) and the hierarchy of the signal classes; the frontend builds the
  palette, the forms and the typed connections from it, with no code per element (REQ-WEB-2).
- **Frontend**: sources in `web/` (Vite, React, TypeScript, `@xyflow/react`, Tailwind CSS, i18next, driver.js),
  built into `src/lumos/web/static/` (not versioned) and served by the backend at `/`; wheels and the Docker
  image are built with it. Without a build, `/` explains how to build it and the API keeps working.

## Alternatives considered

- Server-rendered pages (Jinja, HTMX): a graph editor needs a rich client; React Flow is mature (§8.3).
- Server-sent events for progress: the WebSocket also serves the requests of the client in later steps and is
  what REQ-WEB-5 names.
- Storing the frontend build in git: generated files in reviews; the build is reproducible with `npm ci`.
- Calling the Python API from the routes: no parity by construction and duplicated safeguards (ADR-0015).

## Consequences

- Every feature added to the service is one route away from the web; the web adds no physics.
- Building a wheel or an image needs Node.js; `pip install -e .` from a checkout works without it (API only).
- The authentication, roles and profiles of a public instance come next (P14.2, P14.3, ADR-0019).
