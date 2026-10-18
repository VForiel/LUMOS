# HTTP API

The web interface talks to the backend through a JSON API under `/api`. You can use it from any HTTP client;
the interactive OpenAPI page of a running instance is at `/api/docs`. Every route is one call of the service
layer, so the API, the Python library and the AI assistant give the same results.

## Conventions

- Quantities are text with their unit: `"0.5 mm"`, `"1.6 um"`, `"2 mW"`.
- Ports are written `element.port` (`laser.out`), parameters `element.parameter` (`laser.power`).
- The first request opens a session and sets the cookie `lumos_session`; send it back with every request. The
  contexts and runs of a session are only visible to it.
- Errors are JSON objects `{"error": ..., "hint": ..., "code": ...}`, where `hint` says how to fix the request:

  | `code` | HTTP status | Meaning |
  |--------|-------------|---------|
  | `invalid` | 400 | the request is wrong (unit, port, body) |
  | `forbidden` | 403 | not allowed (operation disabled, role) |
  | `not_found` | 404 | unknown context, element, port, preset... |
  | `failed` | 422 | the computation failed |
  | `quota` | 429 | a quota of the instance is reached |

## Routes

| Route | Purpose |
|-------|---------|
| `GET /api/info` | version, profile, operation enabled, languages |
| `GET /api/element-types?query=` | element types, a page at a time |
| `GET /api/element-types/{type}/schema` | JSON Schema of a type (parameters, units, bounds, ports) |
| `GET /api/palette` | every type with its schema and category, and the signal classes |
| `GET /api/docs/search?q=` | documentation pages |
| `GET, POST /api/contexts` | list, create (`{"name", "mode"}`) |
| `GET, DELETE /api/contexts/{id}` | describe, close |
| `GET /api/presets`, `POST /api/presets/{name}/load` | presets |
| `POST /api/contexts/import` | create a context from a `.lumos` document (request body) |
| `GET /api/contexts/{id}/export.lumos`, `export.py` | download the `.lumos` file, the Python script |
| `POST /api/contexts/{id}/elements` | add an element (`{"type_id", "name", "params"}`) |
| `DELETE /api/contexts/{id}/elements/{path}` | remove an element |
| `POST, DELETE /api/contexts/{id}/connections` | connect (`{"source", "target"}`), disconnect (query) |
| `PATCH /api/contexts/{id}/params` | set a value (`{"path": "laser.power", "value": "3 mW"}`) |
| `PUT /api/contexts/{id}/observation` | observation band (`{"minimum", "maximum", "n_wavelengths"}`) |
| `GET /api/contexts/{id}/validate` | problems with suggested fixes |
| `GET /api/contexts/{id}/sampling/{element}` | propagation method and why |
| `POST /api/contexts/{id}/runs` | start a run (`{"outputs": [...]}`; none: the default outputs) |
| `GET /api/jobs/{job}`, `GET /api/jobs/{job}/results/{port}` | follow a run, read an output |
| `PATCH /api/contexts/{id}` | name, description, seed (`{"seed": 42}`) |
| `PUT /api/contexts/{id}/free` | free a parameter (`{"path", "free", "prior": {"kind": "normal", "mean", "std"}}`) |
| `POST /api/contexts/{id}/studies` | start a sweep (`{"kind": "sweep", "parameters", "mode"}`) or a Monte-Carlo run (`{"kind": "sample", "n"}`) |
| `POST, DELETE /api/contexts/{id}/feedbacks` | close a loop (`{"source", "target", "mode"}`), open it (query) |
| `POST /api/contexts/{id}/loops` | run a loop (`{"steps": 30, "period": "2 ms", "outputs": [...]}`) |
| `GET /api/jobs/{job}/study`, `study.csv` | points of a study or steps of a loop, as JSON or CSV |
| `GET /api/contexts/{id}/ports/{port}` | summary of the signal of any output |
| `GET /api/contexts/{id}/ports/{port}/figure.png` | its figure |
| `GET /api/contexts/{id}/journal` | journal (actions, runs, logs) |
| `GET /api/catalogs/resolve?name=` | astronomical object |
| `GET /api/archives`, `GET /api/archives/metadata?path=` | archives of the runs |

## Live updates

The WebSocket `/api/ws` (same cookie) sends JSON messages:

- `{"type": "job", "job": {...}}` when a run of the session changes (status, progress, error);
- `{"type": "journal", "context": "...", "entry": {...}}` for each new entry of the journal of a context.

## Example

```python
import httpx

with httpx.Client(base_url="http://localhost:8000") as client:
    cid = client.post("/api/contexts", json={"name": "bench"}).json()["id"]
    client.post(
        f"/api/contexts/{cid}/elements",
        json={"type_id": "lumos.sources.GaussianLaser", "name": "laser", "params": {"power": "2 mW"}},
    )
    client.post(
        f"/api/contexts/{cid}/elements", json={"type_id": "lumos.detectors.PowerMeter", "name": "meter"}
    )
    client.post(f"/api/contexts/{cid}/connections", json={"source": "laser.out", "target": "meter.in"})
    job = client.post(f"/api/contexts/{cid}/runs", json={"outputs": ["meter.out"]}).json()
    # ... poll GET /api/jobs/{id} (or listen on /api/ws), then:
    print(client.get(f"/api/jobs/{job['id']}/results/meter.out").json())
```
