# Configuration

An instance is configured by its profile, a TOML file and options of `lumos web`; options win over the file,
and the file wins over the defaults of the profile.

## Profiles

| | `local` (default) | `public` |
|---|---|---|
| Listens on | `127.0.0.1` only | the address you give (`host`) |
| Visitors who did not log in | operators (it is your machine) | simulators (`anonymous_role`) |
| Operation of equipment | with `--enable-operation`, each action confirmed | disabled; can be enabled with TLS and operators |
| Quotas | none | grids, memory, time, concurrency, sessions, requests |

The local profile refuses to listen on another address: use the public profile for that. An unsafe
configuration stops the server at start-up with a message saying what is missing.

## Configuration file

```toml
# lumos-web.toml
profile = "public"
host = "127.0.0.1"                    # behind a reverse proxy on the same machine
port = 8000
allowed_hosts = ["lumos.example.org"] # host names of the instance
trust_proxy = true                    # the proxy terminates TLS and sets X-Forwarded-For/Proto
users_file = "/etc/lumos/users.toml"  # lumos users add ...
audit_log = "/var/log/lumos/audit.log"
anonymous_role = "simulator"          # "viewer", "simulator" or "none" (login required)
session_idle_minutes = 120
requests_per_minute = 600
max_sessions = 500

[service]
archive_root = "/var/lib/lumos/archives"
max_run_seconds = 60
max_grid_pixels = 1024
max_memory_mb = 2048
max_concurrent_runs = 4
max_contexts_per_session = 5
max_jobs_per_session = 1
operation_enabled = false
```

```console
$ lumos web --config lumos-web.toml
```

## Settings

| Setting | Default (local / public) | Meaning |
|---------|--------------------------|---------|
| `profile` | `local` | `local` or `public` |
| `host`, `port` | `127.0.0.1`, `8000` | address to listen on |
| `allowed_hosts` | — | host names accepted in requests (all on a public instance if empty) |
| `allowed_origins` | — | extra origins allowed to open WebSockets (development servers) |
| `trust_proxy` | `false` | behind a reverse proxy terminating TLS |
| `tls_certfile`, `tls_keyfile` | — | serve HTTPS directly |
| `users_file` | — | users and roles ([Accounts](accounts.md)) |
| `anonymous_role` | `operator` / `simulator` | role of visitors who did not log in: `"viewer"` shows without computing, `"none"` requires a login |
| `session_idle_minutes` | 720 | sessions end after this idle time |
| `session_max_hours` | 12 | logged-in sessions end after this time |
| `challenge_seconds` | 120 | validity of an SSH challenge |
| `requests_per_minute` | none / 600 | per client address |
| `login_attempts_per_minute` | 20 / 5 | per client address and per user |
| `max_sessions` | none / 500 | sessions open at the same time |
| `audit_log` | — | JSON-lines journal of the security events |
| `assistant` | `true` | offer the assistant panel |

Quotas of the computations (`[service]`), shared with the MCP server:

| Setting | Default (local / public) | Meaning |
|---------|--------------------------|---------|
| `max_run_seconds` | 600 / 60 | time of a run or of an inspection |
| `max_grid_pixels` | none / 1024 | largest grid side of an element (`n_pixels`, output grids) |
| `max_memory_mb` | none / 2048 | estimated memory: N² × 16 bytes per wavelength, 4 working copies |
| `max_concurrent_runs` | none / 4 | runs in progress, all visitors together |
| `max_jobs_per_session` | 2 / 1 | runs in progress per visitor |
| `max_contexts_per_session` | 20 / 5 | contexts open per visitor |
| `max_workers` | 4 | threads computing the runs |
| `operation_enabled` | `false` | allow actions on real equipment |
| `archive_root`, `presets_dir` | — | directories of the archives and of the presets |
| `session_storage` | `false` / `true` | archives of each visitor in a directory of their own, listed to them only and deleted when their session ends |
| `max_session_bytes` | none / 10⁹ (1 GB) | data kept per session; beyond, its oldest archives and recordings are deleted |
| `min_free_disk_bytes` | none / 2 × 10⁹ | free disk below which archiving is refused; visitors are warned below twice this value |
| `max_jobs_kept` | 50 / 20 | finished runs and studies of a session whose results stay in memory |

A refused computation answers with the code `quota` and says which limit applies, so that the user can
reduce the number of pixels or of wavelengths.

## Enabling the operation on a public instance

Set `operation_enabled = true` in `[service]`, serve the instance over TLS (`tls_certfile` and
`tls_keyfile`, or `trust_proxy = true` behind a proxy with HTTPS) and register at least one operator in the
users file. Anonymous visitors can never be operators there. Each action is still confirmed in the interface and
written to the audit journal.
