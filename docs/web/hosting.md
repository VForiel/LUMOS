# Hosting

## On your machine, with Docker

The image holds the backend and the interface. From the root of the repository:

```console
$ docker compose -f deploy/compose.yaml up --build
```

The interface is at <http://localhost:8000>. The image runs the **local** profile: inside the container the
server listens on all interfaces (`--listen-all`), and the compose file publishes its port on `127.0.0.1` only,
so no other machine can reach it. Archives of the runs are kept in the `lumos-data` volume. Without compose:

```console
$ docker build -f deploy/Dockerfile -t lumos .
$ docker run --rm -p 127.0.0.1:8000:8000 -v lumos-data:/data lumos
```

Never publish the port of the local profile on another address (`-p 8000:8000`): its user has every right.

## A public instance on a VPS

The public instance serves anonymous visitors (simulators by default) with quotas, behind a reverse proxy
that terminates TLS. `deploy/public/` holds a working setup with [Caddy](https://caddyserver.com), which obtains
and renews the certificates by itself.

### Requirements

- A VPS with Docker and Docker Compose; 4 cores and 8 GB of memory serve a few dozen simultaneous visitors with
  the default quotas (the computations are the limit, not the interface).
- A DNS name pointing to the VPS (`lumos.example.org` below); ports 80 and 443 open.

### Steps

1. Clone the repository on the VPS (`deploy/public/update.sh` needs `git` and `python3`), or copy only
   `deploy/` with an image built elsewhere.
2. Replace `lumos.example.org` by your host name in `deploy/public/lumos.toml` (`allowed_hosts`) and
   `deploy/public/Caddyfile`.
3. Adapt the quotas of `lumos.toml` to the machine (see below).
4. Start it:

   ```console
   $ sh deploy/public/update.sh
   $ docker compose -f deploy/public/compose.yaml logs -f lumos
   ```

5. Open `https://lumos.example.org`. `https://lumos.example.org/api/health` answers `{"status": "ok"}`.

### Sizing the quotas

The quotas bound what one visitor and all visitors together may compute ([Configuration](configuration.md)):

| Setting | Rule of thumb |
|---------|---------------|
| `max_workers`, `max_concurrent_runs` | number of cores minus one |
| `max_memory_mb` | memory of the machine divided by `max_concurrent_runs`, minus a margin |
| `max_grid_pixels` | 1024 (a 1024 × 1024 complex field takes 16 MB per wavelength) |
| `max_run_seconds` | 60 for interactive use |
| `max_sessions`, `requests_per_minute` | a few hundred |

The compose file also limits the CPU and memory of the container, so that the proxy stays responsive.

### Behind an existing reverse proxy

On a machine that already runs a reverse proxy for other sites, do not start the bundled Caddy: write
`deploy/public/compose.override.yaml` (not versioned, taken into account by `update.sh`), for instance with a
proxy that reaches its sites through a Docker network named `proxy`:

```yaml
services:
  caddy:
    profiles: ["disabled"]
  lumos:
    networks: [default, proxy]
networks:
  proxy:
    external: true
```

and forward the host name to `lumos:8000` in the configuration of that proxy (WebSockets included).

### Disk

Results do not pile up on a public instance: the archives of a visitor are kept in a directory of their
session (`session-<id>` under `archive_root`), which only they can list and download, and which is deleted
when the session ends (logout, idle time). A session keeps at most `max_session_bytes` (1 GB): beyond, its
oldest data are deleted and the visitor is told so. Visitors see the data of their session in the results
panel and are warned when they reach 80 % of it, or when less than twice `min_free_disk_bytes` stays free
on the disk; below `min_free_disk_bytes` (2 GB), archiving is refused while runs go on. Only the 20 latest
finished runs of a session keep their results in memory (`max_jobs_kept`). Size the volume for
`max_sessions` × `max_session_bytes` in the worst case, or lower these values.

### Accounts

Anonymous visitors need no account. To give some users more rights (or to require a login with
`anonymous_role = "none"`), create the users file in the volume and point `users_file` to it:

```console
$ docker compose -f deploy/public/compose.yaml exec lumos lumos users add alice --role simulator --password --file /data/users.toml
```

See [Accounts and security](accounts.md).

### Operating equipment from a public instance

It stays disabled unless you decide otherwise: set `operation_enabled = true`, keep the instance behind TLS
(`trust_proxy = true` with the proxy above) and register at least one operator. The server refuses to start
when one of these is missing. Every action is confirmed in the interface and written to the audit journal.

### What runs on a public instance

Visitors build and run contexts, procedures made of blocks and declarative components: data, never code.
Procedures written as Python scripts and components with code run only on local and private instances; a
public instance open to anonymous visitors refuses to start with `scripts_enabled = true`, and a script
procedure opened from a file can be read but not run. Element packages you install yourself (entry point
`lumos.elements`, see [Create and share a component](../user_guide/sharing.md)) are loaded at start-up.
Procedures run as jobs, within the same quotas as runs.

### Maintenance

- **Updates**: `sh deploy/public/update.sh` pulls the new version and rebuilds the image with its version
  number (`sh deploy/public/update.sh v1.0.0` deploys a release).
  Sessions are kept in memory: a restart logs everybody out (the interface restores the context of each
  visitor from their browser).
- **Backups**: the `lumos-data` volume holds the users file, the audit journal and the archives.
- **Logs**: `docker compose logs lumos`; security events are in `/data/audit.log` (one JSON object per line).
- **Checklist**: HTTPS only (Caddy redirects HTTP), `allowed_hosts` set, operation disabled or guarded,
  quotas adapted, volume backed up, the image updated with LUMOS.
