# Setting up an assistant

The MCP server ships with LUMOS as the extra `mcp`:

```console
$ pip install "lumos-maxima[mcp]"
$ lumos mcp --help
```

`lumos mcp` starts the server on the standard streams (stdio): the assistant launches it and talks to it
directly, nothing listens on the network. Most clients only need the command `lumos mcp`.

:::{tip}
If LUMOS is installed in a virtual environment, give the client the full path of the command, for example
`/home/me/project/.venv/bin/lumos` on Linux and macOS or `C:\Users\me\project\.venv\Scripts\lumos.exe` on
Windows, since the client does not activate the environment.
:::

## Claude Code

1. Install [Claude Code](https://docs.claude.com/en/docs/claude-code) and LUMOS with the extra `mcp`.
2. Register the server:

   ```console
   $ claude mcp add lumos -- lumos mcp
   ```

   Everything after `--` is the command that starts the server. Add `--scope user` to use LUMOS in all your
   projects, or `--scope project` to share the configuration with your team in a `.mcp.json` file at the root
   of the project.
3. Start `claude` and type `/mcp`: `lumos` is listed as connected, with its tools.
4. Ask, for example: *"Load the LUMOS coronagraph preset and show me the coronagraphic image."*

Options of the server go after `lumos mcp`, for instance a directory for the archives of the runs:

```console
$ claude mcp add lumos -- lumos mcp --archive-root ~/lumos-runs
```

The prompts of LUMOS appear as slash commands (`/mcp__lumos__design_instrument`, ...) and its resources can be
attached to a message with `@` (`@lumos:lumos://presets`).

## Claude Desktop

1. Install [Claude Desktop](https://claude.ai/download) and LUMOS with the extra `mcp`.
2. Open *Settings → Developer → Edit Config*. This opens `claude_desktop_config.json`, found in
   `~/Library/Application Support/Claude/` on macOS and in `%APPDATA%\Claude\` on Windows.
3. Add the server:

   ```json
   {
     "mcpServers": {
       "lumos": {
         "command": "lumos",
         "args": ["mcp"]
       }
     }
   }
   ```

   Use the full path of `lumos` if it is installed in a virtual environment, and add options to `args`, for
   example `["mcp", "--archive-root", "/Users/me/lumos-runs"]`.
4. Restart Claude Desktop. The tools of LUMOS are listed under the tools button of the message box; the
   assistant asks your permission the first time it uses each tool.

Claude Desktop shows the figures returned by `inspect_port` in the conversation.

## Other MCP clients

Any client following the Model Context Protocol works; LUMOS is tested against the specification with a
client written without the SDK.

- **stdio** (recommended for local use): command `lumos`, arguments `["mcp"]`.
- **Streamable HTTP**: start the server yourself, then give the client its URL:

  ```console
  $ lumos mcp --transport http
  ```

  The server listens on `http://127.0.0.1:8765/mcp` (change it with `--host`, `--port` and `--path`). With
  Claude Code:

  ```console
  $ claude mcp add --transport http lumos http://127.0.0.1:8765/mcp
  ```

  Until the web instance brings user accounts, the HTTP server has no authentication: it only listens on your
  machine unless you pass `--allow-remote`, and it never accepts the operation of equipment from other
  machines.

## Options

| Option | Effect |
|--------|--------|
| `--archive-root DIR` | Directory of the archives of the runs (default: the archive directory of LUMOS). |
| `--presets-dir DIR` | Directory of the preset `.lumos` files offered by `list_presets`. |
| `--enable-operation` | Allow actions on real equipment, each one confirmed by you (see [Safety](reference.md#safety)). |
| `--transport http` | Serve over streamable HTTP instead of stdio, with `--host`, `--port`, `--path`. |
| `--allow-remote` | Accept a non-local address for the HTTP server. |

## Troubleshooting

- **The server does not start**: run `lumos mcp --help` in a terminal. "needs the extra 'mcp'" means that
  LUMOS was installed without `[mcp]`; "command not found" means that the client needs the full path of
  `lumos`.
- **The tools are not listed**: in Claude Code, `claude mcp list` shows the state of each server and
  `claude --debug` shows its errors; in Claude Desktop, the logs are in `~/Library/Logs/Claude/` (macOS) or
  `%APPDATA%\Claude\logs\` (Windows), in `mcp-server-lumos.log`.
- **Long runs**: `run` waits up to `wait_seconds` (300 s by default) while reporting the progress, then lets
  the computation continue in the background; the assistant follows it with `get_job`.
