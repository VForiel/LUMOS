# Assistant panel

The **Assistant** tab lets you ask Claude to build, check, run and explain the context open in the interface,
in plain language: "add a camera 200 mm after the lens", "why is the image under-sampled?", "run it and give me
the contrast between 3 and 4 λ/D".

## Your key, your account

The assistant is Claude, called directly from your browser with **your own Anthropic API key** (create one on
[console.anthropic.com](https://console.anthropic.com)). The key is kept in the browser tab (session storage)
and sent to the Claude API only: it never reaches the LUMOS server. Requests are billed to your Anthropic
account. *Forget the key* removes it; closing the tab does too.

## What it can do

The assistant uses the tools of the MCP server of the instance (the same as `lumos mcp`, see the
[AI assistant guide](../assistant/index.md)), in **your session**: what it adds or changes appears at once in
the editor, and you can undo it like any edit. Its actions are journaled with the origin `mcp`. Figures
returned by the tools (inspected ports) are shown in the conversation and given to Claude.

- Viewers cannot use the assistant; simulators can; only operators get the tools that drive equipment, and
  every action on equipment is confirmed in a dialog, whoever asks for it.
- An administrator turns the panel off with `assistant = false` in the configuration.

## Other MCP clients

The same server is at `/mcp` on the instance (streamable HTTP), with the authentication of the interface. A
client such as Claude Code connects with a bearer token from `lumos login`:

```console
$ TOKEN=$(lumos login https://lumos.example.org --user bob --key ~/.ssh/id_ed25519)
$ claude mcp add --transport http lumos https://lumos.example.org/mcp --header "Authorization: Bearer $TOKEN"
```

On your own machine, `lumos mcp` (stdio) remains the simplest way.
