# AI assistant (MCP)

LUMOS provides an MCP server so that an AI assistant can build, run and analyse simulations for you. The
[Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open standard: Claude Code, Claude
Desktop and many other assistants connect to MCP servers and use their tools on your behalf.

With the server connected, you describe what you want in plain language ("simulate the PSF of an 8 m telescope
at 1.6 µm", "why is my simulation under-sampled?") and the assistant uses LUMOS to answer: it finds the
elements, builds the optical system, checks it, runs it and shows the results with their units and figures.
Everything it builds is a normal LUMOS context: you can save it as a `.lumos` file or get the equivalent
Python script.

```{toctree}
:maxdepth: 1

setup
conversations
reference
```

The assistant goes through the same service layer as the Python library and the web interface (ADR-0015), so
the same request gives the same result everywhere. Its actions are recorded in the journal of each context
with the origin `mcp`. It cannot touch real equipment unless you start the server with `--enable-operation`,
and even then you confirm every action (see [Safety](reference.md#safety)).
