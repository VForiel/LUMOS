# Assistant panel

The **Assistant** tab lets you ask a language model to build, check, run and explain the context open in the
interface, in plain language: "add a camera 200 mm after the lens", "why is the image under-sampled?", "run it
and give me the contrast between 3 and 4 λ/D".

## Your model, your account

The assistant is a model of your choice, called **directly from your browser** with your own account. Its key
is kept in the browser tab (session storage) and sent to its provider only: it never reaches the LUMOS server.
Requests are billed to your account. *Forget the key* removes it; closing the tab does too.

| Provider | What you need |
|----------|---------------|
| Claude (Anthropic) | an API key of the [Anthropic console](https://console.anthropic.com) |
| OpenAI | an API key of the [OpenAI platform](https://platform.openai.com/api-keys) |
| Google Gemini | a key of [Google AI Studio](https://aistudio.google.com/apikey) (free tier available) |
| OpenRouter | **Log in with OpenRouter** (no key to copy), or an OpenRouter key; gives access to the models of most providers |
| OpenAI-compatible | the address of the endpoint (`…/v1`), a model and, if needed, a key: Mistral, Groq, DeepSeek, or a model on your machine with Ollama (`http://localhost:11434/v1`), LM Studio or vLLM |

The model can be changed at any time (*Change the model*); a new model starts a new conversation.

**Log in with OpenRouter** opens the OpenRouter login in a popup. Once you accept, OpenRouter creates a key on
your account for LUMOS, with your credits and limits (OAuth with PKCE: LUMOS never sees your password, and the
code of the login is useless without the secret kept by the page). You can revoke that key on OpenRouter.

A compatible endpoint must accept requests from the page of LUMOS (CORS): Ollama does with
`OLLAMA_ORIGINS=<address of LUMOS>`; most hosted APIs do.

## Your Claude or ChatGPT subscription

A Claude.ai or ChatGPT **subscription** cannot be used by other applications: their providers only give API
access with API keys. But both accept LUMOS as a **connector** (a remote MCP server): their own assistant then
works on the contexts of your tab, and you see its edits in the interface as they happen.

1. In the assistant tab, press **Create a connector link**. The link is the address of the MCP server of your
   session, with a secret: anyone with it works in your session, so keep it private and revoke it when you are
   done. It stays valid for 90 days (`connector_link_days`), even after your browser session ended or the
   server restarted: the assistant then works in a new session, and asking it for the link to see its work
   (below) opens that session in your browser. Creating a new link or revoking it ends the old one.
2. Add it to your assistant:
   - **Claude** (Pro, Max, Team or Enterprise): *Settings › Connectors › Add custom connector*, paste the
     link, then enable LUMOS in a conversation;
   - **ChatGPT** (Plus or Pro): *Settings › Apps & Connectors › Advanced settings › Developer mode*, then
     *Create*, paste the link, no authentication;
   - **Claude Code**: `claude mcp add --transport http lumos <link>`.

The page follows the assistant: its edits appear in the open context, and a context it creates or loads is
opened in the editor (the previous one stays in the session). If your page is closed, or you never opened
it, ask the assistant for **the link to see its work**: the tool `get_view_link` gives a link opening the
context in your browser, in the same session. Like the connector link, it gives access to your session:
keep it to yourself. It lasts a day.

Claude.ai and ChatGPT reach the connector from the Internet: use the link of a public instance (HTTPS). On
your own machine (`lumos web`), only local clients can use it: Claude Code, Claude Desktop, or `lumos mcp`
directly.

## What it can do

The assistant uses the tools of the MCP server of the instance (the same as `lumos mcp`, see the
[AI assistant guide](../assistant/index.md)), in **your session**: what it adds or changes appears at once in
the editor, and you can undo it like any edit. Its actions are journaled with the origin `mcp`. Figures
returned by the tools (inspected ports) are shown in the conversation and given to models that read images.

- Viewers cannot use the assistant; simulators can; only operators get the tools that drive equipment, and
  every action on equipment is confirmed in a dialog, whoever asks for it.
- An administrator turns the panel and the connector links off with `assistant = false` in the
  configuration.

## Other MCP clients

The same server is at `/mcp` on the instance (streamable HTTP), with the authentication of the interface. A
client such as Claude Code connects with a bearer token from `lumos login`:

```console
$ TOKEN=$(lumos login https://lumos.example.org --user bob --key ~/.ssh/id_ed25519)
$ claude mcp add --transport http lumos https://lumos.example.org/mcp --header "Authorization: Bearer $TOKEN"
```

On your own machine, `lumos mcp` (stdio) remains the simplest way.
