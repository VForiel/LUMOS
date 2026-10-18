# Versions and stability

LUMOS follows [semantic versioning](https://semver.org/) from 1.0 (ADR-0021).

- **Your scripts** keep working through the 1.x versions: the public API (the names documented in the
  [API reference](../api/index.md)) only grows. A name to be removed is deprecated first, with a
  `DeprecationWarning` and a note in the changelog, during at least one minor version; it disappears in 2.0
  at the earliest. Names starting with `_` are internal.
- **Your files** stay readable: every `.lumos` document written by a 1.x version is read by the later ones.
  When an element type changes its fields, LUMOS converts the older documents when it reads them. A newer
  feature of the format (format 1.1 added feedback loops) raises its minor version; an older LUMOS refuses
  such a document with a message asking to upgrade, instead of dropping what it does not know.
- **Element types** keep their identifiers (`lumos.optics.ThinLens`), the names and units of their fields
  and their ports. Every field has a short description, shown in the web interface, the JSON schemas and the
  MCP tools.
- **Servers and assistants**: the HTTP API of `lumos web`, the MCP tools and the commands of the `lumos` command
  line follow the same rules.

Identifiers use the American spelling of the scientific Python libraries (`normalize`, `center`, `color`),
snake_case, `n_<things>` for counts and `astropy.units.Quantity` for every physical value.
