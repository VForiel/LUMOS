# LUM(os)²

**Lumos is a Universal and Modular Optical Simulator & Operating System.**

LUMOS simulates light end to end, from a scene (astronomical, terrestrial or laboratory) to the detected data,
through any optical system. It also drives real optical equipment with the same object model, builds data
processing pipelines from ready-made blocks, and exposes everything to AI assistants through an MCP server.

> ⚠️ LUMOS is in early development (pre-alpha). The API is not usable yet.

## Installation

```bash
pip install lumos-maxima
```

```python
import lumos
```

## Development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,docs]"   # Windows: .venv\Scripts\python.exe
.venv/bin/python -m pytest
```

See [CLAUDE.md](CLAUDE.md) for the development conventions, [docs/specs/](docs/specs/) for the requirements and
the development plan (in French), and [docs/adr/](docs/adr/) for the architecture decisions.

## Support

LUMOS is developed on spare time. If it helps your work, you can
[sponsor its development](https://github.com/sponsors/VForiel) ❤️

## License

[MIT](LICENSE)
