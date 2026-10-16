# Getting started

## Installation

LUMOS requires Python 3.12 or newer.

```bash
pip install lumos-maxima
```

Optional features are installed with extras, for example:

```bash
pip install "lumos-maxima[jax]"        # differentiable GPU backend
pip install "lumos-maxima[catalogs]"   # astronomical catalog queries
pip install "lumos-maxima[all]"        # everything except GPU-specific packages
```

## First steps

```python
import lumos

print(lumos.__version__)
```

The first simulation tutorial will be added when the core of the library is available.
