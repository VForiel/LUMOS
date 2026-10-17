# Web interface

The visual programming interface of LUMOS: building a context with blocks, running it, inspecting results, and
deploying a local or public instance.

Start it on your machine with:

```console
$ pip install "lumos-maxima[web]"
$ lumos web --open
```

The interface opens at <http://localhost:8000>. It only listens on your own machine.

## First steps

On a first visit, the canvas offers the **presets**: ready-made contexts (a laser focused on a camera, the PSF
of the VLT, a vortex coronagraph, nullers, a debris disk...), the same `.lumos` files as the example notebooks.
Open one and press **Run**. The **guided tour** (graduation cap in the toolbar) walks you through the
interface step by step on the first simulation, from the palette to the results, in English or in French.
The **Presets** button of the toolbar opens the gallery again at any time.

```{toctree}
:maxdepth: 1

editing
running
operation
assistant
configuration
accounts
api
development
```
