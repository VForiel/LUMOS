# Building a context

The editor shows a context as blocks (elements) connected by wires (connections). Light flows from left to
right: the inputs of a block are on its left, its outputs on its right.

## Adding elements

- **Drag** an element from the palette onto the canvas, or **click** it to place it in a free spot.
- Press **Ctrl+K** (or double-click the canvas) and type a few words: `thin lens`, `camera`, `vortex`.
- Drag a wire from a port and **drop it on empty space**: the editor offers the elements that can be connected
  there, and connects the one you choose.

The palette is generated from the library: every element type appears with its documentation, grouped by
category (sources, scenes, telescopes, optics, detectors, processing...). The categories start collapsed:
click one to open it (the interface remembers it), or type in the search box to see every match.

## Connecting

Drag from an output (right) to an input (left), or click a port then the other one: the whole label of a
port is its handle. The colour of a port tells the kind of signal it carries
(wavefront, wavecube, incoherent bundle, scene, data product...). Ports that cannot be connected refuse the wire
while you drag; the server checks again and explains a refusal. To remove a wire alone, click it: the button
at its middle removes it (or press **Delete**); its blocks stay.

A port carrying a **bus** — several guides or telescope arms on one wire — draws one pin per lane, gathered by
a brace (a double stroke when the number of lanes is free, as on a waveguide), and its wire is thicker: one
wire carries all the lanes. To give the lanes different components, split the bus with a **Demux** (category
*Buses*), wire each lane, and gather them again with a **Mux**. Their ports follow their `n_lanes` setting;
lowering it removes the wires of the ports that disappear. A Mux input left unconnected is an empty lane,
reported as a warning in the problems.

## Moving around

- The **pointer mode** (top left of the canvas) chooses what a drag on the empty canvas does: the hand (**H**)
  moves the view, the box (**V**) selects the blocks it touches. In both modes, **Space+drag** and the middle
  or right button move the view, and the wheel zooms.
- **F** fits the view to the blocks; the minimap (bottom right) moves and zooms too.
- Drag the edge of the palette, of the inspector or of the bottom panel to resize it; the button at the top of
  the palette and of the inspector hides it. The interface remembers these sizes in your browser.

## Editing values

Select a block: the inspector on the right shows its parameters, settings and state, generated from the
element type.

- A quantity has a value and a unit. Type `3`, or `3 mm` to give the unit at the same time; the unit menu
  shows the same quantity in another unit (`2 mW` becomes `0.002 W`) without changing it.
- Values outside the bounds of a parameter are flagged and not applied ("must be greater than 0 m").
- The arrow next to a field puts back its default value; the hover text of a name gives its documentation.

With nothing selected, the inspector shows the context: its name, description and **observation band**
(wavelengths sampled by stars, scenes and lamps).

## Undo, copy, groups

- **Ctrl+Z** undoes any edit (values, connections, moves, groups...); **Ctrl+Shift+Z** or **Ctrl+Y** redoes it.
- **Ctrl+C** / **Ctrl+V** copy and paste the selected blocks with the wires between them; **Ctrl+D** duplicates.
- **Shift+click** adds a block to the selection, **Shift+drag** (or a drag in the box mode) selects a region,
  **Ctrl+A** selects everything.
- **Ctrl+G** turns the selection into a **group**: one block whose ports are the free ports of its elements.
  Expand it to see and edit its elements, collapse it to keep the graph readable, or ungroup it.
- **Save as block** keeps a group in *My blocks* (in your browser) to insert it in any context; *Download*
  saves it as a `.lumos` file, which also loads in Python with `lumos.io.load`.

## Files

- **Save** (Ctrl+S) downloads the context as a `.lumos` file, positions of the blocks included.
- **Open** (Ctrl+O) loads a `.lumos` file, made in the interface, by the Python library or by an assistant.
- **Python** shows the script that builds the same context with the library.

A context with changes that are not saved shows **Not saved** next to its name (and a dot in the title of
the tab). Opening another context, a file or a preset then asks first: save, discard the changes or cancel;
closing or reloading the page asks too.

If the server restarts, the interface restores the context you were editing.

Press **?** to see every keyboard shortcut.
