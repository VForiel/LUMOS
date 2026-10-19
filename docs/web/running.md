# Running and inspecting

## Running a context

Press **Run** (or **Ctrl+Enter**). The outputs listed by the context are computed (for a preset, those
chosen by its author; otherwise every output not connected to anything). While it runs, the progress bar
tells which element is being computed; **Cancel** stops the run after the current element.

When the run ends, the **Results** tab shows each output with its figure, its statistics (minimum, maximum,
mean, sum, with units) and details such as the grid, the power or the beam radius. A failed run shows the error
and how to fix it.

Turn on the **archive** button (next to Run) to keep the results of each run on the server, with its
metadata, figures, data and journal; the archive can then be downloaded as a `.zip` file.

## Inspecting any port

Click the **Inspect** button at the top right of a block (its first output), or **Inspect** next to any
output in the inspector, to compute that port alone and see
its summary and figure: the field after the telescope, the image before the coronagraph, the reading of a
detector. Inspecting does not need a run.

## Downloads

Every output and inspected port can be downloaded as:

| Format | Content |
|--------|---------|
| PNG | the figure |
| PltEdit (`.plt`) | the figure, editable in PltEdit (`pltedit.load`) |
| Data (`.npz`) | the data: intensity and axes of a field, or the data product; plain arrays with units, no pickled object (`lumos.results.load_data`, `DataProduct.from_npz`) |
| JSON | the complete signal, as stored in archives (readable by `lumos.load_archive`) |

## Journal and problems

The **Journal** tab lists, as they happen, the runs (outputs and duration), the actions on elements and the
messages of the library. The **Problems** tab lists what would make a run fail (an input not connected, a
missing observation band...), with a fix; click the element to select it. Blocks with a problem are outlined
in red.
