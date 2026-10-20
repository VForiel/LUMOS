# Running and inspecting

## Running a context

Press **Run** (or **Ctrl+Enter**). The outputs listed by the context are computed (for a preset, those
chosen by its author; otherwise every output not connected to anything). While it runs, the progress bar
tells which element is being computed; **Cancel** stops the run after the current element.

When the run ends, the **Outputs** tab shows each output with its figure, its statistics (minimum, maximum,
mean, sum, with units) and details such as the grid, the power or the beam radius. A failed run shows the error
and how to fix it.

The **archive** button (next to Run) opens the archives of the session: a switch keeps each run on the server,
with its context, metadata, figures, data and journal (the button is highlighted while it is on), and the list
of the archives written so far. Choose one to download it as a `.zip` file, delete it, or open its files: a
data file shows its statistics and its figure, a figure is displayed, and every file can be downloaded alone.

In a narrow window, only one side column (elements or inspector) is open at a time, both closed when the page
loads, and the tabs of the bottom panel are gathered in a menu (three lines) next to the name of the open tab.
The top bar keeps the mode switch and **Run**; the commands without room (files, presets, study, help, theme,
language...) are in its **More** menu (three dots).

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

## Browsing the results

**Browse results** (right of the tabs of the bottom panel) opens a window listing the runs and studies of the
session, newest first. A run shows the figures and values of its outputs. A study is browsed point by point:
choose a port, drag the slider (or press play) over the steps of a loop, the points of a sweep or the draws
of a Monte-Carlo run; the figure, the values and the place on the curve of the port follow the slider.

Figures are kept for the ports of a study started with *Record the figures of each point* (ticked when the
archive switch is on), and for the ports tapped by a **Recorder** block (category *Processing*): connect an
output to it, also an output feeding another block; it records its signal at every point, or one point out
of `every`, without changing it. Recordings count in the data of the session and go with it.

## Journal and problems

The **Journal** tab lists, as they happen, the runs (outputs and duration), the actions on elements and the
messages of the library. The **Problems** tab lists what would make a run fail (an input not connected, a
missing observation band...), with a fix; click the element to select it. Blocks with a problem are outlined
in red.
