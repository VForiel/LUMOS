# Procedures

A **procedure** is what you do with a bench, step after step: close the shutter, take twenty frames, average
them into a dark, open the shutter, take ten frames and subtract the dark. It is ordered in time and keeps
**named results** from one step to the next, unlike the graph (a computation) and the studies (sweeps,
Monte-Carlo runs, loops). It runs the same way in simulation and in operation, from Python, from the web
interface (Procedures tab) or from an assistant.

A procedure is data, not code: a list of steps, each naming its kind in `do`.

```python
from lumos.core.procedure import Procedure

ctx.procedures["dark then measure"] = Procedure.from_json(
    {
        "name": "dark then measure",
        "steps": [
            {"do": "action", "element": "shutter", "action": "close"},
            {
                "do": "repeat",
                "times": 20,
                "steps": [
                    {"do": "action", "element": "camera", "action": "expose"},
                    {"do": "measure", "outputs": ["camera.out"], "as": "darks", "collect": True},
                ],
            },
            {"do": "process", "function": "mean", "args": {"frames": "$darks"}, "as": "dark"},
            {"do": "action", "element": "shutter", "action": "open"},
            {
                "do": "repeat",
                "times": 10,
                "steps": [
                    {"do": "action", "element": "camera", "action": "expose"},
                    {"do": "measure", "outputs": ["camera.out"], "as": "images", "collect": True},
                ],
            },
            {"do": "process", "function": "mean", "args": {"frames": "$images"}, "as": "image"},
            {
                "do": "process",
                "function": "subtract_dark",
                "args": {"frame": "$image", "dark": "$dark"},
                "as": "science",
            },
            {"do": "show", "results": ["science"]},
        ],
    }
)
result = ctx.run_procedure("dark then measure")
result["science"]  # a DataProduct
```

## Steps

| `do` | Fields | Effect |
|------|--------|--------|
| `action` | `element`, `action`, `args`, `kwargs` | calls an action of an element (it takes its time on the clock) |
| `measure` | `outputs`, `as`, `collect` | computes ports and names the value (one port) or the mapping (several); `collect` appends to a list |
| `repeat` | `times`, `steps` | runs the steps several times |
| `for_each` | `values`, `as`, `steps` | runs the steps for each value of a list or a range `{"start", "stop", "num", "log"}`; the value is named `as` inside the loop |
| `process` | `function`, `args`, `as` | applies a processing function |
| `set` | `target`, `value` | sets a parameter, a setting or a component (`"lens.focal_length"`, `"dm.commands[3]"`) |
| `use` | `result`, `port` | feeds an input with a named result in the following measurements (a dark for a `DarkSubtraction` block); `result: null` stops it |
| `wait` | `duration` | lets time pass |
| `if` | `left`, `op`, `right`, `then`, `else` | compares two values (`<`, `<=`, `>`, `>=`, `==`, `!=`) |
| `save` | `results`, `archive` | writes named results to an archive |
| `show` | `results` | marks results to display |

Values are numbers, booleans, text, quantities written as text (`"2 mm"`) or named results (`"$dark"`).
The processing functions are `mean`, `median`, `stack` (`method`: mean, median or sigma_clip), `subtract`,
`divide`, `scale`, `subtract_dark`, `flat_field`, `crop`, `rebin`, `total`, `maximum`, `minimum`, `average`,
`element` and `photometry`; a procedure cannot run anything else.

## Python

`procedure.to_python()` writes an equivalent Python script (the steps become calls on `ctx`, the processing
`call("mean", frames=...)`). A procedure may also *be* a script, `{"name": ..., "script": "..."}`, run with
`ctx`, `results`, `u`, `np` and `call` in its namespace: it is code, so the service runs it only where
`scripts_enabled` is set (local and private instances, never the public one), and loading a file never runs it.

## Checks, equipment and saving

`procedure.problems(ctx)` lists what would stop a procedure before it runs: unknown elements, ports, actions,
functions or fields, results used before they are defined, more than 10 000 loop iterations in all. A
procedure that acts on equipment (elements in operation mode) runs only after a human confirmation of the list
of devices (`ctx.run_procedure(name, confirm=...)`, the confirmation dialog of the interface, or
`confirmed=true` given by an assistant after asking the user).

The procedures of a context are saved in its `.lumos` file (format 1.3); `lumos.io.save(procedure, path)` writes
one alone (document of kind `procedure`). In the service, `run_procedure` starts a job (progress, cancellation,
quotas) whose outputs are the named results, read with `get_result`.
