# Your own components

A **component** is an element you define without code: a template filled with your data. It is a `.lumos`
document of kind `component`, valid on every instance (including the public one, where code never runs), and it
travels inside the contexts that use it.

```json
{
  "lumos_format": "1.3",
  "kind": "component",
  "component": {
    "id": "acme.Attenuator",
    "version": 1,
    "title": "Attenuator",
    "description": "A neutral density with an adjustable power transmission.",
    "license": "MIT",
    "authors": ["ACME optics"],
    "template": "gain_phase",
    "params": {
      "transmission": {"unit": "", "default": 0.5, "bounds": [0, 1], "doc": "power transmission"}
    },
    "body": {"gain": "$transmission", "phase": "0 rad"},
    "checks": [{"kind": "power", "wavelength": "1 um", "power": "2 mW", "expect": "1 mW", "rtol": 1e-9}]
  }
}
```

The identifier is `author.Name`; the element type is `components.acme.Attenuator`. The parameters have a unit
(empty for a number), bounds, a default value and a description, like those of the built-in elements: they are
edited in the inspector and can be swept. In the body, `"$transmission"` is the value of the parameter.

## Templates

| Template | Body | Behaviour |
|----------|------|-----------|
| `gain_phase` | `gain`, `phase`, `amplifier` | multiplies the field by √gain e^{iphase}, on any light (each lane of a bus); a gain above 1 needs `"amplifier": true` |
| `spectral_transmission` | `wavelengths`, `transmission` | a power transmission tabulated against the wavelength, interpolated linearly, 0 outside the table |
| `transfer_matrix` | `amplitudes`, `phases`, `wavelength`, `dispersion` | a matrix on a bus of guided lanes, as a `TransferMatrix` (refused if it amplifies) |
| `composite` | `group` | a group of elements (the entry of a `.lumos` group), placed as a group |

## Using and checking a component

```python
import lumos.io
from lumos.components import check_component, create, register_component

document = json.load(open("attenuator.lumos"))
print(check_component(document))  # [] when the component is valid
register_component(document["component"])
nd = create("acme.Attenuator", name="nd", transmission=0.2 * u.dimensionless_unscaled)
```

`lumos component check attenuator.lumos` lists what makes a component invalid: a missing title, description or
license, a parameter without description or with a default outside its bounds, a body that does not fit its
template or amplifies light, and — required for every templated component — at least one **check** with an
independent reference value. A check of kind `power` sends a beam of `power` at `wavelength` (one guided lane
of that power per input of a matrix) and compares the total output power with `expect`: an energy balance,
or an analytic value you know.

A context saved with components holds their definitions under `"components"`, so it opens on any instance.

## In the web interface

The button **New component** of *My blocks* in the palette opens the component editor:

- **Form**: the author and the name make the identifier; the title, description and license are required.
  Add parameters (name, unit, default, bounds, description), pick a template and fill its behaviour (a value
  is a number, a quantity such as `3 dB`, or `$name` for a parameter), then add at least one check. The
  problems of the component are listed while you type, and **Save** is possible only when there is none.
- **Document**: the `.lumos` document the form makes, which **Download .lumos** saves.
- **Code** (local and private instances only, where scripts run): Python code defining element types with
  `@register`, run in the server.

A saved component is a type of the palette of your session, in the category *Components*, added like any
element and edited with the pencil next to it in *My blocks*; a composite component is placed from *My blocks*
as a group. Assistants do the same with the MCP tools `check_component`, `save_component`, `list_components`
and `add_component`.
