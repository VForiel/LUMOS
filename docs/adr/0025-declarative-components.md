# ADR-0025: Declarative components, made by users without code

- **Status**: accepted
- **Date**: 2026-10-20
- **Plan step**: P15.14 (also P15.15, P15.16)

## Context

Users want to make their own components (a fibre coupler with its measured matrix, an attenuator, a filter
with its measured transmission) and share them, from the web interface too (review of 19/10/2026). Python code
can already define elements (plugins, script blocks, ADR-0013), but code cannot run on the public instance:
running the code of a visitor gives them the machine. The decisions of 20/10 (`docs/specs/pistes_evolution.md`,
§2) keep code to local and private instances, and ask for **declarative** components, made with a form from
templates, valid everywhere, and a catalogue in a `community/` folder of the repository.

## Decision

1. **A component is data**: a `.lumos` document of kind `component` holding
   `{"id", "version", "title", "description", "license", "authors", "template", "params", "body",
   "checks"}`. Unknown keys are refused; nothing in it is executed, imported or unpickled.
   - `id`: `author.Name` (a lowercase author, a capitalised name); the element type is
     `components.<author>.<Name>`, its type version is `version`.
   - `params`: the parameters of the component, `{name: {"unit", "default", "bounds", "doc"}}`, as those of
     the built-in elements (units, bounds, description), editable in the inspector and free for studies.
   - `body`: the behaviour, given by the `template`; a value of the body is a quantity (`"3 dB"`), a number,
     an array, or `"$name"`, the value of a parameter.
2. **Templates** (decided on 20/10):
   - `transfer_matrix`: a matrix of moduli and phases at a wavelength, achromatic or with optical paths, on a
     bus of guided lanes (the behaviour of `TransferMatrix`), refused if it amplifies;
   - `gain_phase`: a power gain (at most 1 unless `amplifier` is true) and a phase on any optical signal,
     lane by lane on a bus;
   - `spectral_transmission`: a tabulated power transmission `T(λ)` (linear interpolation, 0 outside the
     table) on any optical signal with wavelengths;
   - `composite`: a group (the entry of a `.lumos` group), placed as a group when the component is added
     (it has no element type: it is copied, like "My blocks").
   The impulse response comes later.
3. **Checks** (`lumos.components.check_component`, `lumos component check FILE`, the editor and the CI of the
   catalogue): the structure and the identifier; a title, a description and a license; every parameter with a
   unit (empty for dimensionless ones), a description and a default inside its bounds; a body valid for the
   template, whose `"$name"` refer to parameters of the right dimension; a passive behaviour unless declared
   otherwise; the round trip through a `.lumos` document; **at least one check with an independent reference
   value**, run each time: `{"kind": "power", "wavelength", "power", "expect", "rtol"}` sends a beam (or one
   guided lane per input of a matrix) of `power` and compares the total output power with `expect`.
4. **Registration**: `register_component(spec)` builds the element class from the template (a pure transfer
   function closed over the body) and registers it; registering the same definition again does nothing,
   another definition under the same identifier and version is refused. A context document using components
   embeds their definitions under `"components"` (format 1.3), so that it opens anywhere.
5. **Code stays local**: components written in Python are plugins or script blocks (ADR-0013), installed on
   local and private instances only (P15.15 offers their code view there, never on the public instance).

## Alternatives considered

- **A safe expression language in the documents** (a formula for the transfer): an interpreter to maintain and
  to secure, for a need the templates cover.
- **Components as Python only**: impossible on the public instance, where the form editor is most useful.
- **Composite components as element types**: a group already saves and reloads itself; a type would add a
  dependency of the document on the definition without benefit.

## Consequences

- A user builds a component in a form, checks it, shares its file, and anyone opens it on any instance.
- The behaviours are those of the templates; new templates are changes of LUMOS, with their tests.
- The catalogue (P15.16) runs `lumos component check` on every contribution.
