# Create and share a component

There are two ways to share an element you made: a **declarative component** in the community catalogue, usable
on every instance, or a **Python package** for elements written in code, installed where code may run.

## The community catalogue

The catalogue is the `community/` folder of the LUMOS repository, shipped with every version of LUMOS and listed
in the **Catalogue** tab of the palette: a component is imported with **Import** (it then is an element of the
palette, in *Components*), a shared preset opens with **Open**. Assistants use the MCP tools `list_catalogue`
and `use_catalogue`; in Python, `lumos.catalogue.read_index(lumos.catalogue.catalogue_dir())` lists it.

```
community/components/<author>/<Name>.lumos   declarative components, identifier <author>.<Name>
community/presets/<author>/<name>.lumos      shared contexts (meta: name, description, authors, license)
community/index.json                         the index, built from the files
```

To contribute a component:

1. make it in the component editor (*My blocks* › *New component*, see [Your own components](components.md))
   and download it with **Download .lumos**, or write the document by hand;
2. give it a license (an SPDX identifier such as `MIT`), its authors, a description of every parameter, and at
   least one check whose expected value you know independently of LUMOS (an energy balance, an analytic case,
   a datasheet), saying where it comes from in the description;
3. put it at `community/components/<author>/<Name>.lumos`, then check it and rebuild the index:

   ```
   lumos component check community/components/<author>/<Name>.lumos
   python tools/build_catalogue.py
   python tools/build_catalogue.py --check
   ```

4. open a pull request: the workflow *Community catalogue* runs the same checks on every change of
   `community/`.

A shared preset is a context saved from the interface or with `lumos.io.save`, whose `meta` gives a `name`, a
`description`, the `authors` and a `license`; the components it uses are embedded in it. It goes to
`community/presets/<author>/<name>.lumos`.

Each entry of the index gives the `lumos_format` of its file, the oldest format that reads it: a contribution
reaches an instance with the next version of LUMOS that ships it.

## Elements with code

An element whose behaviour no template covers is a Python class (see the code view of the component editor on
a local instance, or {doc}`../api/index`). Share it as a package named `lumos-component-<name>` that declares
the entry point `lumos.elements`, a module whose import registers its element types (`@register`):

```toml
[project.entry-points."lumos.elements"]
mine = "lumos_component_mine"
```

Installed with `pip install lumos-component-<name>` where LUMOS runs, its elements are types of the palette:
the service loads the packages of this entry point when it starts (in Python,
`lumos.core.registry.registry.load_plugins()`). Visitors of the public instance cannot install or run code:
there, only the packages installed by its administrator are available.
