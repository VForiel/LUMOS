# LUMOS community catalogue

Contributions of the community, listed in the **Catalogue** tab of the palette of the web interface and shipped
with LUMOS (ADR-0026). Everything here is data: no code runs from this folder.

```
components/<author>/<Name>.lumos   declarative components (documents of kind "component", id <author>.<Name>)
presets/<author>/<name>.lumos      shared contexts (meta: name, description, authors, license)
index.json                         the index, built from the files
```

## Contributing

1. Make a component in the editor of the web interface (*My blocks* › *New component*) and download it
   (*Download .lumos*), or write the document by hand (see the guide "Create and share a component").
2. Put it at `components/<author>/<Name>.lumos`; a preset goes to `presets/<author>/<name>.lumos`, with a name,
   a description, the authors and a license in its `meta` (the components it uses are embedded when it is saved).
3. Check it and rebuild the index:

   ```
   lumos component check community/components/<author>/<Name>.lumos
   python tools/build_catalogue.py
   python tools/build_catalogue.py --check
   ```

4. Open a pull request. The CI runs the same checks on every change of this folder.

A component needs a title, a description, a license (an SPDX identifier), its authors, described parameters
with bounds, and at least one check whose expected value is known independently of LUMOS (an energy balance, an
analytic case, a datasheet): say where it comes from in the description.

Components with code are not accepted here: publish them as a Python package named `lumos-component-…`,
installed with pip where code may run (never on the public instance).
