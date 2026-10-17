"""Description of the presets for the gallery of the web interface (step P14.7, REQ-WEB-7).

Requirements: a preset is listed with its title (the name in its ``.lumos`` file), its description, its
outputs and the types of its elements, so that the gallery can show what it contains without loading it.
"""

from lumos.service import Service


def test_presets_give_their_title_and_the_types_of_their_elements():
    presets = {preset.name: preset for preset in Service().list_presets()}
    coronagraph = presets["coronagraph"]
    assert coronagraph.title == "Vortex coronagraph"
    assert "lumos.coronagraphs.VortexCoronagraph" in coronagraph.elements
    assert coronagraph.description
    assert all(preset.title for preset in presets.values())
