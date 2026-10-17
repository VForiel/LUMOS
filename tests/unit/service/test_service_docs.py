"""Search in the documentation (step P13.1, REQ-MCP-2 ``search_docs``).

A page whose title matches the query but whose text does not (e.g. an element type found by its identifier,
``lumos.coronagraphs.…``, without the word in its docstring) is a hit with an excerpt from its beginning.
"""

from lumos.service import Service


def test_a_word_found_only_in_the_title_of_a_page_is_a_hit():
    hits = Service().search_docs("coronagraphs", limit=50)
    assert any(hit.path.startswith("elements/lumos.coronagraphs.") for hit in hits)
    assert all(hit.excerpt is not None for hit in hits)
