"""Table Access Protocol queries (IVOA TAP 1.1, synchronous) returning astropy tables.

Many catalogue services (SIMBAD, Gaia, the NASA Exoplanet Archive, IRSA, ESO, MAST...) answer ADQL queries
through TAP. :func:`tap_query` posts the query to ``<service>/sync`` through :func:`lumos.catalogs.http.fetch`
(verified TLS, timeout, cache) and parses the VOTable answer.
"""

from __future__ import annotations

import io
import warnings
from typing import Any

from lumos.catalogs.http import CatalogError, fetch

__all__ = ["adql_string", "masked", "tap_query"]


def adql_string(value: str) -> str:
    """Return ``value`` as an ADQL string literal (quotes doubled, no injection)."""
    return "'" + str(value).replace("'", "''") + "'"


def tap_query(url: str, adql: str, *, timeout: float | None = None) -> Any:
    """Run an ADQL query on the synchronous endpoint ``url`` and return an :class:`astropy.table.Table`.

    Raises
    ------
    lumos.catalogs.http.CatalogError
        If the service reports an error.
    """
    from astropy.io.votable import parse

    data = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": adql}
    response = fetch(url, method="POST", data=data, timeout=timeout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # VOTables of services often bend the standard (units, datatypes)
        try:
            document = parse(io.BytesIO(response.content), verify="ignore")
        except Exception as error:
            raise CatalogError(f"{url}: unreadable VOTable answer: {error}") from error
    for resource in document.resources:
        for info in resource.infos:
            if info.name == "QUERY_STATUS" and info.value == "ERROR":
                raise CatalogError(f"{url}: query error: {(info.content or '').strip()}")
    try:
        table = document.get_first_table()
    except (IndexError, ValueError) as error:
        raise CatalogError(f"{url}: the answer holds no table") from error
    return table.to_table(use_names_over_ids=True)


def masked(value: Any) -> bool:
    """Whether a table cell is masked (missing) or empty."""
    import numpy as np

    if value is None or value is np.ma.masked:
        return True
    if isinstance(value, str | bytes):
        return len(value.strip()) == 0
    try:
        return bool(np.ma.is_masked(value)) or bool(np.isnan(value))
    except TypeError:
        return False
