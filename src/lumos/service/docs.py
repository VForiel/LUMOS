"""Search in the documentation: the pages of ``docs/`` when available, and the docstrings of the elements."""

from __future__ import annotations

import re
from pathlib import Path

from lumos.service.models import DocHit

__all__ = ["docs_root", "pages", "search"]

_EXCLUDED = ("specs", "_build", "locales", "jupyter_execute")


def docs_root() -> Path | None:
    """Return the ``docs`` directory of a source checkout of LUMOS, if any."""
    candidate = Path(__file__).resolve().parents[3] / "docs"
    return candidate if (candidate / "index.md").is_file() else None


def pages() -> dict[str, tuple[str, str]]:
    """Documentation pages by relative path: ``(title, text)``, plus one page per element type."""
    found: dict[str, tuple[str, str]] = {}
    root = docs_root()
    if root is not None:
        for path in sorted(root.rglob("*.md")):
            relative = path.relative_to(root).as_posix()
            if any(part in _EXCLUDED for part in path.relative_to(root).parts):
                continue
            text = path.read_text(encoding="utf-8")
            title = next(
                (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), relative
            )
            found[relative] = (title, text)
    import lumos.elements  # noqa: F401 - registers the element types
    from lumos.core.registry import registry

    for type_id in sorted(registry):
        cls = registry.get(type_id)
        found[f"elements/{type_id}"] = (type_id, cls.__doc__ or "")
    return found


def search(query: str, *, limit: int = 5) -> list[DocHit]:
    """Return the pages that best match the words of ``query``, with an excerpt."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
    if not words:
        return []
    hits = []
    for path, (title, text) in pages().items():
        lower = text.lower()
        score = sum(lower.count(w) for w in words) + 5 * sum(w in title.lower() for w in words)
        phrase = query.lower().strip()
        if phrase in lower:
            score += 10
        if score == 0:
            continue
        position = (
            lower.find(phrase)
            if phrase in lower
            else min((lower.find(w) for w in words if w in lower), default=0)
        )
        start = max(position - 120, 0)
        excerpt = " ".join(text[start : start + 360].split())
        hits.append(DocHit(title=title, path=path, excerpt=excerpt, score=float(score)))
    hits.sort(key=lambda hit: -hit.score)
    return hits[:limit]
