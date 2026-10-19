"""Export the untranslated messages of the French catalogues to JSON, and import their translations.

    python tools/translate_po.py export docs/locales/fr/LC_MESSAGES/web/*.po > todo.json
    python tools/translate_po.py import done.json
    python tools/translate_po.py code docs/locales/fr/LC_MESSAGES/**/*.po   # copy the messages made of code

The JSON maps the path of each catalogue (relative to ``docs/locales/fr/LC_MESSAGES``) to ``{msgid: msgstr}``;
empty translations are skipped on import. The API reference (docstrings) and the ADRs stay in English.
"""

import json
import pathlib
import sys

from babel.messages.pofile import read_po, write_po  # pyright: ignore[reportMissingImports] (docs extra)

ROOT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "locales" / "fr" / "LC_MESSAGES"


def export(paths: list[str]) -> None:
    found: dict[str, dict[str, str]] = {}
    for path in paths:
        po = pathlib.Path(path).resolve()
        with po.open(encoding="utf-8") as stream:
            catalog = read_po(stream)
        missing = {
            str(m.id): "" for m in catalog if m.id and (not m.string or m.fuzzy) and not code_only(str(m.id))
        }
        if missing:
            found[po.relative_to(ROOT).as_posix()] = missing
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    json.dump(found, sys.stdout, ensure_ascii=False, indent=1)


def code_only(text: str) -> bool:
    """Whether a message is only inline code, names and punctuation (nothing to translate)."""
    import re

    rest = re.sub(r"`[^`]*`", "", text)
    rest = re.sub(r"https?://\S+", "", rest)
    return not re.search(r"[A-Za-z]{2,}", rest)


def copy_code(paths: list[str]) -> None:
    for path in paths:
        po = pathlib.Path(path).resolve()
        with po.open(encoding="utf-8") as stream:
            catalog = read_po(stream)
        done = 0
        for message in catalog:
            if message.id and not message.string and code_only(str(message.id)):
                message.string = message.id
                done += 1
        with po.open("wb") as stream:
            write_po(stream, catalog, width=0)
        sys.stderr.write(f"{po.relative_to(ROOT).as_posix()}: {done} copied\n")


def import_(path: str) -> None:
    translations = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for name, messages in translations.items():
        po = ROOT / name
        with po.open(encoding="utf-8") as stream:
            catalog = read_po(stream)
        done = 0
        for message in catalog:
            text = messages.get(message.id) if message.id else None
            if text:
                message.string = text
                message.flags.discard("fuzzy")
                done += 1
        with po.open("wb") as stream:
            write_po(stream, catalog, width=0)
        sys.stderr.write(f"{name}: {done} translated\n")


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "export":
        export(arguments)
    elif command == "code":
        copy_code(arguments)
    elif command == "import":
        import_(arguments[0])
    else:
        raise SystemExit(__doc__)
