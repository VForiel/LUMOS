"""Exceptions shared by several LUMOS modules."""

__all__ = ["FormatVersionError", "LumosFormatError"]


class LumosFormatError(ValueError):
    """A ``.lumos`` document is malformed or describes invalid values."""


class FormatVersionError(LumosFormatError):
    """A ``.lumos`` document uses a format or type version this version of LUMOS cannot read."""
