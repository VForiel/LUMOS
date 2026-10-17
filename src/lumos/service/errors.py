"""Errors of the service layer: a cause and a suggested fix, readable by people and by AI assistants."""

from __future__ import annotations

__all__ = ["ServiceError"]


class ServiceError(Exception):
    """An operation of the service failed.

    Attributes
    ----------
    message
        What went wrong.
    hint
        How to fix it.
    code
        Category: ``"invalid"``, ``"not_found"``, ``"forbidden"``, ``"quota"``, ``"failed"``.
    """

    def __init__(self, message: str, *, hint: str = "", code: str = "invalid") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code

    def __str__(self) -> str:
        """Message and hint."""
        return f"{self.message} (hint: {self.hint})" if self.hint else self.message

    def to_json(self) -> dict[str, str]:
        """JSON form of the error."""
        return {"error": self.message, "hint": self.hint, "code": self.code}
