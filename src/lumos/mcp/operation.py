"""Operation tools of the MCP server: devices, actions confirmed by a human, emergency stop.

Step P13.4 (REQ-MCP-4, ADR-0016). The safeguards hold whatever the language model asks:

- access to equipment is disabled unless the server is started with ``--enable-operation``
  (``ServiceConfig.operation_enabled``); then ``discover_devices`` and ``associate_devices`` are offered;
- every action on real equipment is confirmed by the user through an MCP elicitation, asked by the server
  itself: the model cannot answer it, and a client that cannot ask the user cannot act on equipment;
- limits, dry runs and the emergency stop are checked by the drivers in the service (ADR-0014);
- the actions are journaled with the origin ``mcp``.

Actions on simulated elements need no confirmation.
"""

# No ``from __future__ import annotations``: the SDK evaluates the annotation ``Resolve(confirmation)`` of
# execute_action, and ``confirmation`` is local to register_operation_tools.

from typing import TYPE_CHECKING, Annotated, Any

from mcp.server.mcpserver import Context as MCPContext
from mcp.server.mcpserver import Elicit, ElicitationResult, Resolve
from pydantic import BaseModel, Field

from lumos.service import ServiceError
from lumos.service.models import ActionResult, DeviceModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.mcpserver import MCPServer

    from lumos.mcp.server import LumosTools

__all__ = ["Confirmation", "register_operation_tools"]

#: Outcomes of the confirmation resolver that are not answers of the user.
_NOT_NEEDED = "not needed"
_CANNOT_ASK = "cannot ask"


class Confirmation(BaseModel):
    """Answer of the user to a request to act on real equipment."""

    confirm: bool = Field(description="execute this action on the real equipment")


def _call_text(action: str, args: list[Any] | None, kwargs: dict[str, Any] | None) -> str:
    values = [str(a) for a in args or []] + [f"{k}={v}" for k, v in (kwargs or {}).items()]
    return f"{action}({', '.join(values)})"


def register_operation_tools(server: "MCPServer", state: "LumosTools", tool: "Callable[..., Any]") -> None:
    """Add the operation tools to ``server``; ``tool`` registers a function with the error translation."""
    service = state.service
    session = state.session

    def confirmation(
        ctx: MCPContext,
        element: str,
        action: str,
        args: list[Any] | None,
        kwargs: dict[str, Any] | None,
        context: str | None,
    ) -> Any:
        """Ask the user to confirm an action on real equipment, if the action needs it."""
        try:
            identifier = state.context(context)
            device = service.confirmation_needed(session, identifier, element)
        except ServiceError:
            return _NOT_NEEDED  # the tool reports the error
        if device is None:
            return _NOT_NEEDED
        capabilities = ctx.client_capabilities
        if capabilities is None or capabilities.elicitation is None:
            return _CANNOT_ASK
        return Elicit(
            f"An AI assistant asks LUMOS to execute {element}.{_call_text(action, args, kwargs)} on the real "
            f"device {device}. Do you confirm?",
            Confirmation,
        )

    @tool
    def execute_action(
        element: str,
        action: str,
        confirmed: Annotated[ElicitationResult[Confirmation], Resolve(confirmation)],
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        context: str | None = None,
    ) -> ActionResult:
        """Execute an action of an element (e.g. ``move`` of a stage with ``args=["5 mm"]``, ``home``).

        On simulated elements, the action changes the simulation. On real equipment (operation enabled), the
        user is asked to confirm every action; the limits of the devices are enforced. Quantities are written
        with their unit. The result gives the predicted and measured durations.
        """
        data = getattr(confirmed, "data", None)
        if data == _CANNOT_ASK:
            raise ServiceError(
                f"{element}.{action} acts on real equipment and this MCP client cannot ask the user"
                " to confirm",
                hint="use a client that supports elicitation (e.g. Claude Code), or the Python API",
                code="forbidden",
            )
        ok = data == _NOT_NEEDED or (isinstance(data, Confirmation) and data.confirm)
        if not ok:
            raise ServiceError(
                f"the user did not confirm {element}.{action}",
                hint="do not retry unless the user asks for it",
                code="forbidden",
            )
        return service.execute_action(
            session,
            state.context(context),
            element,
            action,
            args=args or [],
            kwargs=kwargs,
            confirmed=isinstance(data, Confirmation) and data.confirm,
        )

    @tool
    def list_devices(context: str | None = None) -> list[dict[str, Any]]:
        """List the devices (drivers) attached to the elements of a context, with their state."""
        return service.list_devices(session, state.context(context))

    @tool
    def device_status(element: str, context: str | None = None) -> dict[str, Any]:
        """State of the device of an element: model, serial, connection, position, limits."""
        return service.device_status(session, state.context(context), element)

    @tool
    def stop_all(context: str | None = None) -> str:
        """Emergency stop of every device of a context. Always allowed, never needs a confirmation."""
        identifier = state.context(context)
        service.stop_all(session, identifier)
        return f"every device of context {identifier} stopped"

    if not service.config.operation_enabled:
        return

    @tool
    def discover_devices(
        tcp: list[str] | None = None, serial: bool = False, visa: bool = False
    ) -> list[DeviceModel]:
        """Find the devices answering on TCP addresses (``"tcp://host:port"``), serial ports or VISA."""
        return service.discover_devices(tcp=tcp or [], serial=serial, visa=visa)

    @tool
    def associate_devices(
        assignments: dict[str, str] | None = None, context: str | None = None
    ) -> dict[str, DeviceModel]:
        """Attach the discovered devices to the elements of a context (by model, or ``{element: serial}``)."""
        return service.associate_devices(session, state.context(context), assignments=assignments)
