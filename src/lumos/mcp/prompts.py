"""Prompts of the MCP server: guided workflows for the main use cases (step P13.3, REQ-MCP-3).

A prompt is a message template the user picks in the client (often as a slash command); it tells the
assistant which tools to use, in which order, and what to check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

__all__ = ["register_prompts"]


def register_prompts(server: MCPServer) -> None:
    """Add the prompts of LUMOS to ``server``."""

    @server.prompt()
    def design_instrument(goal: str) -> str:
        """Design an optical instrument or bench that reaches a goal, then simulate it."""
        return f"""\
Design with LUMOS an optical instrument or bench for this goal: {goal}

1. Look for a close preset with list_presets; load it with load_preset, or start with create_context.
2. Find the element types you need with list_element_types and read their parameters and ports with
   get_element_schema. Write every quantity with its unit ("8 m", "1.6 um", "10 mas").
3. Add the elements (add_element), connect them output to input (connect "element.port"), group repeated
   parts with create_group, and set the observation band if a source needs it (set_observation).
4. Check the context with validate_context and fix every issue it reports.
5. Compute the outputs with run, then summarise them with get_result or inspect_port.
6. Explain the design choices and the results with their units, say what limits the performance, and offer
   the Python script (export_python) or the .lumos file (export_lumos).
"""

    @server.prompt()
    def estimate_detection(target: str, instrument: str = "coronagraph") -> str:
        """Estimate whether an instrument can detect a target (contrast, separation, signal to noise)."""
        return f"""\
Estimate with LUMOS whether the instrument "{instrument}" can detect: {target}

1. Get the physical properties of the target and its host with resolve_object (distance, magnitudes,
   separation, temperature); note the provenance of every value.
2. Load the instrument (load_preset "{instrument}" if it exists, or build it) and set the source to the
   target: separation, contrast and spectrum, with their units.
3. Check it with validate_context, then run it to get the detected image.
4. Compute the detection limit with evaluate_detection on the residual image (peak of the unattenuated star,
   resolution element), and compare the contrast of the target with the curve at its separation.
5. Conclude: detectable or not, at which significance, and what would change the answer (exposure,
   wavelength, coronagraph, wavefront errors).
"""

    @server.prompt()
    def calibrate_bench(bench: str) -> str:
        """Calibrate an optical bench, simulated or real, step by step and safely."""
        return f"""\
Calibrate this optical bench with LUMOS: {bench}

1. Describe the bench in a context (describe_context, or build it); a digital twin runs beside real
   equipment when the operation is enabled.
2. Check every device with list_devices and device_status: connection, position, limits.
3. Plan the calibration steps (homing, scans, reference measurements) and say them to the user before acting.
4. Execute each action with execute_action: on real equipment the user confirms every action, and limits
   and the emergency stop (stop_all) are enforced by the server whatever the request.
5. Read the measurements (inspect_port, get_journal), compare them with the simulation and report the fitted
   calibration values with their units and uncertainties.
"""

    @server.prompt()
    def diagnose_simulation(problem: str, context: str = "") -> str:
        """Find why a simulation gives a wrong, slow or suspicious result."""
        target = f'the context "{context}"' if context else "the current context"
        return f"""\
Diagnose with LUMOS this problem in {target}: {problem}

1. Read the context with describe_context: elements, values with units, connections, observation band.
2. Run validate_context and look at every issue and hint.
3. For each propagation, call explain_sampling: method chosen, reason, grids. Look for under-sampling
   (too few pixels per lambda/D, or a grid smaller than the beam) and aliasing.
4. Inspect intermediate outputs with inspect_port to find the first element where the signal goes wrong;
   check the units and orders of magnitude (power, wavelength, pixel scale).
5. Read the recent journal (get_journal) for warnings.
6. Explain the cause, propose the fix (parameter values with units) and, with the user's agreement, apply it
   with set_param and check the result again.
"""
