"""Evaluation of the MCP server on typical scenarios (step P13.8, REQ-MCP-7, ADR-0017).

Each :class:`Scenario` is a task given to an assistant in plain language (use cases UC1, UC8 and UC9 of the
requirements), with expected values derived from analytic solutions and a check of the final answer. The
assistant ends its answer with a JSON object holding the requested values; :func:`parse_answer` reads it.

- :func:`run_reference` solves a scenario with a scripted sequence of MCP tool calls. It runs in the tests: it
  proves that the scenario can be solved with the tools, and that the checks accept a right answer.
- :func:`run_with_claude` gives the task to Claude through the Claude API, with the tools of the server, and
  checks its answer. ``tools/mcp_eval.py`` runs it periodically (extra ``eval``, an API key).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from scipy import integrate, special

__all__ = [
    "DEFAULT_MODEL",
    "SCENARIOS",
    "EvaluationResult",
    "Scenario",
    "parse_answer",
    "run_reference",
    "run_with_claude",
]

#: Model used by :func:`run_with_claude` unless another one is given.
DEFAULT_MODEL = "claude-opus-5-5"

_ANSWER = (
    "\n\nEnd your answer with a JSON object holding exactly these keys, with plain numbers as values "
    "(no units, in the units given above): {keys}."
)


@dataclass(frozen=True)
class Scenario:
    """A task for an assistant with a verifiable expected result.

    Attributes
    ----------
    name
        Identifier of the scenario.
    use_case
        Use case of the requirements (``"UC1"``...).
    title
        Short title.
    task
        What the assistant is asked, ending with the keys of the JSON answer.
    expected
        Expected value of each key of the answer.
    rules
        How each value is checked: ``("rel", r)`` relative tolerance, ``("abs", a)`` absolute tolerance,
        ``("below", m)`` upper bound (the expected value is then the ideal one).
    reference
        Scripted solution: an async function of an MCP client returning the answer.
    """

    name: str
    use_case: str
    title: str
    task: str
    expected: Mapping[str, float]
    rules: Mapping[str, tuple[str, float]]
    reference: Callable[[Any], Awaitable[dict[str, float]]] = field(repr=False)

    @property
    def answer_keys(self) -> tuple[str, ...]:
        """Keys of the JSON answer."""
        return tuple(self.expected)

    def check(self, answer: Mapping[str, Any]) -> list[str]:
        """Return the reasons why ``answer`` is wrong (an empty list when it is right)."""
        failures = []
        for key, expected in self.expected.items():
            if key not in answer:
                failures.append(f"{key}: missing")
                continue
            try:
                value = float(answer[key])
            except (TypeError, ValueError):
                failures.append(f"{key}: {answer[key]!r} is not a number")
                continue
            kind, tolerance = self.rules[key]
            if kind == "below":
                ok = value <= tolerance
                expectation = f"at most {tolerance:g}"
            elif kind == "abs":
                ok = abs(value - expected) <= tolerance
                expectation = f"{expected:.6g} ± {tolerance:g}"
            else:
                ok = abs(value - expected) <= tolerance * abs(expected)
                expectation = f"{expected:.6g} within {tolerance:.0%}"
            if not (ok and math.isfinite(value)):
                failures.append(f"{key}: {value:.6g}, expected {expectation}")
        return failures


@dataclass
class EvaluationResult:
    """Outcome of a scenario: the answer, what is wrong with it, and the final text of the assistant."""

    scenario: str
    answer: dict[str, Any]
    failures: list[str]
    text: str = ""
    tool_calls: int = 0

    @property
    def passed(self) -> bool:
        """Whether the answer is right."""
        return not self.failures


def parse_answer(text: str) -> dict[str, Any]:
    """Return the last JSON object written in ``text`` (the final answer of an assistant), or ``{}``."""
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


# --- Helpers of the reference solutions ---------------------------------------------------------------------


async def _call(client: Any, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    result = await client.call_tool(tool, arguments or {})
    if result.is_error:
        text = " ".join(getattr(block, "text", "") for block in result.content)
        raise RuntimeError(f"{tool} failed: {text}")
    content = result.structured_content
    if isinstance(content, dict) and set(content) == {"result"}:
        return content["result"]
    return content


# --- UC1: transmission of a Bracewell nuller ----------------------------------------------------------------

_BASELINE, _NULLER_WAVELENGTH = 20.0, 10e-6
_ANGLES = {
    "t_0": 0.0,
    "t_lambda_8b": _NULLER_WAVELENGTH / (8 * _BASELINE),
    "t_lambda_4b": _NULLER_WAVELENGTH / (4 * _BASELINE),
    "t_lambda_2b": _NULLER_WAVELENGTH / (2 * _BASELINE),
}


async def _nuller_reference(client: Any) -> dict[str, float]:
    await _call(client, "load_preset", {"name": "bracewell"})
    await _call(client, "set_param", {"path": "meter.channel", "value": -1})  # power of every output
    answer = {}
    for key, theta in _ANGLES.items():
        await _call(client, "set_param", {"path": "source.theta_x", "value": f"{theta!r} rad"})
        dark, bright = (await _call(client, "inspect_port", {"port": "meter.out", "figure": False}))["values"]
        answer[key] = dark / (dark + bright)
    return answer


UC1 = Scenario(
    name="uc1_nuller_transmission",
    use_case="UC1",
    title="Transmission map of a Bracewell nulling interferometer",
    task=(
        "Load the LUMOS preset 'bracewell': two 2 m telescopes 20 m apart whose beams are recombined by a "
        "nulling "
        "chip at 10 um, observing a point source. Compute the transmission of the dark (nulled) output of the"
        " "
        "chip, that is its power divided by the total power of the outputs, for a source on the baseline axis"
        " at "
        "the angles 0, lambda/(8B), lambda/(4B) and lambda/(2B), where B is the baseline and lambda the "
        "wavelength. Use the keys t_0, t_lambda_8b, t_lambda_4b and t_lambda_2b."
        + _ANSWER.format(keys="t_0, t_lambda_8b, t_lambda_4b, t_lambda_2b")
    ),
    expected={
        key: math.sin(math.pi * _BASELINE * theta / _NULLER_WAVELENGTH) ** 2 for key, theta in _ANGLES.items()
    },
    rules=dict.fromkeys(_ANGLES, ("abs", 0.01)),
    reference=_nuller_reference,
)


# --- UC8: comparison of propagation methods -----------------------------------------------------------------

_W0, _BEAM_WAVELENGTH, _DISTANCE = 0.5e-3, 633e-9, 2.0
_RAYLEIGH = math.pi * _W0**2 / _BEAM_WAVELENGTH
_RADIUS_MM = 1e3 * _W0 * math.sqrt(1 + (_DISTANCE / _RAYLEIGH) ** 2)


async def _propagation_reference(client: Any) -> dict[str, float]:
    await _call(client, "load_preset", {"name": "propagation_methods"})
    answer = {}
    for method, key in (("asm", "radius_asm_mm"), ("fresnel_ir", "radius_fresnel_mm")):
        await _call(client, "set_param", {"path": "far.method", "value": method})
        summary = await _call(client, "inspect_port", {"port": "far.out", "figure": False})
        wx, wy = (float(v) for v in summary["details"]["beam radius"].replace("mm", "").split(","))
        answer[key] = (wx + wy) / 2
    meter = await _call(client, "inspect_port", {"port": "meter.out", "figure": False})
    answer["power_mw"] = meter["values"][0] * 1e3
    return answer


UC8 = Scenario(
    name="uc8_propagation_methods",
    use_case="UC8",
    title="Comparison of propagation methods on a Gaussian beam",
    task=(
        "Load the LUMOS preset 'propagation_methods': a 1 mW Gaussian laser at 633 nm with a waist of 0.5 mm,"
        " "
        "propagated over 2 m to a power meter. Check how LUMOS samples the propagation, then compute the beam"
        " at "
        "2 m with the angular spectrum method ('asm') and with the Fresnel impulse-response method "
        "('fresnel_ir'). Report the 1/e2 beam radius at 2 m in mm for each method (keys radius_asm_mm and "
        "radius_fresnel_mm) and the power received by the meter in mW (key power_mw)."
        + _ANSWER.format(keys="radius_asm_mm, radius_fresnel_mm, power_mw")
    ),
    expected={"radius_asm_mm": _RADIUS_MM, "radius_fresnel_mm": _RADIUS_MM, "power_mw": 1.0},
    rules={"radius_asm_mm": ("rel", 0.02), "radius_fresnel_mm": ("rel", 0.02), "power_mw": ("rel", 0.01)},
    reference=_propagation_reference,
)


# --- UC9: contrast of a coronagraph -------------------------------------------------------------------------


def _airy_ring_mean(inner: float, outer: float) -> float:
    """Area-weighted mean of the Airy pattern over the ring ``inner <= x < outer`` (x in lambda/D)."""

    def airy(x: float) -> float:
        return float((2 * special.j1(math.pi * x) / (math.pi * x)) ** 2)

    weighted, _ = integrate.quad(lambda x: airy(x) * x, inner, outer)
    return weighted / ((outer**2 - inner**2) / 2)


async def _ring_3_to_4(client: Any) -> float:
    # f lambda / D = 100 m x 1.6 um / 8 m = 20 um: rings of 20 um, the fourth one spans 3 to 4 lambda/D
    profile = await _call(client, "radial_profile", {"port": "image.out", "bin_width": "20 um"})
    index = min(range(len(profile["radii"])), key=lambda i: abs(profile["radii"][i] - 70e-6))
    return profile["values"][index]


async def _coronagraph_reference(client: Any) -> dict[str, float]:
    await _call(client, "load_preset", {"name": "coronagraph"})
    await _call(client, "set_param", {"path": "planet.flux_density", "value": "0 Jy"})
    with_coronagraph = await _ring_3_to_4(client)
    for source, target in (("telescope.out", "coronagraph.in"), ("coronagraph.out", "lens.in")):
        await _call(client, "disconnect", {"source": source, "target": target})
    await _call(client, "connect", {"source": "telescope.out", "target": "lens.in"})
    peak = (await _call(client, "inspect_port", {"port": "image.out", "figure": False}))["statistics"]["max"]
    without = await _ring_3_to_4(client)
    return {"contrast_with": with_coronagraph / peak, "contrast_without": without / peak}


UC9 = Scenario(
    name="uc9_coronagraph_contrast",
    use_case="UC9",
    title="Raw contrast of a vortex coronagraph at 3 lambda/D",
    task=(
        "Load the LUMOS preset 'coronagraph' (an 8 m telescope, a charge-2 vortex coronagraph, a star and a "
        "companion). What raw contrast does this coronagraph reach between 3 and 4 lambda/D? Switch the "
        "companion "
        "off, and define the contrast as the mean intensity of the star in the ring between 3 and 4 lambda/D "
        "of "
        "the image, divided by the peak intensity of the image of the star without the coronagraph. Give it "
        "with "
        "the coronagraph (key contrast_with) and without it (key contrast_without)."
        + _ANSWER.format(keys="contrast_with, contrast_without")
    ),
    expected={"contrast_with": 0.0, "contrast_without": _airy_ring_mean(3.0, 4.0)},
    rules={"contrast_with": ("below", 1e-5), "contrast_without": ("rel", 0.10)},
    reference=_coronagraph_reference,
)

#: Scenarios of the evaluation, by name.
SCENARIOS: dict[str, Scenario] = {scenario.name: scenario for scenario in (UC1, UC8, UC9)}


# --- Runners ------------------------------------------------------------------------------------------------


async def run_reference(scenario: Scenario, client: Any) -> EvaluationResult:
    """Solve ``scenario`` with its scripted tool calls through ``client`` and check the answer."""
    answer = await scenario.reference(client)
    return EvaluationResult(scenario.name, answer, scenario.check(answer), text=json.dumps(answer))


def _tool_result_content(result: Any) -> list[dict[str, Any]]:
    """Content blocks of an MCP tool result, in the format of the Claude API."""
    blocks: list[dict[str, Any]] = []
    for block in result.content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "image":
            source = {"type": "base64", "media_type": block.mime_type, "data": block.data}
            blocks.append({"type": "image", "source": source})
    return blocks or [{"type": "text", "text": "(no content)"}]


async def run_with_claude(
    scenario: Scenario, client: Any, *, model: str = DEFAULT_MODEL, max_turns: int = 60
) -> EvaluationResult:
    """Give ``scenario`` to Claude with the tools of the MCP server behind ``client`` and check its answer.

    Needs the package ``anthropic`` (extra ``eval``) and credentials of the Claude API (``ANTHROPIC_API_KEY``
    or a profile of ``ant auth login``).
    """
    import anthropic  # pyright: ignore[reportMissingImports]

    api = anthropic.AsyncAnthropic()
    listed = await client.list_tools()
    tools: list[Any] = [
        {"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema}
        for tool in listed.tools
    ]
    messages: list[Any] = [{"role": "user", "content": scenario.task}]
    calls = 0
    response = None
    for _ in range(max_turns):
        response = await api.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=client.instructions or "",
            tools=tools,
            messages=messages,
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            calls += 1
            result = await client.call_tool(block.name, block.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _tool_result_content(result),
                    "is_error": bool(result.is_error),
                }
            )
        messages.append({"role": "user", "content": results})
    text = "" if response is None else "".join(b.text for b in response.content if b.type == "text")
    answer = parse_answer(text)
    failures = scenario.check(answer)
    if response is not None and response.stop_reason == "refusal":
        failures.insert(0, "the model declined the task")
    elif response is not None and response.stop_reason == "tool_use":
        failures.insert(0, f"no final answer after {max_turns} turns")
    return EvaluationResult(scenario.name, answer, failures, text=text, tool_calls=calls)
