"""Tests for the evaluation suite of the MCP server (step P13.8, REQ-MCP-7).

Requirements: typical scenarios (UC1, UC8, UC9) given to an assistant through MCP, each with an expected
result that a program can verify; the suite runs periodically. In the regular tests, each scenario is solved
by a scripted reference solution that only uses MCP tools, and its answer goes through the same verification
as the answer of an assistant: this proves that the scenarios can be solved with the tools and that the checks
accept right answers and reject wrong ones.

The expected values are analytic and recomputed here independently:

- UC1, Bracewell nuller (Bracewell 1978, Nature 274, 780): the dark output transmits
  T(θ) = sin²(π B θ / λ) of the light of a point source at angle θ along the baseline B;
- UC8, Gaussian beam (Siegman 1986, *Lasers*, §17.1): w(z) = w0 √(1 + (z / zR)²), zR = π w0² / λ, and the
  power is conserved by free-space propagation;
- UC9, coronagraph: without coronagraph the image of a star through a circular unobstructed pupil is the Airy
  pattern I(x) = [2 J1(π x) / (π x)]² (x in λ/D, Born & Wolf §8.5.2), whose mean over the ring 3 ≤ x < 4,
  weighted by area, is computed with scipy; an ideal vortex of charge 2 cancels the on-axis star for such a
  pupil (Mawet et al. 2005, ApJ 633, 1191), so what remains is far below the Airy level.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import anyio
import numpy as np
import pytest
from scipy import integrate, special

pytest.importorskip("mcp")

from mcp import Client

from lumos.mcp import create_server
from lumos.mcp.evaluation import (
    SCENARIOS,
    claude_code_command,
    parse_answer,
    run_reference,
    run_with_claude_code,
)
from lumos.service import Service, ServiceConfig

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def server(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    yield create_server(service)
    service.shutdown()


def airy_ring_mean(inner, outer):
    def airy(x):
        return (2 * special.j1(math.pi * x) / (math.pi * x)) ** 2

    weighted, _ = integrate.quad(lambda x: airy(x) * x, inner, outer)
    return weighted / ((outer**2 - inner**2) / 2)


def test_the_three_use_cases_are_covered():
    assert {scenario.use_case for scenario in SCENARIOS.values()} == {"UC1", "UC8", "UC9"}
    for scenario in SCENARIOS.values():
        assert scenario.task
        assert "JSON" in scenario.task  # the assistant ends with a verifiable answer
        for key in scenario.answer_keys:
            assert key in scenario.task


def test_the_expected_values_are_the_analytic_ones():
    nuller = SCENARIOS["uc1_nuller_transmission"].expected
    baseline, wavelength = 20.0, 10e-6
    for key, theta in [
        ("t_0", 0.0),
        ("t_lambda_8b", wavelength / (8 * baseline)),
        ("t_lambda_4b", wavelength / (4 * baseline)),
        ("t_lambda_2b", wavelength / (2 * baseline)),
    ]:
        assert nuller[key] == pytest.approx(math.sin(math.pi * baseline * theta / wavelength) ** 2, abs=1e-12)

    beam = SCENARIOS["uc8_propagation_methods"].expected
    w0, lam, z = 0.5e-3, 633e-9, 2.0
    w = w0 * math.sqrt(1 + (z / (math.pi * w0**2 / lam)) ** 2)
    assert beam["radius_asm_mm"] == pytest.approx(w * 1e3, rel=1e-9)
    assert beam["radius_fresnel_mm"] == pytest.approx(w * 1e3, rel=1e-9)
    assert beam["power_mw"] == pytest.approx(1.0)

    coronagraph = SCENARIOS["uc9_coronagraph_contrast"].expected
    assert coronagraph["contrast_without"] == pytest.approx(airy_ring_mean(3, 4), rel=1e-6)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_each_scenario_is_solved_through_mcp_and_verified(server, name):
    scenario = SCENARIOS[name]

    async def solve():
        async with Client(server) as client:
            return await run_reference(scenario, client)

    result = anyio.run(solve)
    assert set(result.answer) >= set(scenario.answer_keys)
    assert result.failures == []
    assert result.passed


def test_the_checks_reject_wrong_answers():
    nuller = SCENARIOS["uc1_nuller_transmission"]
    wrong = dict(nuller.expected, t_lambda_2b=0.5)
    assert nuller.check(wrong)
    missing = {k: v for k, v in nuller.expected.items() if k != "t_0"}
    assert any("t_0" in failure for failure in nuller.check(missing))
    coronagraph = SCENARIOS["uc9_coronagraph_contrast"]
    assert coronagraph.check(dict(coronagraph.expected, contrast_with=1e-3))  # no better than without
    beam = SCENARIOS["uc8_propagation_methods"]
    assert beam.check(dict(beam.expected, radius_asm_mm=beam.expected["radius_asm_mm"] * 1.1))
    assert nuller.check(nuller.expected) == []


def test_answers_are_read_from_free_text():
    text = 'The transmission is about 0.5.\n\n```json\n{"t_0": 1e-9, "t_lambda_2b": 0.999}\n```\nDone.'
    assert parse_answer(text) == {"t_0": 1e-9, "t_lambda_2b": 0.999}
    assert parse_answer('first {"a": 1} then {"a": 2}') == {"a": 2}  # the final answer counts
    assert parse_answer("no answer") == {}


def test_beam_radius_is_part_of_the_summary_of_a_field(server):
    """A Gaussian beam at its waist has a 1/e² radius equal to the waist (Siegman 1986, §17.1)."""

    async def scenario():
        async with Client(server) as client:
            await client.call_tool("load_preset", {"name": "propagation_methods"})
            result = await client.call_tool("inspect_port", {"port": "laser.out", "figure": False})
            return result.structured_content

    summary = anyio.run(scenario)
    radius = summary["details"]["beam radius"]
    values = [float(v) for v in radius.replace("mm", "").split(",")]
    np.testing.assert_allclose(values, [0.5, 0.5], rtol=0.01)


def test_radial_profile_tool(server):
    """The azimuthal mean of the focal image of a disk pupil is the Airy pattern (Born & Wolf §8.5.2)."""

    async def scenario():
        async with Client(server) as client:
            await client.call_tool("load_preset", {"name": "coronagraph"})
            await client.call_tool("set_param", {"path": "planet.flux_density", "value": "0 Jy"})
            for source, target in [("telescope.out", "coronagraph.in"), ("coronagraph.out", "lens.in")]:
                await client.call_tool("disconnect", {"source": source, "target": target})
            await client.call_tool("connect", {"source": "telescope.out", "target": "lens.in"})
            result = await client.call_tool(
                "radial_profile", {"port": "image.out", "bin_width": "20 um", "normalize": "peak"}
            )
            return result

    result = anyio.run(scenario)
    assert not result.is_error, result.content
    profile = result.structured_content
    assert profile["radius_unit"] == "m"
    assert profile["radii"][3] == pytest.approx(70e-6)  # centre of the ring 60-80 µm, that is 3-4 λ/D
    assert profile["values"][3] == pytest.approx(airy_ring_mean(3, 4), rel=0.1)


def test_the_evaluation_runner(tmp_path):
    """Without option, the runner solves the scenarios with the scripted solutions (free, no account)."""
    reference = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "mcp_eval.py")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
    )
    assert reference.returncode == 0, reference.stdout + reference.stderr
    assert "3/3 scenarios passed" in reference.stdout


def test_claude_code_is_given_the_lumos_server_only(tmp_path):
    """The evaluation runs Claude Code (``claude -p``), restricted to the tools of LUMOS."""
    scenario = SCENARIOS["uc9_coronagraph_contrast"]
    command = claude_code_command(scenario, archive_root=tmp_path)
    assert command[:3] == ["claude", "-p", scenario.task]
    config = json.loads(command[command.index("--mcp-config") + 1])
    lumos = config["mcpServers"]["lumos"]
    assert lumos["command"] == sys.executable  # no reliance on the PATH
    assert lumos["args"][:3] == ["-m", "lumos.cli", "mcp"]
    assert "--strict-mcp-config" in command
    assert command[command.index("--allowedTools") + 1] == "mcp__lumos"
    assert command[command.index("--output-format") + 1] == "json"


def test_the_answers_of_claude_code_are_checked(tmp_path):
    scenario = SCENARIOS["uc1_nuller_transmission"]
    final = "The transmission follows sin^2.\n```json\n" + json.dumps(dict(scenario.expected)) + "\n```"
    output = json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "result": final, "num_turns": 9}
    )

    def answered(command, **options):
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def failing(command, **options):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="not logged in")

    result = run_with_claude_code(scenario, archive_root=tmp_path, run=answered)
    assert result.passed, result.failures
    assert result.answer == dict(scenario.expected)
    failed = run_with_claude_code(scenario, archive_root=tmp_path, run=failing)
    assert not failed.passed
    assert "not logged in" in failed.failures[0]
    assert not (ROOT / ".github" / "workflows" / "mcp-eval.yml").exists()  # no paid periodic run
