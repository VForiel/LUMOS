"""Run the evaluation scenarios of the MCP server (step P13.8, REQ-MCP-7, ADR-0017).

Each scenario of :mod:`lumos.mcp.evaluation` runs against a fresh server, through an in-process MCP client.

Usage::

    python tools/mcp_eval.py --reference            # scripted solutions (no API key, as in the tests)
    python tools/mcp_eval.py [--model M] [--scenario NAME ...] [--report report.json]
                                                    # Claude solves the scenarios (extra "eval", API key)

Exit status: 0 when every scenario passes, 1 when one fails, 2 when Claude cannot be reached.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from typing import Any

import anyio


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


async def _run(names: list[str], *, reference: bool, model: str, archive_root: str) -> list[Any]:
    from mcp import Client

    from lumos.mcp import create_server
    from lumos.mcp.evaluation import SCENARIOS, run_reference, run_with_claude
    from lumos.service import Service, ServiceConfig

    results = []
    for name in names:
        service = Service(ServiceConfig(archive_root=archive_root))
        try:
            async with Client(create_server(service)) as client:
                scenario = SCENARIOS[name]
                if reference:
                    result = await run_reference(scenario, client)
                else:
                    result = await run_with_claude(scenario, client, model=model)
        finally:
            service.shutdown()
        status = "PASS" if result.passed else "FAIL"
        _write(f"{status} {scenario.use_case} {name}: {json.dumps(result.answer)}")
        for failure in result.failures:
            _write(f"     {failure}")
        results.append(result)
    return results


def _from_anthropic(error: BaseException) -> bool:
    """Whether ``error`` (or an error of its group) comes from the Claude API client."""
    if isinstance(error, BaseExceptionGroup):
        return any(_from_anthropic(inner) for inner in error.exceptions)
    return type(error).__module__.startswith("anthropic")


def main(argv: list[str] | None = None) -> int:
    """Run the scenarios and return the exit status."""
    from lumos.mcp.evaluation import DEFAULT_MODEL, SCENARIOS

    parser = argparse.ArgumentParser(description="Run the evaluation scenarios of the MCP server.")
    parser.add_argument(
        "--reference", action="store_true", help="use the scripted solutions instead of Claude"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--scenario", nargs="*", choices=sorted(SCENARIOS), help="scenarios (default: all)")
    parser.add_argument("--report", help="write the results to this JSON file")
    arguments = parser.parse_args(argv)
    names = arguments.scenario or sorted(SCENARIOS)
    if not arguments.reference:
        try:
            import anthropic  # pyright: ignore[reportMissingImports] # noqa: F401
        except ImportError:
            sys.stderr.write(
                'the evaluation with Claude needs the extra "eval": pip install "lumos-maxima[eval]"\n'
            )
            return 2
    with tempfile.TemporaryDirectory() as archive_root:
        try:
            results = anyio.run(
                lambda: _run(
                    names, reference=arguments.reference, model=arguments.model, archive_root=archive_root
                )
            )
        except Exception as error:
            if _from_anthropic(error):
                sys.stderr.write(f"cannot reach Claude: {error}\n")
                return 2
            raise
    passed = sum(result.passed for result in results)
    _write(f"{passed}/{len(results)} scenarios passed")
    if arguments.report:
        report = {
            "model": None if arguments.reference else arguments.model,
            "results": [asdict(r) for r in results],
        }
        with open(arguments.report, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=str)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
