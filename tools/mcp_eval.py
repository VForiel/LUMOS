"""Run the evaluation scenarios of the MCP server (step P13.8, REQ-MCP-7, ADR-0017).

Usage::

    python tools/mcp_eval.py                     # scripted solutions through MCP (free, as in the tests)
    python tools/mcp_eval.py --claude-code [--model M] [--scenario NAME ...] [--report report.json]
                                                 # Claude Code solves the scenarios with your subscription

With ``--claude-code``, the command ``claude`` (Claude Code) must be installed and logged in; each scenario
starts its own LUMOS server and uses part of the usage of your plan.

Exit status: 0 when every scenario passes, 1 when one fails, 2 when Claude Code cannot be found.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from typing import Any

import anyio


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


async def _run_reference(name: str, archive_root: str) -> Any:
    from mcp import Client

    from lumos.mcp import create_server
    from lumos.mcp.evaluation import SCENARIOS, run_reference
    from lumos.service import Service, ServiceConfig

    service = Service(ServiceConfig(archive_root=archive_root))
    try:
        async with Client(create_server(service)) as client:
            return await run_reference(SCENARIOS[name], client)
    finally:
        service.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Run the scenarios and return the exit status."""
    from lumos.mcp.evaluation import SCENARIOS, run_with_claude_code

    parser = argparse.ArgumentParser(description="Run the evaluation scenarios of the MCP server.")
    parser.add_argument(
        "--claude-code", action="store_true", help="let Claude Code solve the scenarios (your subscription)"
    )
    parser.add_argument("--model", help="model of Claude Code (default: the one of your settings)")
    parser.add_argument("--scenario", nargs="*", choices=sorted(SCENARIOS), help="scenarios (default: all)")
    parser.add_argument("--report", help="write the results to this JSON file")
    arguments = parser.parse_args(argv)
    names = arguments.scenario or sorted(SCENARIOS)
    executable = shutil.which("claude")
    if arguments.claude_code and executable is None:
        sys.stderr.write("Claude Code is not installed or not on the PATH (command 'claude')\n")
        return 2
    results = []
    with tempfile.TemporaryDirectory() as archive_root:
        for name in names:
            if arguments.claude_code and executable is not None:
                result = run_with_claude_code(
                    SCENARIOS[name], archive_root=archive_root, model=arguments.model, executable=executable
                )
            else:
                result = anyio.run(_run_reference, name, archive_root)
            status = "PASS" if result.passed else "FAIL"
            _write(f"{status} {SCENARIOS[name].use_case} {name}: {json.dumps(result.answer)}")
            for failure in result.failures:
                _write(f"     {failure}")
            results.append(result)
    passed = sum(result.passed for result in results)
    _write(f"{passed}/{len(results)} scenarios passed")
    if arguments.report:
        report = {
            "assistant": "claude-code" if arguments.claude_code else "reference",
            "results": [asdict(r) for r in results],
        }
        with open(arguments.report, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=str)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
