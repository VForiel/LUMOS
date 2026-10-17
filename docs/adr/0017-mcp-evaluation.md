# ADR-0017: Evaluation of the MCP server

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P13.8 (REQ-MCP-7)

## Context

The MCP server is useful only if an assistant can reach right results through it. Unit tests call each tool,
but do not show that a task stated in plain language (UC1, UC8, UC9) can be solved with the tools, nor that an
assistant actually solves it. The answer of an assistant is free text, and its quality changes with the model.

## Decision

- `lumos.mcp.evaluation` defines **scenarios**: a task in plain language, expected values derived from analytic
  solutions (Bracewell transmission sin²(π B θ / λ), Gaussian beam radius w(z), Airy ring mean and the rejection
  of an ideal vortex), and a rule per value (relative or absolute tolerance, upper bound). The task asks the
  assistant to end with a JSON object of named numbers; `parse_answer` reads the last one.
- Each scenario has a **scripted reference solution** that only calls MCP tools. The tests run it through an
  in-process client and check its answer with the same rules: the scenario is solvable with the tools, and the
  rules accept a right answer (other tests check that they reject wrong ones).
- The scripted solutions run at every CI run (they are tests): this is the periodic part of REQ-MCP-7.
- `run_with_claude_code` gives the task to Claude Code (`claude -p`) with the MCP server of LUMOS as its only
  tools (`--mcp-config`, `--strict-mcp-config`, `--allowedTools mcp__lumos`) and checks the final answer, read
  from the JSON output. It runs on demand with the subscription of the user: `tools/mcp_eval.py --claude-code`.
  Without option, `tools/mcp_eval.py` runs the scripted solutions.
- Missing capabilities found while writing the scenarios became tools: `radial_profile` (contrast in rings) and
  the 1/e² beam radius in the summaries of fields.

## Alternatives considered

- Judging the answers with a language model: not verifiable, and the results have exact physical references.
- Running the evaluation with Claude in every CI run: costs and flakiness on every commit; the scripted solutions
  keep the scenarios honest at no cost.
- A weekly run through the Claude API (first version of this ADR): needs an API key billed per token, too
  costly for the project; replaced on 2026-10-17 by Claude Code on demand, which uses an existing
  subscription.

## Consequences

- New use cases join the evaluation as a scenario with an analytic expectation and a scripted solution.
- A failure of a run with Claude Code means a regression of the server or of the guidance it gives (instructions,
  descriptions, hints), or a model that needs better guidance; the report holds the answers.
