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
- `run_with_claude` gives the task to Claude (`claude-opus-5-5` by default, adaptive thinking at effort
  `high`, server-side refusal fallback) with the tools of the server, in a manual tool-use loop that forwards
  calls to the MCP client and returns text and images as tool results.
- `tools/mcp_eval.py` runs the scenarios (`--reference` or with Claude) and writes a JSON report; the workflow
  `MCP evaluation` runs it weekly when the secret `ANTHROPIC_API_KEY` exists. The `anthropic` package is the
  extra `eval`, not a dependency of LUMOS.
- Missing capabilities found while writing the scenarios became tools: `radial_profile` (contrast in rings) and
  the 1/e² beam radius in the summaries of fields.

## Alternatives considered

- Judging the answers with a language model: not verifiable, and the results have exact physical references.
- Running the evaluation with Claude in every CI run: costs and flakiness on every commit; the scripted solutions
  keep the scenarios honest at no cost, the weekly run follows the behaviour of the models.
- The MCP conversion helpers of the `anthropic` SDK: written for the SDK 1 of MCP.

## Consequences

- New use cases join the evaluation as a scenario with an analytic expectation and a scripted solution.
- A failure of the weekly run means a regression of the server or of the guidance it gives (instructions,
  descriptions, hints), or a model that needs better guidance; the report holds the answers.
