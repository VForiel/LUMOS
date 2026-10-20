// Blocks of the Procedures tab (step P15.12, ADR-0024): a procedure (JSON of the engine) becomes a Blockly
// workspace (one block per step, nested steps inside statement inputs) and back, without loss.
// Expected values: by construction (the procedure itself must come back unchanged).
import { describe, expect, it } from "vitest";
import { procedureToWorkspace, workspaceToProcedure, type ProcedureJson } from "../../web/src/lib/procedures";

const EVERY_STEP: ProcedureJson = {
  name: "every step",
  description: "one step of each kind",
  steps: [
    { do: "action", element: "stage", action: "move", args: ["2 mm"], kwargs: {} },
    {
      do: "repeat",
      times: 3,
      steps: [
        { do: "action", element: "camera", action: "expose", args: [], kwargs: {} },
        { do: "measure", outputs: ["camera.out"], as: "darks", collect: true },
      ],
    },
    { do: "process", function: "mean", args: { frames: "$darks" }, as: "dark" },
    {
      do: "for_each",
      values: { start: "1 mW", stop: "3 mW", num: 3, log: false },
      as: "power",
      steps: [{ do: "set", target: "laser.power", value: "$power" }],
    },
    { do: "for_each", values: ["0 deg", "90 deg"], as: "angle", steps: [] },
    { do: "use", result: "dark", port: "subtract.dark" },
    { do: "wait", duration: "2 s" },
    {
      do: "if",
      left: "$dark",
      op: ">",
      right: "3 adu",
      then: [{ do: "show", results: ["dark"] }],
      else: [{ do: "save", results: ["dark", "darks"], archive: "darks" }],
    },
  ],
};

describe("procedure blocks", () => {
  it("make one block per step, nested steps inside their parent", () => {
    const workspace = procedureToWorkspace(EVERY_STEP);
    const top = workspace.blocks.blocks;
    expect(top).toHaveLength(1);
    expect(top[0].type).toBe("lumos_procedure");
    expect(top[0].fields.NAME).toBe("every step");
    const first = top[0].inputs?.STEPS?.block;
    expect(first?.type).toBe("lumos_action");
    expect(first?.next?.block.type).toBe("lumos_repeat");
    expect(first?.next?.block.inputs?.STEPS?.block.type).toBe("lumos_action");
  });

  it("come back to the same procedure", () => {
    expect(workspaceToProcedure(procedureToWorkspace(EVERY_STEP))).toEqual(EVERY_STEP);
  });

  it("keep an empty procedure", () => {
    const empty: ProcedureJson = { name: "empty", description: "", steps: [] };
    expect(workspaceToProcedure(procedureToWorkspace(empty))).toEqual(empty);
  });
});
