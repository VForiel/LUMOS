// Free targets of the editor (step P14.17, ADR-0022): arrays written by the server ("[1, 2] nm",
// "[[1, 0], [0, 1]]"), their components "field[i]" / "field[i, j]", and the numeric settings.
// Expected values: by construction (the JSON arrays written by format_value of the service).
import { describe, expect, it } from "vitest";
import { componentsOf, isNumericSetting, parseArray } from "../../web/src/lib/targets";

describe("parseArray", () => {
  it("reads a 1-D array with its unit", () => {
    expect(parseArray("[0.25, 0.5, 1] nm")).toEqual({ values: [0.25, 0.5, 1], shape: [3], unit: "nm" });
  });
  it("reads a dimensionless 2-D array", () => {
    expect(parseArray("[[1, 0], [0, 1]]")).toEqual({ values: [1, 0, 0, 1], shape: [2, 2], unit: "" });
  });
  it("refuses scalars, truncated arrays and ragged arrays", () => {
    expect(parseArray("3 mm")).toBeNull();
    expect(parseArray("[1.0, 2.0]... nm (shape [100])")).toBeNull();
    expect(parseArray("[[1, 2], [3]]")).toBeNull();
  });
});

describe("componentsOf", () => {
  it("names the components in C order", () => {
    expect(componentsOf("weights", "[[1, 2], [3, 4]] W").map((c) => [c.name, c.value, c.unit])).toEqual([
      ["weights[0, 0]", 1, "W"],
      ["weights[0, 1]", 2, "W"],
      ["weights[1, 0]", 3, "W"],
      ["weights[1, 1]", 4, "W"],
    ]);
    expect(componentsOf("t", "[0.5, 0.25]").map((c) => c.name)).toEqual(["t[0]", "t[1]"]);
    expect(componentsOf("t", "2 mm")).toEqual([]);
  });
});

describe("isNumericSetting", () => {
  it("accepts integer and real settings without choices", () => {
    expect(isNumericSetting({ name: "n", kind: "setting", input: "integer", doc: "" })).toBe(true);
    expect(isNumericSetting({ name: "x", kind: "setting", input: "number", doc: "" })).toBe(true);
    expect(isNumericSetting({ name: "b", kind: "setting", input: "boolean", doc: "" })).toBe(false);
    expect(isNumericSetting({ name: "m", kind: "setting", input: "enum", doc: "", choices: [1, 2] })).toBe(false);
    expect(isNumericSetting({ name: "p", kind: "param", input: "quantity", doc: "" })).toBe(false);
  });
});
