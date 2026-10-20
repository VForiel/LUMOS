// Forms and ports generated from the JSON Schema of an element (step P14.4, REQ-WEB-2): no code per element.
// The schema below follows the one LUMOS publishes for a circular aperture (subset of /api/palette).
import { describe, expect, it } from "vitest";
import { checkBounds, fieldsOf, portsOf } from "../../web/src/lib/schema";
import { portNamed } from "../../web/src/lib/schema";

const quantity = (unit: string, extra: object) => ({
  type: "object",
  properties: { value: {}, unit: { type: "string" } },
  required: ["value", "unit"],
  "x-lumos-kind": "param",
  "x-lumos-unit": unit,
  "x-lumos-dimension": "length",
  "x-lumos-shape": [],
  ...extra,
});

const SCHEMA = {
  title: "CircularAperture",
  description: "Circular pupil with an optional central obstruction.",
  properties: {
    params: {
      properties: {
        diameter: quantity("m", {
          description: "",
          "x-lumos-bounds": [{ value: 0, unit: "m" }, null],
          "x-lumos-inclusive": [false, true],
          default: { value: 1, unit: "m" },
        }),
        mask: quantity("", {
          description: "transmission map",
          "x-lumos-shape": null,
          "x-lumos-bounds": [null, null],
        }),
      },
    },
    settings: {
      properties: {
        n_spiders: { "x-lumos-kind": "setting", type: "integer", default: 0, description: "" },
        method: {
          "x-lumos-kind": "setting",
          type: "string",
          enum: ["auto", "fft"],
          default: "auto",
          description: "",
        },
        flip: { "x-lumos-kind": "setting", type: "boolean", default: false, description: "" },
        x: { "x-lumos-kind": "setting", default: { value: 0, unit: "pix" }, description: "centre" },
        output_grid: { "x-lumos-kind": "setting", description: "sampling of the output plane" },
      },
    },
    state: { properties: { position: quantity("mm", { "x-lumos-kind": "state", description: "" }) } },
  },
  "x-lumos-ports": {
    inputs: [{ name: "in", signal: "OpticalSignal", multiple: false, doc: "" }],
    outputs: [{ name: "out", signal: "OpticalSignal", multiple: true, doc: "transmitted field" }],
  },
};

describe("fieldsOf", () => {
  const fields = Object.fromEntries(fieldsOf(SCHEMA).map((f) => [f.name, f]));
  it("gives quantities with their unit and bounds", () => {
    expect(fields.diameter).toMatchObject({ kind: "param", input: "quantity", unit: "m", bounds: [0, null] });
    expect(fields.diameter.inclusive).toEqual([false, true]);
  });
  it("edits arrays as text", () => {
    expect(fields.mask.input).toBe("text");
  });
  it("gives the inputs of the settings", () => {
    expect(fields.n_spiders.input).toBe("integer");
    expect(fields.method).toMatchObject({ input: "enum", choices: ["auto", "fft"] });
    expect(fields.flip.input).toBe("boolean");
    expect(fields.x).toMatchObject({ input: "quantity", unit: "pix" });
    expect(fields.output_grid.input).toBe("json");
  });
  it("keeps the state read-only", () => {
    expect(fields.position.kind).toBe("state");
  });
  it("keeps the order params, settings, state", () => {
    expect(fieldsOf(SCHEMA).map((f) => f.kind)).toEqual([
      "param",
      "param",
      "setting",
      "setting",
      "setting",
      "setting",
      "setting",
      "state",
    ]);
  });
});

describe("portsOf", () => {
  it("lists the inputs and outputs with their signal", () => {
    const ports = portsOf(SCHEMA);
    expect(ports.inputs.map((p) => p.name)).toEqual(["in"]);
    expect(ports.outputs[0]).toMatchObject({ name: "out", signal: "OpticalSignal", doc: "transmitted field" });
  });
});

describe("checkBounds", () => {
  const diameter = fieldsOf(SCHEMA).find((f) => f.name === "diameter")!;
  it("refuses values outside the bounds, exclusive or not", () => {
    expect(checkBounds(diameter, 0)).toBe("below");
    expect(checkBounds(diameter, -1)).toBe("below");
    expect(checkBounds(diameter, 8)).toBeNull();
  });
});

// Repeated ports of Demux and Mux (step P15.7): the schema gives a pattern and the setting counting them.
describe("repeated ports", () => {
  const MUX = {
    "x-lumos-ports": {
      inputs: [{ name: "in{i}", count: "n_lanes", signal: "OpticalSignal", multiple: false, bus: false, optional: true, doc: "lane" }],
      outputs: [{ name: "out", signal: "OpticalSignal", multiple: false, bus: true, doc: "" }],
    },
  };

  it("finds the declaration of a numbered port", () => {
    expect(portNamed(portsOf(MUX).inputs, "in3")?.signal).toBe("OpticalSignal");
    expect(portNamed(portsOf(MUX).inputs, "in3")?.name).toBe("in3");
    expect(portNamed(portsOf(MUX).inputs, "inx")).toBeUndefined();
    expect(portNamed(portsOf(MUX).outputs, "out")?.bus).toBe(true);
  });
});
