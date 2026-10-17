// Typed connections of the editor (step P14.4, REQ-WEB-1): an output feeds an input when its signal is the
// expected one or a subclass of it (lumos.core.port), or a symbolic field converted to a numerical field.
import { describe, expect, it } from "vitest";
import { compatible, signalColor } from "../../web/src/lib/signals";

const SIGNALS = {
  Signal: { parents: [] },
  OpticalSignal: { parents: ["Signal"] },
  DataSignal: { parents: ["Signal"] },
  Wavefront: { parents: ["OpticalSignal", "Signal"] },
  Wavecube: { parents: ["OpticalSignal", "Signal"] },
  DataProduct: { parents: ["DataSignal", "Signal"] },
  SymbolicField: { parents: ["OpticalSignal", "Signal"] },
};

describe("compatible", () => {
  it("accepts the same signal and its subclasses", () => {
    expect(compatible("Wavefront", "Wavefront", SIGNALS)).toBe(true);
    expect(compatible("Wavefront", "OpticalSignal", SIGNALS)).toBe(true);
    expect(compatible("DataProduct", "Signal", SIGNALS)).toBe(true);
  });
  it("refuses other signals", () => {
    expect(compatible("OpticalSignal", "Wavefront", SIGNALS)).toBe(false);
    expect(compatible("DataProduct", "OpticalSignal", SIGNALS)).toBe(false);
    expect(compatible("Wavefront", "DataProduct", SIGNALS)).toBe(false);
  });
  it("converts symbolic fields to numerical fields", () => {
    expect(compatible("SymbolicField", "Wavefront", SIGNALS)).toBe(true);
    expect(compatible("SymbolicField", "Wavecube", SIGNALS)).toBe(true);
    expect(compatible("SymbolicField", "DataProduct", SIGNALS)).toBe(false);
  });
});

describe("signalColor", () => {
  it("gives each kind of signal its own stable colour", () => {
    const names = ["Wavefront", "Wavecube", "DataProduct", "ModalField", "RadianceField", "SourceList"];
    const colours = names.map(signalColor);
    expect(new Set(colours).size).toBe(names.length);
    expect(signalColor("Wavefront")).toBe(colours[0]);
    expect(signalColor("SomethingNew")).toMatch(/^#[0-9a-f]{6}$/);
  });
});
