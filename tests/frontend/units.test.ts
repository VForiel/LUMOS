// Units of the forms (step P14.4, REQ-WEB-3): parsing, conversion and display of quantities.
// Expected values: SI prefixes (1 mm = 1000 um) and plain arithmetic.
import { describe, expect, it } from "vitest";
import { convert, formatNumber, parseQuantity } from "../../web/src/lib/units";

const LENGTHS: [string, number][] = [
  ["nm", 1e-9],
  ["um", 1e-6],
  ["mm", 1e-3],
  ["m", 1],
];

describe("parseQuantity", () => {
  it("reads a value and its unit", () => {
    expect(parseQuantity("3 mm")).toEqual({ value: 3, unit: "mm" });
    expect(parseQuantity("1e-3 W")).toEqual({ value: 0.001, unit: "W" });
    expect(parseQuantity("-2.5e+3 um")).toEqual({ value: -2500, unit: "um" });
    expect(parseQuantity("  0.5   m / s ")).toEqual({ value: 0.5, unit: "m / s" });
  });
  it("reads a bare number as dimensionless", () => {
    expect(parseQuantity("0.95")).toEqual({ value: 0.95, unit: "" });
  });
  it("refuses arrays and text", () => {
    expect(parseQuantity("[1, 2] nm")).toBeNull();
    expect(parseQuantity("mm")).toBeNull();
    expect(parseQuantity("")).toBeNull();
  });
});

describe("convert", () => {
  it("converts with the factors of the units", () => {
    expect(convert(3, "mm", "um", LENGTHS)).toBeCloseTo(3000, 9);
    expect(convert(1550, "nm", "um", LENGTHS)).toBeCloseTo(1.55, 12);
    expect(convert(2, "m", "m", LENGTHS)).toBe(2);
  });
  it("refuses units it does not know", () => {
    expect(() => convert(1, "mm", "s", LENGTHS)).toThrow();
  });
});

describe("formatNumber", () => {
  it("keeps six significant digits without trailing zeros", () => {
    expect(formatNumber(0.002)).toBe("0.002");
    expect(formatNumber(1500)).toBe("1500");
    expect(formatNumber(1e-9)).toBe("1e-9");
    expect(formatNumber(123456789)).toBe("1.23457e+8");
    expect(formatNumber(0.1 + 0.2)).toBe("0.3");
    expect(formatNumber(0)).toBe("0");
  });
});
