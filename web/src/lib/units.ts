// Quantities in the forms: parsing "3 mm", converting between units of the same dimension, compact display.

export type Quantity = { value: number; unit: string };

/** Units of a dimension, with the factor converting a value in each unit to a common reference unit. */
export type UnitChoices = [string, number][];

const NUMBER = /^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$/;

/** Read "3 mm" or "1e-3 W" as a value and a unit; a bare number is dimensionless; arrays and text give null. */
export function parseQuantity(text: string): Quantity | null {
  const match = NUMBER.exec(text);
  if (!match) return null;
  const value = Number(match[1]);
  if (!Number.isFinite(value)) return null;
  const unit = match[2].replace(/\s+/g, " ");
  if (/[[\]{}=,;]/.test(unit)) return null;
  return { value, unit };
}

/** Convert ``value`` from the unit ``from`` to the unit ``to``, both listed in ``choices``. */
export function convert(value: number, from: string, to: string, choices: UnitChoices): number {
  if (from === to) return value;
  const factors = new Map(choices);
  const a = factors.get(from);
  const b = factors.get(to);
  if (a === undefined || b === undefined) throw new Error(`cannot convert ${from} to ${to}`);
  return (value * a) / b;
}

/** Six significant digits at most, without trailing zeros; exponents for very small or large values. */
export function formatNumber(value: number): string {
  if (value === 0) return "0";
  if (!Number.isFinite(value)) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-4 && magnitude < 1e6) {
    return String(Number(value.toPrecision(6)));
  }
  const [mantissa, exponent] = value.toExponential(5).split("e");
  const trimmed = mantissa.includes(".") ? mantissa.replace(/0+$/, "").replace(/\.$/, "") : mantissa;
  const power = Number(exponent);
  return `${trimmed}e${power >= 0 ? "+" : ""}${power}`.replace("e+-", "e-");
}

/** Format a quantity as the service reads it: "3 mm". */
export function formatQuantity(value: number, unit: string): string {
  return unit ? `${formatNumber(value)} ${unit}` : formatNumber(value);
}
