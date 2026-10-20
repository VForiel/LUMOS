// Typed connections: which output may feed which input, and the colour of each kind of signal.

export type Signals = Record<string, { parents: string[] }>;

const NUMERICAL_FIELDS = ["Wavefront", "Wavecube"];

/** Whether an output producing ``source`` can feed an input expecting ``target`` (lumos.core.port). */
export function compatible(source: string, target: string, signals: Signals): boolean {
  const lineage = (name: string) => [name, ...(signals[name]?.parents ?? [])];
  if (lineage(source).includes(target)) return true;
  if (source === "SymbolicField") {
    // converted to a numerical field (P10.6)
    return NUMERICAL_FIELDS.some((field) => lineage(field).includes(target));
  }
  return false;
}

const COLOURS: Record<string, string> = {
  Signal: "#94a3b8",
  OpticalSignal: "#818cf8",
  Wavefront: "#38bdf8",
  Wavecube: "#a78bfa",
  IncoherentBundle: "#22d3ee",
  ModalField: "#2dd4bf",
  Bus: "#2dd4bf",
  RadianceField: "#fb923c",
  SourceList: "#fbbf24",
  SymbolicField: "#f472b6",
  DataSignal: "#86efac",
  DataProduct: "#4ade80",
};

/** Colour of a kind of signal (stable; a colour derived from its name for unknown kinds). */
export function signalColor(name: string): string {
  if (COLOURS[name]) return COLOURS[name];
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  const hue = hash % 360;
  return hslToHex(hue, 70, 62);
}

function hslToHex(h: number, s: number, l: number): string {
  const a = (s / 100) * Math.min(l / 100, 1 - l / 100);
  const channel = (n: number) => {
    const k = (n + h / 30) % 12;
    const value = l / 100 - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(255 * value)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${channel(0)}${channel(8)}${channel(4)}`;
}
