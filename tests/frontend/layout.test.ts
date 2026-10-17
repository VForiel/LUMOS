// Automatic layout of a context without saved positions (step P14.4): each element is to the right of the
// elements feeding it, and no two elements share a position.
import { describe, expect, it } from "vitest";
import { autoLayout } from "../../web/src/lib/layout";

describe("autoLayout", () => {
  it("places the sources on the left and the light flows to the right", () => {
    const nodes = ["star", "planet", "telescope", "coronagraph", "image"];
    const edges: [string, string][] = [
      ["star", "telescope"],
      ["planet", "telescope"],
      ["telescope", "coronagraph"],
      ["coronagraph", "image"],
    ];
    const positions = autoLayout(nodes, edges);
    for (const [a, b] of edges) expect(positions[a][0]).toBeLessThan(positions[b][0]);
    const keys = new Set(Object.values(positions).map(([x, y]) => `${x},${y}`));
    expect(keys.size).toBe(nodes.length);
  });
  it("places isolated elements too", () => {
    expect(Object.keys(autoLayout(["a", "b"], []))).toEqual(["a", "b"]);
  });
});
