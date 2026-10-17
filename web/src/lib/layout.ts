// Automatic layout of a context without saved positions: columns by depth along the light path.

export const COLUMN = 290;
export const ROW = 170;

/** Positions ``[x, y]`` of ``nodes``: each node one column to the right of the deepest node feeding it. */
export function autoLayout(nodes: string[], edges: [string, string][]): Record<string, [number, number]> {
  const incoming = new Map<string, string[]>(nodes.map((n) => [n, []]));
  for (const [source, target] of edges) {
    if (incoming.has(source) && incoming.has(target)) incoming.get(target)!.push(source);
  }
  const depth = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (node: string): number => {
    if (depth.has(node)) return depth.get(node)!;
    if (visiting.has(node)) return 0; // cycle: break it
    visiting.add(node);
    const parents = incoming.get(node) ?? [];
    const value = parents.length ? Math.max(...parents.map(depthOf)) + 1 : 0;
    visiting.delete(node);
    depth.set(node, value);
    return value;
  };
  const columns = new Map<number, string[]>();
  for (const node of nodes) {
    const d = depthOf(node);
    columns.set(d, [...(columns.get(d) ?? []), node]);
  }
  const positions: Record<string, [number, number]> = {};
  for (const node of nodes) {
    const d = depth.get(node)!;
    const column = columns.get(d)!;
    const row = column.indexOf(node);
    const offset = ((column.length - 1) * ROW) / 2;
    positions[node] = [d * COLUMN, row * ROW - offset];
  }
  return positions;
}
