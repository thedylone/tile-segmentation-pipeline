// Procedural PLACEHOLDER scene for the hero viewer: a synthetic city block
// built from polygons, each face carrying a class label, the way the
// pipeline's output tiles do. It is not output from the pipeline.
import { IcosahedronGeometry } from "three";

export interface ClassInfo {
  id: number;
  name: string;
  color: string;
}

// Cityscapes ids, as used by facebook/mask2former-swin-small-cityscapes-semantic.
export const CLASSES: ClassInfo[] = [
  { id: 2, name: "building", color: "#d9a441" },
  { id: 8, name: "vegetation", color: "#b452d1" },
  { id: 0, name: "road", color: "#6b7280" },
  { id: 1, name: "sidewalk", color: "#cfc3a6" },
  { id: 9, name: "terrain", color: "#9db37f" },
];

/** flat triangle soup per class: [x,y,z, x,y,z, x,y,z, ...] */
export type Scene = Map<number, number[]>;

export function buildScene(opts: { lite: boolean }): Scene {
  let seed = 20230717;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const range = (a: number, b: number) => a + rnd() * (b - a);

  const out: Scene = new Map(CLASSES.map((c) => [c.id, []]));
  const tri = (c: number, ...v: number[]) => out.get(c)!.push(...v);
  const quad = (c: number, a: number[], b: number[], d: number[], e: number[]) => {
    tri(c, ...a, ...b, ...d);
    tri(c, ...a, ...d, ...e);
  };

  const HALF = 60;
  const ROAD = 8;
  const WALK = 3;
  const roads = [-24, 30];
  const near = (v: number, w: number) => roads.some((r) => Math.abs(v - r) < w);

  // ground: a gently uneven triangulated grid, labelled per cell
  const N = opts.lite ? 36 : 48;
  const step = (HALF * 2) / N;
  const gh = (x: number, z: number) =>
    near(x, ROAD / 2 + WALK) || near(z, ROAD / 2 + WALK)
      ? 0
      : 0.35 * Math.sin(x * 0.21) * Math.cos(z * 0.17) + 0.2 * Math.sin((x + z) * 0.5);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      const x0 = -HALF + i * step;
      const z0 = -HALF + j * step;
      const x1 = x0 + step;
      const z1 = z0 + step;
      const cx = x0 + step / 2;
      const cz = z0 + step / 2;
      let c = 9;
      if (near(cx, ROAD / 2) || near(cz, ROAD / 2)) c = 0;
      else if (near(cx, ROAD / 2 + WALK) || near(cz, ROAD / 2 + WALK)) c = 1;
      quad(c, [x0, gh(x0, z0), z0], [x0, gh(x0, z1), z1], [x1, gh(x1, z1), z1], [x1, gh(x1, z0), z0]);
    }

  const box = (x0: number, x1: number, z0: number, z1: number, h: number, gable: boolean) => {
    const B = 2;
    const p = (x: number, y: number, z: number) => [x, y, z];
    // walls
    quad(B, p(x0, 0, z0), p(x1, 0, z0), p(x1, h, z0), p(x0, h, z0));
    quad(B, p(x1, 0, z0), p(x1, 0, z1), p(x1, h, z1), p(x1, h, z0));
    quad(B, p(x1, 0, z1), p(x0, 0, z1), p(x0, h, z1), p(x1, h, z1));
    quad(B, p(x0, 0, z1), p(x0, 0, z0), p(x0, h, z0), p(x0, h, z1));
    if (!gable) {
      quad(B, p(x0, h, z0), p(x0, h, z1), p(x1, h, z1), p(x1, h, z0));
      return;
    }
    // gable roof along the longer side
    const r = Math.min(x1 - x0, z1 - z0) * 0.45;
    if (x1 - x0 >= z1 - z0) {
      const zm = (z0 + z1) / 2;
      quad(B, p(x0, h, z0), p(x0, h + r, zm), p(x1, h + r, zm), p(x1, h, z0));
      quad(B, p(x1, h, z1), p(x1, h + r, zm), p(x0, h + r, zm), p(x0, h, z1));
      tri(B, ...p(x0, h, z0), ...p(x0, h, z1), ...p(x0, h + r, zm));
      tri(B, ...p(x1, h, z1), ...p(x1, h, z0), ...p(x1, h + r, zm));
    } else {
      const xm = (x0 + x1) / 2;
      quad(B, p(x0, h, z1), p(xm, h + r, z1), p(xm, h + r, z0), p(x0, h, z0));
      quad(B, p(x1, h, z0), p(xm, h + r, z0), p(xm, h + r, z1), p(x1, h, z1));
      tri(B, ...p(x0, h, z0), ...p(xm, h + r, z0), ...p(x1, h, z0));
      tri(B, ...p(x1, h, z1), ...p(xm, h + r, z1), ...p(x0, h, z1));
    }
  };

  // lumpy canopies: displaced icosahedra, like photogrammetry vegetation
  const ico = new IcosahedronGeometry(1, 1).toNonIndexed().getAttribute("position").array as Float32Array;
  const tree = (x: number, z: number, r: number) => {
    const phase = rnd() * 10;
    const y = gh(x, z) + r * 0.75;
    const V = 8;
    for (let k = 0; k < ico.length; k += 3) {
      const ux = ico[k];
      const uy = ico[k + 1];
      const uz = ico[k + 2];
      // displacement depends only on the direction, so shared vertices stay welded
      const d = 1 + 0.22 * Math.sin(ux * 4.1 + phase) * Math.cos(uz * 3.7 - phase) + 0.12 * Math.sin(uy * 5 + phase);
      out.get(V)!.push(x + ux * r * d, y + uy * r * d * 0.78, z + uz * r * d);
    }
  };

  // blocks between roads
  const edges = [-HALF, ...roads.flatMap((r) => [r - ROAD / 2 - WALK, r + ROAD / 2 + WALK]), HALF];
  const spans: [number, number][] = [];
  for (let i = 0; i < edges.length; i += 2) spans.push([edges[i] + 1.5, edges[i + 1] - 1.5]);

  spans.forEach(([sx0, sx1], bi) =>
    spans.forEach(([sz0, sz1], bj) => {
      const kind = (bi * 3 + bj) % 4;
      if (kind === 3 && bi !== 1) {
        // park
        for (let k = 0; k < 14; k++) tree(range(sx0 + 3, sx1 - 3), range(sz0 + 3, sz1 - 3), range(2.6, 4.4));
        return;
      }
      if (bi === 1 && bj === 1) {
        // condo towers among trees
        const w = sx1 - sx0;
        for (let k = 0; k < 3; k++) {
          const x = sx0 + 4 + k * ((w - 8) / 3);
          box(x, x + 7, sz0 + 6 + k * 3, sz0 + 14 + k * 3, range(26, 34), false);
        }
        for (let k = 0; k < 10; k++) tree(range(sx0 + 2, sx1 - 2), range(sz1 - 12, sz1 - 2), range(2, 3.4));
        return;
      }
      // terraced houses with gable roofs
      const cols = Math.max(2, Math.floor((sx1 - sx0) / 9));
      const rows = Math.max(2, Math.floor((sz1 - sz0) / 11));
      const w = (sx1 - sx0) / cols;
      const d = (sz1 - sz0) / rows;
      for (let a = 0; a < cols; a++)
        for (let b = 0; b < rows; b++) {
          const x0 = sx0 + a * w;
          const z0 = sz0 + b * d;
          if (rnd() < 0.22) {
            tree(x0 + w / 2, z0 + d / 2, range(2.2, 3.6));
            continue;
          }
          box(x0 + 1, x0 + w - 1, z0 + 1.5, z0 + d - 1.5, range(4, 8), true);
          if (rnd() < 0.45) tree(x0 + range(1, w - 1), z0 + d - 0.6, range(1.4, 2.2));
        }
    }),
  );

  // street trees along the verges
  for (const r of roads)
    for (let t = -HALF + 4; t < HALF - 2; t += 7) {
      if (near(t, ROAD / 2 + WALK + 2)) continue;
      for (const side of [-1, 1]) {
        if (rnd() < 0.3) continue;
        const off = r + side * (ROAD / 2 + WALK * 0.5);
        tree(off, t, range(1.5, 2.1));
        tree(t, off, range(1.5, 2.1));
      }
    }

  return out;
}
