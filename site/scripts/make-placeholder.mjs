// Generates a clearly-labelled PLACEHOLDER segmented point cloud for the
// project page viewer. This is a synthetic city block, NOT pipeline output.
// Swap public/data/segmented_cloud.json for real output when available
// (same schema: classes[], positions[x,y,z,...], labels[]).
import { writeFileSync } from "node:fs";

let seed = 20230717;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const range = (a, b) => a + rnd() * (b - a);

// Class ids follow the Cityscapes label set used by
// facebook/mask2former-swin-small-cityscapes-semantic.
const classes = [
  { id: 0, name: "road", color: "#5b6474" },
  { id: 1, name: "sidewalk", color: "#c9bfa6" },
  { id: 2, name: "building", color: "#e08a3c" },
  { id: 8, name: "vegetation", color: "#6a3d9a" },
  { id: 9, name: "terrain", color: "#8fa77a" },
  { id: 13, name: "car", color: "#1e2f4d" },
];

const P = [], L = [];
const push = (x, y, z, c) => { P.push(+x.toFixed(1), +y.toFixed(1), +z.toFixed(1)); L.push(c); };

const HALF = 60, ROAD = 7, WALK = 3;
const roadsAt = [-30, 30]; // road centrelines along x and z

const onRoad = (v) => roadsAt.some((r) => Math.abs(v - r) < ROAD / 2);
const onWalk = (v) => roadsAt.some((r) => Math.abs(v - r) < ROAD / 2 + WALK);

// ground
for (let i = 0; i < 16000; i++) {
  const x = range(-HALF, HALF), z = range(-HALF, HALF);
  let c = 9;
  if (onRoad(x) || onRoad(z)) c = 0;
  else if (onWalk(x) || onWalk(z)) c = 1;
  push(x, range(-0.1, 0.1), z, c);
}

// blocks: 3x3 grid between roads
const cells = [-HALF, ...roadsAt.flatMap((r) => [r - ROAD / 2 - WALK, r + ROAD / 2 + WALK]), HALF];
const spans = [];
for (let i = 0; i < cells.length; i += 2) spans.push([cells[i] + 2, cells[i + 1] - 2]);

const building = (x0, x1, z0, z1, h) => {
  const area = (x1 - x0) * (z1 - z0);
  for (let i = 0; i < area * 0.9; i++) push(range(x0, x1), h + range(-0.05, 0.05), range(z0, z1), 2);
  const walls = [[x0, z0, x1, z0], [x1, z0, x1, z1], [x1, z1, x0, z1], [x0, z1, x0, z0]];
  for (const [ax, az, bx, bz] of walls) {
    const len = Math.hypot(bx - ax, bz - az);
    for (let i = 0; i < len * h * 0.55; i++) {
      const t = rnd();
      push(ax + (bx - ax) * t, range(0, h), az + (bz - az) * t, 2);
    }
  }
};

const tree = (x, z, r, h) => {
  for (let i = 0; i < r * r * 16; i++) {
    const u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, s = Math.sqrt(1 - u * u);
    const rr = r * Math.cbrt(range(0.15, 1));
    push(x + rr * s * Math.cos(th), h + rr * u * 0.8, z + rr * s * Math.sin(th), 8);
  }
};

for (const [sx0, sx1] of spans) for (const [sz0, sz1] of spans) {
  const park = rnd() < 0.2;
  if (park) {
    for (let k = 0; k < 9; k++) tree(range(sx0 + 3, sx1 - 3), range(sz0 + 3, sz1 - 3), range(2.5, 4.2), range(4, 7));
    continue;
  }
  const n = 2;
  const w = (sx1 - sx0) / n, d = (sz1 - sz0) / n;
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
    if (rnd() < 0.18) { tree(sx0 + w * (a + 0.5), sz0 + d * (b + 0.5), range(3, 4.5), range(4, 6)); continue; }
    const x0 = sx0 + w * a + 1, z0 = sz0 + d * b + 1;
    building(x0, x0 + w - 2, z0, z0 + d - 2, range(6, 30));
  }
}

// street trees along sidewalks
for (const r of roadsAt) for (let t = -HALF + 4; t < HALF; t += 9) {
  if (roadsAt.some((q) => Math.abs(t - q) < 8)) continue;
  for (const side of [-1, 1]) {
    const off = r + side * (ROAD / 2 + WALK / 2);
    tree(off, t, 1.6, 4.5); tree(t, off, 1.6, 4.5);
  }
}

// cars
const car = (x, z, alongX) => {
  const [lx, lz] = alongX ? [4.2, 1.8] : [1.8, 4.2];
  for (let i = 0; i < 70; i++) {
    const top = rnd() < 0.5;
    push(x + range(-lx / 2, lx / 2), top ? 1.5 : range(0.3, 1.5), z + range(-lz / 2, lz / 2), 13);
  }
};
for (const r of roadsAt) for (let k = 0; k < 10; k++) {
  const t = range(-HALF + 3, HALF - 3);
  if (roadsAt.some((q) => Math.abs(t - q) < 6)) continue;
  const lane = r + (rnd() < 0.5 ? -1.6 : 1.6);
  if (rnd() < 0.5) car(t, lane, true); else car(lane, t, false);
}

const out = {
  placeholder: true,
  note: "Synthetic illustrative city block. Not output from the pipeline.",
  classes,
  positions: P,
  labels: L,
};
writeFileSync(new URL("../public/data/segmented_cloud.json", import.meta.url), JSON.stringify(out));
console.log(`${L.length} points`, Object.fromEntries(classes.map((c) => [c.name, L.filter((l) => l === c.id).length])));
