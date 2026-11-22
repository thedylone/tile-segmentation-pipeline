import {
  BufferGeometry,
  CanvasTexture,
  Color,
  Float32BufferAttribute,
  Group,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Scene,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface ClassInfo {
  id: number;
  name: string;
  color: string;
}

interface CloudData {
  placeholder?: boolean;
  note?: string;
  classes: ClassInfo[];
  positions: number[];
  labels: number[];
}

const DATA_URL = `${import.meta.env.BASE_URL}data/segmented_cloud.json`;

function circleTexture(): CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  g.beginPath();
  g.arc(32, 32, 30, 0, Math.PI * 2);
  g.fillStyle = "#fff";
  g.fill();
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  return t;
}

export async function mountViewer(stage: HTMLElement): Promise<void> {
  const status = document.getElementById("viewer-status");
  const legend = document.getElementById("legend");
  const note = document.getElementById("viewer-note");
  const badge = document.getElementById("placeholder-badge");

  const res = await fetch(DATA_URL);
  if (!res.ok) throw new Error(`Failed to load ${DATA_URL}: ${res.status}`);
  const data = (await res.json()) as CloudData;

  // Keep phones light: fewer points, lower pixel ratio, no MSAA.
  const lite = window.matchMedia("(max-width: 640px), (pointer: coarse)").matches;
  const stride = lite ? 2 : 1;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Centre the cloud on the origin; data uses y-up.
  const pos = data.positions;
  const min = new Vector3(Infinity, Infinity, Infinity);
  const max = new Vector3(-Infinity, -Infinity, -Infinity);
  for (let i = 0; i < pos.length; i += 3) {
    min.min(new Vector3(pos[i], pos[i + 1], pos[i + 2]));
    max.max(new Vector3(pos[i], pos[i + 1], pos[i + 2]));
  }
  const centre = new Vector3().addVectors(min, max).multiplyScalar(0.5);
  centre.y = min.y;
  const radius = Math.max(max.x - min.x, max.z - min.z) / 2;
  const height = max.y - min.y || 1;

  // Bucket points per class so each class can be toggled.
  const buckets = new Map<number, { p: number[]; c: number[] }>();
  for (const cls of data.classes) buckets.set(cls.id, { p: [], c: [] });
  const base = new Map(data.classes.map((c) => [c.id, new Color(c.color)]));
  const tmp = new Color();
  let seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0, n = 0; i < data.labels.length; i++) {
    const label = data.labels[i];
    const b = buckets.get(label);
    const col = base.get(label);
    if (!b || !col) continue;
    if (n++ % stride !== 0) continue;
    const x = pos[i * 3] - centre.x;
    const y = pos[i * 3 + 1] - centre.y;
    const z = pos[i * 3 + 2] - centre.z;
    b.p.push(x, y, z);
    // subtle height + jitter shading so surfaces read as solid
    const shade = 0.82 + 0.18 * Math.min(1, y / height + 0.35) + (rnd() - 0.5) * 0.1;
    tmp.copy(col).multiplyScalar(shade);
    b.c.push(tmp.r, tmp.g, tmp.b);
  }

  const renderer = new WebGLRenderer({ antialias: !lite, alpha: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, lite ? 1.5 : 2));
  renderer.outputColorSpace = SRGBColorSpace;
  stage.appendChild(renderer.domElement);

  const scene = new Scene();
  const camera = new PerspectiveCamera(38, 1, 0.5, radius * 20);
  const viewDir = new Vector3(1, 0.82, 1.15).normalize();
  const sphereR = Math.hypot(radius * Math.SQRT2, height * 0.6);
  let userMoved = false;
  // Fit the bounding sphere in whichever field of view is narrower.
  const fit = () => {
    const vfov = (camera.fov * Math.PI) / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
    const dist = (sphereR / Math.sin(Math.min(vfov, hfov) / 2)) * 0.74;
    camera.position.copy(viewDir).multiplyScalar(dist).add(controls.target);
    controls.maxDistance = dist * 1.8;
    controls.update();
  };

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, height * 0.08, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = radius * 0.6;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 0.6;
  fit();

  const sprite = circleTexture();
  const pointSize = Math.max(0.6, radius / 70);
  const group = new Group();
  const layers = new Map<number, Points>();
  for (const cls of data.classes) {
    const b = buckets.get(cls.id)!;
    if (!b.p.length) continue;
    const geom = new BufferGeometry();
    geom.setAttribute("position", new Float32BufferAttribute(b.p, 3));
    geom.setAttribute("color", new Float32BufferAttribute(b.c, 3));
    const mat = new PointsMaterial({
      size: pointSize * (lite ? 1.25 : 1),
      sizeAttenuation: true,
      vertexColors: true,
      map: sprite,
      alphaTest: 0.5,
      transparent: false,
    });
    const pts = new Points(geom, mat);
    layers.set(cls.id, pts);
    group.add(pts);
  }
  scene.add(group);

  // Legend with per-class toggles.
  if (legend) {
    legend.innerHTML = "";
    for (const cls of data.classes) {
      if (!layers.has(cls.id)) continue;
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.setAttribute("aria-pressed", "true");
      btn.title = `Toggle ${cls.name}`;
      btn.innerHTML = `<span class="swatch" style="background:${cls.color}"></span>${cls.name} <span class="cid">${cls.id}</span>`;
      btn.addEventListener("click", () => {
        const layer = layers.get(cls.id)!;
        layer.visible = !layer.visible;
        btn.setAttribute("aria-pressed", String(layer.visible));
        requestRender();
      });
      li.appendChild(btn);
      legend.appendChild(li);
    }
  }
  if (data.placeholder) {
    if (badge) badge.hidden = false;
    if (note)
      note.textContent =
        "Placeholder data: a synthetic city block coloured with Cityscapes class IDs, the label set the pipeline's model uses. It is not output from the pipeline.";
  }

  const resize = () => {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (!userMoved) fit();
    requestRender();
  };
  new ResizeObserver(resize).observe(stage);

  // Only animate while visible on screen.
  let inView = true;
  let frame = 0;
  let dirty = true;
  function requestRender() {
    dirty = true;
    if (!frame && inView) frame = requestAnimationFrame(tick);
  }
  function tick() {
    frame = 0;
    const moved = controls.update();
    if (moved || dirty || controls.autoRotate) {
      renderer.render(scene, camera);
      dirty = false;
    }
    if (inView && !document.hidden && (controls.autoRotate || moved)) {
      frame = requestAnimationFrame(tick);
    }
  }
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
    userMoved = true;
  });
  controls.addEventListener("change", requestRender);

  new IntersectionObserver((entries) => {
    inView = entries.some((e) => e.isIntersecting);
    if (inView) requestRender();
  }).observe(stage);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) requestRender();
  });

  resize();
  status?.remove();
  stage.dataset.ready = "true";
  requestRender();
}
