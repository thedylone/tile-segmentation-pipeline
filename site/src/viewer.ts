import {
  AmbientLight,
  BufferGeometry,
  Color,
  DirectionalLight,
  Float32BufferAttribute,
  Group,
  HemisphereLight,
  Mesh,
  MeshLambertMaterial,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { buildScene, CLASSES } from "./scene";

export function mountViewer(stage: HTMLElement): void {
  const status = document.getElementById("viewer-status");
  const legend = document.getElementById("legend");

  // Keep phones light: coarser ground mesh, lower pixel ratio, no MSAA.
  const lite = window.matchMedia("(max-width: 640px), (pointer: coarse)").matches;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const tris = buildScene({ lite });

  // bounds
  const min = new Vector3(Infinity, Infinity, Infinity);
  const max = new Vector3(-Infinity, -Infinity, -Infinity);
  for (const arr of tris.values())
    for (let i = 0; i < arr.length; i += 3) {
      min.set(Math.min(min.x, arr[i]), Math.min(min.y, arr[i + 1]), Math.min(min.z, arr[i + 2]));
      max.set(Math.max(max.x, arr[i]), Math.max(max.y, arr[i + 1]), Math.max(max.z, arr[i + 2]));
    }
  const radius = Math.max(max.x - min.x, max.z - min.z) / 2;
  const height = max.y - min.y || 1;

  const renderer = new WebGLRenderer({ antialias: !lite, alpha: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, lite ? 1.5 : 2));
  renderer.outputColorSpace = SRGBColorSpace;
  stage.appendChild(renderer.domElement);

  const scene = new Scene();
  scene.add(new HemisphereLight(0xffffff, 0xd8d2c0, 1.6));
  scene.add(new AmbientLight(0xffffff, 0.35));
  const sun = new DirectionalLight(0xffffff, 1.9);
  sun.position.set(-0.6, 1, 0.35);
  scene.add(sun);

  const camera = new PerspectiveCamera(36, 1, 0.5, radius * 20);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, height * 0.05, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = radius * 0.5;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 0.5;

  const viewDir = new Vector3(1, 0.85, 1.2).normalize();
  const sphereR = Math.hypot(radius * Math.SQRT2, height * 0.5);
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
  fit();

  // One mesh per class so each can be toggled. Every face gets a slight
  // colour jitter so the individual polygons stay readable.
  let seed = 11;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const group = new Group();
  const layers = new Map<number, Mesh>();
  const tmp = new Color();
  for (const cls of CLASSES) {
    const pos = tris.get(cls.id);
    if (!pos || !pos.length) continue;
    const base = new Color(cls.color);
    const colors = new Float32Array(pos.length);
    for (let f = 0; f < pos.length; f += 9) {
      tmp.copy(base).multiplyScalar(0.93 + rnd() * 0.12);
      for (let v = 0; v < 9; v += 3) {
        colors[f + v] = tmp.r;
        colors[f + v + 1] = tmp.g;
        colors[f + v + 2] = tmp.b;
      }
    }
    const geom = new BufferGeometry();
    geom.setAttribute("position", new Float32BufferAttribute(pos, 3));
    geom.setAttribute("color", new Float32BufferAttribute(colors, 3));
    geom.computeVertexNormals();
    const mesh = new Mesh(geom, new MeshLambertMaterial({ vertexColors: true, flatShading: true }));
    layers.set(cls.id, mesh);
    group.add(mesh);
  }
  group.position.y = -min.y;
  scene.add(group);

  // Legend with per-class toggles.
  if (legend) {
    legend.innerHTML = "";
    for (const cls of CLASSES) {
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
