import { defineConfig } from "vite";

// Served at https://projects.thedylone.com/tile-segmentation-pipeline/
export default defineConfig({
  base: "/tile-segmentation-pipeline/",
  build: { target: "es2020", outDir: "dist" },
});
