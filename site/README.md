# Project page

The GitHub Pages site for this repository, served at
<https://projects.thedylone.com/tile-segmentation-pipeline/>. It is built with Vite and
TypeScript and is separate from the Python pipeline.

```bash
cd site
npm install
npm run dev      # local dev server
npm run build    # production build in site/dist
```

`.github/workflows/pages.yml` builds and deploys the page on pushes to `main`
that change `site/`. It can also be run by hand. In the repository settings, set **Pages → Source** to
**GitHub Actions**.

## 3D viewer

The hero viewer renders a **synthetic placeholder scene** generated in `src/scene.ts`: a procedural
city block made of polygons, each labelled with a Cityscapes class ID. The page labels it as a
placeholder. It is not pipeline output. The real results in the Results section are images from the
Medium article, in `public/img/article/`.

To show a real tile instead, replace `buildScene()` with per-class triangle data taken from a
segmented `.glb` file (for example, the `class_{id}.glb` submeshes from `--submeshes`).

`public/gallery-cover.webp` (960×600) is a screenshot of the top of the page for the projects gallery.
