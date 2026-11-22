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

`.github/workflows/pages.yml` builds and deploys the page on pushes to `main` that change `site/`.
It can also be run by hand. In the repository settings, set **Pages → Source** to **GitHub Actions**.

## Point cloud data

The viewer loads `public/data/segmented_cloud.json`:

```jsonc
{
  "placeholder": true,             // shows the "Placeholder data" badge and note
  "classes": [{ "id": 2, "name": "building", "color": "#e08a3c" }],
  "positions": [x, y, z, ...],     // y-up, any units
  "labels": [2, ...]               // one class id per point
}
```

The current file is a **synthetic placeholder** made by `npm run make-placeholder`. It is not
pipeline output. To show real results, replace the file with real output in the same schema and
set `placeholder` to `false`.

`public/gallery-cover.webp` (960×600) is a screenshot of the top of the page for the projects gallery.
