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

## Shared kit

`public/project-kit.css` (v1.2) and `public/project-kit.js` (v1.1) are the shared style kit for every
project page under projects.thedylone.com. They come from `public/kit/` on the `projects-gallery` branch
of thedylone/thedylone.github.io. Copy them verbatim and don't edit them here. Page-specific styles live
in `src/style.css`, which loads after the kit and uses only kit tokens.

## Images

The hero and Results images come from the Medium article and live in `public/img/article/`.
`public/img/pipeline-readme.webp` is the diagram from the repository README.

`public/gallery-cover.webp` (960×600) is a screenshot of the top of the page for the projects gallery.
