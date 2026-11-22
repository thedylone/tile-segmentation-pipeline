import "./style.css";

const stage = document.getElementById("stage");
if (stage) {
  // Load three.js only once the viewer is near the viewport.
  const start = () =>
    import("./viewer")
      .then(({ mountViewer }) => mountViewer(stage))
      .catch((err) => {
        console.error(err);
        const status = document.getElementById("viewer-status");
        if (status) status.textContent = "3D viewer unavailable";
      });

  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          start();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(stage);
  } else {
    start();
  }
}
