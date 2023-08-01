"""pipeline for segmenting glb files"""
from get_vertices import load_glb, MeshSegment
from image_segment import predict_semantic, get_labels
from PIL.Image import Image
from trimesh import Trimesh
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import logging


LOG: logging.Logger = logging.getLogger(__name__)


def segment_glb(path: str) -> None:
    """segment glb"""
    meshes: list[Trimesh] = load_glb(path)
    for i, _mesh in tqdm(
        enumerate(meshes),
        desc="segmenting meshes",
        unit="mesh",
        total=len(meshes),
    ):
        mesh = MeshSegment(_mesh)
        if mesh.visual is None:
            LOG.warning(f"Mesh {i} has no visual")
            continue
        texture: Image = mesh.visual.material.baseColorTexture
        if texture is None:
            LOG.warning(f"Mesh {i} has no texture")
            continue
        LOG.info("Predicting semantic segmentation")
        mesh.seg = predict_semantic(texture)
        # logging.info("Exporting submeshes")
        submeshes: dict = mesh.get_submeshes()
        for class_id in tqdm(submeshes, desc="exporting", unit="submesh"):
            for submesh in submeshes[class_id]:
                labels: dict = get_labels()
                label: str | int = labels.get(class_id, class_id)
                submesh.export(f"output/submesh_mesh{i}_{label}.glb")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        LOG.info("Starting segmentation")
        segment_glb("model.gltf")
        LOG.info("Segmentation complete")
