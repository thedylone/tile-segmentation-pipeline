from collections import defaultdict
from get_vertices import load_glb, sort_uv_by_seg_class, get_submeshes_by_class
from image_segment import predict_semantic, get_labels
from PIL.Image import Image
from trimesh import Trimesh
import numpy as np
import logging


def segment_glb(path: str) -> None:
    """segment glb"""
    meshes: list[Trimesh] = load_glb(path)
    for i, mesh in enumerate(meshes):
        if mesh.visual is None:
            logging.warning(f"Mesh {i} has no visual")
            continue
        texture: Image = mesh.visual.material.baseColorTexture
        if texture is None:
            logging.warning(f"Mesh {i} has no texture")
            continue
        seg: np.ndarray = predict_semantic(texture)
        objects: defaultdict = sort_uv_by_seg_class(mesh, seg)
        submeshes_dict: dict = get_submeshes_by_class(mesh, objects)
        for class_id in submeshes_dict:
            for submesh in submeshes_dict[class_id]:
                labels: dict = get_labels()
                label = labels.get(class_id, class_id)
                submesh.export(f"output/submesh_mesh{i}_{label}.glb")


if __name__ == "__main__":
    segment_glb("model.gltf")
