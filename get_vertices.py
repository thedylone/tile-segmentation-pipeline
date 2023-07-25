"""get vertices from a glb file - save to a file"""
from collections import defaultdict
import numpy as np
from trimesh import Trimesh, load
import torch


def sort_uv_by_seg_class(mesh, seg) -> defaultdict:
    """returns a dictionary of vertices sorted by class"""
    objects = defaultdict(list)
    visual = mesh.visual
    if not visual:
        raise ValueError("Mesh has no visual")
    uv_map = visual.uv
    width, height = seg.shape[1], seg.shape[0]
    for i, coords in enumerate(uv_map):
        # origin is bottom left
        col = int(coords[0] * width - 1)
        row = int((1 - coords[1]) * height - 1)
        class_id = seg[row][col].item()
        objects[class_id].append(i)
    return objects


def get_submesh_from_class(mesh: Trimesh, objects: defaultdict, class_id):
    """returns a submesh from a class"""
    selected_faces = [
        i
        for i, face in enumerate(mesh.faces)
        if np.any(np.isin(face, objects[class_id]))
    ]
    return mesh.submesh([selected_faces], append=False)


def main() -> None:
    """main"""
    seg = torch.load("map.pt")
    mesh: Trimesh = load("model.glb", force="mesh")
    objects: defaultdict = sort_uv_by_seg_class(mesh, seg)
    for class_id in objects:
        submeshes = get_submesh_from_class(mesh, objects, class_id)
        for i, submesh in enumerate(submeshes):
            submesh.export(f"sub_{class_id}_{i}.glb")


if __name__ == "__main__":
    main()
