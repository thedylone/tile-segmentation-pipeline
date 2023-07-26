"""sort vertices by class and export submeshes"""
from collections import defaultdict
import numpy as np
from trimesh import Trimesh, load, Scene
import torch


def sort_uv_by_seg_class(mesh, seg) -> defaultdict:
    """returns a dictionary of vertices sorted by class"""
    objects = defaultdict(list)
    visual = mesh.visual
    if not visual:
        raise ValueError("Mesh has no visual")
    uv_map = visual.uv
    width: int
    height: int
    width, height = seg.shape[1], seg.shape[0]
    for i, coords in enumerate(uv_map):
        # origin is bottom left
        col: int = int(coords[0] * width) % width
        row: int = int((1 - coords[1]) * height) % height
        class_id: int = seg[row][col].item()
        objects[class_id].append(i)  # add vertex index to class
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
    glb = load("model.glb")
    meshes: list[Trimesh] = []
    if isinstance(glb, Scene):
        meshes = list(glb.geometry.values())
    elif isinstance(glb, Trimesh):
        meshes = [glb]
    else:
        raise ValueError("Unsupported GLB type")
    for i, mesh in enumerate(meshes):
        objects: defaultdict = sort_uv_by_seg_class(mesh, seg)
        for class_id in objects:
            submeshes = get_submesh_from_class(mesh, objects, class_id)
            for j, submesh in enumerate(submeshes):
                submesh.export(f"sub_{class_id}_{i}_{j}.glb")


if __name__ == "__main__":
    main()
