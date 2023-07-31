"""sort vertices by class and export submeshes"""
from collections import defaultdict
import numpy as np
from trimesh import Trimesh, load, Scene
import torch


def sort_uv_by_seg_class(mesh: Trimesh, seg: np.ndarray) -> defaultdict:
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


def get_submeshes_by_class(mesh: Trimesh, objects: defaultdict) -> dict:
    """returns a dictionary of submeshes by class"""
    submeshes = defaultdict(list)
    for i, face in enumerate(mesh.faces):
        for class_id in objects:
            if np.any(np.isin(face, objects[class_id])):
                submeshes[class_id].append(i)
    return {
        class_id: mesh.submesh([submeshes[class_id]], append=False)
        for class_id in submeshes
    }
    # selected_faces = [
    #     i
    #     for i, face in enumerate(mesh.faces)
    #     if np.any(np.isin(face, objects[class_id]))
    # ]
    # return mesh.submesh([selected_faces], append=False)
    # TODO: fix uv
    # for submesh in submeshes:
    #     submesh.visual.uv = mesh.visual.[submesh.faces.reshape(-1)].reshape(-1, 2)


def load_glb(path: str) -> list[Trimesh]:
    """load glb into list of meshes"""
    glb = load(path)
    meshes: list[Trimesh] = []
    if isinstance(glb, Scene):
        meshes = list(glb.geometry.values())
    elif isinstance(glb, Trimesh):
        meshes = [glb]
    else:
        raise ValueError("Unsupported GLB type")
    return meshes


if __name__ == "__main__":
    seg = torch.load("map.pt")
    # seg = np.load("map.npy")
    # split_mesh("model.gltf", seg)
    meshes: list[Trimesh] = load_glb("model.glb")
    for i, mesh in enumerate(meshes):
        objects: defaultdict = sort_uv_by_seg_class(mesh, seg)
        submeshes: dict = get_submeshes_by_class(mesh, objects)
        for class_id in submeshes:
            for submesh in submeshes[class_id]:
                submesh.export(f"output/submesh_mesh{i}_{class_id}.glb")
