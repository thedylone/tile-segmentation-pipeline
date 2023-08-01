"""sort vertices by class and export submeshes"""
from collections import defaultdict
import numpy as np
from trimesh import Trimesh, load, Scene
import torch


class MeshSegment:
    """MeshSegment"""

    def __init__(self, mesh: Trimesh) -> None:
        self.mesh: Trimesh = mesh
        self.visual = mesh.visual
        self.vertices_to_class: list[int] = [-1] * len(mesh.vertices)

    @property
    def seg(self) -> np.ndarray:
        """returns segmentation map"""
        return self._seg

    @seg.setter
    def seg(self, seg: np.ndarray) -> None:
        """sets segmentation map"""
        self._seg: np.ndarray = seg
        self._height: int = seg.shape[0]
        self._width: int = seg.shape[1]

    @property
    def visual(self):
        """returns visual"""
        return self._visual

    @visual.setter
    def visual(self, visual) -> None:
        """sets visual"""
        if visual is None:
            raise ValueError("Mesh has no visual")
        self._visual = visual

    def get_vertex_class(self, vertex: int) -> int:
        """returns class of vertex"""
        if self.vertices_to_class[vertex] != -1:
            return self.vertices_to_class[vertex]
        coords = self.visual.uv[vertex]
        col: int = int(coords[0] * self._width) % self._width
        row: int = int((1 - coords[1]) * self._height) % self._height
        class_id = self.seg[row][col].item()
        self.vertices_to_class[vertex] = class_id
        return class_id

    def get_submeshes(self) -> dict:
        """returns a dictionary of submeshes by class"""
        if self._height == 0 or self._width == 0:
            raise ValueError("Segmentation map is empty")
        submeshes = defaultdict(list)
        for i, face in enumerate(self.mesh.faces):
            for vertex in face:
                class_id: int = self.get_vertex_class(vertex)
                submeshes[class_id].append(i)
        return {
            class_id: self.mesh.submesh([submeshes[class_id]], append=False)
            for class_id in submeshes
        }


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
    meshes: list[Trimesh] = load_glb("model.glb")
    for i, mesh in enumerate(meshes):
        mesh = MeshSegment(mesh)
        mesh.seg = seg
        submeshes: dict = mesh.get_submeshes()
        for class_id in submeshes:
            for submesh in submeshes[class_id]:
                submesh.export(f"output/submesh_mesh{i}_{class_id}.glb")
