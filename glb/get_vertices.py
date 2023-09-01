"""sort vertices by class and export submeshes. for uncompressed glTF files"""
from collections import defaultdict
from pathlib import Path
from trimesh import Trimesh, load, Scene
from tqdm import tqdm


class MeshSegment:
    """MeshSegment"""

    def __init__(self, mesh: Trimesh, index: int = 0) -> None:
        self.mesh: Trimesh = mesh
        self.index: int = index
        self.vertices_to_class: list[int] = [-1] * len(mesh.vertices)

    @property
    def submeshes(self) -> dict[int, Trimesh]:
        """returns submeshes"""
        if not hasattr(self, "_submeshes") or self._submeshes is None:
            self._submeshes: dict[int, Trimesh] = self._get_submeshes()
        return self._submeshes

    @submeshes.setter
    def submeshes(self, submeshes: dict[int, Trimesh]) -> None:
        """sets submeshes"""
        self._submeshes = submeshes

    @staticmethod
    def load_by_path(path: Path) -> list["MeshSegment"]:
        """load glb into list of MeshSegments"""
        glb = load(path)
        if isinstance(glb, Scene):
            return [
                MeshSegment(mesh, i)
                for i, mesh in enumerate(glb.geometry.values())
            ]
        if isinstance(glb, Trimesh):
            return [MeshSegment(glb)]

        raise ValueError("Unsupported GLB type")

    def _get_submeshes(self) -> dict[int, Trimesh]:
        """returns a dictionary of submeshes by class"""
        sub_faces = defaultdict(list)
        for i, face in enumerate(self.mesh.faces):
            for vertex_index in face:
                class_id: int = self.vertices_to_class[vertex_index]
                sub_faces[class_id].append(i)
        submeshes: dict[int, Trimesh] = {}
        for class_id, faces in sub_faces.items():
            submesh = self.mesh.submesh([faces], append=True)
            if not isinstance(submesh, Trimesh):
                raise ValueError("Submesh is not a Trimesh")
            submeshes[class_id] = submesh
        return submeshes

    def export_submeshes(self, path: Path) -> None:
        """exports submeshes by class"""
        submeshes: dict[int, Trimesh] = self.submeshes
        for class_id, submesh in tqdm(
            submeshes.items(), desc="Exporting", unit="submesh"
        ):
            submesh.export(path / f"submesh{self.index}_{class_id}.glb")


def main() -> None:
    """main"""
    meshes: list[MeshSegment] = MeshSegment.load_by_path(Path("model.gltf"))
    for mesh in meshes:
        # ADD VERTICES TO CLASS HERE
        mesh.export_submeshes(Path("output"))


if __name__ == "__main__":
    main()
