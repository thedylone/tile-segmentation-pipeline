import DracoPy
import struct
import operator
from collections import defaultdict
import numpy as np
from pathlib import Path
from PIL.Image import Image
from extract_textures import glb_to_pillow
from gltflib.gltf import GLTF
from gltflib import (
    GLTFModel,
    Asset,
    Scene,
    Node,
    Mesh,
    Primitive,
    Attributes,
    Buffer,
    BufferView,
    Accessor,
    AccessorType,
    BufferTarget,
    ComponentType,
    GLBResource,
    FileResource,
)


class MeshSegment:
    """MeshSegment"""

    def __init__(self, path: Path):
        self.glb = GLTF.load(str(path))
        self.vertices_to_class: list[int] = [-1] * len(self._data.points)

    @property
    def glb(self) -> GLTF:
        """returns glb"""
        return self._glb

    @glb.setter
    def glb(self, glb: GLTF) -> None:
        """sets glb"""
        self._glb: GLTF = glb
        self._data = DracoPy.decode(glb.resources[0].data)

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
    def subvertices(self) -> dict[int, list[np.ndarray]]:
        """returns subvertices"""
        if not hasattr(self, "_subvertices") or self._subvertices is None:
            self._subvertices: dict[
                int, list[np.ndarray]
            ] = self._get_subvertices()
        return self._subvertices

    @subvertices.setter
    def subvertices(self, subvertices: dict[int, list[np.ndarray]]) -> None:
        """sets subvertices"""
        self._subvertices = subvertices

    def _get_vertex_class(self, vertex: int) -> int:
        """returns class of vertex"""
        if self.vertices_to_class[vertex] != -1:
            return self.vertices_to_class[vertex]
        coords = self._data.tex_coord[vertex]
        col: int = int(coords[0] * self._width) % self._width
        row: int = int((1 - coords[1]) * self._height) % self._height
        class_id = self.seg[row][col].item()
        self.vertices_to_class[vertex] = class_id
        return class_id

    def _get_subvertices(self) -> defaultdict[int, list[np.ndarray]]:
        """returns subvertices"""
        sub_vertices = defaultdict(list)
        for face in self._data.faces:
            for vertex in face:
                class_id: int = self._get_vertex_class(vertex)
                sub_vertices[class_id].append(self._data.points[vertex])
        return sub_vertices


def main():
    mesh = MeshSegment(Path("model.glb"))
    # textures: list[Image] = glb_to_pillow(mesh.glb)
    mesh.seg = np.load("map.npy")
    for class_id, subvertices in mesh.subvertices.items():
        vertex_bytearray = bytearray()
        for vertex in subvertices:
            for value in vertex:
                vertex_bytearray.extend(struct.pack("f", value))
        bytelen = len(vertex_bytearray)
        mins = [
            min([operator.itemgetter(i)(vertex) for vertex in subvertices])
            for i in range(3)
        ]
        maxs = [
            max([operator.itemgetter(i)(vertex) for vertex in subvertices])
            for i in range(3)
        ]
        model = GLTFModel(
            asset=Asset(version="2.0"),
            scenes=[Scene(nodes=[0])],
            nodes=[Node(mesh=0)],
            meshes=[
                Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0))])
            ],
            buffers=[Buffer(byteLength=bytelen, uri="vertices.bin")],
            bufferViews=[
                BufferView(
                    buffer=0,
                    byteOffset=0,
                    byteLength=bytelen,
                    target=BufferTarget.ARRAY_BUFFER.value,
                )
            ],
            accessors=[
                Accessor(
                    bufferView=0,
                    byteOffset=0,
                    componentType=ComponentType.FLOAT.value,
                    count=len(subvertices),
                    type=AccessorType.VEC3.value,
                    min=mins,
                    max=maxs,
                )
            ],
        )
        resource = FileResource("vertices.bin", data=vertex_bytearray)
        glb = GLTF(model=model, resources=[resource])
        glb.export(f"model_{class_id}.glb")


if __name__ == "__main__":
    main()
