"""segment glb with draco compression"""
import argparse
import io
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from typing import Optional, Union

import DracoPy
import numpy as np
from gltflib import (
    Accessor,
    AccessorType,
    Attributes,
    Buffer,
    BufferTarget,
    BufferView,
    ComponentType,
    Primitive,
    Material,
    Image,
    Texture,
)
from gltflib.gltf import GLTF
from gltflib.gltf_resource import GLTFResource, GLBResource, FileResource
from PIL import Image as PIL_Image
from tqdm import tqdm
from trimesh import Trimesh
from trimesh.visual import TextureVisuals


@dataclass
class MeshData:
    """MeshData"""

    points: np.ndarray
    faces: np.ndarray
    tex_coord: np.ndarray

    def all(self) -> bool:
        """returns true if all data is present"""
        return all(
            [
                self.points.size > 0,
                self.faces.size > 0,
                self.tex_coord.size > 0,
            ]
        )

    def to_trimesh(self) -> Trimesh:
        """returns trimesh"""
        return Trimesh(
            vertices=self.points,
            faces=self.faces,
            texture_coords=self.tex_coord,
        )


class SubPrimitiveSeg:
    """SubPrimitiveSeg"""

    points: np.ndarray
    tex_coord: np.ndarray

    def __init__(self) -> None:
        self.vertices: list[np.ndarray] = []
        self.faces: list[np.ndarray] = []
        self.uv_coords: list[np.ndarray] = []
        self.vertex_map: dict[int, int] = {}

    def add_face(self, face: np.ndarray) -> None:
        """adds face"""
        new_face: list[int] = []
        for vertex in face:
            if vertex not in self.vertex_map:
                self.vertex_map[vertex] = len(self.vertices)
                self.vertices.append(self.points[vertex])
                self.uv_coords.append(self.tex_coord[vertex])
            new_face.append(self.vertex_map[vertex])
        self.faces.append(np.array(new_face))

    def to_dict(self) -> dict[str, np.ndarray]:
        """returns dict"""
        return {
            "vertices": np.array(self.vertices),
            "faces": np.array(self.faces),
            "uv_coords": np.array(self.uv_coords),
        }

    def to_mesh_data(self) -> MeshData:
        """returns mesh data"""
        return MeshData(
            points=np.array(self.vertices),
            faces=np.array(self.faces),
            tex_coord=np.array(self.uv_coords),
        )

    def to_trimesh(self) -> Trimesh:
        """returns trimesh"""
        return Trimesh(
            vertices=np.array(self.vertices),
            faces=np.array(self.faces),
            texture_coords=np.array(self.uv_coords),
        )


class BufferAccessor:
    """BufferAccessor"""

    glb: GLTF

    def get_accessor(
        self, accessor_index: Optional[int]
    ) -> Optional[Accessor]:
        """returns accessor"""
        if accessor_index is None:
            return None
        return self._get_accessor(accessor_index)

    def _get_accessor(self, accessor_index: int) -> Accessor:
        """returns accessor"""
        accessors: Optional[list[Accessor]] = self.glb.model.accessors
        if accessors is None:
            raise ValueError("No accessors found")
        if accessor_index >= len(accessors):
            raise ValueError("Accessor index out of range")
        return accessors[accessor_index]

    def get_buffer_data(self, buffer: Buffer) -> bytes:
        """get the data from a buffer"""
        resource: Union[GLBResource, GLTFResource] = (
            self.glb.get_glb_resource()
            if buffer.uri is None
            else self.glb.get_resource(buffer.uri)
        )
        if isinstance(resource, FileResource):
            resource.load()
        return resource.data

    def retrieve_bufferview(self, buffer_view_index: int) -> bytes:
        """returns buffer data"""
        buffer_views: Optional[list[BufferView]] = self.glb.model.bufferViews
        if buffer_views is None:
            raise ValueError("No buffer views found")
        if buffer_view_index >= len(buffer_views):
            raise ValueError("Buffer view index out of range")
        buffer_view: BufferView = buffer_views[buffer_view_index]
        buffers: Optional[list[Buffer]] = self.glb.model.buffers
        if buffers is None:
            raise ValueError("No buffers found")
        buffer: Buffer = buffers[buffer_view.buffer]
        data: bytes = self.get_buffer_data(buffer)
        start: int = buffer_view.byteOffset or 0
        end: int = start + buffer_view.byteLength
        return data[start:end]

    def access_buffer(self, accessor_index: Optional[int]) -> bytes:
        """returns buffer data"""
        if accessor_index is None:
            return b""
        accessor: Accessor = self._get_accessor(accessor_index)
        if accessor.bufferView is None:
            raise ValueError("No buffer view found")
        return self.retrieve_bufferview(accessor.bufferView)


class _Attributes(BufferAccessor):
    """Attributes of a primitive, containing accessors to the data"""

    def __init__(self, attributes: Attributes) -> None:
        self.attributes: Attributes = attributes

    @property
    def position(self) -> Optional[Accessor]:
        """returns position"""
        return self.get_accessor(self.attributes.POSITION)

    @property
    def normal(self) -> Optional[Accessor]:
        """returns normal"""
        return self.get_accessor(self.attributes.NORMAL)

    @property
    def tangent(self) -> Optional[Accessor]:
        """returns tangent"""
        return self.get_accessor(self.attributes.TANGENT)

    @property
    def texcoord_0(self) -> Optional[Accessor]:
        """returns texcoord_0"""
        return self.get_accessor(self.attributes.TEXCOORD_0)

    @property
    def texcoord_1(self) -> Optional[Accessor]:
        """returns texcoord_1"""
        return self.get_accessor(self.attributes.TEXCOORD_1)

    @property
    def color_0(self) -> Optional[Accessor]:
        """returns color_0"""
        return self.get_accessor(self.attributes.COLOR_0)

    @property
    def joints_0(self) -> Optional[Accessor]:
        """returns joints_0"""
        return self.get_accessor(self.attributes.JOINTS_0)

    @property
    def weights_0(self) -> Optional[Accessor]:
        """returns weights_0"""
        return self.get_accessor(self.attributes.WEIGHTS_0)


class PrimitiveSeg(BufferAccessor):
    """PrimitiveSeg"""

    def __init__(self, primitive: Primitive) -> None:
        self.primitive: Primitive = primitive
        self.attributes: _Attributes = _Attributes(primitive.attributes)
        self.indices: Optional[Accessor] = self.get_accessor(primitive.indices)
        self.data: MeshData = self._get_data()
        self.vertices_to_class: list[int] = [-1] * self.data.points.shape[0]
        self._subprimitives: dict[int, SubPrimitiveSeg]
        self._material: Optional[Material]

    @property
    def subprimitives(self) -> dict[int, SubPrimitiveSeg]:
        """returns submeshes"""
        if not hasattr(self, "_subprimitives") or self._subprimitives is None:
            self._subprimitives = self._get_subprimitives()
        return self._subprimitives

    @subprimitives.setter
    def subprimitives(self, subprimitives: dict[int, SubPrimitiveSeg]) -> None:
        """sets submeshes"""
        self._subprimitives = subprimitives

    @property
    def material(self) -> Optional[Material]:
        """returns material"""
        if not hasattr(self, "_material"):
            self._material = self.get_material(self.primitive.material)
        return self._material

    def get_material(
        self, material_index: Optional[int]
    ) -> Optional[Material]:
        """sets material"""
        if material_index is None:
            return None
        materials: Optional[list[Material]] = self.glb.model.materials
        if materials is None:
            raise ValueError("No materials found")
        if material_index >= len(materials):
            raise ValueError("Material index out of range")
        return materials[material_index]

    def get_texture_image_bytes(self) -> Optional[bytes]:
        """returns texture image bytes"""
        if (
            self.material is None
            or self.material.pbrMetallicRoughness is None
            or self.material.pbrMetallicRoughness.baseColorTexture is None
        ):
            return None
        texture_index: int = (
            self.material.pbrMetallicRoughness.baseColorTexture.index
        )
        textures: Optional[list[Texture]] = self.glb.model.textures
        if textures is None or texture_index >= len(textures):
            return None
        texture: Texture = textures[texture_index]
        if texture.source is None:
            return None
        image_index: int = texture.source
        images: Optional[list[Image]] = self.glb.model.images
        if images is None or image_index >= len(images):
            return None
        image: Image = images[image_index]
        if image.bufferView is None:
            return None
        return self.retrieve_bufferview(image.bufferView)

    def get_texture_image(self) -> Optional[PIL_Image.Image]:
        """returns texture image"""
        data: Optional[bytes] = self.get_texture_image_bytes()
        if data is None:
            return None
        return PIL_Image.open(io.BytesIO(data))

    def _get_data(self) -> MeshData:
        data: MeshData = self.retrieve_data()
        if data.all():
            return data
        try_data = self.try_finding_data()
        if try_data:
            return try_data
        raise ValueError("No data found")

    def retrieve_data(self) -> MeshData:
        """retrieves data from attributes accessors"""
        points: np.ndarray = np.array([])  # vec3 float
        faces: np.ndarray = np.array([])  # vec3 int
        tex: np.ndarray = np.array([])  # vec2 float
        attr: _Attributes = self.attributes
        if attr.position and attr.position.bufferView:
            index: int = attr.position.bufferView
            buffer: bytes = self.access_buffer(index)
            points = np.frombuffer(buffer, dtype=np.float32).reshape(-1, 3)
        if self.indices and self.indices.bufferView:
            index = self.indices.bufferView
            buffer: bytes = self.access_buffer(index)
            faces = np.frombuffer(buffer, dtype=np.int32).reshape(-1, 3)
        if attr.texcoord_0 and attr.texcoord_0.bufferView:
            index = attr.texcoord_0.bufferView
            buffer: bytes = self.access_buffer(index)
            tex = np.frombuffer(buffer, dtype=np.float32).reshape(-1, 2)
        return MeshData(points, faces, tex)

    def try_finding_data(self):
        """tries to match decoded buffer data with accessor counts"""
        if self.glb.model.bufferViews is None:
            return None
        if (
            self.indices is None
            or self.attributes.position is None
            or self.attributes.texcoord_0 is None
        ):
            return None
        for index in range(len(self.glb.model.bufferViews)):
            data: bytes = self.retrieve_bufferview(index)
            # check if data is Draco compressed
            if data[:4] != b"DRAC":
                continue
            decoded = DracoPy.decode(data)
            if decoded is None:
                continue
            # compare decoded data with accessor counts
            # faces to indices
            if decoded.faces.size != self.indices.count:
                continue
            # points to positions
            if decoded.points.size != self.attributes.position.count * 3:
                continue
            # texture coordinates to texcoord_0
            if decoded.tex_coord.size != self.attributes.texcoord_0.count * 2:
                continue
            # if all checks passed, return decoded data
            return decoded
        return None

    def _get_subprimitives(self) -> dict[int, SubPrimitiveSeg]:
        """returns subprimitives"""
        if self.data is None:
            raise ValueError("No data found")
        if (
            self.data.points is None
            or self.data.faces is None
            or self.data.tex_coord is None
        ):
            raise ValueError("No data found")
        SubPrimitiveSeg.points = self.data.points
        SubPrimitiveSeg.tex_coord = self.data.tex_coord
        submeshes = defaultdict(SubPrimitiveSeg)
        for face in tqdm(
            self.data.faces,
            desc="Loading faces",
            unit="face",
            leave=False,
        ):
            for vertex in face:
                class_id: int = self.vertices_to_class[vertex]
                submeshes[class_id].add_face(face)
        return submeshes

    def export_subprimitives(self, output_dir: Path) -> None:
        """exports separate subprimitives by class"""
        for class_id, subprimitive in tqdm(
            self.subprimitives.items(), desc="Exporting", unit="subprimitive"
        ):
            subtrimesh: Trimesh = subprimitive.to_trimesh()
            subtrimesh.visual = TextureVisuals(
                uv=np.array(subprimitive.uv_coords),
                image=self.get_texture_image(),
            )
            subtrimesh.export(f"{output_dir}_{class_id}.glb")

    def export_subprimitive(
        self, subprimitive: SubPrimitiveSeg, path: Path
    ) -> None:
        """exports separate subprimitives by class"""
        subtrimesh: Trimesh = subprimitive.to_trimesh()
        subtrimesh.visual = TextureVisuals(
            uv=np.array(subprimitive.uv_coords),
            image=self.get_texture_image(),
        )
        Path.mkdir(path.parent, parents=True, exist_ok=True)
        subtrimesh.export(path)


class GLBSegment(BufferAccessor):
    """GLBSegment"""

    def __init__(self, path: Path) -> None:
        BufferAccessor.glb = GLTF.load(str(path))
        self.meshes: list[list[PrimitiveSeg]] = []

    def load_meshes(self) -> None:
        """loads meshes"""
        if self.glb.model.meshes is None:
            return
        for mesh in tqdm(
            self.glb.model.meshes,
            desc="Loading meshes",
            unit="mesh",
            leave=False,
        ):
            primitives: list[PrimitiveSeg] = []
            for primitive in tqdm(
                mesh.primitives,
                desc="Loading primitives",
                unit="primitive",
                leave=False,
            ):
                primitives.append(PrimitiveSeg(primitive))
            self.meshes.append(primitives)

    def export(self, path: Path) -> None:
        """exports glb with metadata"""
        if self.glb.model.meshes is None:
            return
        if self.glb.model.buffers is None:
            self.glb.model.buffers = []
        if self.glb.model.bufferViews is None:
            self.glb.model.bufferViews = []
        if self.glb.model.accessors is None:
            self.glb.model.accessors = []
        if self.glb.model.extensionsUsed is None:
            self.glb.model.extensionsUsed = []
        for mesh in tqdm(
            self.meshes,
            desc="Exporting mesh",
            unit="mesh",
            leave=False,
        ):
            for primitive_seg in mesh:
                primitive: Primitive = primitive_seg.primitive
                feature_bytearray = bytearray()
                for vertex in primitive_seg.vertices_to_class:
                    feature_bytearray.extend(struct.pack("f", vertex))
                feature_bytelen: int = len(feature_bytearray)
                if primitive.extensions is None:
                    primitive.extensions = {}
                primitive.extensions["EXT_mesh_features"] = {
                    "featureIds": [
                        {
                            "featureCount": len(
                                set(primitive_seg.vertices_to_class)
                            ),
                            "attribute": 0,
                        }
                    ]
                }
                # TO DO: FIX THIS
                # THIS REQUIRES ADDING _FEATURE_ID_0 TO THE ATTRIBUTES CLASS
                primitive.attributes._FEATURE_ID_0 = len(
                    self.glb.model.accessors
                )
                self.glb.model.accessors.append(
                    Accessor(
                        bufferView=len(self.glb.model.bufferViews),
                        byteOffset=0,
                        componentType=ComponentType.FLOAT.value,
                        count=primitive_seg.data.points.shape[0],
                        type=AccessorType.SCALAR.value,
                        normalized=False,
                    )
                )
                self.glb.model.bufferViews.append(
                    BufferView(
                        buffer=0,
                        byteOffset=len(self.glb.get_glb_resource().data),
                        byteLength=feature_bytelen,
                        byteStride=4,
                        target=BufferTarget.ARRAY_BUFFER.value,
                    )
                )
                self.glb.model.buffers[0].byteLength += feature_bytelen
                self.glb.glb_resources[0].data += feature_bytearray
        self.glb.model.extensionsUsed.append("EXT_mesh_features")
        Path.mkdir(path.parent, parents=True, exist_ok=True)
        self.glb.export(str(path))

    def export_submeshes(self, output_dir: Path) -> None:
        """exports submeshes by class"""
        for i, mesh in enumerate(self.meshes):
            for primitive_seg in tqdm(
                mesh,
                desc="Exporting mesh",
                unit="mesh",
                leave=False,
            ):
                primitive_seg.export_subprimitives(output_dir / f"mesh_{i}")


def main(glb_path: Path, output_dir: Path, submeshes: bool) -> None:
    """main"""
    glb = GLBSegment(glb_path)
    glb.load_meshes()
    # ADD VERTICES TO CLASS HERE
    for mesh in glb.meshes:
        for primitive in mesh:
            primitive.vertices_to_class = np.random.randint(
                0, 10, primitive.data.points.shape[0]
            ).tolist()
    if submeshes:
        glb.export_submeshes(output_dir)
    else:
        glb.export(output_dir / glb_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment GLB")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="GLB file to segment",
        default="model.glb",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory",
        default="output",
    )
    parser.add_argument(
        "-s",
        "--submeshes",
        action="store_true",
        help="Export submeshes",
    )
    args: argparse.Namespace = parser.parse_args()
    main(Path(args.file), Path(args.output_dir), args.submeshes)
