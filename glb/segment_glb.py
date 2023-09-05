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
from gltflib import GLTFModel, Asset, Mesh
from gltflib.gltf import GLTF
from gltflib.gltf_resource import GLTFResource, GLBResource, FileResource
from PIL import Image as PIL_Image
from tqdm import tqdm


@dataclass
class MeshData:
    """dataclass storing mesh data

    parameters
    ----------
    points: np.ndarray
        array containing the x, y, and z coordinates of each point
    faces: np.ndarray
        array containing the vertex indices of each face
    tex_coord: np.ndarray
        array containing the u and v coordinates of each texture coordinate

    attributes
    ----------
    points: np.ndarray
        array containing the x, y, and z coordinates of each point
    faces: np.ndarray
        array containing the vertex indices of each face
    tex_coord: np.ndarray
        array containing the u and v coordinates of each texture coordinate

    """

    points: np.ndarray
    """array containing the x, y, and z coordinates of each point"""

    faces: np.ndarray
    """array containing the vertex indices of each face"""

    tex_coord: np.ndarray
    """array containing the u and v coordinates of each texture coordinate"""

    def all(self) -> bool:
        """checks if all data is present

        returns
        -------
        bool
            True if all data is present, False otherwise

        """
        return all(
            [
                self.points.size > 0,
                self.faces.size > 0,
                self.tex_coord.size > 0,
            ]
        )


class SubPrimitiveSeg:
    """a subset of a primitive

    class attributes `points` and `tex_coord` are used to store the data of the
    parent primitive.

    attributes
    ----------
    vertices: list[np.ndarray]
        list of coordinates of each vertex
    faces: list[np.ndarray]
        list of vertex indices of each face in the subprimitive
    uv_coords: list[np.ndarray]
        list of texture coordinates of each vertex
    vertex_map: dict[int, int]
        dictionary mapping parent vertex indices to subprimitive vertex indices

    examples
    --------
    >>> SubPrimitiveSeg.points = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]
        ])
    >>> SubPrimitiveSeg.tex_coord = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]
        ])
    >>> subprimitive = SubPrimitiveSeg()
    >>> subprimitive.add_face(np.array([0, 2, 4]))
    >>> subprimitive.to_dict()
    {
        "vertices": np.array([
            [0, 0, 0], [0, 1, 0], [1, 0, 0]
            ]),
        "faces": np.array([
            [0, 1, 2]
            ]),
        "uv_coords": np.array([
            [0, 0], [0, 1], [0.5, 0.5]
            ])
    }

    """

    points: np.ndarray
    """points of the parent primitive"""

    tex_coord: np.ndarray
    """texture coordinates of the parent primitive"""

    def __init__(self) -> None:
        self.vertices: list[np.ndarray] = []
        """list of coordinates of each vertex"""

        self.faces: list[np.ndarray] = []
        """list of vertex indices of each face in the subprimitive"""

        self.uv_coords: list[np.ndarray] = []
        """list of texture coordinates of each vertex"""

        self.vertex_map: dict[int, int] = {}
        """dictionary mapping parent vertex indices to subprimitive vertex"""

    def add_face(self, face: np.ndarray) -> None:
        """adds a face from the parent primitive to the subprimitive

        appends vertices, faces, and texture coordinates with updated
        vertex indices for the subprimitive

        parameters
        ----------
        face: np.ndarray
            face from the parent primitive containing vertex indices

        """
        new_face: list[int] = []
        for vertex in face:
            if vertex not in self.vertex_map:
                self.vertex_map[vertex] = len(self.vertices)
                self.vertices.append(self.points[vertex])
                self.uv_coords.append(self.tex_coord[vertex])
            new_face.append(self.vertex_map[vertex])
        self.faces.append(np.array(new_face))

    def to_dict(self) -> dict[str, np.ndarray]:
        """converts to a dictionary containing numpy arrays of vertices, faces,
        and texture coordinates

        returns
        -------
        dict[str, np.ndarray]
            dictionary containing numpy arrays of vertices, faces, and texture
            coordinates

        """
        return {
            "vertices": np.array(self.vertices),
            "faces": np.array(self.faces),
            "uv_coords": np.array(self.uv_coords),
        }

    def to_mesh_data(self) -> MeshData:
        """converts to a MeshData object

        returns
        -------
        MeshData
            MeshData object containing vertices, faces, and texture coordinates

        """
        return MeshData(
            points=np.array(self.vertices),
            faces=np.array(self.faces),
            tex_coord=np.array(self.uv_coords),
        )


class BufferAccessor:
    """accesses buffer data contained in a glb

    attributes
    ----------
    glb: gltflib.GLTF
        the glTF object to access

    examples
    --------
    >>> BufferAccessor.glb = GLTF.load("model.glb")
    >>> BufferAccessor.get_accessor(0)
    Accessor(...)
    >>> BufferAccessor.retrieve_bufferview(0)
    b"..."
    >>> BufferAccessor.access_buffer(0)
    b"..."

    """

    glb: GLTF
    """the glTF object to access"""

    def get_accessor(
        self, accessor_index: Optional[int]
    ) -> Optional[Accessor]:
        """retrieve the accessor by accessor index

        parameters
        ----------
        accessor_index: int or None
            index of the accessor

        returns
        -------
        gltflib.Accessor or None
            accessor in the glTF if found, None otherwise

        """
        if accessor_index is None:
            return None
        return self._get_accessor(accessor_index)

    def _get_accessor(self, accessor_index: int) -> Accessor:
        """retrieve the accessor by accessor index

        parameters
        ----------
        accessor_index: int
            index of the accessor

        returns
        -------
        Accessor
            accessor in the glTF

        raises
        ------
        ValueError
            if no accessors are found
        ValueError
            if accessor index is out of range

        """
        accessors: Optional[list[Accessor]] = self.glb.model.accessors
        if accessors is None:
            raise ValueError("No accessors found")
        if accessor_index >= len(accessors):
            raise ValueError("Accessor index out of range")
        return accessors[accessor_index]

    def get_buffer_data(self, buffer: Buffer) -> bytes:
        """retrieve the data from a buffer

        parameters
        ----------
        buffer: gltflib.Buffer
            buffer to retrieve data from

        returns
        -------
        bytes
            data of the buffer

        """
        resource: Union[GLBResource, GLTFResource] = (
            self.glb.get_glb_resource()
            if buffer.uri is None
            else self.glb.get_resource(buffer.uri)
        )
        if isinstance(resource, FileResource):
            resource.load()
        return resource.data

    def retrieve_bufferview(self, buffer_view_index: int) -> bytes:
        """retrieve the data from a buffer referenced from a buffer view by
        index of the buffer view in the glTF

        parameters
        ----------
        buffer_view_index: int
            index of the buffer view in the glTF

        returns
        -------
        bytes
            data of the buffer referenced from the buffer view
        """
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
        """retrieve the data from a buffer referenced from an accessor by
        index of the accessor in the glTF

        parameters
        ----------
        accessor_index: int or None
            index of the accessor in the glTF

        returns
        -------
        bytes
            data of the buffer referenced from the accessor

        """
        if accessor_index is None:
            return b""
        accessor: Accessor = self._get_accessor(accessor_index)
        if accessor.bufferView is None:
            raise ValueError("No buffer view found")
        return self.retrieve_bufferview(accessor.bufferView)


class _Attributes(BufferAccessor):
    """Attributes of a primitive, containing accessors to the data

    attributes
    ----------
    attributes: gltflib.Attributes
        the raw attributes object containing the accessor indices
    position: gltflib.Accessor or None
        accessor to the position data
    normal: gltflib.Accessor or None
        accessor to the normal data
    tangent: gltflib.Accessor or None
        accessor to the tangent data
    texcoord_0: gltflib.Accessor or None
        accessor to the texture coordinate 0 data
    texcoord_1: gltflib.Accessor or None
        accessor to the texture coordinate 1 data
    color_0: gltflib.Accessor or None
        accessor to the color data
    joints_0: gltflib.Accessor or None
        accessor to the joint data
    weights_0: gltflib.Accessor or None
        accessor to the weight data

    """

    def __init__(self, attributes: Attributes) -> None:
        self.attributes: Attributes = attributes
        """the raw attributes object containing the accessor indices"""

    @property
    def position(self) -> Optional[Accessor]:
        """retrieve the position data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the position data if found, None otherwise

        """
        return self.get_accessor(self.attributes.POSITION)

    @property
    def normal(self) -> Optional[Accessor]:
        """retrieve the normal data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the normal data if found, None otherwise
        """
        return self.get_accessor(self.attributes.NORMAL)

    @property
    def tangent(self) -> Optional[Accessor]:
        """retrieve the tangent data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the tangent data if found, None otherwise
        """
        return self.get_accessor(self.attributes.TANGENT)

    @property
    def texcoord_0(self) -> Optional[Accessor]:
        """retrieve the texture coordinate 0 data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the texture coordinate 0 data if found, None otherwise

        """
        return self.get_accessor(self.attributes.TEXCOORD_0)

    @property
    def texcoord_1(self) -> Optional[Accessor]:
        """retrieve the texture coordinate 1 data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the texture coordinate 1 data if found, None otherwise
        """
        return self.get_accessor(self.attributes.TEXCOORD_1)

    @property
    def color_0(self) -> Optional[Accessor]:
        """retrieve the color data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the color data if found, None otherwise

        """
        return self.get_accessor(self.attributes.COLOR_0)

    @property
    def joints_0(self) -> Optional[Accessor]:
        """retrieve the joints data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the joint data if found, None otherwise

        """
        return self.get_accessor(self.attributes.JOINTS_0)

    @property
    def weights_0(self) -> Optional[Accessor]:
        """retrieve the weights data accessor

        returns
        -------
        gltflib.Accessor or None
            accessor to the weight data if found, None otherwise

        """
        return self.get_accessor(self.attributes.WEIGHTS_0)


class PrimitiveSeg(BufferAccessor):
    """loads a primitive and sorts vertices by class. exports submeshes
    by class into separate glb files

    parameters
    ----------
    primitive: gltflib.Primitive
        primitive to load and segment

    attributes
    ----------
    primitive: gltflib.Primitive
        primitive to load and segment
    attributes: _Attributes
        attributes of the primitive containing accessors to the data
    indices: gltflib.Accessor or None
        accessor to the indices data
    data: MeshData
        data of the primitive containing vertices, faces,
        and texture coordinates
    vertices_to_class: list[int]
        list of classes for each vertex
    subprimitives: dict[int, SubPrimitiveSeg]
        dictionary of subprimitives sorted by class
    material: gltflib.Material or None
        material of the primitive

    examples
    --------
    >>> primitive_seg = PrimitiveSeg(primitive)
    >>> primitive_seg.vertices_to_class = list_of_classes
    >>> primitive_seg.export_subprimitives(Path("output"))

    """

    def __init__(self, primitive: Primitive) -> None:
        self.primitive: Primitive = primitive
        """primitive to load and segment"""

        self.attributes: _Attributes = _Attributes(primitive.attributes)
        """attributes of the primitive containing accessors to the data"""

        self.indices: Optional[Accessor] = self.get_accessor(primitive.indices)
        """accessor to the indices data"""

        self.data: MeshData = self._get_data()
        """data of the primitive containing vertices, faces, and texture
        coordinates"""

        self.vertices_to_class: list[int] = [-1] * self.data.points.shape[0]
        """list of classes for each vertex"""

        self._subprimitives: dict[int, SubPrimitiveSeg]
        self._material: Optional[Material]

    @property
    def subprimitives(self) -> dict[int, SubPrimitiveSeg]:
        """dictionary of subprimitives sorted by class

        subprimitives are only generated once when this property is accessed

        returns
        -------
        dict[int, SubPrimitiveSeg]
            dictionary of subprimitives by class

        """
        if not hasattr(self, "_subprimitives") or self._subprimitives is None:
            self.load_subprimitives()
        return self._subprimitives

    @subprimitives.setter
    def subprimitives(self, subprimitives: dict[int, SubPrimitiveSeg]) -> None:
        self._subprimitives = subprimitives

    @property
    def material(self) -> Optional[Material]:
        """material of the primitive if found

        returns
        -------
        gltflib.Material or None
            material of the primitive if found, None otherwise

        """
        if not hasattr(self, "_material"):
            self._material = self.get_material(self.primitive.material)
        return self._material

    def get_material(
        self, material_index: Optional[int]
    ) -> Optional[Material]:
        """retrieve the material in the glTF by index

        parameters
        ----------
        material_index: int or None
            index of the material in the glTF

        returns
        -------
        gltflib.Material or None
            material in the glTF if found, None otherwise

        """
        if material_index is None:
            return None
        materials: Optional[list[Material]] = self.glb.model.materials
        if materials is None:
            raise ValueError("No materials found")
        if material_index >= len(materials):
            raise ValueError("Material index out of range")
        return materials[material_index]

    def get_texture_image_bytes(self) -> Optional[bytes]:
        """retrieve the bytes data of the texture image from self.material

        self.material should be set before calling this method. uses
        pbrMetallicRoughness.baseColorTexture.index to retrieve the texture

        returns
        -------
        bytes or None
            bytes data of the texture image if found, None otherwise

        """
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
        """retrieve the texture image from self.material

        self.material should be set before calling this method. uses
        pbrMetallicRoughness.baseColorTexture.index to retrieve the texture

        returns
        -------
        PIL.Image.Image or None
            texture image if found, None otherwise

        """
        data: Optional[bytes] = self.get_texture_image_bytes()
        if data is None:
            return None
        return PIL_Image.open(io.BytesIO(data))

    def _get_data(self) -> MeshData:
        """attempts to retrieve data from attributes accessors and Draco
        compression

        tries to retrieve data from attributes accessors first. if no data is
        found, tries to retrieve data from Draco compression. if no data is
        found, raises ValueError

        returns
        -------
        MeshData
            data of the primitive containing vertices, faces,
            and texture coordinates

        """
        data: MeshData = self.retrieve_data()
        if data.all():
            return data
        try_data = self.try_finding_data()
        if try_data:
            return try_data
        raise ValueError("No data found")

    def retrieve_data(self) -> MeshData:
        """retrieves data from attributes accessors and indices accessor

        tries to retrieve data from accessors. if no data is found, returns
        empty arrays for points, faces, and texture coordinates in MeshData

        returns
        -------
        MeshData
            data of the primitive containing vertices, faces,
            and texture coordinates

        """
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
        """tries to find data from Draco decompression and matches the data
        with the counts of each property from the accessors

        returns
        -------
        MeshData or None
            data of the primitive containing vertices, faces,
            and texture coordinates if found, None otherwise

        """
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

    def load_subprimitives(self) -> None:
        """loads subprimitives and saves them to the _subprimitives attribute

        if vertices_to_class is changed, this method must be called again to
        update the subprimitives

        """
        self._subprimitives = self._get_subprimitives()

    def _get_subprimitives(self) -> dict[int, SubPrimitiveSeg]:
        """returns a dictionary of subprimitives sorted by class

        avoid calling this method directly. use the subprimitves property
        instead to save the subprimitves to the subprimitves attribute, or use
        the _load_subprimitves method to update the subprimitves

        returns
        -------
        dict[int, SubPrimitiveSeg]
            dictionary of subprimitives by class

        """
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

    def export_subprimitives(self, path: Path) -> None:
        """exports subprimitives by class to the given path

        exported subprimitives are saved as glb files with the following naming
        convention: {`path`}/class_{`class_id`}.glb

        parameters
        ----------
        path: pathlib.Path
            path to export subprimitives to

        """
        Path.mkdir(path, parents=True, exist_ok=True)
        for class_id, subprimitive in tqdm(
            self.subprimitives.items(), desc="Exporting", unit="subprimitive"
        ):
            self.export_subprimitive(
                subprimitive, path / f"class_{class_id}.glb"
            )

    def export_subprimitive(
        self, subprimitive: SubPrimitiveSeg, path: Path
    ) -> None:
        """exports subprimitive to the given path

        path should specify the entire path including the file name and
        extension (e.g. "output/subprimitive.glb")

        parameters
        ----------
        subprimitive: SubPrimitiveSeg
            subprimitive to export
        path: pathlib.Path
            path to export subprimitive to

        """
        vertex_bytearray = bytearray()
        vertex_max: list[float] = [float("-inf")] * 3
        vertex_min: list[float] = [float("inf")] * 3
        face_bytearray = bytearray()
        tex_coord_bytearray = bytearray()
        # image_bytearray: bytes = self.get_texture_image_bytes() or b""
        for vertex in subprimitive.vertices:
            for i, value in enumerate(vertex):
                vertex_bytearray.extend(struct.pack("f", value))
                vertex_max[i] = max(vertex_max[i], value)
                vertex_min[i] = min(vertex_min[i], value)
        for face in subprimitive.faces:
            for value in face:
                face_bytearray.extend(struct.pack("H", value))
        for tex_coord in subprimitive.uv_coords:
            for value in tex_coord:
                tex_coord_bytearray.extend(struct.pack("f", value))
        vertex_bytelen: int = len(vertex_bytearray)
        face_bytelen: int = len(face_bytearray)
        tex_coord_bytelen: int = len(tex_coord_bytearray)
        # image_bytelen: int = len(image_bytearray) if image_bytearray else 0
        model = GLTFModel(
            accessors=[
                # position
                Accessor(
                    bufferView=0,
                    byteOffset=0,
                    componentType=ComponentType.FLOAT.value,
                    count=len(subprimitive.vertices),
                    type=AccessorType.VEC3.value,
                    max=vertex_max,
                    min=vertex_min,
                ),
                # faces
                Accessor(
                    bufferView=1,
                    byteOffset=0,
                    componentType=ComponentType.UNSIGNED_SHORT.value,
                    count=len(subprimitive.faces) * 3,
                    type=AccessorType.SCALAR.value,
                ),
                # tex coords
                Accessor(
                    bufferView=2,
                    byteOffset=0,
                    componentType=ComponentType.FLOAT.value,
                    count=len(subprimitive.uv_coords),
                    type=AccessorType.VEC2.value,
                ),
            ],
            asset=Asset(version="2.0"),
            extensionsUsed=self.glb.model.extensionsUsed,
            scenes=self.glb.model.scenes,
            nodes=self.glb.model.nodes,
            meshes=[
                Mesh(
                    primitives=[
                        Primitive(
                            attributes=Attributes(POSITION=0, TEXCOORD_0=2),
                            indices=1,
                            material=0,
                        )
                    ]
                )
            ],
            bufferViews=[
                BufferView(
                    buffer=0,
                    byteOffset=0,
                    byteLength=vertex_bytelen,
                    target=BufferTarget.ARRAY_BUFFER.value,
                ),
                BufferView(
                    buffer=0,
                    byteOffset=vertex_bytelen,
                    byteLength=face_bytelen,
                    target=BufferTarget.ELEMENT_ARRAY_BUFFER.value,
                ),
                BufferView(
                    buffer=0,
                    byteOffset=vertex_bytelen + face_bytelen,
                    byteLength=tex_coord_bytelen,
                    target=BufferTarget.ARRAY_BUFFER.value,
                ),
            ],
            materials=[self.material] if self.material else None,
            samplers=self.glb.model.samplers,
            textures=self.glb.model.textures,
        )
        image_bytearray: bytes = bytearray()
        for image in self.glb.model.images or []:
            bufferview: Optional[int] = image.bufferView
            if bufferview is None:
                continue
            image_bytes = self.retrieve_bufferview(bufferview)
            if model.images is None:
                model.images = []
            model.images.append(
                Image(
                    mimeType=image.mimeType,
                    bufferView=len(model.bufferViews or []),
                )
            )
            if model.bufferViews is None:
                model.bufferViews = []
            model.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=vertex_bytelen
                    + face_bytelen
                    + tex_coord_bytelen
                    + len(image_bytearray),
                    byteLength=len(image_bytes),
                )
            )
            image_bytearray.extend(image_bytes)

        model.buffers = [
            Buffer(
                byteLength=vertex_bytelen
                + face_bytelen
                + tex_coord_bytelen
                + len(image_bytearray),
                uri="submesh.bin",
            )
        ]

        resource = FileResource(
            "submesh.bin",
            data=vertex_bytearray
            + face_bytearray
            + tex_coord_bytearray
            + image_bytearray,
        )

        GLTF(model=model, resources=[resource]).export(str(path))


class GLBSegment(BufferAccessor):
    """loads a glb and loads primitives of each mesh. exports a glb with
    metadata from each PrimitiveSeg

    class attribute `glb` is used to store the glTF object to access

    parameters
    ----------
    path: pathlib.Path
        path to glb file

    attributes
    ----------
    glb: gltflib.GLTF
        the glTF object to access and segment
    meshes: list[list[PrimitiveSeg]]
        list of meshes containing lists of primitives

    examples
    --------
    >>> glb = GLBSegment(Path("model.glb"))
    >>> glb.load_meshes()
    >>> glb.export(Path("output/model.glb"))
    >>> glb.export_submeshes(Path("output"))

    """

    def __init__(self, path: Path) -> None:
        BufferAccessor.glb = GLTF.load(str(path))
        self.meshes: list[list[PrimitiveSeg]] = []
        """list of meshes containing lists of primitives"""

    def load_meshes(self) -> None:
        """loads meshes and primitives from the glb and saves the loaded
        PrimitiveSegs to the meshes attribute

        running this method twice will result in duplicate PrimitiveSegs in the
        meshes attribute, so set the meshes attribute to an empty list before
        calling this method again

        """
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
        """exports the glb with metadata from each PrimitiveSeg to the given
        path

        the path should specify the entire path including the file name and
        extension (e.g. "output/model.glb")

        parameters
        ----------
        path: pathlib.Path
            path to export the glb to

        """
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
        """exports submeshes by class to the given path

        exported submeshes are saved as glb files with the following naming
        convention: {`output_dir`}/mesh_{`mesh_index`}/class_{`class_id`}.glb

        parameters
        ----------
        output_dir: pathlib.Path
            path to export submeshes to

        """
        for i, mesh in enumerate(self.meshes):
            for j, primitive_seg in enumerate(
                tqdm(
                    mesh,
                    desc="Exporting mesh",
                    unit="mesh",
                    leave=False,
                )
            ):
                primitive_seg.export_subprimitives(output_dir / f"mesh{i}/{j}")


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
