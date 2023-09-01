"""
contains functions to load and modify glTF files
"""


from .extract_textures import glb_to_pillow

from .get_vertices import MeshSegment

from .segment_glb import GLBSegment, PrimitiveSeg


__all__: list[str] = [
    "glb_to_pillow",
    "MeshSegment",
    "GLBSegment",
    "PrimitiveSeg",
]
