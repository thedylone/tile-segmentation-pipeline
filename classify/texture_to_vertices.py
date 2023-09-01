"""converts segmentation map to class for each vertex"""
import numpy as np
from trimesh import Trimesh


def get_map_value(seg: np.ndarray, coords: tuple[float, float]) -> int:
    """returns the value of the segmentation map at the given coordinates"""
    height: int
    width: int
    height, width = seg.shape
    col: int = int(coords[0] * width) % width
    row: int = int(coords[1] * height) % height
    return seg[row][col].item()


def texture_to_vertices_trimesh(mesh: Trimesh, seg: np.ndarray) -> list[int]:
    """converts segmentation map to class for each vertex"""
    total: int = len(mesh.vertices)
    vertices_to_class: list[int] = [-1] * total
    if mesh.visual is None:
        return vertices_to_class
    for index in range(total):
        vertices_to_class[index] = get_map_value(seg, mesh.visual.uv[index])
    return vertices_to_class


def texture_to_vertices(
    vertices: np.ndarray, tex_coords: np.ndarray, seg: np.ndarray
) -> list[int]:
    """converts segmentation map to class for each vertex"""
    total: int = len(vertices)
    vertices_to_class: list[int] = [-1] * total
    for index in range(total):
        vertices_to_class[index] = get_map_value(seg, tex_coords[index])
    return vertices_to_class
