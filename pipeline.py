"""pipeline for segmenting glb files"""
from pathlib import Path
import logging
from PIL.Image import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from py3dtiles.tileset.tileset import TileSet, Tile
from tileset_converter import convert_tileset
from image_segment import predict_semantic, get_labels
from segment_glb import GLBSegment, MeshSegment


LOG: logging.Logger = logging.getLogger(__name__)
OUTPUT_DIR: Path = Path("webserver/public/")


def get_meshes_segmented(path: Path) -> list[MeshSegment]:
    """get meshes segmented"""
    glb = GLBSegment(path)
    glb.load_meshes()
    meshes: list[MeshSegment] = glb.meshes
    for i, mesh in enumerate(
        tqdm(meshes, desc="Segmenting meshes", unit="mesh")
    ):
        texture: Image | None = mesh.get_texture_image()
        if texture is None:
            LOG.warning(f"Mesh {i} has no texture")
            continue
        LOG.info("Predicting semantic segmentation")
        mesh.seg = predict_semantic(texture)
    return meshes


def rewrite_tile(tile: Tile, meshes: list[MeshSegment]) -> None:
    """rewrite tile"""
    if meshes is None:
        return
    tile.content = None
    if tile.contents is None:
        tile.contents = []
    for i, mesh in enumerate(meshes):
        for class_id in tqdm(
            mesh.submeshes, desc="Exporting submeshes", unit="submesh"
        ):
            SUBMESH_DIR: str = f"output/submesh{i}_{class_id}.glb"
            mesh.export_submesh(
                mesh.submeshes[class_id], OUTPUT_DIR / SUBMESH_DIR
            )
            tile.contents.append({"uri": SUBMESH_DIR, "group": class_id})


def pipeline(path: Path) -> None:
    """pipeline"""
    LOG.info("Loading tileset")
    tileset: TileSet = TileSet.from_file(path)
    tile: Tile = tileset.root_tile
    if tile.content is None:
        return
    meshes: list[MeshSegment] = get_meshes_segmented(Path(tile.content["uri"]))
    if meshes is None:
        return
    labels = get_labels()
    LOG.info("Converting tileset")
    convert_tileset(tileset, labels)
    LOG.info("Rewriting tile")
    rewrite_tile(tile, meshes)
    tileset.write_as_json(OUTPUT_DIR / "tileset.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        LOG.info("Starting segmentation")
        # segment_glb("model.gltf")
        pipeline(Path("tileset.json"))
        LOG.info("Segmentation complete")
