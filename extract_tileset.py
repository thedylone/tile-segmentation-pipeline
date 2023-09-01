"""extract a branch of the tileset from the full tileset"""
import argparse
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from gltflib.gltf import GLTF
from pyproj import Transformer
from py3dtiles.tileset import TileSet, Tile, BoundingVolumeBox
from tileset import TilesetTraverser

LOG: logging.Logger = logging.getLogger(__name__)


class TilesetExtractor(TilesetTraverser):
    """extract a branch of the tileset from the full tileset"""

    OUTPUT_DIR: Path = Path(".")
    with_neighbours: bool = False
    TILE_PBAR = tqdm(desc="Tiles checked", unit=" tile")

    def __init__(self, find_name: str) -> None:
        self.find_name: str = find_name

    def traverse_tileset(self, tileset: TileSet) -> bool:
        """traverse tileset"""
        root_uri: Path = self.root_uri or tileset.root_uri
        traverse: bool = self.traverse_tile(self.get_tileset_tile(tileset))
        self.root_uri = root_uri
        return traverse

    def traverse_tile(self, tile: Tile) -> bool:
        """traverse tile"""
        self.TILE_PBAR.update()
        return self.traverse_tile_content(tile) or self.traverse_tile_children(
            tile
        )

    def traverse_tile_content(self, tile: Tile) -> bool:
        """traverse tile content"""
        content_uri: Optional[Path] = self.get_tile_content(tile)
        if content_uri is None:
            return False
        if self.find_name in content_uri.name:
            LOG.warning("File %s found", self.find_name)
            return True
        if content_uri.suffix == ".json":
            tileset: TileSet = TileSet.from_file(content_uri)
            if self.traverse_tileset(tileset):
                self.write_tileset(tileset, content_uri.name)
                return True
        return False

    def traverse_tile_children(self, tile: Tile) -> bool:
        """traverse tile children"""
        for child in tile.children:
            if not self.traverse_tile(child):
                continue
            if self.with_neighbours:
                self.with_neighbours = False
            else:
                tile.children = [child]
            return True
        return False

    def extract_tileset(self, tileset_name: str) -> bool:
        """extract tileset"""
        LOG.info("Loading tileset")
        tileset: TileSet = TileSet.from_file(self.INPUT_DIR / tileset_name)
        if self.traverse_tileset(tileset):
            LOG.info("Writing tileset")
            self.write_tileset(tileset, tileset_name)
            return True
        return False


class LatLongExtractor(TilesetExtractor):
    """extract a branch of the tileset from the full tileset by lat long"""

    ECEF_WGS84: Transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326")

    def __init__(self, bounds: np.ndarray) -> None:
        super().__init__("")
        if bounds.shape == (4,):
            bounds = bounds.reshape((2, 2))
        if bounds.shape != (2, 2):
            raise ValueError("Bounds must be a 2x2 array")
        self.bounds: np.ndarray = bounds

    @staticmethod
    def get_gltf_coords(gltf: GLTF) -> list[tuple[float, float]]:
        # pylint: disable=unpacking-non-sequence
        """get list of lat longs from gltf"""
        if (
            gltf.model.scene is None
            or gltf.model.scenes is None
            or len(gltf.model.scenes) == 0
            or gltf.model.nodes is None
        ):
            return []
        scene = gltf.model.scenes[gltf.model.scene]
        if scene.nodes is None or len(scene.nodes) == 0:
            return []
        coords: list[tuple[float, float]] = []
        lat: float
        long: float
        for node_index in scene.nodes:
            node = gltf.model.nodes[node_index]
            if node.translation is None:
                continue
            # y-up to z-up
            lat, long, _ = LatLongExtractor.ECEF_WGS84.transform(
                xx=node.translation[0],
                yy=-node.translation[2],
                zz=node.translation[1],
            )
            coords.append((lat, long))
        return coords

    @staticmethod
    def get_tile_centre_coords(tile: Tile) -> tuple[float, float]:
        # pylint: disable=unpacking-non-sequence
        """get lat long of centre from tile bounding volume"""
        volume = tile.bounding_volume
        if not isinstance(volume, BoundingVolumeBox):
            raise ValueError("Bounding volume is not a box")
        x_coord, y_coord, z_coord = volume.get_center()
        (
            lat,
            long,
            _,
        ) = LatLongExtractor.ECEF_WGS84.transform(
            xx=x_coord,
            yy=y_coord,
            zz=z_coord,
        )
        return (lat, long)

    @staticmethod
    def check_tile_centre_in_bounds(tile: Tile, bounds: np.ndarray) -> bool:
        """check if tile centre is in bounds"""
        if tile.bounding_volume is None:
            return False
        lat, long = LatLongExtractor.get_tile_centre_coords(tile)
        lat1, lat2 = sorted(bounds.transpose()[0])
        long1, long2 = sorted(bounds.transpose()[1])
        if lat1 <= lat <= lat2 and long1 <= long <= long2:
            return True
        return False

    def traverse_tile_content(self, tile: Tile) -> bool:
        """traverse tile content"""
        if self.check_tile_centre_in_bounds(tile, self.bounds):
            LOG.warning("Tile within bounds found")
            return True
        content_uri: Optional[Path] = self.get_tile_content(tile)
        if content_uri is None:
            return False
        if content_uri.suffix == ".json":
            tileset: TileSet = TileSet.from_file(content_uri)
            if self.traverse_tileset(tileset):
                self.write_tileset(tileset, content_uri.name)
                return True
        return False


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="filename of tileset",
        default="tileset.json",
    )
    parser.add_argument(
        "-s",
        "--search-name",
        type=str,
        help="name of file to search for",
    )
    parser.add_argument(
        "-b",
        "--bounds",
        type=float,
        nargs=4,
        help="bounds of tileset to extract",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="input directory for tileset",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="output directory for tileset",
    )
    parser.add_argument(
        "-n",
        "--with-neighbours",
        action="store_true",
        help="include neighbours",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )
    args: argparse.Namespace = parser.parse_args()
    if not args.search_name and not args.bounds:
        parser.error("Either --search-name or --bounds is required")
    if args.input_dir:
        TilesetExtractor.INPUT_DIR = Path(args.input_dir)
    if args.output_dir:
        TilesetExtractor.OUTPUT_DIR = Path(args.output_dir)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    TilesetExtractor.with_neighbours = args.with_neighbours
    with logging_redirect_tqdm():
        extractor: Optional[Union[TilesetExtractor, LatLongExtractor]] = None
        if args.bounds:
            extractor = LatLongExtractor(np.array(args.bounds))
        if args.search_name:
            extractor = TilesetExtractor(args.search_name)
        if extractor is None:
            raise RuntimeError("Extractor not set")
        if extractor.extract_tileset(args.filename):
            LOG.info("Tileset extracted")
        else:
            LOG.warning("Tileset not extracted")
