"""extract a branch of the tileset from the full tileset"""
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from py3dtiles.tileset.tileset import TileSet, Tile

LOG: logging.Logger = logging.getLogger(__name__)


class TilesetExtractor:
    """extract a branch of the tileset from the full tileset"""

    INPUT_DIR: Path = Path(".")
    OUTPUT_DIR: Path = Path(".")
    with_neighbours: bool = False
    TILE_PBAR = tqdm(desc="Tiles checked", unit=" tile")

    @staticmethod
    def search_tileset(tileset: TileSet, find_name: str) -> bool:
        """search tileset"""
        return TilesetExtractor.search_tile(tileset.root_tile, find_name)

    @staticmethod
    def search_tile(tile: Tile, find_name: str) -> bool:
        """search tile"""
        TilesetExtractor.TILE_PBAR.update()
        if TilesetExtractor.search_tile_content(tile, find_name):
            return True
        if TilesetExtractor.search_tile_children(tile, find_name):
            return True
        return False

    @classmethod
    def search_tile_content(cls, tile: Tile, find_name: str) -> bool:
        """search tile content"""
        if tile.content is None:
            return False
        uri_: str = tile.content["uri"]
        if uri_[0] == "/":
            uri_ = uri_[1:]
        uri: Path = cls.INPUT_DIR / uri_
        if not uri.exists():
            LOG.info("File %s does not exist", uri)
            return False
        if uri.name == find_name:
            return True
        if uri.suffix == ".json":
            tileset: TileSet = TileSet.from_file(uri)
            if cls.search_tileset(tileset, find_name):
                # tileset.write_as_json(cls.OUTPUT_DIR / uri.name)
                tileset.write_to_directory(cls.OUTPUT_DIR / uri.name)
                return True
        return False

    @classmethod
    def search_tile_children(cls, tile: Tile, find_name: str) -> bool:
        """search tile children"""
        for child in tile.children:
            if not cls.search_tile(child, find_name):
                continue
            if cls.with_neighbours:
                cls.with_neighbours = False
            else:
                tile.children = [child]
            return True
        return False

    @classmethod
    def extract_tileset(cls, tileset_name: str, find_name: str) -> None:
        """extract tileset"""
        LOG.info("Loading tileset")
        tileset: TileSet = TileSet.from_file(cls.INPUT_DIR / tileset_name)
        if cls.search_tileset(tileset, find_name):
            LOG.info("File %s found", find_name)
            LOG.info("Writing tileset")
            # tileset.write_as_json(cls.OUTPUT_DIR / tileset_name)
            tileset.write_to_directory(cls.OUTPUT_DIR / tileset_name)
        else:
            LOG.warning("File %s not found", find_name)


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
        required=True,
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
    if args.input_dir:
        TilesetExtractor.INPUT_DIR = Path(args.input_dir)
    if args.output_dir:
        TilesetExtractor.OUTPUT_DIR = Path(args.output_dir)
    if args.verbose:
        LOG.setLevel(logging.INFO)
    TilesetExtractor.with_neighbours = args.with_neighbours
    with logging_redirect_tqdm():
        TilesetExtractor.extract_tileset(args.filename, args.search_name)
