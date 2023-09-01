"""traverse a 3d tileset JSON file"""
import logging
from pathlib import Path
from typing import Optional
from py3dtiles.tileset import TileSet, Tile


class TilesetTraverser:
    """traverse a 3d tileset JSON file"""

    INPUT_DIR: Path = Path(".")
    OUTPUT_DIR: Path = Path(".")
    root_uri: Path = Path(".")

    @classmethod
    def get_tileset_tile(cls, tileset: TileSet) -> Tile:
        """return root tile and set root uri"""
        if tileset.root_uri is not None:
            cls.root_uri = tileset.root_uri
        return tileset.root_tile

    @classmethod
    def get_tile_content(cls, tile: Tile) -> Optional[Path]:
        """returns tile content as a Path"""
        if tile.content is None:
            return None
        uri_: str = tile.content["uri"]
        if uri_[0] == "/":
            uri_ = uri_[1:]
        uri: Path = cls.INPUT_DIR / uri_
        if not uri.exists():
            # try tileset root dir
            uri = cls.root_uri / uri_
        if not uri.exists():
            logging.info("File %s does not exist", uri)
            tile.content = None
            return None
        return uri

    @classmethod
    def write_tileset(cls, tileset: TileSet, name="tileset.json") -> None:
        """write tileset to output dir"""
        root_uri: Path = tileset.root_uri or cls.root_uri
        rel_path: Path = root_uri.relative_to(cls.INPUT_DIR)
        Path.mkdir(
            cls.OUTPUT_DIR / rel_path,
            parents=True,
            exist_ok=True,
        )
        tileset.write_as_json(cls.OUTPUT_DIR / rel_path / name)
