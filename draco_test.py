import DracoPy
from gltflib.gltf import GLTF


def decode_data(data: bytes):
    """decode the data from a buffer"""
    return DracoPy.decode(data)


def main():
    gltf: GLTF = GLTF.load("model.glb")
    data = decode_data(gltf.resources[0].data)
    print(data.faces)


if __name__ == "__main__":
    main()
