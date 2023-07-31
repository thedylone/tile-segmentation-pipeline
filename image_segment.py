"""run image segmentation on a single image"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


URL = "facebook/mask2former-swin-small-cityscapes-semantic"
MODEL = Mask2FormerForUniversalSegmentation.from_pretrained(URL)
PROCESSOR = AutoImageProcessor.from_pretrained(URL)


def get_labels() -> dict:
    """returns list of labels"""
    if MODEL is None:
        raise RuntimeError("Model not initialised")
    if MODEL.config is None:
        raise RuntimeError("Model config not initialised")
    return MODEL.config.id2label or {}


def predict_semantic(image: Image.Image) -> np.ndarray:
    """predict semantic segmentation on a single image"""
    inputs = PROCESSOR(image, return_tensors="pt")
    with torch.no_grad():
        outputs = MODEL(**inputs)
    return PROCESSOR.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]


def visualise(image: Image.Image, seg: np.ndarray) -> None:
    """visualise image segmentation"""
    color_seg: np.ndarray = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    color_palette = [
        list(np.random.choice(range(256), size=3))
        for _ in range(len(get_labels()))
    ]
    palette = np.array(color_palette)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def save_seg(seg: np.ndarray) -> None:
    """save segmentation to a file"""
    torch.save(seg, "map.pt")


def main(image_path: str, save=False) -> None:
    """main"""
    image: Image.Image = Image.open(image_path)
    seg: np.ndarray = predict_semantic(image)
    if save:
        save_seg(seg)
    visualise(image, seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", required=True, type=str, help="path to image file"
    )
    parser.add_argument("-s", "--save", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    main(args.file, args.save)
