from pathlib import Path

import numpy as np
from PIL import Image


INPUT_PATH = Path("example.png")
DILATION_OUTPUT_PATH = Path("example_dilation_cross3x3.png")
EROSION_OUTPUT_PATH = Path("example_erosion_cross3x3.png")

# 3x3 cross structuring element:
# 0 1 0
# 1 1 1
# 0 1 0
CROSS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def cross_dilation(image: np.ndarray) -> np.ndarray:
    # The drawing is black foreground on a white background, so dilation
    # should expand dark pixels.
    padded = np.pad(image, 1, mode="constant", constant_values=255)
    candidates = [
        padded[1 + dy : 1 + dy + image.shape[0], 1 + dx : 1 + dx + image.shape[1]]
        for dy, dx in CROSS_OFFSETS
    ]
    return np.minimum.reduce(candidates).astype(np.uint8)


def cross_erosion(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="constant", constant_values=0)
    candidates = [
        padded[1 + dy : 1 + dy + image.shape[0], 1 + dx : 1 + dx + image.shape[1]]
        for dy, dx in CROSS_OFFSETS
    ]
    return np.maximum.reduce(candidates).astype(np.uint8)


def main() -> None:
    image = np.array(Image.open(INPUT_PATH).convert("L"), dtype=np.uint8)

    dilation = cross_dilation(image)
    erosion = cross_erosion(image)

    Image.fromarray(dilation).save(DILATION_OUTPUT_PATH)
    Image.fromarray(erosion).save(EROSION_OUTPUT_PATH)

    print(f"saved: {DILATION_OUTPUT_PATH}")
    print(f"saved: {EROSION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
