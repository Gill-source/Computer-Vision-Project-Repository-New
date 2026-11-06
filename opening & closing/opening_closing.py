from pathlib import Path

import numpy as np
from PIL import Image


INPUT_PATH = Path("input.png")
DILATION_OUTPUT_PATH = Path("input_dilation_square7x7.png")
EROSION_OUTPUT_PATH = Path("input_erosion_square7x7.png")
OPENING_OUTPUT_PATH = Path("input_opening_square7x7.png")
CLOSING_OUTPUT_PATH = Path("input_closing_square7x7.png")

# 7x7 square structuring element.
SE_RADIUS = 3
SQUARE_OFFSETS = tuple(
    (dy, dx)
    for dy in range(-SE_RADIUS, SE_RADIUS + 1)
    for dx in range(-SE_RADIUS, SE_RADIUS + 1)
)


def square_dilation(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, SE_RADIUS, mode="constant", constant_values=0)
    candidates = [
        padded[
            SE_RADIUS + dy : SE_RADIUS + dy + image.shape[0],
            SE_RADIUS + dx : SE_RADIUS + dx + image.shape[1],
        ]
        for dy, dx in SQUARE_OFFSETS
    ]
    return np.maximum.reduce(candidates).astype(np.uint8)


def square_erosion(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, SE_RADIUS, mode="constant", constant_values=255)
    candidates = [
        padded[
            SE_RADIUS + dy : SE_RADIUS + dy + image.shape[0],
            SE_RADIUS + dx : SE_RADIUS + dx + image.shape[1],
        ]
        for dy, dx in SQUARE_OFFSETS
    ]
    return np.minimum.reduce(candidates).astype(np.uint8)


def square_opening(image: np.ndarray) -> np.ndarray:
    return square_dilation(square_erosion(image))


def square_closing(image: np.ndarray) -> np.ndarray:
    return square_erosion(square_dilation(image))


def main() -> None:
    image = np.array(Image.open(INPUT_PATH).convert("L"), dtype=np.uint8)

    dilation = square_dilation(image)
    erosion = square_erosion(image)
    opening = square_opening(image)
    closing = square_closing(image)

    Image.fromarray(dilation).save(DILATION_OUTPUT_PATH)
    Image.fromarray(erosion).save(EROSION_OUTPUT_PATH)
    Image.fromarray(opening).save(OPENING_OUTPUT_PATH)
    Image.fromarray(closing).save(CLOSING_OUTPUT_PATH)

    print(f"saved: {DILATION_OUTPUT_PATH}")
    print(f"saved: {EROSION_OUTPUT_PATH}")
    print(f"saved: {OPENING_OUTPUT_PATH}")
    print(f"saved: {CLOSING_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
