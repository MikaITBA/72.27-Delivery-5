"""Create data/fish.npz from Google's Quick, Draw! dataset.

Downloads every "fish" drawing (about 134,000 images of 28x28 pixels,
105 MB), keeps a fixed random subset and turns the grey pixels into black and
white. Only the small subset is stored in the repository.

Run from the project root:
    uv run python scripts/make_fish_dataset.py

Data: Quick, Draw! by Google, https://quickdraw.withgoogle.com/data,
licensed under CC BY 4.0.
"""

import tempfile
import urllib.request
from pathlib import Path

import numpy as np

SOURCE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/fish.npy"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "fish.npz"
NUMBER_OF_IMAGES = 2000
SEED = 0  # fixed seed, so everyone gets exactly the same subset


def main():
    with tempfile.TemporaryDirectory() as download_folder:
        download_path = Path(download_folder) / "fish.npy"
        print(f"Downloading {SOURCE_URL} ...")
        urllib.request.urlretrieve(SOURCE_URL, download_path)
        all_drawings = np.load(download_path, allow_pickle=False)  # (134150, 784), values 0-255

    rng = np.random.default_rng(SEED)
    chosen_rows = rng.choice(len(all_drawings), size=NUMBER_OF_IMAGES, replace=False)
    grey_images = all_drawings[np.sort(chosen_rows)]

    # Pixels are 0 (background) to 255 (pen stroke); everything above the
    # middle becomes a black pixel (1), the rest white (0).
    binary_images = (grey_images > 127).astype(np.uint8)

    np.savez_compressed(OUTPUT_PATH, images=binary_images)
    print(f"Saved {len(binary_images)} images to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
