"""Hand-drawn fish: the dataset for exercise 2 (VAE).

A fixed subset of 2,000 drawings from Google's Quick, Draw! "fish" category,
as black-and-white 28x28 images. The file data/fish.npz is created by
scripts/make_fish_dataset.py.

Data: Quick, Draw! by Google, https://quickdraw.withgoogle.com/data,
licensed under CC BY 4.0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 28  # images are IMAGE_SIZE x IMAGE_SIZE pixels

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "fish.npz"


def load_fish(path=_DEFAULT_PATH):
    """Return the fish drawings as an array of shape (2000, 784) with values 0/1.

    Each row is one drawing with its 28 rows of pixels concatenated
    (1 = pen stroke, 0 = background), like load_font does for the letters.
    """
    return np.load(path)["images"].astype(int)


def show_images(images, titles=None, columns=10, image_shape=(IMAGE_SIZE, IMAGE_SIZE)):
    """Plot each row of `images` as a small picture in a grid of subplots."""
    images = np.atleast_2d(images)
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.2 * columns, 1.3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i].reshape(image_shape), cmap="binary", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    return fig
