"""Cartoon faces in colour: an experimental dataset for exercise 2 (VAE).

A fixed subset of 2,000 faces from Google's Cartoon Set, shrunk to 32x32
colour images. The file data/cartoon_faces.npz is created by
scripts/make_cartoon_dataset.py.

Data: Cartoon Set by Google, https://google.github.io/cartoonset/, licensed
under CC BY 4.0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IMAGE_SHAPE = (32, 32, 3)  # rows, columns, colour channels (red, green, blue)

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "cartoon_faces.npz"


def load_cartoons(path=_DEFAULT_PATH):
    """Return the faces as "ink": an array of shape (2000, 3072) with values between 0 and 1.

    Each row is one face: its 32 rows of 32 pixels, each pixel given as three
    amounts of ink (cyan, magenta, yellow), like a printer. 0 is no ink (white
    paper) and 1 is full ink. This is the same convention as the fish
    (0 = paper, 1 = pen), and the inverse of the usual red/green/blue.

    Why ink and not red/green/blue: with red/green/blue the white background
    is 1, so about 60% of the inputs are a constant 1. In our tests that
    drowned out the small differences between faces and the VAE ignored z
    (posterior collapse: validation recon 805, no active latent unit). As ink
    the background is 0 and only the face itself feeds the network (recon 650,
    both latent units active). See docs/vae_decisions.md.
    """
    rgb = np.load(path)["images"] / 255
    return 1 - rgb


def to_rgb(ink):
    """Turn ink values (0 = paper, 1 = full ink) back into red/green/blue for plotting."""
    return np.clip(1 - ink, 0, 1)  # clip: generated values can be a hair outside 0..1


def load_attributes(path=_DEFAULT_PATH):
    """Return (values, names): the attributes that generated each face.

    values has shape (2000, 18) with whole numbers (e.g. which hair style,
    which glasses); names lists the 18 attribute names in column order.
    """
    data = np.load(path)
    return data["attributes"], [str(name) for name in data["attribute_names"]]


def show_images(images, titles=None, columns=10):
    """Plot each row of `images` (ink values, as from load_cartoons) as a small colour picture."""
    images = np.atleast_2d(images)
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.2 * columns, 1.3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(to_rgb(images[i]).reshape(IMAGE_SHAPE))
            if titles is not None:
                ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    return fig
