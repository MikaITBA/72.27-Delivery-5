"""Compare a few Quick, Draw! categories to pick a more consistent dataset.

The fish drawings are messy (fish face both ways, some drawings hold two
fish), so we look for a category with one object in a fixed orientation.
For each category we take the same kind of subset as make_fish_dataset.py
(2,000 drawings, seed 0, black and white) and save three figures:

  samples.png      20 random drawings per category
  mean_images.png  the average drawing (sharp = consistent, blurry = varied)
  vae_samples.png  10 new drawings from a short VAE run per category

Run from the project root (downloads about 100 MB per category, once):
    uv run python scripts/compare_quickdraw_categories.py

Data: Quick, Draw! by Google, https://quickdraw.withgoogle.com/data,
licensed under CC BY 4.0.
"""

import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tp5.vae_study import run_config, split  # noqa: E402

CATEGORIES = ["fish", "apple", "cup", "mushroom", "light bulb", "house", "umbrella"]
URL_TEMPLATE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
DOWNLOAD_FOLDER = ROOT / "results" / "quickdraw_raw"  # results/ is not committed
OUTPUT_FOLDER = ROOT / "results" / "category_comparison"
NUMBER_OF_IMAGES = 2000
SEED = 0  # same subset rule as make_fish_dataset.py
VAE_EPOCHS = 150  # same as the final model in notebook 03


def download(category):
    """Download the full .npy file of a category (skipped if already there)."""
    path = DOWNLOAD_FOLDER / f"{category}.npy"
    if not path.exists():
        url = URL_TEMPLATE.format(urllib.parse.quote(category))
        print(f"Downloading {url} ...", flush=True)
        DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_subset(category):
    """Return 2,000 black-and-white drawings, chosen exactly like the fish."""
    all_drawings = np.load(download(category), allow_pickle=False)
    rng = np.random.default_rng(SEED)
    chosen_rows = rng.choice(len(all_drawings), size=NUMBER_OF_IMAGES, replace=False)
    return (all_drawings[np.sort(chosen_rows)] > 127).astype(float)


def plot_rows(images_per_category, filename, columns):
    """One row of small pictures per category, with the category name on the left."""
    fig, axes = plt.subplots(
        len(images_per_category), columns, figsize=(columns, 1.1 * len(images_per_category)),
        squeeze=False,
    )
    for row_axes, (category, images) in zip(axes, images_per_category.items()):
        for ax, image in zip(row_axes, images):
            ax.imshow(image.reshape(28, 28), cmap="binary", vmin=0, vmax=1)
        for ax in row_axes:
            ax.axis("off")
        row_axes[0].set_title(category, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / filename, dpi=120)
    plt.close(fig)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    datasets = {category: load_subset(category) for category in CATEGORIES}

    # 1. Random drawings: how varied and how messy is each category?
    show_rng = np.random.default_rng(1)
    plot_rows(
        {c: X[show_rng.choice(len(X), 20, replace=False)] for c, X in datasets.items()},
        "samples.png", columns=20,
    )

    # 2. Average drawing: a clear shape means the drawings agree with each other.
    fig, axes = plt.subplots(1, len(datasets), figsize=(1.6 * len(datasets), 2))
    for ax, (category, X) in zip(axes, datasets.items()):
        ax.imshow(X.mean(axis=0).reshape(28, 28), cmap="binary")
        ax.set_title(category, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "mean_images.png", dpi=120)
    plt.close(fig)

    # 3. Short VAE run per category: do the generated drawings look like the object?
    print(f"\n{'category':<12}{'ink':>6}{'val recon':>11}{'KL':>7}{'total':>8}{'seconds':>9}")
    generated = {}
    z = np.random.default_rng(1).standard_normal((10, 2))  # same z for every category
    for category, X in datasets.items():
        X_train, X_val = split(X)
        result = run_config(X_train, X_val, latent_dim=2, epochs=VAE_EPOCHS, seeds=(0,))[0]
        generated[category] = result["model"].decode(z)
        ink = X.sum(axis=1).mean()  # average number of black pixels per drawing
        print(
            f"{category:<12}{ink:>6.0f}{result['val_recon']:>11.1f}{result['val_kl']:>7.1f}"
            f"{result['val_total']:>8.1f}{result['seconds']:>9.0f}",
            flush=True,
        )
    plot_rows(generated, "vae_samples.png", columns=10)
    print("\nFigures saved to", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
