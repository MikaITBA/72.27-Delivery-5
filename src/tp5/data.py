"""Loading, noise and plotting helpers for the 5x7 bitmap font in font.h."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROWS, COLS = 7, 5

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "font.h"


def load_font(path: str | Path = _DEFAULT_PATH) -> tuple[np.ndarray, list[str]]:
    """Read font.h and return the letters as flat 0/1 vectors.

    Each character is a line like ``{0x00, 0x0e, ...},   // 0x61, a``: seven
    row bytes of which the lowest 5 bits are used, bit 4 being the leftmost
    pixel.

    Returns:
        X: array of shape (n_chars, 35) with values 0/1, rows concatenated.
        chars: the character of each row, taken from the line comments.
    """
    patterns, chars = [], []
    for line in Path(path).read_text().splitlines():
        match = re.search(r"\{([^}]*)\}", line)
        if match is None:
            continue
        rows = [int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
        if len(rows) != ROWS:
            continue
        bits = [(row >> (COLS - 1 - c)) & 1 for row in rows for c in range(COLS)]
        patterns.append(bits)
        comment = line.split("//", 1)[1] if "//" in line else ""
        chars.append(comment.split(",", 1)[-1].strip())
    return np.array(patterns, dtype=int), chars


def pixel_errors(X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    """Number of wrong pixels per letter after rounding X_hat to 0/1."""
    return np.sum(np.round(X_hat).astype(int) != np.asarray(X).astype(int), axis=1)


def add_noise(X: np.ndarray, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Flip every pixel independently with probability p."""
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X)
    flip = rng.random(X.shape) < p
    return np.where(flip, 1 - X, X)


def show_letters(X: np.ndarray, titles: list[str] | None = None, ncols: int = 8):
    """Plot each row of X as a 7x5 grid in a raster of subplots."""
    X = np.atleast_2d(X)
    nrows = int(np.ceil(len(X) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.2 * ncols, 1.6 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(X):
            ax.imshow(X[i].reshape(ROWS, COLS), cmap="binary", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i])
    fig.tight_layout()
    return fig
