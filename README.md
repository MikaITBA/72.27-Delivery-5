# TP5 — Deep Learning

Sistemas de Inteligencia Artificial (72.27), ITBA.

Autoencoders on the 5x7 bitmap font (`data/font.h`): a plain autoencoder with a
2D latent space, a denoising autoencoder, and a variational autoencoder (VAE).

## Setup

Requires Python 3.11+. Run every command from the project root.

```bash
uv sync --extra dev
```

Or without uv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests with `uv run pytest` and open the notebooks with `uv run jupyter lab notebooks/`.

## Data

Both datasets are already in the repository, so there is nothing to download
or move before running the code.

| File | Used in | Content | Load with |
|---|---|---|---|
| `data/font.h` | Exercise 1 | 32 characters of 7x5 pixels | `tp5.data.load_font()` |
| `data/fish.npz` | Exercise 2 (VAE) | 2,000 hand-drawn fish of 28x28 pixels, black and white | `tp5.fish.load_fish()` |

The fish are a fixed random subset of the "fish" category of Google's
[Quick, Draw! dataset](https://quickdraw.withgoogle.com/data) (about 134,000
drawings, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). The original
files are listed on
[GitHub](https://github.com/googlecreativelab/quickdraw-dataset); we use the
"numpy bitmap" file
([fish.npy](https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/fish.npy), 105 MB).

To recreate `data/fish.npz` from scratch (normally not needed), run:

```bash
uv run python scripts/make_fish_dataset.py
```

The script downloads the full file to a temporary folder, picks the same 2,000
drawings every time (fixed seed), turns the pixels into black and white and
saves only the subset. The full 105 MB file is not kept.

## Notebooks

| Notebook | Section |
|---|---|
| `00_data_check.ipynb` | Font loading, plots and noise |
| `01_autoencoder_architecture.ipynb` | 1a.1–1a.2 Autoencoder architecture and optimisation study |

The autoencoder (`src/tp5/autoencoder.py`) is written from scratch with numpy:
dense layers, backpropagation with binary cross-entropy, and SGD, momentum and
Adam optimizers.

## Structure

```
data/        input data (font.h, fish.npz)
docs/        assignment text, decision log for exercise 2
notebooks/   experiments and figures
scripts/     one-off scripts (rebuilding the fish dataset)
src/tp5/     package code
tests/       pytest tests
```

## Quick start

```python
from tp5.data import load_font, pixel_errors, add_noise, show_letters
from tp5.autoencoder import Autoencoder, Adam

X, chars = load_font()                        # (32, 35) array of 0/1
ae = Autoencoder(hidden=(20, 10), latent_dim=2, seed=0)
ae.fit(X, epochs=5000, optimizer=Adam(lr=0.01), stop_at_pixels=1)
Z = ae.encode(X)                              # (32, 2) latent codes
X_hat = ae.decode(Z)                          # reconstructions
print(pixel_errors(X, X_hat).max())           # <= 1
```

For the denoising autoencoder, `ae.gradients(X_noisy, X_clean)` already supports
a noisy input with a clean target; `fit` currently trains on clean input only.

## Task division

| Part | Who | Content | Status |
|---|---|---|---|
| Data & tools | Mika | font loading, noise, plots, error metric | done |
| 1a.1–1a.2 architecture | Mika | build the autoencoder and tune it to ≤ 1 wrong pixel | done (`01_autoencoder_architecture.ipynb`) |
| 1a.3–1a.4 analysis | Mika | plot the latent space, generate a new letter | to do |
| 1b | 1 person | denoising architecture and noise experiments | to do |
| 2 | 2 people | choose a dataset, build the VAE, generate a new sample | to do |

## Contributing

`main` is protected: direct pushes are blocked, so every change goes through a
pull request. No approval is needed, so you can merge your own PR.

```bash
git checkout main && git pull
git checkout -b feat/<your-part>         # e.g. feat/denoising
# ... work, run `uv run pytest` ...
git push -u origin feat/<your-part>
gh pr create --base main                 # or open the PR on github.com
```

Add new notebooks as `notebooks/NN_<topic>.ipynb` and reusable code under
`src/tp5/`.
