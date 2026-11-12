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
data/        input data (font.h)
docs/        assignment text
notebooks/   experiments and figures
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
| 1b | 1 person - Stella | denoising architecture and noise experiments | In progress se branch: feat/denoising |
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
