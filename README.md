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

## Structure

```
data/        input data (font.h)
docs/        assignment and report material (not tracked)
notebooks/   experiments and figures
src/tp5/     package code
tests/       pytest tests
```

## Task division

| Part | Topic | Members |
|---|---|---|
| 1a | Autoencoder | |
| 1b | Denoising autoencoder | |
| 2 | VAE | |
