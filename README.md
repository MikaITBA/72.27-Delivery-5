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
docs/        assignment text
notebooks/   experiments and figures
src/tp5/     package code
tests/       pytest tests
```

## Task division

| Deel | Wie | Inhoud |
|---|---|---|
| Data & tools | Mika | font inladen, ruis, plots, error-metric |
| 1a architectuur | 2 personen | autoencoder bouwen en tunen tot max 1 pixel fout |
| 1a analyse | Mika | latente ruimte plotten, nieuwe letter genereren |
| 1b | 1 persoon | denoising-architectuur en ruisexperimenten |
| 2 | 2 personen | dataset kiezen, VAE bouwen, nieuw sample genereren |
