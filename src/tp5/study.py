"""Repeated training runs used to compare autoencoder configurations."""

import numpy as np

from tp5.autoencoder import OPTIMIZERS, Autoencoder
from tp5.data import pixel_errors

MAX_EPOCHS = 20_000


def run_config(
    X,
    hidden=(20, 10),
    activation="tanh",
    optimizer="adam",
    lr=1e-3,
    batch_size=None,
    seeds=range(5),
    epochs=MAX_EPOCHS,
):
    """Train one configuration once per seed, stopping at <= 1 wrong pixel.

    Returns:
        List of dicts, one per seed, with the configuration, the epoch at
        which training stopped, whether every letter reached <= 1 wrong
        pixel, the worst letter's error, the number of letters learned and
        the training history.
    """
    rows = []
    for seed in seeds:
        ae = Autoencoder(hidden=hidden, activation=activation, seed=seed)
        history = ae.fit(
            X,
            epochs=epochs,
            optimizer=OPTIMIZERS[optimizer](lr=lr),
            batch_size=batch_size,
            stop_at_pixels=1,
            seed=seed,
        )
        errors = pixel_errors(X, ae.reconstruct(X))
        rows.append(
            {
                "hidden": hidden,
                "activation": activation,
                "optimizer": optimizer,
                "lr": lr,
                "batch_size": batch_size or len(X),
                "seed": seed,
                "epochs": history["epoch"][-1],
                "solved": bool(errors.max() <= 1),
                "max_error": int(errors.max()),
                "letters_ok": int(np.sum(errors <= 1)),
                "history": history,
            }
        )
    return rows


def summarize(rows, key):
    """Aggregate run_config rows per value of `key`.

    Returns:
        List of dicts with the value of `key`, the success rate, the median
        epochs of the successful runs (None if none succeeded) and the mean
        number of letters learned.
    """
    summary = []
    for value in dict.fromkeys(str(r[key]) for r in rows):
        group = [r for r in rows if str(r[key]) == value]
        solved = [r["epochs"] for r in group if r["solved"]]
        summary.append(
            {
                key: value,
                "success_rate": len(solved) / len(group),
                "median_epochs": float(np.median(solved)) if solved else None,
                "mean_letters_ok": float(np.mean([r["letters_ok"] for r in group])),
            }
        )
    return summary
