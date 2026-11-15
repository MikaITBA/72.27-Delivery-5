"""Repeated training runs used to compare VAE configurations."""

import time

import numpy as np

from tp5.autoencoder import OPTIMIZERS
from tp5.vae import VAE

ACTIVE_VAR = 0.01  # a latent unit is "active" if the variance of its mu exceeds this


def split(X, n_val=200, seed=0):
    """Shuffle once with a fixed seed and return (X_train, X_val)."""
    order = np.random.default_rng(seed).permutation(len(X))
    return X[order[n_val:]], X[order[:n_val]]


def run_config(
    X_train,
    X_val,
    latent_dim=2,
    hidden=(256, 64),
    activation="tanh",
    lr=1e-3,
    batch_size=64,
    epochs=50,
    beta=1.0,
    seeds=(0, 1),
):
    """Train one configuration once per seed and evaluate it on X_val.

    Training uses recon + beta * kl. The validation numbers are always the
    unweighted recon and kl, and val_total = recon + kl (the beta = 1 loss), so
    configurations with different beta can still be compared on val_total.

    Returns:
        List of dicts, one per seed, with the configuration, the validation
        reconstruction and KL (both per image, averaged over X_val), their sum,
        the number of active latent units, the training time in seconds and
        the trained model.
    """
    rows = []
    for seed in seeds:
        vae = VAE(
            X_train.shape[1], hidden, latent_dim, seed=seed, beta=beta, activation=activation
        )
        start = time.time()
        vae.fit(
            X_train,
            epochs=epochs,
            optimizer=OPTIMIZERS["adam"](lr=lr),
            batch_size=batch_size,
            log_every=epochs,  # only log at the end: logging costs a full pass
            seed=seed,
        )
        recon, kl = vae.loss_parts(X_val)
        mu, _ = vae.encode(X_val)
        rows.append(
            {
                "latent_dim": latent_dim,
                "hidden": hidden,
                "activation": activation,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "beta": beta,
                "seed": seed,
                "val_recon": float(recon),
                "val_kl": float(kl),
                "val_total": float(recon + kl),
                "active_units": int(np.sum(mu.var(axis=0) > ACTIVE_VAR)),
                "seconds": time.time() - start,
                "model": vae,
            }
        )
    return rows


def summarize(rows, key):
    """Aggregate run_config rows per value of `key` (mean and std over seeds).

    Returns:
        List of dicts with the value of `key`, the number of runs and the mean
        (and std) of the validation total, reconstruction and KL, plus the mean
        number of active units.
    """
    summary = []
    for value in dict.fromkeys(str(r[key]) for r in rows):
        group = [r for r in rows if str(r[key]) == value]
        total = [r["val_total"] for r in group]
        summary.append(
            {
                key: value,
                "runs": len(group),
                "val_total": float(np.mean(total)),
                "val_total_std": float(np.std(total)),
                "val_recon": float(np.mean([r["val_recon"] for r in group])),
                "val_kl": float(np.mean([r["val_kl"] for r in group])),
                "active_units": float(np.mean([r["active_units"] for r in group])),
                "seconds": float(np.mean([r["seconds"] for r in group])),
            }
        )
    return summary
