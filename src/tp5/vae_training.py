"""Train a VAE one epoch at a time: progress bar and optional KL warm-up.

The training loop itself is the inherited Autoencoder.fit (in
autoencoder.py, which we do not change). Here we call it for one epoch at a
time with the same optimizer, so that between epochs we can

  * print a progress bar with the current reconstruction and KL, and
  * change the KL weight beta (KL warm-up, see train()).
"""

import time

from tp5.autoencoder import OPTIMIZERS


def show_progress(epoch, epochs, recon, kl, beta, seconds):
    """Print a one-line progress bar that overwrites itself."""
    width = 30
    filled = int(width * epoch / epochs)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if epoch == epochs else ""  # move to a new line once finished
    print(
        f"\repoch {epoch:>4}/{epochs} [{bar}] recon {recon:7.1f}  KL {kl:5.1f}  "
        f"beta {beta:4.2f}  {seconds:4.0f}s",
        end=end,
        flush=True,
    )


def train(vae, X, epochs, lr=1e-3, batch_size=64, seed=0, warmup_epochs=0, progress=True,
          X_val=None):
    """Train `vae` on X with Adam and return the history (epoch, recon, kl per epoch).

    If X_val is given, the history also holds val_recon and val_kl after
    every epoch (unweighted, from loss_parts), so overfitting becomes visible:
    the training recon keeps falling while val_recon stops or rises.

    KL warm-up: during the first `warmup_epochs` epochs the KL weight beta
    grows in a straight line from 0 to the beta the VAE was created with.
    With beta near 0 the decoder first learns to use z, before the KL term
    starts pulling every code towards N(0, I). Without it, the KL term can win
    from the start and the VAE ignores z (posterior collapse: every image is
    decoded as the same average image). warmup_epochs=0 is plain training.

    Each epoch is one call to the inherited fit(); the same optimizer object
    is reused, so Adam keeps its running averages between epochs. The shuffle
    seed changes every epoch, otherwise every epoch would use the same order.
    """
    optimizer = OPTIMIZERS["adam"](lr=lr)
    target_beta = vae.beta
    history = {"epoch": [], "recon": [], "kl": [], "beta": [], "val_recon": [], "val_kl": []}
    start = time.time()
    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0:
            vae.beta = target_beta * min(1.0, epoch / warmup_epochs)
        vae.fit(X, epochs=1, optimizer=optimizer, batch_size=batch_size, log_every=1,
                seed=seed * 100_000 + epoch)
        recon, kl = vae.recon_log[-1], vae.kl_log[-1]  # fit() logged this epoch's loss
        history["epoch"].append(epoch)
        history["recon"].append(recon)
        history["kl"].append(kl)
        history["beta"].append(vae.beta)
        if X_val is not None:
            val_recon, val_kl = vae.loss_parts(X_val)  # loss_parts does not touch the logs
            history["val_recon"].append(float(val_recon))
            history["val_kl"].append(float(val_kl))
        if progress:
            show_progress(epoch, epochs, recon, kl, vae.beta, time.time() - start)
    vae.beta = target_beta
    return history
