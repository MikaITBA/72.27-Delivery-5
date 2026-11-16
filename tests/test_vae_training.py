import numpy as np

from tp5.vae import VAE
from tp5.vae_training import train


def small_data():
    rng = np.random.default_rng(0)
    return (rng.random((40, 20)) < 0.3).astype(float)


def test_history_has_one_entry_per_epoch():
    X = small_data()
    vae = VAE(20, (12, 6), 2, seed=0)
    history = train(vae, X[:30], epochs=3, progress=False, X_val=X[30:])
    for key in ("epoch", "recon", "kl", "beta", "val_recon", "val_kl"):
        assert len(history[key]) == 3
    assert history["epoch"] == [1, 2, 3]


def test_warmup_raises_beta_and_restores_it():
    vae = VAE(20, (12, 6), 2, seed=0, beta=1.0)
    history = train(vae, small_data(), epochs=4, warmup_epochs=4, progress=False)
    assert history["beta"] == [0.25, 0.5, 0.75, 1.0]
    assert vae.beta == 1.0


def test_training_lowers_the_loss():
    X = small_data()
    vae = VAE(20, (12, 6), 2, seed=0)
    history = train(vae, X, epochs=30, lr=1e-2, progress=False)
    assert history["recon"][-1] < history["recon"][0]
