import numpy as np

from tp5.autoencoder import Adam, Autoencoder
from tp5.data import load_font, pixel_errors


def test_shapes():
    X, _ = load_font()
    ae = Autoencoder(hidden=(20, 10), latent_dim=2, seed=0)
    assert ae.encode(X).shape == (32, 2)
    assert ae.decode(np.zeros((3, 2))).shape == (3, 35)
    assert ae.reconstruct(X).shape == (32, 35)


def test_gradients_match_finite_differences():
    X, _ = load_font()
    X = X.astype(float)
    ae = Autoencoder(hidden=(6,), seed=0)
    grads = ae.gradients(X, X)
    for p, g in zip(ae.params, grads):
        idx = (0,) * p.ndim
        old = p[idx]
        p[idx] = old + 1e-6
        loss_plus = ae.loss(X)
        p[idx] = old - 1e-6
        loss_minus = ae.loss(X)
        p[idx] = old
        assert np.isclose(g[idx], (loss_plus - loss_minus) / 2e-6, rtol=1e-4, atol=1e-9)


def test_learns_font_with_at_most_one_wrong_pixel():
    X, _ = load_font()
    X = X.astype(float)
    ae = Autoencoder(hidden=(20, 10), seed=0)
    ae.fit(X, epochs=5000, optimizer=Adam(lr=1e-2), log_every=50, stop_at_pixels=1)
    assert pixel_errors(X, ae.reconstruct(X)).max() <= 1
