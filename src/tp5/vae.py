"""Variational autoencoder built on top of the numpy Autoencoder.

Layer layout (L = self.latent_index):
    trunk    layers 0 .. L-1   shared encoder hidden layers, output h
    mu head  layer  L          h -> mu      (inherited latent layer)
    lv head  W_lv, b_lv        h -> logvar  (new, separate weights)
    decoder  layers L+1 ..     z -> reconstruction (sigmoid output)

Training samples z = mu + exp(0.5 * logvar) * eps with eps ~ N(0, I) and
minimises  BCE summed over pixels + KL(N(mu, exp(logvar)) || N(0, I)),
both averaged over the batch.
"""

import numpy as np
from tp5.autoencoder import Autoencoder, ACTIVATIONS


class VAE(Autoencoder):
    """Variational autoencoder; see the module docstring for the layout.

    Args:
        n_inputs: Size of each input vector.
        hidden: Encoder hidden layer sizes (at least one), e.g. (256, 64).
            The decoder uses the same sizes in reverse order.
        latent_dim: Size of mu, logvar and z.
        seed: Seed for the weights and for the sampling noise eps.
        **kw: Passed on to Autoencoder (activation, latent_activation).
    """

    def __init__(self, n_inputs: int, hidden: tuple, latent_dim: int, seed=None, **kw):
        super().__init__(n_inputs, hidden, latent_dim, seed=seed, **kw)
        # separate generator for the logvar head weights, so they do not start
        # as a copy of the random numbers used by the parent
        head_rng = np.random.default_rng(None if seed is None else seed + 1)
        n_in = list(hidden)[-1]  # size of the last trunk layer (size of h)
        self.W_lv = head_rng.normal(0, np.sqrt(1 / n_in), (n_in, latent_dim))
        self.b_lv = np.zeros(latent_dim)
        self.rng = np.random.default_rng(seed)  # noise eps used in training
        self.recon_log, self.kl_log = [], []  # filled by loss(), see below

    @property
    def params(self):
        # same order as the gradients returned by gradients()
        return self.weights + self.biases + [self.W_lv, self.b_lv]

    def encode(self, X: np.ndarray):
        """Return (mu, logvar), each of shape (n_samples, latent_dim)."""
        L = self.latent_index
        h = self._forward(np.atleast_2d(X), stop=L)[-1]
        mu = h @ self.weights[L] + self.biases[L]
        logvar = h @ self.W_lv + self.b_lv
        return mu, logvar

    def loss_parts(self, X: np.ndarray, X_hat=None):
        """Return (recon, kl), both averaged over the samples.

        recon is the binary cross-entropy summed over the pixels of one image.
        kl is -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) over the latent units.
        X_hat defaults to the deterministic reconstruction (z = mu).
        """
        mu, logvar = self.encode(X)
        kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / len(X)
        recon = super().loss(X, X_hat) * X.shape[1]  # parent gives the mean per pixel
        return recon, kl

    def loss(self, X: np.ndarray, X_hat=None):
        """Return recon + kl. Also appends both parts to recon_log / kl_log.

        The inherited fit() calls this once per logged epoch, so the two logs
        line up with history["epoch"]. Use loss_parts() for any other call
        that should not add log entries.
        """
        recon, kl = self.loss_parts(X, X_hat)
        self.recon_log.append(recon)
        self.kl_log.append(kl)
        return recon + kl

    def gradients(self, X_in, X_target=None):
        """Backpropagate recon + kl and return the gradients in params order.

        X_target is ignored (the target is X_in itself); it only exists so the
        inherited fit(), which calls gradients(batch, batch), keeps working.
        The noise eps is drawn from self.rng. Assumes at least one trunk layer.
        """
        # Forward pass, keeping what the backward pass needs
        L = self.latent_index
        trunk = self._forward(X_in, stop=L)  # trunk[0] = X_in, trunk[-1] = h
        h = trunk[-1]
        mu = h @ self.weights[L] + self.biases[L]
        logvar = h @ self.W_lv + self.b_lv
        eps, std, z = self.sampling((mu, logvar), self.rng)
        dec = self._forward(z, start=L + 1)  # dec[0] = z, dec[-1] = reconstruction
        n = len(X_in)

        # Decoder backward. Sigmoid + cross-entropy: d(recon)/d(pre-activation)
        # of the output layer is (x_hat - x), divided by n for the batch mean.
        delta = (dec[-1] - X_in) / n
        gW, gb = {}, {}
        for k in reversed(range(len(dec) - 1)):
            i = L + 1 + k  # decoder layer k is layer i of the full stack
            gW[i] = dec[k].T @ delta
            gb[i] = delta.sum(0)
            back = delta @ self.weights[i].T  # gradient w.r.t. the layer's input
            if k > 0:
                delta = back * ACTIVATIONS[self.activations[i - 1]][1](dec[k])
        dz = back  # after the last iteration the layer input is z

        # Gradient w.r.t. mu and logvar: path through z (reconstruction)
        # plus the KL term (also divided by n for the batch mean)
        dmu = dz + mu / n
        dlogvar = dz * 0.5 * eps * std + 0.5 * (np.exp(logvar) - 1) / n

        # Heads
        gW[L] = h.T @ dmu
        gb[L] = dmu.sum(0)
        gW_lv = h.T @ dlogvar
        gb_lv = dlogvar.sum(0)
        dh = dmu @ self.weights[L].T + dlogvar @ self.W_lv.T  # both heads feed h

        # Trunk backward
        delta = dh * ACTIVATIONS[self.activations[L - 1]][1](h)
        for i in reversed(range(L)):
            gW[i] = trunk[i].T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * ACTIVATIONS[self.activations[i - 1]][1](
                    trunk[i]
                )

        n_layers = len(self.weights)
        return (
            [gW[i] for i in range(n_layers)]
            + [gb[i] for i in range(n_layers)]
            + [gW_lv, gb_lv]
        )

    def sampling(self, Z: tuple, rng: np.random.Generator):
        """Reparameterisation trick: z = mu + exp(0.5 * logvar) * eps.

        Z is the tuple (mu, logvar). Returns (eps, std, z); eps and std are
        kept because the backward pass needs them.
        """
        mean, log_var = Z
        std = np.exp(0.5 * log_var)
        epsilon = rng.standard_normal(size=mean.shape)
        z = mean + std * epsilon
        return epsilon, std, z
