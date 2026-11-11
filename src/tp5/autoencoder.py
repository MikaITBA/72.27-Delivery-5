"""Multilayer perceptron autoencoder implemented from scratch with numpy.

The network is one stack of dense layers; the layer that outputs the latent
code splits it into an encoder and a decoder. Training uses backpropagation
with a choice of optimizer (SGD, momentum or Adam).
"""

import numpy as np

# ------------------------------------------------------------------
# Activations: (function, derivative expressed in terms of the output)
# ------------------------------------------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


ACTIVATIONS = {
    "sigmoid": (_sigmoid, lambda a: a * (1 - a)),
    "tanh": (np.tanh, lambda a: 1 - a**2),
    "relu": (lambda z: np.maximum(z, 0), lambda a: (a > 0).astype(float)),
    "identity": (lambda z: z, lambda a: np.ones_like(a)),
}


# ------------------------------------------------------------------
# Optimizers
# ------------------------------------------------------------------


class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    def __init__(self, lr=0.05, beta=0.9):
        self.lr, self.beta = lr, beta
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.beta
            v -= self.lr * g
            p += v


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}


# ------------------------------------------------------------------
# Autoencoder
# ------------------------------------------------------------------


class Autoencoder:
    """Dense autoencoder with a mirrored decoder and a sigmoid output layer.

    Args:
        n_inputs: Size of each input vector (35 for the font).
        hidden: Encoder hidden layer sizes, e.g. [20, 10]. The decoder uses
            the same sizes in reverse order.
        latent_dim: Size of the latent code.
        activation: Activation of the hidden layers.
        latent_activation: Activation of the latent layer.
        seed: Seed for the weight initialisation.
    """

    def __init__(
        self,
        n_inputs=35,
        hidden=(20, 10),
        latent_dim=2,
        activation="tanh",
        latent_activation="identity",
        seed=None,
    ):
        rng = np.random.default_rng(seed)
        hidden = list(hidden)
        sizes = [n_inputs, *hidden, latent_dim, *hidden[::-1], n_inputs]
        acts = (
            [activation] * len(hidden)
            + [latent_activation]
            + [activation] * len(hidden)
            + ["sigmoid"]
        )
        self.latent_index = len(hidden)  # layer whose output is the latent code
        self.activations = acts
        self.weights, self.biases = [], []
        for n_in, n_out, act in zip(sizes[:-1], sizes[1:], acts):
            # He init for ReLU, Xavier/Glorot for the rest
            scale = np.sqrt(2 / n_in) if act == "relu" else np.sqrt(1 / n_in)
            self.weights.append(rng.normal(0, scale, (n_in, n_out)))
            self.biases.append(np.zeros(n_out))

    @property
    def params(self):
        return self.weights + self.biases

    def _forward(self, X, start=0, stop=None):
        """Run layers [start, stop) and return the output of every layer."""
        stop = len(self.weights) if stop is None else stop
        outputs = [X]
        for W, b, act in zip(
            self.weights[start:stop], self.biases[start:stop], self.activations[start:stop]
        ):
            outputs.append(ACTIVATIONS[act][0](outputs[-1] @ W + b))
        return outputs

    def encode(self, X):
        return self._forward(np.atleast_2d(X), stop=self.latent_index + 1)[-1]

    def decode(self, Z):
        return self._forward(np.atleast_2d(Z), start=self.latent_index + 1)[-1]

    def reconstruct(self, X):
        return self._forward(np.atleast_2d(X))[-1]

    def loss(self, X, X_hat=None):
        """Mean binary cross-entropy per pixel."""
        X_hat = self.reconstruct(X) if X_hat is None else X_hat
        X_hat = np.clip(X_hat, 1e-12, 1 - 1e-12)
        return -np.mean(X * np.log(X_hat) + (1 - X) * np.log(1 - X_hat))

    def gradients(self, X_in, X_target):
        """Backpropagate the binary cross-entropy loss.

        X_in and X_target differ for the denoising autoencoder; for the plain
        autoencoder both are the clean letters.
        """
        outputs = self._forward(X_in)
        # sigmoid + cross-entropy: dL/dz of the output layer is (y_hat - y)
        delta = (outputs[-1] - X_target) / X_target.size
        grads_W, grads_b = [], []
        for i in reversed(range(len(self.weights))):
            grads_W.append(outputs[i].T @ delta)
            grads_b.append(delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.weights[i].T) * ACTIVATIONS[self.activations[i - 1]][1](
                    outputs[i]
                )
        return grads_W[::-1] + grads_b[::-1]

    def fit(
        self,
        X,
        epochs=10_000,
        optimizer=None,
        batch_size=None,
        log_every=100,
        stop_at_pixels=None,
        seed=None,
    ):
        """Train on X and return a history dict.

        Args:
            X: Training data, shape (n_samples, n_inputs), values 0/1.
            epochs: Number of passes over the data.
            optimizer: Optimizer instance; defaults to Adam(lr=1e-3).
            batch_size: Mini-batch size; None means full batch.
            log_every: Record loss and pixel errors every this many epochs.
            stop_at_pixels: Stop early once every pattern has at most this
                many wrong pixels.
            seed: Seed for shuffling.

        Returns:
            Dict with lists "epoch", "loss" and "max_pixel_error".
        """
        optimizer = Adam() if optimizer is None else optimizer
        rng = np.random.default_rng(seed)
        n = len(X)
        batch_size = n if batch_size is None else batch_size
        history = {"epoch": [], "loss": [], "max_pixel_error": []}
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = X[order[start : start + batch_size]]
                optimizer.step(self.params, self.gradients(batch, batch))
            if epoch % log_every == 0 or epoch == epochs:
                X_hat = self.reconstruct(X)
                max_err = int(np.max(np.sum(np.round(X_hat) != X, axis=1)))
                history["epoch"].append(epoch)
                history["loss"].append(self.loss(X, X_hat))
                history["max_pixel_error"].append(max_err)
                if stop_at_pixels is not None and max_err <= stop_at_pixels:
                    break
        return history
