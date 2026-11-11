# TP5 — Deep Learning

Sistemas de Inteligencia Artificial (72.27), ITBA. Group of 6.

## 1. Autoencoders

### a) Basic autoencoder

Basic autoencoder for the 32 binary font patterns (7x5 pixels) in `data/font.h`.

1. Design an encoder and decoder architecture with a 2-dimensional latent space.
2. Describe and study the architectures and optimisation techniques used. Goal: learn all 32 patterns in a 2D latent space with at most 1 wrong pixel per pattern. If only a subset can be learned, show why the full set could not.
3. Plot the input data in the 2D latent space.
4. Show how the network can generate a new letter that is not in the training set.

### b) Denoising autoencoder

On the same dataset.

1. Choose a suitable architecture and explain the choice.
2. Corrupt the inputs at different noise levels and study how well the autoencoder removes the noise.

## 2. Variational Autoencoder (VAE)

a) Choose or create a new dataset (e.g. emojis).

b) Adapt the autoencoder to a variational scheme to solve the problem of the representation in the latent space.

c) Use the VAE to generate a new sample that clearly belongs to the dataset.

## Font data

`data/font.h` holds 32 patterns of 7 rows of 5 bits per character; bit 4 is the leftmost pixel.
