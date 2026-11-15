# Exercise 2 (VAE): decisions and trade-offs

A running log of every design decision made while building the variational
autoencoder. Each entry says what we chose, what else we considered, why, and
what we gave up. This is the source material for the presentation.

---

## D1. Dataset: hand-drawn fish from Google's Quick, Draw!

- **Chosen:** The "fish" category of Quick, Draw! (about 134,000 drawings of
  28×28 pixels made by people in an online game; licence CC BY 4.0).
- **How we got here (three versions):**
  1. *Emoji faces generated in code* (16×16, 5 eye types × 5 mouths ×
     3 extras = 75 images). Dropped because 75 artificial images is small for
     a generative model: a VAE learns a smoother latent space, and generates
     better samples, from many real examples.
  2. *Quick, Draw! "smiley face"* (real drawings, plenty of data). Dropped
     because we were told **not to use emojis** for the dataset, and smiley
     faces count as emojis.
  3. *Quick, Draw! "fish"*: same source, format and code as version 2; only
     the category changed.
- **Other alternatives considered:** real emoji designs (OpenMoji / Twemoji;
  ruled out with the emoji ban, and very varied colour images that turn into
  blobs when shrunk); MNIST digits (the classic VAE dataset, but very common in
  tutorials); Fashion-MNIST (grayscale detail is lost in black and white);
  other Quick, Draw! categories (house, star, tree) or a mix of several.
- **Why fish:** a simple shape (body + tail) that varies in clear ways
  (direction, body shape, fins, eye), so the latent space should show smooth
  variation, and it is easy to judge whether a generated drawing looks like a
  fish (task 2c).
- **Trade-off:** These are quick doodles, and some are messy or unfinished.
  Fish face both left and right; drawings that are halfway between the two
  may blend into odd shapes (e.g. a tail on both ends). The data comes from
  outside, so we credit the source.

## D2. Only add new files; never edit teammates' code

- **Chosen:** The VAE lives in new files (`scripts/make_fish_dataset.py`,
  `src/tp5/fish.py`, `src/tp5/vae.py`, `tests/test_fish.py`,
  `tests/test_vae.py`, `notebooks/03_vae.ipynb`). We *import* the activations
  and optimizers from `src/tp5/autoencoder.py` but do not change that file.
- **Alternatives:** Extend the existing `Autoencoder` class with a variational
  option.
- **Why:** Exercise 1 belongs to teammates, and the open denoising PR (#3)
  already changes `autoencoder.py`. Separate files avoid merge conflicts and
  keep the VAE easy to read on its own.
- **Trade-off:** Some forward/backward-pass logic is similar to the plain
  autoencoder and is written again instead of shared.

## D3. Camera demo postponed

- **Chosen:** No camera demo for now.
- **Idea for later:** Use the webcam either to "snap" a handwritten letter to
  the closest font character, or to redraw the live image as a mosaic of font
  characters.
- **Why:** It is not part of the assignment. We finish exercise 2 first.
- **Trade-off:** None for the grade; possibly a nice extra for the presentation
  if there is time.

## D4. Keep a fixed random subset of 2,000 drawings, without cleaning

- **Chosen:** `scripts/make_fish_dataset.py` downloads the full file (105 MB),
  picks 2,000 drawings at random with a fixed seed (0) and stores only those in
  `data/fish.npz` (106 KB, compressed).
- **Alternatives:** Use all 124,000 drawings; download the data inside the
  notebook every time; filter out bad drawings (e.g. ones without a face
  outline).
- **Why:** 2,000 images is plenty of variation for a 2D latent space while
  keeping training fast with our numpy network. Storing the small subset means
  teammates need no download, and the fixed seed means the script always
  recreates exactly the same file.
- **Trade-off:** We use about 1.5% of the available data, and we keep the messy
  drawings (some are scribbles or barely look like a fish), which adds noise.
  Filtering them would need a rule for what counts as a "good" fish, so we kept it
  simple.

## D5. Keep 28×28, convert to black and white

- **Chosen:** Keep the original 28×28 = 784 pixels and turn every grey value
  above 127 into black (1) and the rest into white (0). On average 110 of the
  784 pixels are black.
- **Alternatives:** Keep the grey values (0–255); shrink the images to 16×16.
- **Why:** Black and white means we can keep the same sigmoid output and binary
  cross-entropy loss as in exercise 1. Not resizing keeps the preprocessing to
  one line and keeps the strokes sharp.
- **Trade-off:** The soft anti-aliased edges of the strokes are lost, and 784
  inputs make the network bigger and slower than 16×16 images would.
