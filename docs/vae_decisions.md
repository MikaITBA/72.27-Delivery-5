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

---

# Experiment (branch `exp/vae-cartoon-colour`): colour cartoon faces

## D6. Look for a more consistent dataset than the fish

- **Problem:** Fish face both left and right, and some drawings contain two
  fish or are scribbles. The latent space mixes these, and many generated
  samples come out as noise.
- **Tried first:** Other Quick, Draw! categories (apple, cup, mushroom, light
  bulb, house, umbrella; `scripts/compare_quickdraw_categories.py`). Umbrella
  and mushroom looked the most consistent.
- **Experiment:** Google's Cartoon Set (cartoon faces drawn by a program from
  18 attributes; CC BY 4.0). Every face is centred, the same size and looks
  straight ahead, so the data is perfectly "stable". Google's own download no
  longer allows anonymous access, so we use the identical copy on Hugging Face.
- **Open question:** The team was told not to use emojis. Cartoon avatar faces
  are not emojis, but they are close, so this must be checked with the teacher.

## D7. Colour 32×32, 2,000 faces, cropped border

- **Chosen:** `scripts/make_cartoon_dataset.py` crops the empty border (every
  face lies inside the same 453×453 square), shrinks to 32×32 and keeps the
  colour: 32 × 32 × 3 = 3,072 inputs. 2,000 faces with seed 0, like the fish.
  The 18 attributes of each face are stored as well.
- **Alternatives:** Black and white at 28×28 (the faces disappear: only eyes
  and glasses survive, because the drawings are coloured areas, not pen
  strokes); grey 28×28 or 32×32 (recognisable, but loses hair and skin colour).
- **Why:** Colour is what makes the faces recognisable. The VAE needs no code
  change: the cross-entropy works with any value between 0 and 1.
- **Trade-off:** About 4× more inputs than the fish, so training is about 4×
  slower (≈0.25 s per epoch). The loss numbers are summed over 3,072 values and
  cannot be compared with the fish.

## D8. Feed the network "ink" (0 = paper), not red/green/blue

- **Problem:** On red/green/blue values the VAE collapsed: validation recon
  805, KL ≈ 1, no active latent unit. Every face was decoded as the same
  average face. Even with beta = 0 (no KL at all) training sat on that plateau
  for 13 epochs, so the cause was the input, not the KL term.
- **Chosen:** `load_cartoons()` returns 1 − RGB, the amount of cyan, magenta
  and yellow ink, as a printer does. White background becomes 0, the same
  convention as the fish (0 = paper, 1 = pen). `show_images` converts back.
- **Results** (60 epochs, 256-64, 2D latent, one seed):

  | input | activation | val recon | KL | active units |
  |---|---|---|---|---|
  | RGB | tanh | 805.5 | 1.3 | 0 |
  | RGB | relu | 703.4 | 6.2 | 2 |
  | ink | tanh | 649.8 | 7.0 | 2 |
  | ink | tanh + KL warm-up | 648.3 | 7.1 | 2 |
  | ink | relu | 640.7 | 7.1 | 2 |

- **Why:** With a white background 60% of the 3,072 inputs are a constant 1,
  which drowns the small differences between faces in the first layer. As ink,
  only the face itself feeds the network.
- **Trade-off:** One extra conversion when plotting. KL warm-up turned out not
  to be needed once the input was fixed.

## D9. Train one epoch at a time (progress bar, optional KL warm-up)

- **Chosen:** `src/tp5/vae_training.py` calls the inherited `fit()` for one
  epoch at a time with the same Adam optimizer, prints a progress bar with the
  current recon and KL, and can raise beta from 0 to 1 over the first epochs
  (KL warm-up).
- **Why:** Training on 3,072 inputs takes minutes, and we want to see that it
  is moving. `fit()` lives in the teammates' `autoencoder.py`, which we do not
  change (D2), so we wrap it instead.
- **Trade-off:** `fit()` computes the loss on the whole training set after
  every epoch, which costs about one extra forward pass per epoch.

## D10. Has training matured? Epochs vs. more data vs. a larger latent space

- **Question:** With 2,000 faces, latent 2 and 150 epochs (validation recon
  627), would more epochs or more data give better faces?
- **Experiment:** All 10,000 faces processed the same way, the same 500
  held-out faces for every run, validation loss recorded after every epoch
  (`train(..., X_val=...)`). ReLU, 256-64, one seed (reruns give identical
  numbers).

  | run | epoch 50 | epoch 100 | epoch 150 | epoch 300 |
  |---|---|---|---|---|
  | 1,500 faces, latent 2 | 654.5 | 636.2 | 631.4 | 624.0 |
  | 4,500 faces, latent 2 | 629.3 | 615.3 | 609.2 | – |
  | 9,500 faces, latent 2 | 611.5 | 599.3 | – | – |
  | 1,500 faces, latent 8 | 585.2 | 576.7 | 574.2 | 570.1 |

  (validation recon + KL, averaged over 5 epochs; lower is better)
- **Findings:**
  - *Epochs:* mostly matured. 150 → 300 epochs gains only 631 → 624 for
    twice the time; no overfitting (the best epoch was 292 of 300).
  - *More data:* a real gain, not only from more training steps. At the same
    total amount of training (450,000 face-passes) 9,500 faces reach 612 vs.
    624 for 1,500. The train/validation gap shrinks from 18 to under 1.
    Pictures: hair colours come out right more often, and the edges of the
    latent space decode to clean faces instead of noise.
  - *Latent size:* the biggest gain. Latent 8 with only 1,500 faces (570)
    beats latent 2 with 9,500 (599). Its reconstructions keep beards, hair
    styles and sunglasses, and its generated faces are far more varied. Two
    numbers per face are the real bottleneck: with latent 2 every face is
    the same "generic face" recoloured.
- **Next (not done yet):** use all 10,000 faces; a latent 2 model for the
  latent-space plots and a latent 8 model for generating new faces (2c), as
  in notebook 03 for the fish.
- **Trade-off:** 10,000 faces make training about 5× slower per epoch and
  the data file about 12 MB instead of 2.4 MB. A latent 8 space can no
  longer be plotted directly.
