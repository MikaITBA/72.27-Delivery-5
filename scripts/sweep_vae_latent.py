"""Compare VAE latent sizes on a held-out set of fish.

Run from the repository root:
    python scripts/sweep_vae_latent.py [output_dir]

Takes about 5 to 10 minutes. Prints a table and writes sweep_latent.json and
sweep_latent_samples.png (10 random samples per latent size) to output_dir
(default: the current directory).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tp5.fish import load_fish  # noqa: E402
from tp5.vae_study import run_config, split, summarize  # noqa: E402

LATENT_DIMS = (2, 8, 16)
EPOCHS = 50
SEEDS = (0, 1)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)

X_train, X_val = split(load_fish().astype(float))
rows = []
for d in LATENT_DIMS:
    for r in run_config(X_train, X_val, latent_dim=d, epochs=EPOCHS, seeds=SEEDS):
        rows.append(r)
        print(
            f"latent {d} seed {r['seed']}: val recon {r['val_recon']:.1f} kl {r['val_kl']:.1f} "
            f"total {r['val_total']:.1f} active {r['active_units']} ({r['seconds']:.0f}s)",
            flush=True,
        )

print()
for s in summarize(rows, "latent_dim"):
    print(s)

with open(out / "sweep_latent.json", "w") as f:
    json.dump([{k: v for k, v in r.items() if k != "model"} for r in rows], f, indent=1)

rng = np.random.default_rng(1)
fig, axes = plt.subplots(len(LATENT_DIMS), 10, figsize=(10, 1.1 * len(LATENT_DIMS)))
for row, d in zip(axes, LATENT_DIMS):
    model = next(r["model"] for r in rows if r["latent_dim"] == d and r["seed"] == SEEDS[0])
    for ax, img in zip(row, model.decode(rng.standard_normal((10, d)))):
        ax.imshow(img.reshape(28, 28), cmap="binary", vmin=0, vmax=1)
        ax.axis("off")
    row[0].set_title(f"latent {d}", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(out / "sweep_latent_samples.png", dpi=150)
print("saved to", out.resolve())
