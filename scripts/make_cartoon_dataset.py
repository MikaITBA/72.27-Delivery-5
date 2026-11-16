"""Create data/cartoon_faces.npz from Google's Cartoon Set (10k version).

Downloads the archive (477 MB, kept in results/ so it is only downloaded
once), picks a fixed random subset of 2,000 faces, and turns each 500x500
colour PNG into a small 32x32 colour image. The 18 attributes that generated
each face (hair, glasses, face shape, ...) are stored too.

Run from the project root:
    uv run python scripts/make_cartoon_dataset.py

Data: Cartoon Set by Google, https://google.github.io/cartoonset/, licensed
under CC BY 4.0. Google's own download link no longer allows anonymous
downloads, so we use the identical copy on Hugging Face.
"""

import csv
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://huggingface.co/datasets/cgarciae/cartoonset/resolve/main/data/cartoonset10k.tgz"
DOWNLOAD_PATH = ROOT / "results" / "cartoonset_raw" / "cartoonset10k.tgz"  # results/ is not committed
OUTPUT_PATH = ROOT / "data" / "cartoon_faces.npz"
NUMBER_OF_IMAGES = 2000  # same as the fish, so training times stay comparable
SEED = 0  # fixed seed, so everyone gets exactly the same subset
IMAGE_SIZE = 32  # final images are IMAGE_SIZE x IMAGE_SIZE x 3 (red, green, blue)

# Every face lies inside x 48..447 and y 7..460 of the 500x500 picture
# (measured on 1,000 faces). Cutting off the empty border first gives the
# face more of the 32x32 pixels.
CROP_BOX = (22, 7, 475, 460)  # left, top, right, bottom: a 453x453 square


def show_progress(done, total, label):
    """Print a one-line progress bar that overwrites itself, e.g. 'faces [#####-----]  50%'."""
    width = 30
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if done >= total else ""  # move to a new line once finished
    print(f"\r{label} [{bar}] {100 * done / total:3.0f}% ({done}/{total})", end=end, flush=True)


def download_progress(blocks_done, block_size, total_size):
    """Called by urlretrieve after each downloaded block; shows the progress in MB."""
    total_mb = total_size // 1_000_000
    done_mb = min(blocks_done * block_size, total_size) // 1_000_000
    show_progress(done_mb, total_mb, "download (MB)")


def to_small_colour_image(png_bytes):
    """Turn one 500x500 RGBA PNG into a (32, 32, 3) array of values 0-255."""
    face = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    # the background is transparent; put the face on white, like the paper of a drawing
    white = Image.new("RGBA", face.size, (255, 255, 255, 255))
    face = Image.alpha_composite(white, face).convert("RGB")
    face = face.crop(CROP_BOX).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return np.asarray(face)


def read_attributes(csv_bytes):
    """Return {attribute name: value} from one face's csv ("name", value, number of options)."""
    rows = csv.reader(io.StringIO(csv_bytes.decode()), skipinitialspace=True)
    return {name: int(value) for name, value, _ in rows}


def main():
    if not DOWNLOAD_PATH.exists():
        print(f"Downloading {SOURCE_URL} ...")
        DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, DOWNLOAD_PATH, reporthook=download_progress)

    # A .tgz can only be read efficiently from front to back: jumping back to
    # an earlier file makes Python unpack everything again from the start.
    # So we go through the archive twice, front to back, and never jump.
    # Nothing is extracted to disk; the files are read straight from the archive.

    # Pass 1: list the file names and choose the faces.
    print("Pass 1: listing the files in the archive ...", flush=True)
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as archive:
        all_names = archive.getnames()
    png_names = sorted(name for name in all_names if name.endswith(".png"))
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(png_names), size=NUMBER_OF_IMAGES, replace=False))
    chosen_faces = [png_names[i].removesuffix(".png") for i in chosen]  # e.g. "cartoonset10k/cs1000..."
    wanted = {face + ".png" for face in chosen_faces} | {face + ".csv" for face in chosen_faces}

    # Pass 2: walk through the archive once and keep the bytes of the chosen files.
    print("Pass 2: reading the chosen faces ...", flush=True)
    file_bytes = {}
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as archive:
        for done, member in enumerate(archive, start=1):
            if member.name in wanted:
                file_bytes[member.name] = archive.extractfile(member).read()
            if done % 200 == 0 or done == len(all_names):  # redraw now and then, not every file
                show_progress(done, len(all_names), "archive files")

    # Turn the bytes into small images and attribute values, in sorted name order.
    images = [to_small_colour_image(file_bytes[face + ".png"]) for face in chosen_faces]
    attributes = [read_attributes(file_bytes[face + ".csv"]) for face in chosen_faces]

    attribute_names = list(attributes[0])
    np.savez_compressed(
        OUTPUT_PATH,
        # one row per face: the 32 rows of 32 pixels of (red, green, blue), flattened
        images=np.array(images, dtype=np.uint8).reshape(NUMBER_OF_IMAGES, -1),
        attributes=np.array([[a[name] for name in attribute_names] for a in attributes]),
        attribute_names=np.array(attribute_names),
    )
    print(f"Saved {NUMBER_OF_IMAGES} faces to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
