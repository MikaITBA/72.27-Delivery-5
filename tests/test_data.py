import numpy as np

from tp5.data import _DEFAULT_PATH, add_noise, load_font, pixel_errors

FONT_SAMPLE = """\
static const unsigned char font[][7] = {
   {0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00},   // 0x60, `
   {0x00, 0x0e, 0x01, 0x0d, 0x13, 0x13, 0x0d},   // 0x61, a
};
"""


def test_load_font_parses_bits(tmp_path):
    path = tmp_path / "font.h"
    path.write_text(FONT_SAMPLE)
    X, chars = load_font(path)
    assert X.shape == (2, 35)
    assert chars == ["`", "a"]
    # 0x04 = 00100 -> middle pixel of the first row
    assert X[0, :5].tolist() == [0, 0, 1, 0, 0]
    # 0x13 = 10011 -> bit 4 is the leftmost pixel
    assert X[1, 20:25].tolist() == [1, 0, 0, 1, 1]


def test_load_font_real_file():
    X, chars = load_font(_DEFAULT_PATH)
    assert X.shape == (32, 35)
    assert len(chars) == 32
    assert set(np.unique(X)) <= {0, 1}


def test_add_noise_zero_and_one():
    X = np.random.default_rng(0).integers(0, 2, size=(32, 35))
    assert np.array_equal(add_noise(X, 0.0), X)
    assert np.array_equal(add_noise(X, 1.0), 1 - X)


def test_add_noise_rate():
    X = np.zeros((1000, 35), dtype=int)
    noisy = add_noise(X, 0.1, rng=np.random.default_rng(0))
    assert set(np.unique(noisy)) <= {0, 1}
    assert abs(noisy.mean() - 0.1) < 0.01
    assert pixel_errors(X, noisy).sum() == noisy.sum()
