import numpy as np

from tp5.cartoon import IMAGE_SHAPE, load_attributes, load_cartoons


def test_shape_and_values():
    images = load_cartoons()
    assert images.shape == (2000, np.prod(IMAGE_SHAPE))
    assert images.min() >= 0 and images.max() <= 1


def test_every_image_has_a_face():
    # an image without any ink (all zeros) would mean the face got lost
    images = load_cartoons()
    assert (images > 0).sum(axis=1).min() > 0


def test_background_is_white_paper():
    # the corners are background: no ink, so the VAE input there is 0
    images = load_cartoons().reshape(-1, *IMAGE_SHAPE)
    assert images[:, 0, 0, :].max() == 0


def test_attributes_match_images():
    values, names = load_attributes()
    assert values.shape == (2000, len(names))
    assert "hair" in names and "glasses" in names
