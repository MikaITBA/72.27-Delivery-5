import numpy as np

from tp5.fish import IMAGE_SIZE, load_fish


def test_shape_and_values():
    images = load_fish()
    assert images.shape == (2000, IMAGE_SIZE * IMAGE_SIZE)
    assert set(np.unique(images)) == {0, 1}


def test_every_image_has_a_drawing():
    # an empty image would teach the network nothing
    images = load_fish()
    assert images.sum(axis=1).min() > 0
