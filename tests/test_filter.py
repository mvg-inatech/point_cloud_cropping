import numpy as np
import pytest

from common.filter import (
    filter_for_exponential,
    filter_for_gaussian,
    filter_for_linear,
    filter_for_range_box,
    filter_for_range_cylinder,
    filter_for_range_sphere,
    filter_prediction_knn,
    get_sub_idx_function,
    numba_fast_norm,
)


def test_numba_fast_norm_matches_numpy():
    point = np.array([1.0, 2.0, 3.0])
    xyz = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [1.0, 4.0, 3.0]])
    expected = np.linalg.norm(xyz - point, axis=1)
    result = numba_fast_norm(point, xyz)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("box", filter_for_range_box),
        ("exponential", filter_for_exponential),
        ("sphere", filter_for_range_sphere),
        ("gaussian", filter_for_gaussian),
        ("cylinder", filter_for_range_cylinder),
        ("linear", filter_for_linear),
    ],
)
def test_get_sub_idx_function_known(method, expected):
    assert get_sub_idx_function(method) is expected


def test_get_sub_idx_function_unknown():
    with pytest.raises(NotImplementedError):
        get_sub_idx_function("unknown")


def test_filter_for_range_box_basic():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.9, 0.9],
            [1.1, 0.0, 0.0],
            [0.0, -1.2, 0.0],
        ]
    )
    pos = np.array([0.0, 0.0, 0.0])
    idx, probabilities = filter_for_range_box(pts, range_box=1.0, pos=pos)

    expected_idx = np.array([0, 1])
    expected_dist = np.linalg.norm(pts[:, :3] - pos, axis=1)
    expected_prob = 1 - (expected_dist / 1.0)

    assert np.array_equal(np.sort(idx), expected_idx)
    assert np.allclose(probabilities, expected_prob)


def test_filter_for_range_sphere_median_pos():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )
    idx, probabilities = filter_for_range_sphere(pts, range_sphere=1.5, pos=None)

    mean = np.median(pts[:, :3], axis=0)
    dist = np.linalg.norm(pts[:, :3] - mean, axis=1)
    expected_idx = np.argwhere(dist < 1.5).flatten()
    expected_prob = 1 - (dist / 1.5)

    assert np.array_equal(np.sort(idx), np.sort(expected_idx))
    assert np.allclose(probabilities, expected_prob)


def test_filter_for_range_cylinder_ignores_z():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 10.0],
            [2.0, 0.0, 0.0],
        ]
    )
    pos = np.array([0.0, 0.0, 0.0])
    idx, probabilities = filter_for_range_cylinder(pts, range_cylinder=1.0, pos=pos)

    dist_xy = np.linalg.norm(pts[:, :2] - pos[:2], axis=1)
    expected_idx = np.argwhere(dist_xy < 1.0).flatten()
    expected_prob = 1 - (dist_xy / 1.0)

    assert np.array_equal(np.sort(idx), np.sort(expected_idx))
    assert np.allclose(probabilities, expected_prob)


def test_filter_for_exponential_deterministic():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    pos = np.array([0.0, 0.0, 0.0])
    lambda_p = 0.5

    np.random.seed(0)
    idx, probabilities = filter_for_exponential(pts, lambda_p=lambda_p, pos=pos)

    dist = np.linalg.norm(pts[:, :3] - pos, axis=1)
    expected_prob = np.exp(-lambda_p * dist)
    np.random.seed(0)
    expected_random = np.random.random(len(expected_prob))
    expected_idx = np.argwhere(expected_random < expected_prob).flatten()

    assert np.array_equal(idx, expected_idx)
    assert np.allclose(probabilities, expected_prob)


def test_filter_for_gaussian_deterministic():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    pos = np.array([0.0, 0.0, 0.0])
    std = 0.75

    np.random.seed(1)
    idx, probabilities = filter_for_gaussian(pts, std=std, pos=pos)

    dist = np.linalg.norm(pts[:, :3] - pos, axis=1)
    expected_prob = np.exp(-((dist / std) ** 2))
    np.random.seed(1)
    expected_random = np.random.random(len(expected_prob))
    expected_idx = np.argwhere(expected_random < expected_prob).flatten()

    assert np.array_equal(idx, expected_idx)
    assert np.allclose(probabilities, expected_prob)


def test_filter_for_linear_deterministic():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    pos = np.array([0.0, 0.0, 0.0])
    range_max = 2.5

    np.random.seed(2)
    idx, probabilities = filter_for_linear(pts, range_max=range_max, pos=pos)

    dist = np.linalg.norm(pts[:, :3] - pos, axis=1)
    expected_prob = (range_max - dist) / range_max
    np.random.seed(2)
    expected_random = np.random.random(len(expected_prob))
    expected_idx = np.argwhere(expected_random < expected_prob).flatten()

    assert np.array_equal(idx, expected_idx)
    assert np.allclose(probabilities, expected_prob)


def test_filter_prediction_knn_k1():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    full_probs = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    )
    labels = filter_prediction_knn(pts, full_probs, k=1)
    expected = np.argmax(full_probs, axis=1)
    assert np.array_equal(labels, expected)


def test_filter_methods_require_three_dimensions():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    pos = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        filter_for_range_sphere(pts, range_sphere=1.0, pos=pos)
