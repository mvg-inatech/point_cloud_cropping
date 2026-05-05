"""Originally from
https://github.com/Pointcept/PointTransformerV2/blob/5386c4d71f3d6c42c24a8105fce8750e9355dc54/pcr/datasets/transform.py
"""

import numpy as np


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def voxelize(coords, voxel_size=0.05):
    assert coords.shape[1] in [2, 3], "Input coordinates must be 2D or 3D"
    discrete_coords = np.floor((coords - np.min(coords, axis=0)) / np.array(voxel_size))
    key = fnv_hash_vec(discrete_coords)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[0:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]
    return idx_unique, discrete_coords[idx_unique]


def voxelize_each_point(coords, voxel_size=0.05):
    assert coords.shape[1] in [2, 3], "Input coordinates must be 2D or 3D"
    discrete_coords = np.floor((coords - np.min(coords, axis=0)) / np.array(voxel_size))
    key = fnv_hash_vec(discrete_coords)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    idx_unique = []
    discrete_coords_list = []
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_unique.append(idx_sort[idx_select])
        discrete_coords_list.append(discrete_coords[idx_unique[-1]])
    return idx_unique, discrete_coords_list
