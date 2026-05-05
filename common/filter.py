import numpy as np
from sklearn.neighbors import NearestNeighbors
from numba import jit, prange

#################################################################
# sub cloud get method

def get_sub_idx_function(sub_cloud_method):
    """Get the sub cloud filtering function based on the dataset configuration."""

    if sub_cloud_method == "box":
        get_sub_idx = filter_for_range_box
    elif sub_cloud_method == "exponential":
        get_sub_idx = filter_for_exponential
    elif sub_cloud_method == "sphere":
        get_sub_idx = filter_for_range_sphere
    elif sub_cloud_method == "gaussian":
        get_sub_idx = filter_for_gaussian
    elif sub_cloud_method == "cylinder":
        get_sub_idx = filter_for_range_cylinder
    elif sub_cloud_method == "linear":
        get_sub_idx = filter_for_linear
    else:
        raise NotImplementedError(
            f"Sub cloud method {sub_cloud_method} not implemented. "
            f"Please choose from [box, exponential, sphere, gaussian, cylinder]"
        )
    return get_sub_idx


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_fast_norm(point, xyz):
    """
    Compute squared Euclidean distances from each point in xyz to the given point.
    """
    n_points = xyz.shape[0]
    n_dims = xyz.shape[1]
    result = np.empty(n_points, dtype=np.float64)

    for i in prange(n_points):
        sum_sq = 0.0
        for j in range(n_dims):
            diff = xyz[i, j] - point[j]
            sum_sq += diff * diff
        result[i] = np.sqrt(sum_sq)

    return result


#################################################################
# sub cloud calculation main function


def filter_for_range_box(
    pts: np.ndarray,
    range_box: float,
    pos: np.ndarray = None,
) -> np.ndarray:
    """
    Returns indicies of all points within given range relative to position,
    if given or mean otherwise
    Args:
        pts (np.ndarray) : pointcloud points as a `np.array` [n, 3]
        range_box (float): range in which points should be selected
        pos (np.ndarray): position of scanner for instance
    Returns:
        - idx (np.ndarray): idx of points in defined range as a `np.array` [n]
        - dist (np.ndarray): distances of points from the position as a `np.array` [n]
    """
    # 3D case
    if pts.shape[1] < 3:
        raise ValueError("Input points must have at least 3 dimensions (x, y, z).")

    xyz = pts[:, :3]
    if pos is not None:
        mean = pos.reshape((3))
    else:
        mean = np.median(xyz, axis=0)

    x_filt = np.logical_and(
        (xyz[:, 0] < mean[0] + range_box),
        (xyz[:, 0] > mean[0] - range_box),
    )
    y_filt = np.logical_and(
        (xyz[:, 1] < mean[1] + range_box),
        (xyz[:, 1] > mean[1] - range_box),
    )
    z_filt = np.logical_and(
        (xyz[:, 2] < mean[2] + range_box),
        (xyz[:, 2] > mean[2] - range_box),
    )

    filter = np.logical_and(x_filt, y_filt)
    filter = np.logical_and(filter, z_filt)
    idx = np.argwhere(filter).flatten()

    dist = np.linalg.norm(xyz - mean, axis=1)
    probabilities = 1 - (dist / range_box)

    return idx, probabilities


def filter_for_range_sphere(
    pts: np.ndarray,
    range_sphere: float,
    pos: np.ndarray = None,
) -> np.ndarray:
    """
    Returns indicies of all points within given range relative to position,
    if given or mean otherwise
    Args:
        pts (np.ndarray) : pointcloud points as a `np.array`
        range_sphere (float): range in which points should be selected
        pos (np.ndarray): position of scanner for instance
    Returns:
        - idx (np.ndarray): idx of points in defined range as a `np.array` [n]
        - dist (np.ndarray): distances of points from the position as a `np.array` [n]
    """
    # 3D case
    if pts.shape[1] < 3:
        raise ValueError("Input points must have at least 3 dimensions (x, y, z).")

    xyz = pts[:, :3]
    if pos is not None:
        mean = pos.reshape((3))
    else:
        mean = np.median(xyz, axis=0)

    dist = np.linalg.norm(xyz - mean, axis=1)
    idx = np.argwhere(dist < range_sphere).flatten()

    probabilities = 1 - (dist / range_sphere)
    return idx, probabilities


def filter_for_range_cylinder(
    pts: np.ndarray,
    range_cylinder: float,
    pos: np.ndarray = None,
) -> np.ndarray:
    """
    Returns indicies of all points within given cylinder range relative to position,
    if given or mean otherwise. Omitting z-axis!!
    Args:
        pts (np.ndarray) : pointcloud points as a `np.array`
        range_cylinder (float): range in which points should be selected
        pos (np.ndarray): position of scanner for instance
    Returns:
        - idx (np.ndarray): idx of points in defined range as a `np.array` [n]
        - dist (np.ndarray): distances of points from the position as a `np.array` [n]
    """
    if pts.shape[1] < 3:
        raise ValueError("Input points must have at least 3 dimensions (x, y, z).")

    xyz = pts[:, :3]
    if pos is not None:
        mean = pos.reshape((3))
    else:
        mean = np.median(xyz, axis=0)

    dist = np.linalg.norm(xyz[:, :2] - mean[:2], axis=1)
    idx = np.argwhere(dist < range_cylinder).flatten()

    probabilities = 1 - (dist / range_cylinder)

    return idx, probabilities


def filter_for_exponential(
    pts: np.ndarray,
    lambda_p: float,
    pos: np.ndarray = None,
) -> np.ndarray:
    """
    Returns indicies of all points within given range relative to position,
    if given or mean otherwise
    Args:
        pts (np.ndarray) : pointcloud points as a `np.array`
        lambda_p (float): lambda parameter for exponential filtering
        pos (np.ndarray): position of scanner for instance
    Returns:
        - idx (np.ndarray): idx of points in defined range as a `np.array` [n]
        - probabilities (np.ndarray): probabilities of points being kept as a `np.array` [n]
    """
    # 3D case
    if pts.shape[1] < 3:
        raise ValueError("Input points must have at least 3 dimensions (x, y, z).")

    xyz = pts[:, :3]
    if pos is not None:
        mean = pos.reshape((3))
    else:
        mean = np.median(xyz, axis=0)

    distance = np.linalg.norm(xyz - mean, axis=1)
    probabilities = np.exp(-lambda_p * distance)
    random_vals = np.random.random(len(probabilities))
    keep_mask = random_vals < probabilities  # Keep if random < drop_probability
    idx = np.argwhere(keep_mask).flatten()

    return idx, probabilities


def filter_for_gaussian(
    pts: np.ndarray,
    std: float,
    pos: np.ndarray = None,
) -> np.ndarray:
    """
    Returns indicies of all points within given range relative to position,
    if given or mean otherwise
    Args:
        pts (np.ndarray) : pointcloud points as a `np.array`
        std (float): standard deviation for gaussian filtering
        pos (np.ndarray): position of scanner for instance
    Returns:
        - idx (np.ndarray): idx of points in defined range as a `np.array` [n]
        - probabilities (np.ndarray): probabilities of points being kept as a `np.array` [n]
    """
    # 3D case
    if pts.shape[1] < 3:
        raise ValueError("Input points must have at least 3 dimensions (x, y, z).")

    xyz = pts[:, :3]
    if pos is not None:
        mean = pos.reshape((3))
    else:
        mean = np.median(xyz, axis=0)

    distance = np.linalg.norm(xyz - mean, axis=1)
    probabilities = np.exp(-((distance / std) ** 2))
    random_vals = np.random.random(len(probabilities))
    keep_mask = random_vals < probabilities  # Keep if random < drop_probability
    idx = np.argwhere(keep_mask).flatten()

    return idx, probabilities


def filter_for_linear(
    pts: np.ndarray,
    range_max: float,
    pos: np.ndarray = None,
) -> np.ndarray:
    """
    Returns indicies of all points within given range relative to position,
    if given or mean otherwise
    Args:
        pts (np.ndarray) : pointcloud points as a `np.array`
        range (float): max allowed range for linear filtering
        pos (np.ndarray): position of scanner for instance
    Returns:
        - idx (np.ndarray): idx of points in defined range as a `np.array` [n]
        - probabilities (np.ndarray): probabilities of points being kept as a `np.array` [n]
    """
    # 3D case
    if pts.shape[1] < 3:
        raise ValueError("Input points must have at least 3 dimensions (x, y, z).")

    xyz = pts[:, :3]
    if pos is not None:
        mean = pos.reshape((3))
    else:
        mean = np.median(xyz, axis=0)

    distance = np.linalg.norm(xyz - mean, axis=1)
    probabilities = (range_max - distance) / range_max
    random_vals = np.random.random(len(probabilities))
    keep_mask = random_vals < probabilities  # Keep if random < drop_probability
    idx = np.argwhere(keep_mask).flatten()

    return idx, probabilities


#################################################################
# other filtering stuff


def filter_prediction_knn(
    pts: np.ndarray,
    full_probs: np.ndarray,
    k=16,
) -> np.ndarray:
    """Use knn to filter the prediction by taking the most common class in the hood.

    Args:
        pts (np.ndarray): point cloud points as a `np.array` [n, 3]
        full_probs (np.ndarray): prediction probabilities as a `np.array` [n, c]
        k (int, optional): number of nearest neighbors to consider. Defaults to 16.

    Returns:
        np.ndarray: prediction labels as a `np.array` [n]
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(pts)
    _, indices = nbrs.kneighbors(pts)
    pp = full_probs[indices]
    pp = np.argmax(pp.sum(1), axis=1)
    return pp
