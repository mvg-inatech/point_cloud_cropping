from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy as np
from functools import partial
from collections import defaultdict

from common.filter import (
    filter_for_range_box,
    filter_for_range_sphere,
    filter_for_range_cylinder,
)
from common.voxelize import voxelize


#################################################################
# simple idx holding sub cloud class


class SubCloud:
    def __init__(self, file_name, center, idx, probabilities):
        self.file_name = file_name
        self.center = center
        self.idx = idx
        self.probabilities = probabilities


#################################################################
# grid overlay for center calculations


def create_3d_grid_overlay(points, box_size):
    """
    Create 3D grid overlay and return centers of occupied boxes.
    Args:
        points (np.ndarray): Input point cloud of shape (N, 3).
        box_size (float or tuple): Size of the grid boxes. If a float is provided,
                                   it is used for all dimensions. If a tuple,
                                   it should be (dx, dy, dz).
    Returns:
        - occupied_centers (np.ndarray): Centers of occupied boxes of shape (M, 3).
        - grid_dict (dict): Dictionary with grid cell indices as keys and list of
                            point indices as values.
    """

    # Handle box size input
    if isinstance(box_size, (int, float)):
        dx, dy, dz = box_size, box_size, box_size
    else:
        dx, dy, dz = box_size

    # Get point cloud bounds
    min_coords = np.min(points, axis=0)

    # Calculate grid indices for each point
    grid_indices_x = np.floor((points[:, 0] - min_coords[0]) / dx).astype(int)
    grid_indices_y = np.floor((points[:, 1] - min_coords[1]) / dy).astype(int)
    grid_indices_z = np.floor((points[:, 2] - min_coords[2]) / dz).astype(int)

    # Group points by grid cell
    grid_dict = defaultdict(list)
    for i, (gx, gy, gz) in enumerate(
        zip(grid_indices_x, grid_indices_y, grid_indices_z)
    ):
        grid_dict[(gx, gy, gz)].append(i)

    # Calculate centers of occupied boxes
    occupied_centers = []
    for gx, gy, gz in grid_dict.keys():
        center_x = min_coords[0] + (gx + 0.5) * dx
        center_y = min_coords[1] + (gy + 0.5) * dy
        center_z = min_coords[2] + (gz + 0.5) * dz
        occupied_centers.append([center_x, center_y, center_z])

    return np.array(occupied_centers), dict(grid_dict)


#################################################################
# sub cloud calculation main function

    
def calculate_sub_clouds(
    pts,
    file_name,
    selected_range,
    filter_method,
    min_appearance=1,
    min_pts=500,
):
    sub_clouds = []
    potentials = np.zeros((len(pts)))
    current_pos = pts[np.random.randint(0, len(potentials)), ...]
    while min(potentials) <= min_appearance:
        # probabilty is distance for box and sphere
        # for exponential and gaussian its the probability of being kept
        idx_pts, probabilty = filter_method(
            pts[:, :3],
            selected_range.get_lambda(),
            current_pos,
        )
        if len(idx_pts) > min_pts:
            sub_clouds.append(SubCloud(file_name, current_pos, idx_pts, probabilty))
        if filter_method in [
            filter_for_range_box,
            filter_for_range_sphere,
            filter_for_range_cylinder,
        ]:
            potentials[idx_pts] += 1
        else:
            potentials += probabilty
        choosen_idx = np.random.choice(np.where(potentials == np.min(potentials))[0])
        current_pos = pts[choosen_idx, :3]
    return sub_clouds


def calculate_sub_clouds_grid(
    data_dict,
    file_name,
    selected_range,
    grid_size,
    filter_method,
    min_pts=500,
):
    """
    Calculate sub clouds based on 3D grid overlay.
    Takes availaible CPUS/2 for processing clouds...
    """
    pts = data_dict["coords"]
    idx_voxel, _ = voxelize(pts, voxel_size=0.5)
    center_list, _ = create_3d_grid_overlay(pts[idx_voxel], grid_size)

    process_func = partial(
        _process_single_center,
        pts=pts,
        selected_range=selected_range,
        filter_method=filter_method,
        min_pts=min_pts,
        file_name=file_name,
    )

    num_workers = int(multiprocessing.cpu_count() / 2)

    # Parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_func, center_list))

    # Filter None values
    sub_clouds = [sc for sc in results if sc is not None]
    return sub_clouds


def _process_single_center(
    center,
    pts,
    selected_range,
    filter_method,
    min_pts,
    file_name,
):
    """Helper function for parallel processing"""
    idx_pts, probabilty = filter_method(
        pts[:, :3],
        selected_range,
        center,
    )

    if len(idx_pts) > min_pts:
        return SubCloud(
            file_name,
            center,
            idx_pts,
            probabilty,
        )
    return None
