import numpy as np
from plyfile import PlyData
import laspy


############################################################################
# saving las stuff


def save_scalar_to_laspy(pts, scalar, scalar_name, filename: str):
    """
    Saves points and scalar values to a LAS file using laspy.
    :param pts: Nx3 numpy array of points
    :param scalar: Nx1 numpy array of scalar values
    :param scalar_name: Name of the scalar field to be saved
    :param filename: Path to the output LAS file
    """
    header = laspy.LasHeader(point_format=7)

    # header offset and scale have to be specified
    xmin = np.floor(np.min(pts[:, 0]))
    ymin = np.floor(np.min(pts[:, 1]))
    zmin = np.floor(np.min(pts[:, 2]))
    header.offset = [xmin, ymin, zmin]
    header.scale = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.add_extra_dim(
        laspy.ExtraBytesParams(name=scalar_name, type=np.uint64, description="HUI")
    )

    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    try:
        las.red = pts[:, 3].astype(np.uint8)
        las.green = pts[:, 4].astype(np.uint8)
        las.blue = pts[:, 5].astype(np.uint8)
    except:
        print("Unable to write colorinformation")

    setattr(las, scalar_name, scalar)
    las.write(filename)


def save_dict_to_laspy(data_dict: dict, output_path: str):
    """
    Save a dictionary of point cloud data to a LAS file using laspy.

    Parameters:
    - data_dict: A dictionary where keys are attribute names and values are numpy arrays of shape (N,).
    - output_path: The path to save the LAS file.
    """
    header = laspy.LasHeader(point_format=7)

    pts = data_dict.pop("coords")
    xmin = np.floor(np.min(pts[:, 0]))
    ymin = np.floor(np.min(pts[:, 1]))
    zmin = np.floor(np.min(pts[:, 2]))
    header.offset = [xmin, ymin, zmin]
    header.scale = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)

    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]

    if "colors" in data_dict:
        colors = data_dict.pop("colors")
        scale = 255.0 / np.max(colors)
        colors = (colors * scale).astype(np.uint8)
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

    for key, value in data_dict.items():
        if value.ndim != 1 or value.shape[0] != pts.shape[0]:
            raise ValueError(
                f"Attribute {key} must be a 1D array with the same length as coords."
            )
        if not hasattr(las, key):
            las.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=key,
                    type=value.dtype,
                    description="HUI",
                )
            )
        setattr(las, key, value)

    las.write(output_path)


def save_list_to_laspy(
    pts: np.ndarray,
    scalar_list: list,
    scalar_names: list,
    scalar_types: list,
    filename: str,
):
    """
    Saves a point cloud with a scalar list to a LAS file using laspy
    :param pts: Nx3 array of points
    :param lst: List of scalar values (length N)
    :param scalar_name: Name of the scalar field
    :param filename: Output LAS file path
    """

    assert (
        len(scalar_list) == len(scalar_names) == len(scalar_types)
    ), "Length of scalar_list, scalar_names and scalar_types must be the same"

    header = laspy.LasHeader(point_format=7)
    xmin = np.floor(np.min(pts[:, 0]))
    ymin = np.floor(np.min(pts[:, 1]))
    zmin = np.floor(np.min(pts[:, 2]))
    header.offset = [xmin, ymin, zmin]
    header.scale = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)

    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    try:
        las.red = pts[:, 3].astype(np.uint8)
        las.green = pts[:, 4].astype(np.uint8)
        las.blue = pts[:, 5].astype(np.uint8)
    except:
        print("Unable to write colorinformation")

    for i in range(len(scalar_list)):
        if not hasattr(las, scalar_names[i]):
            las.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=scalar_names[i],
                    type=scalar_types[i],
                    description="HUI",
                )
            )
        setattr(las, scalar_names[i], scalar_list[i])

    las.write(filename)
