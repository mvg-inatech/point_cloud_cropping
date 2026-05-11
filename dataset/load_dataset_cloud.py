"""
Specific loaders for point cloud datasets. Each loader should return a dictionary with the following keys, depending on the dataset and the available data.
The final naming:
coords: [N, 3] with xyz
colors: [N, 3] with rgb
normals: [N, 3] with normal vectors
labels: [N] with int labels
intensity: [N] with float intensity/reflectance values
"""

import numpy as np
from plyfile import PlyData
import laspy

############################################################################
# General loader function


def get_pc_loader(dataset_name: str):
    if dataset_name == "paris_lille_3d":
        return load_paris_lille_3d_cloud
    elif dataset_name == "semantic_bridge":
        return load_semantic_bridge_cloud
    elif dataset_name == "s3dis":
        return load_s3dis_cloud
    elif dataset_name == "toronto_3d":
        return load_toronto_3d_cloud
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


############################################################################
# Dataset-specific loaders


def load_paris_lille_3d_cloud(file_path: str) -> dict:
    """
    Load a paris lille 3D (pl3d) point cloud from a .ply file and return it as a dictionary.
    Can be directly used on the PL3D dataset without preprocessing of the .ply files.
    """
    data_dict = {}
    with open(file_path, "rb") as quad_file:
        plydata = PlyData.read(quad_file)
    coords = np.zeros((len(plydata.elements[0].data["x"]), 3))
    coords[:, 0] = plydata.elements[0].data["x"]
    coords[:, 1] = plydata.elements[0].data["y"]
    coords[:, 2] = plydata.elements[0].data["z"]
    data_dict["coords"] = coords.astype(np.float32)

    if "class" in plydata.elements[0].data.dtype.names:
        labels = np.asarray(plydata.elements[0]["class"]).astype(np.int64)
        data_dict["labels"] = labels

    intensity = np.asarray(plydata.elements[0]["reflectance"]).astype(np.float32)
    data_dict["intensity"] = intensity
    return data_dict


def load_semantic_bridge_cloud(file_path: str) -> dict:
    """
    Load a semantic bridge point cloud from a .ply file and return it as a dictionary.
    Can be directly used on the SemanticBridge dataset without preprocessing of the .ply files.
    """
    data_dict = {}
    with open(file_path, "rb") as quad_file:
        plydata = PlyData.read(quad_file)

    coords = np.zeros((len(plydata.elements[0].data["x"]), 3))
    coords[:, 0] = plydata.elements[0].data["x"]
    coords[:, 1] = plydata.elements[0].data["y"]
    coords[:, 2] = plydata.elements[0].data["z"]
    data_dict["coords"] = coords.astype(np.float32)

    colors = np.zeros((len(plydata.elements[1].data["red"]), 3))
    colors[:, 0] = plydata.elements[1].data["red"]
    colors[:, 1] = plydata.elements[1].data["green"]
    colors[:, 2] = plydata.elements[1].data["blue"]
    data_dict["colors"] = colors.astype(np.uint8)

    if "label" in plydata.header:
        labels = plydata.elements[2].data["label"].astype(np.int64)
        data_dict["labels"] = labels
    return data_dict


def load_s3dis_cloud(file_path: str) -> dict:
    """
    Load a S3DIS point cloud from a .las file and return it as a dictionary.
    Can NOT be directly used on the S3DIS dataset without preprocessing the original .txt files.
    """
    data_dict = {}
    with laspy.open(file_path) as fh:
        data = fh.read()

    coords = np.zeros((len(data), 3))
    coords[:, 0] = data.x
    coords[:, 1] = data.y
    coords[:, 2] = data.z
    data_dict["coords"] = coords.astype(np.float32)

    colors = np.zeros((len(data), 3))
    colors[:, 0] = data.red
    colors[:, 1] = data.green
    colors[:, 2] = data.blue
    data_dict["colors"] = colors.astype(np.uint8)

    if hasattr(data, "label"):
        labels = data.label.astype(np.int64)
        data_dict["labels"] = labels

    if hasattr(data, "normal_x"):
        normals = np.zeros((len(data), 3))
        normals[:, 0] = data.normal_x
        normals[:, 1] = data.normal_y
        normals[:, 2] = data.normal_z
        data_dict["normals"] = normals.astype(np.float32)
    return data_dict


def load_toronto_3d_cloud(file_path: str) -> dict:
    """
    Load a Toronto-3D point cloud from a .ply file and return it as a dictionary.
    Can be directly used on the Toronto-3D dataset without preprocessing of the .ply files.
    NaNs are replaced with 0.0 for intensity and colors, as they can contain NaN values in the original dataset.
    The coords are centered, as they are in UTM coordinates and can have very large values, which can lead
    to numerical instability during training.
    """
    UTM_OFFSET = [627285, 4841948, 0]
    data_dict = {}
    with open(file_path, "rb") as quad_file:
        plydata = PlyData.read(quad_file)

    coords = np.zeros((len(plydata.elements[0].data["x"]), 3))
    coords[:, 0] = plydata.elements[0].data["x"]
    coords[:, 1] = plydata.elements[0].data["y"]
    coords[:, 2] = plydata.elements[0].data["z"]

    coords -= np.array(UTM_OFFSET, dtype=np.float32)
    data_dict["coords"] = coords.astype(np.float32)

    colors = np.zeros((len(plydata.elements[0].data["red"]), 3))
    colors[:, 0] = plydata.elements[0].data["red"]
    colors[:, 1] = plydata.elements[0].data["green"]
    colors[:, 2] = plydata.elements[0].data["blue"]
    np.nan_to_num(colors, copy=False, nan=0.0)
    data_dict["colors"] = colors.astype(np.uint8)

    intensity = np.asarray(plydata.elements[0]["scalar_Intensity"]).astype(np.float32)
    np.nan_to_num(intensity, copy=False, nan=0.0)
    data_dict["intensity"] = intensity

    if "scalar_Label" in plydata.elements[0].data.dtype.names:
        labels = np.asarray(plydata.elements[0]["scalar_Label"]).astype(np.int64)
        data_dict["labels"] = labels
    return data_dict
