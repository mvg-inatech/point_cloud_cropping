import numpy as np
import random

from common.geometry import rotate_point_cloud_yaw
from dataset.utils import dict_from_idx

######################################################################
# Point Augmentation


class RandomPointJitter(object):
    """
    Jitters the data by a uniform distributed random amount
        return data_dict
    """

    def __init__(self, jitter=((-1, 1), (-1, 1), (-1, 1))):
        self.jitter = jitter

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            jitter_x = random.uniform(self.jitter[0][0], self.jitter[0][1])
            jitter_y = random.uniform(self.jitter[1][0], self.jitter[1][1])
            jitter_z = random.uniform(self.jitter[2][0], self.jitter[2][1])
            data_dict["coords"][:, 0] += jitter_x
            data_dict["coords"][:, 1] += jitter_y
            data_dict["coords"][:, 2] += jitter_z
        return data_dict


class RandomPointDrop(object):
    """
    Randomly drops points from the data
    """

    def __init__(self, drop_rate=0.2):
        self.drop_rate = drop_rate

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            num_points = len(data_dict["coords"])
            if num_points == 0:
                return data_dict
            keep_count = int(num_points * (1 - self.drop_rate))
            keep_count = max(1, min(num_points, keep_count))
            points_to_keep = np.random.choice(
                num_points,
                size=keep_count,
                replace=False,
            )
            data_dict = dict_from_idx(data_dict, points_to_keep)

        return data_dict


class RandomPointScale(object):
    """
    Randomly scale the point cloud.
    """

    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            scan = data_dict["coords"]
            factor = random.uniform(self.scale_range[0], self.scale_range[1])
            scan[:, :3] = scan[:, :3] * factor
        return data_dict


class RandomPointRotateZ(object):
    """
    Randomly rotate the point cloud arround z
    """

    def __init__(self, angle_range=(-180, 180)):
        self.angle_range = angle_range

    def __call__(self, data_dict):
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        if "coords" in data_dict.keys():
            data_dict["coords"] = rotate_point_cloud_yaw(data_dict["coords"], angle)
        if "normals" in data_dict.keys():
            data_dict["normals"] = rotate_point_cloud_yaw(data_dict["normals"], angle)
        return data_dict


class RandomPointFlip(object):
    """
    Randomly flip the point cloud (only x and y)
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            if random.random() < self.prob:
                data_dict["coords"][:, 0] = -data_dict["coords"][:, 0]
                if "normals" in data_dict.keys():
                    data_dict["normals"][:, 0] = -data_dict["normals"][:, 0]
            if random.random() < self.prob:
                data_dict["coords"][:, 1] = -data_dict["coords"][:, 1]
                if "normals" in data_dict.keys():
                    data_dict["normals"][:, 1] = -data_dict["normals"][:, 1]
        return data_dict


class GaussianPointNoise(object):
    """
    Add Gaussian noise to the point cloud
    """

    def __init__(self, sigma=0.005):
        self.sigma = sigma

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            noise = np.random.normal(0, self.sigma, data_dict["coords"].shape)
            data_dict["coords"] += noise
        return data_dict


######################################################################
# Color Augmentation


class GaussianColorNoise(object):
    """
    Add Gaussian noise to the point cloud
    """

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, data_dict):
        if "colors" in data_dict.keys():
            noise = np.random.normal(0, self.sigma, data_dict["colors"].shape)
            data_dict["colors"] += noise
        return data_dict


class ChromaticColorTranslation(object):
    """
    Add chromatic color translation to the point cloud
    """

    def __init__(self, ratio=0.2):
        self.ratio = ratio

    def __call__(self, data_dict):
        if "colors" in data_dict.keys():
            tr = (np.random.rand(1, 3) - 0.5) * 2 * self.ratio
            data_dict["colors"] = np.clip(data_dict["colors"] + tr, 0, 1)
        return data_dict


class RandomColorDrop(object):
    """
    Drop random colors or intensity values from the point cloud
    """

    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate

    def __call__(self, data_dict):
        if "colors" in data_dict.keys():
            num_colors = len(data_dict["colors"])
            drop_count = int(num_colors * self.drop_rate)
            if drop_count <= 0 or num_colors == 0:
                return data_dict
            drop_count = min(num_colors, drop_count)
            colors_to_drop = np.random.choice(
                num_colors,
                size=drop_count,
                replace=False,
            )
            data_dict["colors"][colors_to_drop] *= 0
        if "intensity" in data_dict.keys():
            num_intensity = len(data_dict["intensity"])
            drop_count = int(num_intensity * self.drop_rate)
            if drop_count <= 0 or num_intensity == 0:
                return data_dict
            drop_count = min(num_intensity, drop_count)
            intensity_to_drop = np.random.choice(
                num_intensity,
                size=drop_count,
                replace=False,
            )
            data_dict["intensity"][intensity_to_drop] *= 0
        return data_dict


######################################################################
# Normal Augmentation


class RandomNormalDrop(object):
    """
    Randomly drops normals from the data
    """

    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate

    def __call__(self, data_dict):
        if "normals" in data_dict.keys():
            num_normals = len(data_dict["normals"])
            drop_count = int(num_normals * self.drop_rate)
            if drop_count <= 0 or num_normals == 0:
                return data_dict
            drop_count = min(num_normals, drop_count)
            normals_to_drop = np.random.choice(
                num_normals,
                size=drop_count,
                replace=False,
            )
            data_dict["normals"][normals_to_drop] *= 0
        return data_dict


class GaussianNormalNoise(object):
    """
    Add Gaussian noise to the normals
    """

    def __init__(self, sigma=0.005):
        self.sigma = sigma

    def __call__(self, data_dict):
        if "normals" in data_dict.keys():
            noise = np.random.normal(0, self.sigma, data_dict["normals"].shape)
            data_dict["normals"] += noise
        return data_dict
